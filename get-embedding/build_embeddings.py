"""对 MM-SafetyBench 采样数据抽取「样本级 feature」并落盘。

每条数据的流程（参考 vlm-hidden-state-inspect-demo）：
  1. 用 load_mm_safetybench_sample 拿到 image_path / question / 类目；
  2. 构造 (image, question) 的多模态输入；
  3. 生成 response；
  4. 对「输入 + 生成」的完整序列做一次前向，取指定层的 hidden states；
  5. 只截取输入部分 [0, input_len) 的 token 向量做平均 → 本条数据的 feature。

输出（默认写到 get-embedding/outputs/，文件名 stem 由 image_type 和 --tag 决定）：
  <stem>.npz          主产物，自包含：features 矩阵 + 5 个等长元数据数组
  <stem>.jsonl        可读镜像，每行一条记录（含 feature），字段顺序与 npz 一致
  <stem>.manifest.json 本次运行的配置与统计，用于复现
  <stem>.partial.jsonl 增量检查点（跑到最后会保留，删掉即从头重跑）

用法示例：
  python build_embeddings.py --sample-size 200
  python build_embeddings.py --layer -4 --dtype bf16 --tag layer4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# load_data.py 在仓库根目录，脚本所在目录是它的子目录，直接跑脚本时根目录不在 sys.path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from load_data import MMImageTypes, load_mm_safetybench_sample

from model_runner import (  # noqa: E402
    DTYPES,
    extract_hidden_states,
    generate,
    load_model,
    mean_pool_input,
    prepare_inputs,
)

DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEFAULT_LOCAL_MODEL_DIR = (
    r"C:\Users\kzlin\.cache\huggingface\hub"
    r"\models--Qwen--Qwen3.5-0.8B\snapshots\2fc06364715b967f1860aea9cf38778875588b17"
)

# 落盘的字段顺序，npz 与 jsonl 共用
FIELDS = ("index", "image_path", "question", "unsafe_category", "unsafe_category_id", "response")

IMAGE_TYPE_BY_VALUE: dict[str, MMImageTypes] = {t.value: t for t in MMImageTypes}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抽取 MM-SafetyBench 样本级 feature",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        default=str(REPO_ROOT / "datasets" / "MM-SafetyBench"),
        help="MM-SafetyBench 根目录（含 imgs/ 与 processed_questions/）",
    )
    parser.add_argument(
        "--image-type",
        default=MMImageTypes.IMAGE_WITH_TEXT.value,
        choices=sorted(IMAGE_TYPE_BY_VALUE),
        help="SD=纯图, SD_TYPO=图上叠字, TYPO=纯文字图",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="按类别比例分层抽样的条数；不传则用全量 1680 条",
    )
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HF Hub ID 或模型名")
    parser.add_argument(
        "--local-model-dir",
        default=DEFAULT_LOCAL_MODEL_DIR,
        help="本地快照目录，存在则优先本地加载；传空字符串则强制走在线/缓存",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=sorted(DTYPES),
        help="加载精度；fp16 出 NaN 时换 bf16",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="取 hidden_states 的哪一层，支持负数；-1 是最后一层。该模型共 25 项(embedding + 24 层)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="生成 response 的最大新 token 数")

    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="输出目录",
    )
    parser.add_argument("--tag", default="", help="附加到输出文件名 stem 之后的标签")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有检查点，从头重跑",
    )
    return parser.parse_args(argv)


def output_stem(args: argparse.Namespace) -> str:
    stem = f"MMSB_{args.image_type.lower()}"
    if args.tag:
        stem += f"_{args.tag}"
    return stem


def read_checkpoint(path: Path) -> dict[int, dict]:
    """读增量检查点，返回 {index: record}。文件不存在或行损坏时尽力而为。"""
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done[int(record["index"])] = record
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"  [warn] 检查点有一行无法解析，已跳过: {path}", file=sys.stderr)
    return done


def append_checkpoint(fp, record: dict) -> None:
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    fp.flush()
    os.fsync(fp.fileno())


def save_npz(path: Path, records: list[dict]) -> None:
    """features 存成 (N, D) float32 矩阵，元数据存成等长的 numpy 数组。

    字符串用 numpy 的 unicode dtype 而不是 object dtype，这样 np.load 不需要
    allow_pickle=True。
    """
    features = np.stack([np.asarray(r["feature"], dtype=np.float32) for r in records])

    arrays: dict[str, np.ndarray] = {"features": features}
    for field in FIELDS:
        if field == "unsafe_category_id":
            arrays[field] = np.asarray([r[field] for r in records], dtype=np.int64)
        elif field == "index":
            arrays[field] = np.asarray([r[field] for r in records], dtype=np.int64)
        else:
            arrays[field] = np.asarray([r[field] for r in records])

    np.savez(path, **arrays)


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    dataset_path = os.path.abspath(args.dataset_path)
    image_type = IMAGE_TYPE_BY_VALUE[args.image_type]
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = output_stem(args)
    npz_path = output_dir / f"{stem}.npz"
    jsonl_path = output_dir / f"{stem}.jsonl"
    manifest_path = output_dir / f"{stem}.manifest.json"
    ckpt_path = output_dir / f"{stem}.partial.jsonl"

    dataset = load_mm_safetybench_sample(
        dataset_path=dataset_path,
        image_type=image_type,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"数据集: {dataset_path}")
    print(f"图片类型: {image_type.name} ({image_type.value})")
    print(f"样本数: {len(dataset)} (sample_size={args.sample_size}, seed={args.seed})")

    if args.no_resume and ckpt_path.exists():
        ckpt_path.unlink()
        print(f"检查点: --no-resume，已清空 {ckpt_path}")
    done = {} if args.no_resume else read_checkpoint(ckpt_path)
    if done:
        print(f"检查点: 已有 {len(done)} 条，将跳过这些 index 继续跑 ({ckpt_path})")

    todo = [i for i in range(len(dataset)) if i not in done]
    if not todo:
        print("所有样本都已在检查点里，直接汇总落盘。")

    processor = model = None
    if todo:
        dtype = DTYPES[args.dtype]
        print(f"\n加载模型: {args.model_name}")
        print(f"本地目录: {args.local_model_dir or '(未指定，走在线下载 / 本地缓存)'}")
        processor, model, source, is_local = load_model(
            args.model_name,
            hf_token=os.environ.get("HF_TOKEN"),
            dtype=dtype,
            local_dir=args.local_model_dir or None,
        )
        print(f"加载来源: {'本地目录' if is_local else '在线 (HF Hub / 缓存)'} → {source}")
        print(f"设备: {model.device}  精度: {args.dtype}  取层: {args.layer}")

    failures: list[dict] = []
    hidden_entries: int | None = None
    feature_dim: int | None = None

    if todo:
        with ckpt_path.open("a", encoding="utf-8") as fp:
            for i in tqdm(todo, desc="抽取 feature", unit="条"):
                item = dataset[i]
                try:
                    inputs = prepare_inputs(
                        processor, model, item["question"], image=item["image_path"]
                    )
                    input_len = int(inputs["input_ids"].shape[-1])

                    outputs, response, _ = generate(
                        model, processor, inputs, max_new_tokens=args.max_new_tokens
                    )

                    hidden, hidden_entries = extract_hidden_states(
                        model, inputs, outputs, layer=args.layer
                    )
                    feature = mean_pool_input(hidden, input_len)

                    if not np.isfinite(feature).all():
                        raise ValueError(
                            f"feature 含非有限值 (NaN/Inf)，input_len={input_len}；"
                            f"试试 --dtype bf16"
                        )

                    record = {
                        "index": i,
                        "image_path": item["image_path"],
                        "question": item["question"],
                        "unsafe_category": item["unsafe_category"],
                        "unsafe_category_id": int(item["unsafe_category_id"]),
                        "response": response,
                        "feature": feature.tolist(),
                    }
                    append_checkpoint(fp, record)
                    done[i] = record
                except Exception as exc:  # noqa: BLE001 - 单条失败不该中断整轮
                    failures.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
                    tqdm.write(f"  [fail] index={i}: {type(exc).__name__}: {exc}")

    if not done:
        print("没有任何成功样本，不落盘。", file=sys.stderr)
        return 1

    records = [done[i] for i in sorted(done)]
    if len(records) != len(dataset):
        print(
            f"\n注意: 只拿到 {len(records)}/{len(dataset)} 条，落盘的产物是不完整的。"
            f"重跑本脚本会从检查点续跑。",
            file=sys.stderr,
        )

    # 维度从记录里推，这样「全部命中检查点、本轮没跑前向」时 manifest 也有值；
    # 同时挡住「检查点混了不同模型/不同层的结果」这种会让 save_npz 报晦涩错误的情况
    feature_dims = {len(r["feature"]) for r in records}
    if len(feature_dims) != 1:
        print(
            f"feature 维度不一致 {sorted(feature_dims)}：检查点里混了不同模型/不同层的运行结果，"
            f"删掉 {ckpt_path} 重跑。",
            file=sys.stderr,
        )
        return 1
    feature_dim = feature_dims.pop()

    save_npz(npz_path, records)
    save_jsonl(jsonl_path, records)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": dataset_path,
        "image_type": image_type.name,
        "image_type_value": image_type.value,
        "sample_size_arg": args.sample_size,
        "seed": args.seed,
        "num_samples_in_dataset": len(dataset),
        "num_records": len(records),
        "num_failed": len(failures),
        "failures": failures,
        "model_name": args.model_name,
        "local_model_dir": args.local_model_dir or None,
        "dtype": args.dtype,
        "layer": args.layer,
        "hidden_states_entries": hidden_entries,
        "pooling": "mean over hidden[0, 0:input_len, :] in float32 (包含图片占位 token)",
        "max_new_tokens": args.max_new_tokens,
        "feature_dim": feature_dim,
        "fields": list(FIELDS) + ["feature"],
        "outputs": {
            "npz": npz_path.name,
            "jsonl": jsonl_path.name,
            "checkpoint": ckpt_path.name,
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _version_of("transformers"),
            "numpy": np.__version__,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"\n完成 {len(records)} 条，feature 维度 {feature_dim}")
    print(f"  {npz_path}")
    print(f"  {jsonl_path}")
    print(f"  {manifest_path}")
    return 0


def _version_of(pkg: str) -> str:
    try:
        module = __import__(pkg)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return "not installed"


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
