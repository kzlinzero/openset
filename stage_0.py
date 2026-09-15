#!/usr/bin/env python3
# GPU_PROBE_VERSION: 2026-09-12 | feature extraction + Ridge training on cuda:0
"""Frozen InternVL response mean pooling, GPU Ridge probes, dev-MSE ranking.

Layer numbers are 1-based decoder BLOCK outputs, before the final model RMSNorm.
See README_stage_0.md for installation, layer choices, and cache semantics.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
from pathlib import Path

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

# 控制当前的实验版本，防止旧实验结果和新实验代码混用
CACHE_VERSION = 1

# Defaults paths for this machine; command-line flags can override them.
MODEL_PATH = Path("/home/user/models/InternVL3_5-8B")
TRAIN_PATH = Path("/home/user/dataset/HelpSteer/train.jsonl/train.jsonl")
VALIDATION_PATH = Path("/home/user/dataset/HelpSteer/validation.jsonl/val.jsonl")

# 1.候选层 spec指定层号 preset预设策略
def parse_layers(spec: str | None, total: int, preset: str = "middle-dense") -> list[int]:
    if spec:
        layers = set()
        for item in spec.split(","):
            parts = item.strip().split("-")
            if len(parts) == 1:
                layers.add(int(parts[0]))
            elif len(parts) == 2:
                start, end = map(int, parts)
                if start > end:
                    raise ValueError(f"层区间起点大于终点：{item}")
                layers.update(range(start, end + 1))
            else:
                raise ValueError(f"无法解析层编号：{item}")
    elif preset == "all":
        layers = set(range(1, total + 1))
    else:
    
        low, high = max(1, total // 3), max(1, math.ceil(total * 0.75) - 1)
        layers = set(range(low, high + 1))
        if preset == "middle-dense":
            layers.update(max(1, round(total * x / 36)) for x in (8, 10, 28, 30, 32, 36))
    if not layers or min(layers) < 1 or max(layers) > total:
        raise ValueError(f"候选层必须在 1..{total} 内，得到 {sorted(layers)}")
    return sorted(layers)

# 2.解析数据
def read_records(path: Path, attributes: list[str], limit: int | None, seed: int) -> list[dict]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".jsonl", ".ndjson", ".json"})
    else:
        raise FileNotFoundError(f"数据路径不存在：{path}；请设置 --train-path / --validation-path")
    if not files:
        raise ValueError(f"目录内没有 JSONL/JSON 文件：{path}")
    records = []
    for file in files:
        with file.open(encoding="utf-8-sig") as stream:
            first = next((char for char in iter(lambda: stream.read(1), "") if not char.isspace()), "")
            stream.seek(0)
            try:
                entries = (enumerate(json.load(stream), 1) if first == "[" else
                           ((i, json.loads(line)) for i, line in enumerate(stream, 1) if line.strip()))
                for line, record in entries:
                    location = f"{file}:{line}"
                    if not isinstance(record, dict):
                        raise ValueError(f"{location} 必须是 JSON 对象")
                    for name in ("prompt", "response"):
                        if not isinstance(record.get(name), str) or not record[name].strip():
                            raise ValueError(f"{location} 缺少非空字符串字段 {name}")
                    labels = []
                    for name in attributes:
                        value = record.get(name)
                        if isinstance(value, bool) or not isinstance(value, (float, int)):
                            raise ValueError(f"{location} 的 {name} 必须是数值")
                        if not math.isfinite(value) or not 0 <= value <= 4:
                            raise ValueError(f"{location} 的 {name} 必须是 0..4 的有限评分")
                        labels.append(float(value))
                    records.append({"prompt": record["prompt"], "response": record["response"],
                                    "labels": labels, "source": location})
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 解析失败：{file}: {exc}") from exc
    if not records:
        raise ValueError(f"数据集为空：{path}")
    if limit is not None and limit < len(records):
        indices = sorted(random.Random(seed).sample(range(len(records)), limit))
        records = [records[i] for i in indices]
    return records

# 工具函数：管理数据以及模型版本
def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def model_signature(path: Path) -> str:
    # No need to read ~17 GB of weights merely to validate the feature cache.
    files = sorted(p for p in path.rglob("*") if p.is_file() and
                   p.suffix in {".json", ".py", ".safetensors", ".bin", ".model", ".txt"})
    return digest([(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns) for p in files])


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

# 3.编码数据，prompt+response一同编码
def encode_record(tokenizer, template, record: dict, max_length: int, max_prompt_tokens: int) -> dict:
    conversation = template.copy()
    conversation.messages = []
    conversation.append_message(conversation.roles[0], record["prompt"])
    conversation.append_message(conversation.roles[1], None)
    prefix = conversation.get_prompt()
    # One tokenization handles BPE boundary overlaps; no response EOS is appended.
    encoded = tokenizer(prefix + record["response"], add_special_tokens=False,
                        return_offsets_mapping=True, truncation=False)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    boundary = len(prefix)
    response_start = next((i for i, (_, end) in enumerate(offsets) if end > boundary), None)
    if response_start is None:
        raise ValueError(f"没有可提取的 response token：{record['source']}")
    prompt_ids, response_ids = ids[:response_start], ids[response_start:]
    kept_prompt = prompt_ids[-min(max_prompt_tokens, max_length - 1):]
    kept_response = response_ids[:max_length - len(kept_prompt)]
    specials = set(tokenizer.all_special_ids)
    response_mask = [False] * len(kept_prompt) + [token not in specials for token in kept_response]
    if not any(response_mask):
        raise ValueError(f"截断后没有普通 response token：{record['source']}")
    return {"input_ids": kept_prompt + kept_response, "response_mask": response_mask,
            "prompt_truncated": len(kept_prompt) < len(prompt_ids),
            "response_truncated": len(kept_response) < len(response_ids)}

# 只对回答的隐藏状态进行处理，取平均
def pool_response(hidden, response_mask):
    import torch

    weights = response_mask.to(torch.float32).unsqueeze(-1)
    denominator = weights.sum(dim=1)
    if (denominator == 0).any():
        raise ValueError("response mask 为空")
    return (hidden.float() * weights).sum(dim=1) / denominator

# 取不同层的隐藏状态，注册hook,相当于告诉这个layer每一次forward之后都调用一下我指定的函数
class LayerCollector:
    """Retain only [batch, hidden] vectors; never retain all token states."""

    def __init__(self, decoder, layers):
        self.mask = None
        self.values = {}
        self.handles = [decoder.layers[layer - 1].register_forward_hook(self._hook(layer)) for layer in layers]

    def _hook(self, layer):
        def collect(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.values[layer] = pool_response(hidden, self.mask).to(device="cpu").numpy()
        return collect

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_backbone(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；RTX 5090 请安装 requirements.txt 中的 CUDA 12.8 PyTorch")
    if device.type == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16；可用 --dtype float16")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                             use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("需要 fast tokenizer 的 offset_mapping 来准确识别 response token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 缺少 pad_token 和 eos_token")
    print("CPU 上载入本地 InternVL，然后仅将语言 decoder 放到指定 GPU...", flush=True)
    wrapper = AutoModel.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True,
        device_map={"": "cpu"}, use_flash_attn=False, trust_remote_code=True, local_files_only=True,
    ).eval()
    if not hasattr(wrapper, "conv_template") or not hasattr(wrapper, "language_model"):
        raise ValueError("此脚本面向 InternVL3_5-8B 原始格式（非 -HF 格式）")
    template = wrapper.conv_template.copy()
    template.system_message = wrapper.system_message
    decoder = wrapper.language_model.model
    # Qwen3 chooses attention from config on each forward. InternVL forces eager
    # when flash-attn is disabled, so override that here to use native PyTorch SDPA.
    decoder.config._attn_implementation = "sdpa"
    decoder.config.use_cache = False
    decoder.requires_grad_(False)
    del wrapper  # Release vision tower, projector, and vocabulary LM head.
    gc.collect()
    return decoder.to(device).eval(), tokenizer, template


def valid_cache(directory: Path, expected: dict) -> bool:
    import numpy as np

    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata["identity"] != expected:
        raise ValueError(f"{directory} 缓存与本次配置不同；请指定新的 --output-dir，避免误用旧特征")
    n, h = metadata["num_samples"], metadata["hidden_size"]
    try:
        for layer in expected["layers"]:
            x = np.load(directory / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
            if x.shape != (n, h) or x.dtype != np.float16:
                return False
        y = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
        return n > 0 and y.shape == (n, len(expected["attributes"])) and bool(np.isfinite(y).all())
    except (OSError, ValueError):
        return False

# 特征提取阶段，准备训练所需要的X(layer_.npy)和Y(labels.npy)
def extract_split(decoder, tokenizer, template, records, directory, identity, args):
    import numpy as np
    import torch
    from tqdm import tqdm

    directory.mkdir(parents=True, exist_ok=True)
    # Metadata is written only after a COMPLETE split, allowing failed splits to restart.
    (directory / "metadata.json").unlink(missing_ok=True)
    hidden_size = decoder.config.hidden_size
    arrays = {layer: np.lib.format.open_memmap(directory / f"layer_{layer:02d}.npy", mode="w+",
              dtype=np.float16, shape=(len(records), hidden_size)) for layer in identity["layers"]}
    collector = LayerCollector(decoder, identity["layers"])
    stats = {"prompt_truncated": 0, "response_truncated": 0, "response_tokens": 0}
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(records), args.batch_size), desc=f"提取 {directory.name}"):
                batch = [encode_record(tokenizer, template, record, args.max_length, args.max_prompt_tokens)
                         for record in records[start:start + args.batch_size]]
                length = max(len(row["input_ids"]) for row in batch)
                ids = torch.full((len(batch), length), tokenizer.pad_token_id, dtype=torch.long, device=args.device)
                attention = torch.zeros_like(ids)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for i, row in enumerate(batch):
                    size = len(row["input_ids"])
                    ids[i, :size] = torch.tensor(row["input_ids"], device=args.device)
                    attention[i, :size] = 1
                    mask[i, :size] = torch.tensor(row["response_mask"], device=args.device)
                    for key in ("prompt_truncated", "response_truncated"):
                        stats[key] += int(row[key])
                    stats["response_tokens"] += sum(row["response_mask"])
                collector.mask = mask
                collector.values.clear()
                # No LM head logits, KV cache, or autograd graph.真的forward
                decoder(input_ids=ids, attention_mask=attention, use_cache=False,
                        output_hidden_states=False, output_attentions=False, return_dict=True)
                if set(collector.values) != set(arrays):
                    raise RuntimeError("部分候选层 hook 未执行")
                for layer, array in arrays.items():
                    values = collector.values[layer]
                    if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(np.float16).max:
                        raise ValueError(f"第 {layer} 层的表示无法安全保存为 FP16")
                    array[start:start + len(batch)] = values
    finally:
        collector.close()
        for array in arrays.values():
            array.flush()
    np.save(directory / "labels.npy", np.asarray([row["labels"] for row in records], dtype=np.float32))
    with (directory / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps({"source": row["source"], "sha256": digest(row)}, ensure_ascii=False) + "\n")
    write_json(directory / "metadata.json", {"identity": identity, "num_samples": len(records),
                                            "hidden_size": hidden_size, "stats": stats})
    print(f"{directory.name}: {len(records)} 条，截断统计 {stats}", flush=True)

# 各种指标的计算，mse是5个属性的平均
def regression_metrics(target, prediction, attributes):
    import numpy as np

    result = {}
    for column, attribute in enumerate(attributes):
        y, p = target[:, column].astype(np.float64), prediction[:, column].astype(np.float64)
        residual = y - p
        variance = np.sum((y - y.mean()) ** 2)
        denom = np.linalg.norm(y - y.mean()) * np.linalg.norm(p - p.mean())
        result[attribute] = {
            "mse": float(np.mean(residual ** 2)),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1 - np.sum(residual ** 2) / variance) if variance > 0 else None,
            "pearson": float(np.dot(y - y.mean(), p - p.mean()) / denom) if denom > 0 else None,
        }
    return {"mean_mse": float(np.mean([x["mse"] for x in result.values()])),
            "per_attribute": result}

# 训练某一层的线性映射，ridge probe岭回归，加上了一个L2正则项
def fit_ridge_torch(train_x, train_y, alpha):
    """Fit independent Ridge targets on the tensors' device, with an unpenalized intercept."""
    import torch

    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("Ridge alpha 必须为正有限数")
    if train_x.ndim != 2 or train_y.ndim != 2 or len(train_x) != len(train_y) or len(train_x) < 2:
        raise ValueError("Probe 需要至少两条样本及匹配的二维 X/Y")
    if train_x.device != train_y.device or train_x.dtype != train_y.dtype:
        raise ValueError("Probe X/Y 必须位于同一设备且 dtype 相同")
    with torch.inference_mode():
        # All statistics are fit on train only. var_mean uses a stable variance reduction.
        variance, feature_mean = torch.var_mean(train_x, dim=0, correction=0)
        feature_scale = variance.clamp_min(0).sqrt()
        feature_scale = torch.where(feature_scale > 0, feature_scale, torch.ones_like(feature_scale))
        # In-place normalization bounds memory; caller owns this disposable tensor.
        train_x.sub_(feature_mean).div_(feature_scale)
        x_offset = train_x.mean(dim=0)
        train_x.sub_(x_offset)
        y_offset = train_y.mean(dim=0)
        centered_y = train_y - y_offset
        n, h = train_x.shape
        # Small smoke datasets use the mathematically equivalent dual solve.
        gram = train_x.T @ train_x if n >= h else train_x @ train_x.T
        gram.diagonal().add_(alpha)
        try:
            factor = torch.linalg.cholesky(gram)
        except torch.linalg.LinAlgError as exc:
            raise ValueError("Ridge 矩阵分解失败；请检查特征或增大 --ridge-alpha") from exc
        if n >= h:
            weights = torch.cholesky_solve(train_x.T @ centered_y, factor)
        else:
            weights = train_x.T @ torch.cholesky_solve(centered_y, factor)
        intercept = y_offset - x_offset @ weights
        if not torch.isfinite(weights).all() or not torch.isfinite(intercept).all():
            raise ValueError("Probe 参数包含非有限值；请检查缓存或增大 --ridge-alpha")
        return {"coef": weights.T.contiguous(), "intercept": intercept,
                "feature_mean": feature_mean, "feature_scale": feature_scale}

# 总控制函数，每个候选层都训练一个线性映射，然后使用验证集评价每层，最后选择最佳层
def fit_probes(output_dir, layers, attributes, alpha, device="cuda:0", cpu_threads=8):
    """Train Ridge probes with PyTorch GPU matrix operations; never silently fall back to CPU."""
    import numpy as np
    import torch
    import time

    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Probe 训练需要 CUDA；请在 .yy_value_subspace 中安装 requirements.txt")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        print(f"Probe 训练设备：{device} / {torch.cuda.get_device_name(device)}；FP32 GPU Ridge", flush=True)
    elif device.type == "cpu":
        print("Probe 训练设备：CPU（由 --probe-device cpu 显式指定）", flush=True)
    else:
        raise ValueError("--probe-device 只支持 cuda / cuda:N / cpu")
    torch.set_num_threads(cpu_threads)
    train_dir, val_dir = output_dir / "features/train", output_dir / "features/validation"
    train_y = np.load(train_dir / "labels.npy", allow_pickle=False)
    val_y = np.load(val_dir / "labels.npy", allow_pickle=False)
    if (train_y.ndim != 2 or val_y.ndim != 2 or train_y.shape[1] != len(attributes)
            or val_y.shape[1] != len(attributes) or not len(val_y)
            or not np.isfinite(train_y).all() or not np.isfinite(val_y).all()):
        raise ValueError("Probe 标签缓存的形状或数值不合法")
    baseline = regression_metrics(val_y, np.broadcast_to(train_y.mean(axis=0), val_y.shape), attributes)
    y_gpu = torch.tensor(train_y, dtype=torch.float32, device=device)
    probe_dir = output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    results = []
    previous_precision = torch.get_float32_matmul_precision()
    # Use full FP32 for the regularized linear system, independent of extraction BF16.
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.inference_mode():
            for layer in layers:
                started = time.perf_counter()
                print(f"在 {device} 上训练第 {layer} 层 Probe ...", flush=True)
                train_array = np.load(train_dir / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
                val_array = np.load(val_dir / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
                if (train_array.ndim != 2 or val_array.ndim != 2 or len(train_array) != len(train_y)
                        or len(val_array) != len(val_y) or train_array.shape[1] != val_array.shape[1]):
                    raise ValueError(f"第 {layer} 层特征与标签形状不匹配")
                train_x = torch.tensor(np.array(train_array, dtype=np.float32), device=device)
                if not torch.isfinite(train_x).all():
                    raise ValueError(f"第 {layer} 层缓存包含非有限值，请重新提取")
                model = fit_ridge_torch(train_x, y_gpu, alpha)
                del train_x, train_array
                val_x = torch.tensor(np.array(val_array, dtype=np.float32), device=device)
                if not torch.isfinite(val_x).all():
                    raise ValueError(f"第 {layer} 层验证缓存包含非有限值，请重新提取")
                val_x.sub_(model["feature_mean"]).div_(model["feature_scale"])
                prediction = (val_x @ model["coef"].T + model["intercept"]).cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError(f"第 {layer} 层 Probe 预测包含非有限值")
                metrics = regression_metrics(val_y, prediction, attributes)
                # Prediction: ((x - feature_mean) / feature_scale) @ coef.T + intercept.
                np.savez(probe_dir / f"layer_{layer:02d}.npz",
                         **{key: tensor.cpu().numpy() for key, tensor in model.items()},
                         attributes=np.asarray(attributes), layer=layer, ridge_alpha=alpha,
                         backend="torch_ridge", training_device=str(device))
                seconds = time.perf_counter() - started
                results.append({"layer": layer, "ridge_alpha": alpha, "training_seconds": seconds, **metrics})
                print(f"  validation mean MSE = {metrics['mean_mse']:.6f}；用时 {seconds:.2f}s", flush=True)
                del val_x, val_array, model
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    ranking = sorted(results, key=lambda row: (row["mean_mse"], row["layer"]))
    report = {"metric": "mean validation MSE across requested attributes (lower is better)",
              "backend": "torch_ridge", "training_device": str(device),
              "attributes": attributes, "baseline_train_mean": baseline, "ranking": ranking}
    if device.type == "cuda":
        report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        print(f"Probe 阶段 CUDA 峰值 allocated：{report['peak_cuda_allocated_gib']:.2f} GiB", flush=True)
    write_json(output_dir / "layer_metrics.json", report)
    with (output_dir / "layer_ranking.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["rank", "layer", "mean_mse"] + [f"{a}_mse" for a in attributes]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranking, 1):
            writer.writerow({"rank": rank, "layer": row["layer"], "mean_mse": row["mean_mse"],
                             **{f"{a}_mse": row["per_attribute"][a]["mse"] for a in attributes}})
    return report


def select_layers(report, eligible_layers, top_k):
    eligible = set(eligible_layers)
    ranking = [row for row in report["ranking"] if row["layer"] in eligible]
    if not 1 <= top_k <= len(ranking):
        raise ValueError(f"--top-k 必须在 1..{len(ranking)} 内")
    selected = ranking[:top_k]
    return {
        "selection_metric": report["metric"],
        "eligible_layers": sorted(eligible),
        "top_k": top_k,
        "selected_layers": [row["layer"] for row in selected],
        "selected_layers_in_model_order": sorted(row["layer"] for row in selected),
        "selected_scores": [{"layer": row["layer"], "mean_mse": row["mean_mse"]} for row in selected],
        "best_layer_per_attribute": {
            a: min(ranking, key=lambda row: (row["per_attribute"][a]["mse"], row["layer"]))["layer"]
            for a in report["attributes"]
        },
        "note": "Validation is used for layer selection; these are not held-out test scores. "
                "Selected layers are individually ranked, not evaluated as a joint ensemble.",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--validation-path", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs/stage_0")
    parser.add_argument("--stage", choices=["all", "extract", "probe", "select"], default="all",
                        help="select 只读取已有排名；probe 复用完整特征缓存")
    parser.add_argument("--layers", help="1-based decoder 层编号，如 8,10,12-26,28,30,32,36")
    parser.add_argument("--layer-preset", choices=["middle-dense", "middle", "all"], default="middle-dense")
    parser.add_argument("--selection-layers", help="最终选层允许的范围，须为候选层子集，如 12-26")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--attributes", nargs="+", choices=ATTRIBUTES, default=ATTRIBUTES)
    parser.add_argument("--ridge-alpha", type=float, default=100.0, help="固定 L2 正则强度，所有层使用相同值")
    parser.add_argument("--cpu-threads", type=int, default=8, help="CPU 辅助操作线程数，不控制 GPU Probe")
    parser.add_argument("--device", default="cuda:0", help="表示提取设备")
    parser.add_argument("--probe-device", default="cuda:0", help="Probe 训练设备，默认单卡 GPU，不自动回退 CPU")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--train-limit", type=int, help="随机抽样，用于冒烟测试")
    parser.add_argument("--validation-limit", type=int, help="随机抽样，用于冒烟测试")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-data", action="store_true", help="只检查路径、数据与空间估算，无需第三方依赖")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for name in ("top_k", "batch_size", "max_prompt_tokens", "cpu_threads", "train_limit", "validation_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.max_length < 2:
        parser.error("--max-length 必须至少为 2")
    if not math.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0:
        parser.error("--ridge-alpha 必须为正有限数")
    if len(set(args.attributes)) != len(args.attributes):
        parser.error("--attributes 不可重复")
    args.output_dir = args.output_dir.resolve()
    if args.stage == "select" and not args.check_data:
        report = json.loads((args.output_dir / "layer_metrics.json").read_text(encoding="utf-8"))
        candidates = [row["layer"] for row in report["ranking"]]
        eligible = parse_layers(args.selection_layers, max(candidates)) if args.selection_layers else candidates
        if not set(eligible) <= set(candidates):
            parser.error("--selection-layers 必须是已测试候选层的子集")
        result = select_layers(report, eligible, args.top_k)
        write_json(args.output_dir / "selected_layers.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    args.model_path = args.model_path.resolve()
    args.train_path = args.train_path.resolve()
    args.validation_path = args.validation_path.resolve()
    print(f"执行脚本：{Path(__file__).resolve()}", flush=True)
    print(f"模型路径：{args.model_path}", flush=True)
    print(f"训练文件：{args.train_path}", flush=True)
    print(f"验证文件：{args.validation_path}", flush=True)
    print(f"提取设备：{args.device}；Probe 设备：{args.probe_device}", flush=True)
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    llm_config = config["llm_config"]
    total, hidden = llm_config["num_hidden_layers"], llm_config["hidden_size"]
    layers = parse_layers(args.layers, total, args.layer_preset)
    eligible = parse_layers(args.selection_layers, total) if args.selection_layers else layers
    if not set(eligible) <= set(layers):
        parser.error("--selection-layers 必须是 --layers 候选层的子集")
    if args.top_k > len(eligible) and args.stage != "extract":
        parser.error("--top-k 不能超过可选层数量")
    records = {
        "train": read_records(args.train_path, args.attributes, args.train_limit, args.seed),
        "validation": read_records(args.validation_path, args.attributes, args.validation_limit, args.seed),
    }
    if len(records["train"]) < 2:
        parser.error("训练集至少需要两条样本")
    print(f"模型：{total} 层，hidden_size={hidden}；候选层：{layers}", flush=True)
    counts = {split: len(rows) for split, rows in records.items()}
    print(f"数据量：{counts}；FP16 特征约 {sum(counts.values()) * len(layers) * hidden * 2 / 2**30:.2f} GiB", flush=True)
    overlap = len({row["prompt"] for row in records["train"]} &
                  {row["prompt"] for row in records["validation"]})
    print(f"训练/验证之间重复 prompt 数：{overlap}（相同 split 内的多回答不会被拆分）", flush=True)
    if args.check_data:
        return

    # Import heavy dependencies only after the stdlib-only preflight.
    import numpy as np
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"cache_version": CACHE_VERSION, "model_path": str(args.model_path),
                "model_signature": model_signature(args.model_path),
                "layers": layers, "attributes": args.attributes,
                "max_length": args.max_length, "max_prompt_tokens": args.max_prompt_tokens,
                "dtype": args.dtype, "pooling": "response_mean_before_final_norm"}
    identities = {split: {**identity, "records_sha256": digest(rows)} for split, rows in records.items()}
    cache_dirs = {split: args.output_dir / "features" / split for split in records}
    cached = {split: valid_cache(cache_dirs[split], identities[split]) for split in records}
    if args.stage == "probe" and not all(cached.values()):
        raise ValueError("--stage probe 需要与本次参数一致的完整 train/validation 缓存；先运行 extract 或 all")
    if args.stage in {"all", "extract"} and not all(cached.values()):
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.cpu_threads)
        decoder, tokenizer, template = load_backbone(args)
        if len(decoder.layers) != total or decoder.config.hidden_size != hidden:
            raise ValueError("实际 decoder 与模型配置不一致")
        try:
            for split in records:
                if cached[split]:
                    print(f"复用缓存：{cache_dirs[split]}", flush=True)
                else:
                    extract_split(decoder, tokenizer, template, records[split], cache_dirs[split],
                                  identities[split], args)
        finally:
            del decoder, tokenizer, template
            gc.collect()
            if torch.cuda.is_available():
                print(f"CUDA 峰值 allocated：{torch.cuda.max_memory_allocated(args.device) / 2**30:.2f} GiB"
                      if str(args.device).startswith("cuda") else "CPU 提取完成", flush=True)
                torch.cuda.empty_cache()
    if args.stage == "extract":
        print(f"特征已保存：{args.output_dir / 'features'}", flush=True)
        return
    report = fit_probes(args.output_dir, layers, args.attributes, args.ridge_alpha,
                        device=args.probe_device, cpu_threads=args.cpu_threads)
    result = select_layers(report, eligible, args.top_k)
    write_json(args.output_dir / "selected_layers.json", result)
    write_json(args.output_dir / "run_config.json",
               {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    print(f"选中的层（按验证 MSE）：{result['selected_layers']}", flush=True)
    print(f"结果目录：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
