"""把 13 类不安全类别合并成 7 类（粒度 B），产出一份新的 .npz。

合并**只动标签**：features 原样保留，仅把 `unsafe_category`（以及配套的
`unsafe_category_id`）按 MERGE_MAP 重映射。这样 openset-proto/proto_openset.py
（不改一行）拿这个 npz + `--n-unknown 2` 跑，就等于"从 7 个合并类里随机抽 2 个
当未知类"。

用法：
  python merge_taxonomy.py                 # 默认生成 layer 24 与 32
  python merge_taxonomy.py --layer 32      # 只生成某层
  python merge_taxonomy.py --src <path>    # 手动指定源 npz（层号仍要传，用于命名）

源特征在 get-embedding/outputs/ 下的带日期子目录里，按文件名递归查找。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EMB_ROOT = REPO_ROOT / "get-embedding" / "outputs"
OUT_DIR = SCRIPT_DIR / "data"

# ---------------------------------------------------------------- 合并映射（粒度 B）
# 原类别 → 合并后类别。左列是 npz 里 unsafe_category 的原始取值。
MERGE_MAP: dict[str, str] = {
    # 经济 / 金融 / 欺诈
    "05-EconomicHarm": "M1-Economic-Financial-Fraud",
    "06-Fraud": "M1-Economic-Financial-Fraud",
    "11-Financial_Advice": "M1-Economic-Financial-Fraud",
    # 法律意见
    "10-Legal_Opinion": "M2-Legal-Opinion",
    # 违法 / 人身伤害
    "01-Illegal_Activity": "M3-Illegal-PhysicalHarm",
    "04-Physical_Harm": "M3-Illegal-PhysicalHarm",
    # 恶意软件 / 隐私暴力
    "03-Malware_Generation": "M4-Malware-PrivacyViolence",
    "09-Privacy_Violence": "M4-Malware-PrivacyViolence",
    # 言论 / 政治
    "02-HateSpeech": "M5-HateSpeech-Political",
    "08-Political_Lobbying": "M5-HateSpeech-Political",
    "13-Gov_Decision": "M5-HateSpeech-Political",
    # 色情
    "07-Sex": "M6-Sex",
    # 健康
    "12-Health_Consultation": "M7-Health",
}

DEFAULT_LAYERS = (24, 32)


def find_source(layer: int | str) -> Path:
    """在 get-embedding/outputs/**/ 下递归查找该层的 npz。

    注意：文件名里的 `[24]` 会被 glob 当成字符集，所以这里先通配 *.npz 再按文件名
    精确比对，不能直接把带方括号的名字喂给 rglob。
    """
    name = f"MMSB_sd_typo_decoder[{layer}].npz"
    hits = sorted(p for p in EMB_ROOT.rglob("*.npz") if p.name == name)
    if not hits:
        raise SystemExit(
            f"[错] 找不到 {name}（在 {EMB_ROOT} 下递归搜过）。\n"
            f"     先确认 get-embedding 已抽过第 {layer} 层特征。")
    if len(hits) > 1:
        newest = max(hits, key=lambda p: p.stat().st_mtime)
        print(f"[提示] {name} 有 {len(hits)} 份，用最新的：{newest}")
        return newest
    return hits[0]


def build_merged(src: Path, layer: int | str) -> Path:
    with np.load(src, allow_pickle=True) as d:
        arrays = {k: np.asarray(d[k]) for k in d.files}
    if "unsafe_category" not in arrays:
        raise SystemExit(f"[错] {src} 里没有 unsafe_category 字段")

    cats = np.asarray(arrays["unsafe_category"]).astype(str)
    unknown_cats = sorted(set(cats.tolist()))
    missing = [c for c in unknown_cats if c not in MERGE_MAP]
    if missing:
        raise SystemExit(
            f"[错] 这些原始类别不在 MERGE_MAP 里：{missing}\n"
            f"     请先在 merge_taxonomy.py 的 MERGE_MAP 中补上。")
    unused = [c for c in MERGE_MAP if c not in unknown_cats]
    if unused:
        print(f"[提示] MERGE_MAP 里有原数据未出现的类别（忽略）：{unused}")

    # 源 npz 的 unsafe_category 是定宽 <U22，直接沿用会把更长的合并类名截断，
    # 所以显式用足够宽的 dtype。
    width = max(len(MERGE_MAP[c]) for c in cats)
    merged = np.array([MERGE_MAP[c] for c in cats], dtype=f"<U{width}")
    merged_names = sorted(set(merged.tolist()))
    id_of = {name: i for i, name in enumerate(merged_names)}

    out_arrays = dict(arrays)
    out_arrays["unsafe_category"] = merged
    if "unsafe_category_id" in out_arrays:
        out_arrays["unsafe_category_id"] = np.array(
            [id_of[c] for c in merged], dtype=np.int64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"MMSB_sd_typo_mergeB7_layer{layer}.npz"
    with open(out_path, "wb") as f:
        np.savez(f, **out_arrays)

    print(f"\n[层 {layer}] 源 : {src}")
    print(f"         产物 : {out_path}")
    print(f"         样本 : {len(merged)}  合并前类别 {len(unknown_cats)} → 合并后 {len(merged_names)}")
    print("         各类样本数：")
    for name in merged_names:
        n = int((merged == name).sum())
        members = sorted({o for o, m in MERGE_MAP.items() if m == name})
        print(f"           {name:<32s} {n:4d}   <- {', '.join(members)}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=str, action="append", default=None,
                   help="要生成的层（可重复，如 --layer 24 --layer 32）；默认 24 与 32")
    p.add_argument("--src", default=None,
                   help="手动指定源 npz；只对单层有意义（此时 --layer 必须只给一个）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layers = args.layer if args.layer else [str(x) for x in DEFAULT_LAYERS]
    if args.src and len(layers) != 1:
        raise SystemExit("[错] --src 只能配合单个 --layer 使用")
    for layer in layers:
        src = Path(args.src) if args.src else find_source(layer)
        build_merged(src, layer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
