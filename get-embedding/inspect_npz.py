"""读取 get-embedding 产物 <stem>.npz，打印结构、统计信息和若干条样本。

用法：
  python inspect_npz.py outputs/MMSB_sd_typo.npz
  python inspect_npz.py outputs/MMSB_sd_typo.npz --show 5
  python inspect_npz.py outputs/MMSB_sd_typo.npz --category 01-Illegal_Activity
  python inspect_npz.py outputs/MMSB_sd_typo.npz --index 3 --index 17
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# npz 里可能出现的字段
FEATURE_KEY = "features"
META_KEYS = (
    "index",
    "image_path",
    "question",
    "unsafe_category",
    "unsafe_category_id",
    "response",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 build_embeddings.py 落盘的 npz 内容",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz_path", help="<stem>.npz 路径", default="./outputs/MMSB_sd_typo.npz",)
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="打印多少条完整样本（0 表示不打印）",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="每条样本里 question/response 截断到多少字符",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="只看某个 unsafe_category（前缀匹配，例如 01-Illegal_Activity）",
    )
    parser.add_argument(
        "--index",
        type=int,
        action="append",
        default=None,
        help="只看指定 index，可重复传多次",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="不尝试读取同目录下的 <stem>.manifest.json",
    )
    return parser.parse_args(argv)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def print_structure(arrays: dict[str, np.ndarray]) -> None:
    print("=" * 78)
    print("数组清单")
    print("=" * 78)
    n = None
    for key, arr in arrays.items():
        print(f"  {key:<20} dtype={str(arr.dtype):<12} shape={arr.shape}")
        if arr.ndim >= 1:
            if n is None:
                n = arr.shape[0]
            elif n != arr.shape[0]:
                print(f"    [warn] 第 0 维与其它数组不一致: {arr.shape[0]} != {n}")
    if n is not None:
        print(f"\n  记录数 N = {n}")


def print_feature_stats(features: np.ndarray) -> None:
    print("\n" + "=" * 78)
    print("feature 矩阵统计")
    print("=" * 78)

    if features.ndim != 2:
        print(f"  [warn] features 不是二维矩阵，实际 shape={features.shape}")
        return

    n, d = features.shape
    print(f"  形状            : ({n}, {d})")
    print(f"  dtype           : {features.dtype}")

    finite_mask = np.isfinite(features)
    n_bad = int((~finite_mask).sum())
    print(f"  非有限元素个数  : {n_bad}")
    if n_bad:
        bad_rows = np.where(~finite_mask.all(axis=1))[0]
        print(f"  含 NaN/Inf 的行 : {bad_rows[:20].tolist()}"
              f"{' ...' if len(bad_rows) > 20 else ''}")

    print(f"  全局 min/max    : {features.min():.6f} / {features.max():.6f}")
    print(f"  全局 mean/std   : {features.mean():.6f} / {features.std():.6f}")

    # 逐行范数：feature 是平均出来的，范数反映向量尺度
    norms = np.linalg.norm(features, axis=1)
    print(f"  L2 范数 min/中位/max : "
          f"{norms.min():.4f} / {np.median(norms):.4f} / {norms.max():.4f}")

    # 维度级统计：哪些维几乎不变、哪些变化剧烈
    dim_std = features.std(axis=0)
    print(f"  各维 std min/中位/max: "
          f"{dim_std.min():.6f} / {np.median(dim_std):.6f} / {dim_std.max():.6f}")
    near_const = int((dim_std < 1e-6).sum())
    print(f"  近似常量维数(σ<1e-6) : {near_const} / {d}")

    # 行之间是否高度相似（自相关抽样）
    if n >= 2:
        sample = features[: min(n, 200)].astype(np.float32)
        normed = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-12)
        sim = normed @ normed.T
        off = sim[~np.eye(sim.shape[0], dtype=bool)]
        print(f"  前 {sample.shape[0]} 条两两余弦相似: "
              f"min={off.min():.4f} 中位={np.median(off):.4f} max={off.max():.4f}")


def print_meta_summary(arrays: dict[str, np.ndarray], mask: np.ndarray | None) -> None:
    print("\n" + "=" * 78)
    print("元数据概览" + ("（已按过滤条件筛选）" if mask is not None else ""))
    print("=" * 78)

    def take(key: str) -> np.ndarray | None:
        arr = arrays.get(key)
        if arr is None:
            return None
        return arr if mask is None else arr[mask]

    cats = take("unsafe_category")
    if cats is not None:
        counter = Counter(str(c) for c in cats.tolist())
        print(f"  unsafe_category 取值 ({len(counter)} 类):")
        for cat, cnt in sorted(counter.items()):
            print(f"    {cat:<40} {cnt:>5}")

    cids = take("unsafe_category_id")
    if cids is not None:
        pairs = sorted(set(zip(cats.tolist(), cids.tolist()))) if cats is not None else []
        if pairs:
            print("  category ↔ id 对应:")
            for cat, cid in pairs:
                print(f"    {cid:>3}  {cat}")

    resp = take("response")
    if resp is not None and resp.size:
        lens = np.asarray([len(str(r)) for r in resp.tolist()])
        print(f"  response 字符长度 min/中位/max: "
              f"{lens.min()} / {int(np.median(lens))} / {lens.max()}")
        empty = int((lens == 0).sum())
        if empty:
            print(f"  [warn] 空 response 条数: {empty}")

    q = take("question")
    if q is not None and q.size:
        lens = np.asarray([len(str(x)) for x in q.tolist()])
        print(f"  question 字符长度 min/中位/max: "
              f"{lens.min()} / {int(np.median(lens))} / {lens.max()}")

    idx = take("index")
    if idx is not None and idx.size:
        print(f"  index 范围: {idx.min()} ~ {idx.max()}，"
              f"唯一值 {len(set(idx.tolist()))} 个（总数 {idx.size}）")


def build_mask(arrays: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray | None:
    n = len(next(iter(arrays.values())))
    mask = np.ones(n, dtype=bool)

    if args.category:
        cats = arrays.get("unsafe_category")
        if cats is None:
            print("[warn] npz 里没有 unsafe_category，--category 忽略")
        else:
            mask &= np.asarray([str(c).startswith(args.category) for c in cats.tolist()])

    if args.index:
        idx = arrays.get("index")
        if idx is None:
            print("[warn] npz 里没有 index，--index 按行号解释")
            wanted = set(args.index)
            mask &= np.asarray([i in wanted for i in range(n)])
        else:
            wanted = set(args.index)
            mask &= np.asarray([int(i) in wanted for i in idx.tolist()])

    if mask.all():
        return None
    return mask


def print_samples(
    arrays: dict[str, np.ndarray],
    mask: np.ndarray | None,
    show: int,
    max_chars: int,
) -> None:
    if show <= 0:
        return

    print("\n" + "=" * 78)
    print(f"样本明细（最多 {show} 条）")
    print("=" * 78)

    n = len(next(iter(arrays.values())))
    rows = np.arange(n) if mask is None else np.where(mask)[0]
    if rows.size == 0:
        print("  没有匹配的记录。")
        return

    features = arrays.get(FEATURE_KEY)
    for row in rows[:show]:
        print(f"\n--- row {row} " + "-" * 55)
        for key in META_KEYS:
            arr = arrays.get(key)
            if arr is None:
                continue
            value = arr[row]
            if key in ("question", "response"):
                text = str(value)
                if len(text) > max_chars:
                    text = text[:max_chars] + f" …(+{len(text) - max_chars} chars)"
                print(f"  {key:<18}: {text}")
            else:
                print(f"  {key:<18}: {value}")
        if features is not None and features.ndim == 2:
            vec = features[row].astype(np.float32)
            print(f"  feature            : dim={vec.shape[0]} "
                  f"norm={np.linalg.norm(vec):.4f} "
                  f"mean={vec.mean():.6f} std={vec.std():.6f}")
            head = ", ".join(f"{x:+.4f}" for x in vec[:8])
            print(f"  前 8 维            : [{head}, ...]")


def print_manifest(npz_path: Path) -> None:
    manifest_path = npz_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        print(f"\n[info] 未找到 {manifest_path.name}，跳过 manifest")
        return
    print("\n" + "=" * 78)
    print(f"manifest: {manifest_path.name}")
    print("=" * 78)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  [warn] manifest 解析失败: {exc}")
        return
    for key, value in manifest.items():
        if isinstance(value, (dict, list)) and len(json.dumps(value, ensure_ascii=False)) > 200:
            print(f"  {key}: <{type(value).__name__}, {len(value)} 项>")
        else:
            print(f"  {key}: {value}")


def main() -> int:
    args = parse_args()
    npz_path = Path(args.npz_path).expanduser().resolve()
    if not npz_path.exists():
        raise SystemExit(f"文件不存在: {npz_path}")

    print(f"读取: {npz_path}")
    print(f"文件大小: {npz_path.stat().st_size / 1024 / 1024:.2f} MB")

    arrays = load_npz(npz_path)
    if not arrays:
        raise SystemExit("npz 是空的。")

    if FEATURE_KEY not in arrays:
        print(f"[warn] 缺少 {FEATURE_KEY} 字段，只有: {list(arrays)}")

    print_structure(arrays)
    if FEATURE_KEY in arrays:
        print_feature_stats(arrays[FEATURE_KEY])

    mask = build_mask(arrays, args)
    print_meta_summary(arrays, mask)
    print_samples(arrays, mask, args.show, args.max_chars)

    if not args.no_manifest:
        print_manifest(npz_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())