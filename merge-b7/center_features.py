"""把合并后的 7 类特征做「去均值 / all-but-the-top」预处理，产出可选的 npz 变体。

动机：LLM 隐藏态各向异性极强——原始特征里各类均值的两两余弦高达 0.99+，真正的判别
信号被一个巨大的公共分量淹没（这也是 NCM 失效的原因）。减掉全局均值、再去掉最大的
若干主方向，就能把这个公共分量移除。

变体：
  mean     只减全局均值
  abtt{k}  减全局均值后，再去掉前 k 个主方向（all-but-the-top）

只改 `features`，其余字段（unsafe_category / question / index / ...）原样保留。
下游 `openset-proto/proto_openset.py` **不改一行**，它拿到变体 npz 后会照常做 PCA(默认关)
+ 逐行 L2 归一化，于是「去均值 + L2 归一化」= 标准的 centered-cosine 表示。

拟合口径：均值与主方向都用**全部样本**估计（无标签、不看已知/未知划分），
和 merge-b7 其余数据层预处理一致。

用法：
  python center_features.py                        # 默认 layer 24,32；变体 mean + abtt1
  python center_features.py --layer 32 --variants mean,abtt1,abtt3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_LAYERS = (24, 32)
DEFAULT_VARIANTS = ("mean", "abtt1")


def parse_variant(spec: str) -> tuple[str, int]:
    """'mean' -> ('mean', 0)；'abtt3' -> ('abtt', 3)。"""
    s = spec.strip().lower()
    if s == "mean":
        return "mean", 0
    if s.startswith("abtt"):
        tail = s[4:]
        if not tail.isdigit():
            raise SystemExit(f"[错] 认不出的变体 {spec!r}（abtt 后面要跟主方向个数，如 abtt1）")
        return "abtt", int(tail)
    raise SystemExit(f"[错] 认不出的变体 {spec!r}（只支持 mean / abtt{k}）")


def source_path(layer: str) -> Path:
    p = DATA_DIR / f"MMSB_sd_typo_mergeB7_layer{layer}.npz"
    if not p.exists():
        raise SystemExit(f"[错] 找不到合并特征 {p}，先跑 merge_taxonomy.py")
    return p


def class_mean_cosine(feats: np.ndarray, cats: np.ndarray) -> tuple[float, float, float]:
    """各类均值（L2 归一化后）两两余弦的 (min, mean, max)，用来直观衡量「公共分量」有多大。"""
    X = feats.astype(np.float64)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    protos = np.stack([X[cats == c].mean(0) for c in sorted(set(cats.tolist()))])
    protos = protos / np.clip(np.linalg.norm(protos, axis=1, keepdims=True), 1e-12, None)
    sim = protos @ protos.T
    off = sim[~np.eye(len(protos), dtype=bool)]
    return float(off.min()), float(off.mean()), float(off.max())


def build_variant(src: Path, layer: str, spec: str) -> Path:
    kind, k = parse_variant(spec)
    with np.load(src, allow_pickle=True) as d:
        arrays = {name: np.asarray(d[name]) for name in d.files}
    X = arrays["features"].astype(np.float64)
    cats = np.asarray(arrays["unsafe_category"]).astype(str)

    before = class_mean_cosine(X, cats)

    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    removed_var = None
    if kind == "abtt" and k > 0:
        # 在去均值矩阵上取前 k 个主方向（右奇异向量），把投影减掉
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:k]                              # (k, D)
        Xc = Xc - (Xc @ V.T) @ V
        energy = S ** 2
        removed_var = float(energy[:k].sum() / energy.sum())

    after = class_mean_cosine(Xc, cats)

    out = dict(arrays)
    out["features"] = Xc.astype(np.float32)
    out_path = DATA_DIR / f"MMSB_sd_typo_mergeB7_layer{layer}_center-{spec}.npz"
    with open(out_path, "wb") as f:
        np.savez(f, **out)

    print(f"\n[层 {layer}] 变体 {spec}  ({kind}{'' if k == 0 else f', k={k}'})")
    print(f"  源   : {src}")
    print(f"  产物 : {out_path}")
    print(f"  类均值两两余弦： 去均值前 min/mean/max = "
          f"{before[0]:.4f} / {before[1]:.4f} / {before[2]:.4f}")
    print(f"                   去均值后 min/mean/max = "
          f"{after[0]:.4f} / {after[1]:.4f} / {after[2]:.4f}")
    if removed_var is not None:
        print(f"  被移除的前 {k} 个主方向占总能量 {removed_var * 100:.2f}%")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=str, action="append", default=None,
                   help="要处理的层（可重复）；默认 24 与 32")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                   help="逗号分隔的变体，如 mean,abtt1,abtt3")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layers = args.layer if args.layer else [str(x) for x in DEFAULT_LAYERS]
    variants = [s.strip() for s in args.variants.split(",") if s.strip()]
    for spec in variants:          # 提前校验，避免跑到一半才报错
        parse_variant(spec)
    for layer in layers:
        src = source_path(layer)
        for spec in variants:
            build_variant(src, layer, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
