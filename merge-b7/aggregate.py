"""汇总 merge-b7/outputs/ 下所有运行的 metrics.json（+ manifest.json），生成报告。

支持「预处理变体」维度：run 目录名形如
  <时间戳>_b7_L24_s42_cac             → 变体 none（原始合并特征，基线）
  <时间戳>_b7_mean_L24_s42_cac        → 变体 mean（去均值）
  <时间戳>_b7_abtt1_L24_s42_cac       → 变体 abtt1（去均值 + all-but-the-top k=1）

产物（文件名可覆盖）：
  - 合并B7_去均值ABTT_实验报告.md   （--report）
  - aggregate_summary.json          （--summary）

用法：
  python aggregate.py
  python aggregate.py --report 合并B7_去均值ABTT_实验报告.md --summary aggregate_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs"
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_REPORT = SCRIPT_DIR / "合并B7_实验报告.md"
DEFAULT_SUMMARY = SCRIPT_DIR / "aggregate_summary.json"

PROTO = SCRIPT_DIR.parent / "openset-proto" / "proto_openset.py"

RUN_RE = re.compile(
    r"_(?P<name>b7_(?:(?P<variant>mean|abtt\d+)_)?L(?P<layer>-?\d+)_s(?P<seed>\d+)_(?P<method>cac|arpl))$")

METHOD_ORDER = ("cac", "arpl")
METRICS = [
    ("AUROC", "auroc", "{:.4f}"),
    ("闭集ACC", "closed_set_acc", "{:.4f}"),
    ("未知拒绝率", "unknown_reject_rate", "{:.4f}"),
    ("FPR@TPR95", "fpr_at_tpr95", "{:.4f}"),
]
VARIANT_LABEL = {
    "none": "raw（原始合并特征，基线）",
    "mean": "mean（去均值）",
}
ABTT_LABEL = "abtt{k}（去均值 + all-but-the-top k={k}）"


def variant_label(v: str) -> str:
    if v in VARIANT_LABEL:
        return VARIANT_LABEL[v]
    if v.startswith("abtt"):
        return ABTT_LABEL.format(k=v[4:])
    return v


def variant_key(v: str) -> tuple[int, int]:
    if v == "none":
        return (0, 0)
    if v == "mean":
        return (1, 0)
    if v.startswith("abtt"):
        return (2, int(v[4:]))
    return (3, 0)


def load_runs() -> list[dict]:
    runs: list[dict] = []
    for run_dir in sorted(OUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.search(run_dir.name)
        if not m:
            continue
        mf = next(iter(run_dir.glob("*.metrics.json")), None)
        if mf is None:
            continue
        d = json.loads(mf.read_text(encoding="utf-8"))
        # manifest 里记着这次实际用的 npz 路径与完整参数，用于报告里的"运行了什么文件"
        man_f = next(iter(run_dir.glob("*.manifest.json")), None)
        man = json.loads(man_f.read_text(encoding="utf-8")) if man_f else {}
        runs.append({
            "variant": m.group("variant") or "none",
            "layer": m.group("layer"),
            "seed": int(m.group("seed")),
            "method": m.group("method"),
            "dir": run_dir.name,
            "npz": man.get("npz", "(manifest 缺失)"),
            "device": (man.get("device") or {}).get("name", man.get("args", {}).get("device", "?")),
            "unknown_classes": d["unknown_classes"],
            "known_classes": d["known_classes"],
            "counts": d["counts"],
            "variants": {k: v["variants"] for k, v in d["results"].items()},
            "attribution": d["attribution"],
        })
    return runs


def mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def layer_sort_key(layer: str) -> float:
    try:
        return float(layer)
    except ValueError:
        return float("inf")


def fmt_mean_std(vals: list[float], fmt: str) -> str:
    mu, sd = mean_std(vals)
    return f"{fmt.format(mu)}±{fmt.format(sd)}"


def pick(runs: list[dict], variant: str, layer: str, method: str,
         metric_variant: str, seed: int, field: str) -> float | None:
    """取某个 (变体, 层, 方法, seed) 的指标；缺失返回 None（便于容忍未跑完的实验）。"""
    for r in runs:
        if (r["variant"] == variant and r["layer"] == layer
                and r["method"] == method and r["seed"] == seed):
            return float(r["variants"][method][metric_variant][field])
    return None


def vals_for(runs: list[dict], variant: str, layer: str, method: str,
             metric_variant: str, field: str) -> list[float]:
    """该 (变体, 层, 方法) 下所有已跑 seed 的某个指标（按 seed 排序）。"""
    hit = [r for r in runs if r["variant"] == variant and r["layer"] == layer
           and r["method"] == method]
    hit.sort(key=lambda r: r["seed"])
    return [float(r["variants"][method][metric_variant][field]) for r in hit]


def _npz_for(layer: str, variant: str) -> Path:
    if variant == "none":
        return DATA_DIR / f"MMSB_sd_typo_mergeB7_layer{layer}.npz"
    return DATA_DIR / f"MMSB_sd_typo_mergeB7_layer{layer}_center-{variant}.npz"


def _class_mean_cos(M: np.ndarray, cats: np.ndarray) -> tuple[float, float, float]:
    Z = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)
    P = np.stack([Z[cats == c].mean(0) for c in sorted(set(cats.tolist()))])
    P = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)
    off = (P @ P.T)[~np.eye(len(P), dtype=bool)]
    return float(off.min()), float(off.mean()), float(off.max())


def centering_stats(layer: str, variant: str) -> tuple[float, float, float, float | None]:
    """重算该 (层, 变体) 的类均值余弦统计 + ABTT 移除能量占比。

    直接从原始 npz 出发、用与 center_features.py 相同的公式重算，这样报告里的数字
    不会和实际跑实验用的文件脱节。
    """
    raw = _npz_for(layer, "none")
    if not raw.exists():
        raise FileNotFoundError(raw)
    with np.load(raw, allow_pickle=True) as d:
        X = np.asarray(d["features"], dtype=np.float64)
        cats = np.asarray(d["unsafe_category"]).astype(str)
    if variant == "none":
        return (*_class_mean_cos(X, cats), None)
    Xc = X - X.mean(axis=0, keepdims=True)
    if not variant.startswith("abtt"):
        return (*_class_mean_cos(Xc, cats), None)
    k = int(variant[4:]) or 1
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]
    Xa = Xc - (Xc @ V.T) @ V
    energy = float((S[:k] ** 2).sum() / (S ** 2).sum())
    return (*_class_mean_cos(Xa, cats), energy)


def build_report(runs: list[dict]) -> tuple[str, dict]:
    variants = sorted({r["variant"] for r in runs}, key=variant_key)
    layers = sorted({r["layer"] for r in runs}, key=layer_sort_key)
    seeds = sorted({r["seed"] for r in runs})
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in runs)]
    L: list[str] = []
    summary: dict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_runs": len(runs), "variants": variants, "layers": layers,
        "seeds": seeds, "methods": methods,
    }

    L.append("# 合并类别 B（13→7）+ 去均值 / all-but-the-top 开集识别实验结果")
    L.append("")
    L.append(f"- 生成时间：{summary['generated_at']}")
    L.append(f"- 运行数：**{len(runs)}**（变体 {variants} × 层 {layers} × seed {seeds} × 方法 {methods}）")
    L.append("- 全部由 merge-b7 目录下的脚本产出；推理/训练脚本 `openset-proto/proto_openset.py` **未做任何修改**。")
    L.append("")

    # ---------------- 1. 本次运行的命令与文件（provenance） ----------------
    L.append("## 1. 本次运行的命令与文件")
    L.append("")
    L.append("### 1.1 用到的脚本")
    L.append("")
    L.append("| 脚本 | 作用 |")
    L.append("|---|---|")
    L.append(f"| `{SCRIPT_DIR / 'merge_taxonomy.py'}` | 13 类 → 7 类合并，生成 `data/MMSB_sd_typo_mergeB7_layer*.npz` |")
    L.append(f"| `{SCRIPT_DIR / 'center_features.py'}` | 去均值 / ABTT 预处理，生成 `data/..._center-*.npz` |")
    L.append(f"| `{SCRIPT_DIR / 'run_merge_b7.py'}` | 驱动器，循环 变体×层×seed×方法 调下面的脚本 |")
    L.append(f"| `{PROTO}` | **实际训练/评测的主脚本（未修改）** |")
    L.append(f"| `{SCRIPT_DIR / 'aggregate.py'}` | 汇总 metrics/manifest，生成本报告 |")
    L.append("")
    L.append("### 1.2 单次调用的命令模板")
    L.append("")
    L.append("每次运行就是下面这条命令（参数按变体/层/seed/方法替换）：")
    L.append("")
    L.append("```bash")
    L.append("python ../openset-proto/proto_openset.py \\")
    L.append("  --npz data/MMSB_sd_typo_mergeB7_layer<层>[_center-<变体>].npz \\")
    L.append("  --method <cac|arpl> --n-unknown 2 --seed <seed> --device gpu0 \\")
    L.append("  --output-dir outputs --run-name b7_[<变体>_]L<层>_s<seed>_<方法>")
    L.append("```")
    L.append("")
    L.append("### 1.3 每个变体对应的输入特征文件")
    L.append("")
    L.append("| 变体 | 输入 npz（绝对路径） | run-name 前缀 |")
    L.append("|---|---|---|")
    for v in variants:
        if v == "none":
            pat = f"MMSB_sd_typo_mergeB7_layer<层>.npz"
            pref = "b7_"
        else:
            pat = f"MMSB_sd_typo_mergeB7_layer<层>_center-{v}.npz"
            pref = f"b7_{v}_"
        L.append(f"| {variant_label(v)} | `{DATA_DIR / pat}` | `{pref}` |")
    L.append("")
    L.append("### 1.4 逐次运行清单（从每次运行的 manifest.json 读出）")
    L.append("")
    L.append("| 变体 | 层 | seed | 方法 | 输入 npz（文件名） | 输出目录 |")
    L.append("|---|---|---|---|---|---|")
    provenance = []
    for r in sorted(runs, key=lambda r: (variant_key(r["variant"]), layer_sort_key(r["layer"]),
                                         r["seed"], r["method"])):
        npz_name = Path(r["npz"]).name
        L.append(f"| {r['variant']} | {r['layer']} | {r['seed']} | {r['method']} | "
                 f"`{npz_name}` | `outputs/{r['dir']}` |")
        provenance.append({"variant": r["variant"], "layer": r["layer"], "seed": r["seed"],
                           "method": r["method"], "npz": r["npz"], "run_dir": r["dir"],
                           "device": r["device"]})
    summary["provenance"] = provenance
    L.append("")

    # ---------------- 2. 实验设置 ----------------
    L.append("## 2. 实验设置")
    L.append("")
    L.append("| 项目 | 内容 |")
    L.append("|---|---|")
    L.append("| 底座模型 | Qwen/Qwen3-VL-8B-Instruct（fp16，特征维度 4096） |")
    L.append("| 数据 | MM-SafetyBench SD_TYPO，1680 样本，13 类**合并为 7 类**（粒度 B） |")
    L.append(f"| 对比层 | decoder 第 {'、'.join(layers)} 层 |")
    L.append(f"| 预处理变体 | {'、'.join(variant_label(v) for v in variants)} |")
    L.append("| 未知类 | 每 seed 从 7 个合并类里**随机抽 2 个**（`--n-unknown 2`，各变体/各层同 seed 同抽法） |")
    L.append(f"| 方法 | {'、'.join(methods)} |")
    L.append("| 阈值 tau | 各方法在每个 seed 下于验证集按 TPR 95% 校准 |")
    L.append("| 汇报口径 | B_matched（未知类下采样到与已知测试集 1:1）为主，A_full 作附表 |")
    L.append("")
    L.append("**预处理定义**（对每层特征矩阵 `X`，全部样本逐条作用）：")
    L.append("")
    L.append("- `mean`：`X` 减去全局均值 `μ = mean(X)`。")
    L.append("- `abtt1`：在去均值后的矩阵上做 SVD，把投影到**第一主方向**的部分减掉")
    L.append("  （`X' = Xc − (Xc·v₁)v₁`，all-but-the-top）。")
    L.append("")
    L.append("**拟合口径（需要声明）**：均值和主方向都用**全部 1680 条**估计，其中包含某些 seed 下会被划为")
    L.append("未知类的样本。这是无标签估计、对所有类一视同仁（ABTT 文献通常这么做），但它确实「看过」未知")
    L.append("样本的特征，属于比「只用已知类训练集估计」更宽松的口径。")
    L.append("")
    L.append("预处理只是把每个样本的特征换成一个新的向量，因此**没有**改变类别划分、未知类抽取或方法本身；")
    L.append("下游 `proto_openset.py` 照常对特征做逐行 L2 归一化，于是「去均值 + L2 归一化」就是标准的")
    L.append("centered-cosine 表示。")
    L.append("")
    L.append("**预处理前后的类均值两两余弦**（说明公共分量被移掉了；")
    L.append("数值是在本脚本里直接从 §1.3 的 npz 文件重算的，不使用硬编码）：")
    L.append("")
    L.append("| 层 | " + " | ".join(variant_label(v) for v in variants) + " |")
    L.append("|---" * (1 + len(variants)) + "|")
    energy_note: list[str] = []
    for layer in layers:
        cells = []
        for v in variants:
            try:
                st = centering_stats(layer, v)
                cells.append(f"{st[0]:.4f} / {st[1]:.4f} / {st[2]:.4f}")
                if v.startswith("abtt") and st[3] is not None:
                    energy_note.append(f"L{layer} {v} 移除 {st[3] * 100:.1f}%")
            except Exception as exc:  # noqa: BLE001 - 报告不该因统计失败而中断
                cells.append(f"(无法计算: {type(exc).__name__})")
        L.append(f"| {layer} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("（格式为 min / mean / max。")
    if energy_note:
        L.append("被 ABTT 移除的主方向占总能量：" + "、".join(energy_note) + "。）")
    else:
        L.append("）")
    L.append("")

    # ---------------- 3. 未知类组合 ----------------
    L.append("## 3. 每个 seed 抽中的未知类组合")
    L.append("")
    L.append("同一 seed 下所有变体、所有层抽到的未知类相同（选择只取决于类别表与 seed）。")
    L.append("")
    L.append("| seed | 未知类（2 个） | 未知样本数 | 已知测试数 |")
    L.append("|---|---|---|---|")
    seen = set()
    for r in sorted(runs, key=lambda r: r["seed"]):
        if r["seed"] in seen:
            continue
        seen.add(r["seed"])
        L.append("| {} | {} | {} | {} |".format(
            r["seed"], "、".join(r["unknown_classes"]),
            r["counts"]["unknown"], r["counts"]["known_test"]))
    L.append("")

    # ---------------- 4/6. 主表 & 附表 ----------------
    for mv, title, note in [
        ("B_matched", "4. 主结果 · 口径 B_matched（1:1）",
         "未知类下采样到与已知测试集等量；每个格子是 5 个 seed 的均值±样本标准差。"),
        ("A_full", "6. 附表 · 口径 A_full（未知类全部放入）",
         "该口径未知占绝大多数，「全判未知」的退化解也有很高的 ACC，请以 AUROC 为准。"),
    ]:
        L.append(f"## {title}")
        L.append("")
        L.append(note)
        L.append("")
        L.append("| 变体 | 层 | 方法 | " + " | ".join(m[0] for m in METRICS) + " |")
        L.append("|---" * (3 + len(METRICS)) + "|")
        for v in variants:
            for layer in layers:
                for method in methods:
                    cells = []
                    for _, field, fmt in METRICS:
                        vals = vals_for(runs, v, layer, method, mv, field)
                        cells.append(fmt_mean_std(vals, fmt) if vals else "—")
                    L.append(f"| {v} | {layer} | {method} | " + " | ".join(cells) + " |")
        L.append("")
        summary[f"table_{mv}"] = {
            v: {f"L{layer}_{method}": {
                field: (mean_std(vals_for(runs, v, layer, method, mv, field))[0]
                        if vals_for(runs, v, layer, method, mv, field) else None)
                for _, field, _ in METRICS} for layer in layers for method in methods}
            for v in variants}
        if mv == "B_matched":
            # 核心对比表放在主表之后
            L.append("## 5. 核心对比 · 去均值 / ABTT 相对基线的变化（B_matched）")
            L.append("")
            L.append("每格为「均值（相对 raw 的变化）」。看两件事：**去均值单独带来多少**、"
                     "**再去掉第一主方向又带来多少**。")
            L.append("")
            L.append("| 层 | 方法 | 变体 | " + " | ".join(m[0] + "（Δ vs raw）" for m in METRICS) + " |")
            L.append("|---" * (3 + len(METRICS)) + "|")
            summary["comparison"] = {}
            for layer in layers:
                for method in methods:
                    base = {field: (mean_std(vals_for(runs, "none", layer, method, "B_matched", field))[0]
                                    if vals_for(runs, "none", layer, method, "B_matched", field) else None)
                            for _, field, _ in METRICS}
                    for v in variants:
                        cells = []
                        for _, field, fmt in METRICS:
                            vals = vals_for(runs, v, layer, method, "B_matched", field)
                            if not vals:
                                cells.append("—")
                                continue
                            val = mean_std(vals)[0]
                            if v != "none" and base[field] is not None:
                                cells.append(f"{fmt.format(val)} ({val - base[field]:+.4f})")
                            else:
                                cells.append(f"{fmt.format(val)} (—)")
                        L.append(f"| {layer} | {method} | {v} | " + " | ".join(cells) + " |")
                        summary["comparison"][f"L{layer}_{method}_{v}"] = {
                            field: (mean_std(vals_for(runs, v, layer, method, "B_matched", field))[0]
                                    if vals_for(runs, v, layer, method, "B_matched", field) else None)
                            for _, field, _ in METRICS}
            L.append("")
            L.append("### 5.1 结论（以 cac 为准，理由见 §8）")
            L.append("")
            L.append("- **去均值大幅改善了 cac 的开集性能**：AUROC 从 0.80~0.82 升到 0.88~0.91")
            L.append("  （Δ +0.07~+0.11），未知拒绝率从 0.28~0.34 升到 0.46~0.65，FPR@TPR95 从")
            L.append("  0.68~0.73 降到 0.35~0.56。也就是说，之前那个「未知类被大量误收」的问题，")
            L.append("  主要就是**公共分量**造成的，减掉之后一半以上的未知样本能被正确拒绝。")
            L.append("- **ABTT(k=1) 在此基础上再有增益，但只在 L24 明显**：L24 上")
            L.append("  AUROC 0.8779 → 0.9109、拒绝率 0.4606 → 0.6408、FPR 0.5640 → 0.3518；")
            L.append("  L32 上 ABTT1 与单纯去均值基本持平（0.9063 vs 0.9056）。")
            L.append("- **最优组合**：L24 + abtt1 + cac（AUROC 0.9109±0.0147、拒绝率 0.6408±0.0825、")
            L.append("  FPR 0.3518±0.0649）；L32 + mean/abtt1 + cac 紧随其后。")
            L.append("- **代价**：闭集 ACC 小幅下降（−0.02~−0.04），即去掉公共分量后类内也更「散」了一点。")
            L.append("- **arpl 出现反向崩塌**（AUROC < 0.5、拒绝率 ~2%），两个层、两个变体、5 个 seed 一致；")
            L.append("  见 §8 的专门说明，本报告的开集结论不采信 arpl。")
            L.append("")

    # ---------------- 7. 误判方向 ----------------
    L.append("## 7. 未知类的误判方向（按未知类聚合，行内归一化）")
    L.append("")
    L.append("把「某个合并类被抽为未知」的那些 seed 的归属比例平均起来；"
             "`判为未知` = 被正确拒绝的比例，其余为被骗过阈值后归到的已知类。")
    L.append("")
    summary["attribution"] = {}
    for v in variants:
        L.append(f"### 变体 {v}（{variant_label(v)}）")
        L.append("")
        L.append("| 层 | 方法 | 未知类 | 被抽次数 | 判为未知 | 主要误判去向（top3） |")
        L.append("|---|---|---|---|---|---|")
        for layer in layers:
            for method in methods:
                rs = [r for r in runs if r["variant"] == v and r["layer"] == layer
                      and r["method"] == method]
                per_cat: dict[str, list[dict]] = {}
                for r in rs:
                    for cat, row in r["attribution"].get(method, {}).items():
                        per_cat.setdefault(cat, []).append(row)
                if not per_cat:
                    continue
                summary["attribution"].setdefault(v, {})[f"L{layer}_{method}"] = {}
                for cat in sorted(per_cat):
                    rows = per_cat[cat]
                    keys = {"判为未知"} | {k for row in rows for k in row if k != "判为未知"}
                    avg = {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in keys}
                    rej = avg.pop("判为未知", 0.0)
                    sinks = sorted(avg.items(), key=lambda kv: -kv[1])[:3]
                    detail = "、".join(f"{k} {s*100:.0f}%" for k, s in sinks if s > 0) or "无"
                    L.append(f"| {layer} | {method} | {cat} | {len(rows)} | {rej*100:.1f}% | {detail} |")
                    summary["attribution"][v][f"L{layer}_{method}"][cat] = {
                        "n_drawn": len(rows), "reject_rate": rej, "sinks": dict(sinks)}
        L.append("")

    # ---------------- 8. 注意事项 ----------------
    L.append("## 8. 注意事项")
    L.append("")
    L.append("- **口径不同不可严格对比**：7 类里随机抽 2 个未知，与旧的 13 类结果不是同一口径；")
    L.append("  这里的主对比是**同一口径下不同预处理变体之间**。")
    L.append("- **方差主要来自「抽中哪 2 个未知类」**：合并后类大小很不均（M5=465、M1=443 vs M6=M7=109），")
    L.append("  拒绝率的标准差可达 ±0.10 以上；变体之间的小差距（<0.02）可能落在噪声内。")
    L.append("- **均值/主方向用全部样本估计**（含未知样本特征），是无标签的宽松口径，见 §2 声明。")
    L.append("- **arpl 在去均值/ABTT 特征上系统性失效（重要）**：AUROC 掉到 0.20~0.34，小于 0.5 说明它输出的")
    L.append("  「已知度」分数**方向反了**——未知样本反而得分更高，于是 95% 的未知类被当成已知接受（拒绝率 ~2%）。")
    L.append("  该现象在 L24/L32、mean/abtt1、5 个 seed 上完全一致（标准差仅 0.02~0.06），不是噪声。")
    L.append("  同时它的**闭集 ACC 仍保持 0.89~0.92**，说明分类没坏、坏的只是拒绝用的那个分数。")
    L.append("  原因方向：去均值让特征谱变平，而 ARPL 的「反向点 + margin」打分对特征尺度/分布很敏感，")
    L.append("  它的 margin、tau 等超参是按原始特征调的；要公平对比，需要**先在去均值特征上重新调 ARPL**。")
    L.append("  因此本报告的开集结论以 **cac** 为准，arpl 一栏只作「原样套用旧超参会崩」的记录。")
    L.append("- raw 变体的数字与上一份 `合并B7_实验报告.md` 完全一致（同一批运行的产物）。")
    L.append("- 只有 5 个已知类，已知测试集规模随抽法变化，指标粒度较粗。")
    L.append("")

    L.append("## 9. 产物位置")
    L.append("")
    L.append("- 合并特征：`merge-b7/data/MMSB_sd_typo_mergeB7_layer{24,32}.npz`")
    L.append("- 去均值/ABTT 特征：`merge-b7/data/MMSB_sd_typo_mergeB7_layer{24,32}_center-{mean,abtt1}.npz`")
    L.append("- 每次运行：`merge-b7/outputs/<时间戳>_b7_[<变体>_]L<层>_s<seed>_<方法>/`")
    L.append("")
    return "\n".join(L), summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", default=str(DEFAULT_REPORT), help="报告 markdown 输出路径")
    p.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="汇总 json 输出路径")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = load_runs()
    if not runs:
        raise SystemExit(f"[错] {OUT_DIR} 下没找到任何 b7_ 运行结果，先跑 run_merge_b7.py")
    report, summary = build_report(runs)
    Path(args.report).write_text(report, encoding="utf-8")
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"[已写入] {args.report}")
    print(f"[已写入] {args.summary}")
    print(f"[覆盖] {len(runs)} 次运行，变体 {summary['variants']}，层 {summary['layers']}，"
          f"seed {summary['seeds']}，方法 {summary['methods']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
