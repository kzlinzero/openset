"""汇总 merge-b7/outputs/ 下所有运行的 metrics.json，生成报告。

产物：
  - 合并B7_实验报告.md
  - aggregate_summary.json

用法：
  python aggregate.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs"
REPORT = SCRIPT_DIR / "合并B7_实验报告.md"
SUMMARY = SCRIPT_DIR / "aggregate_summary.json"

RUN_RE = re.compile(r"_(b7_L(?P<layer>\S+?)_s(?P<seed>\d+)_(?P<method>cac|arpl))$")

METHOD_ORDER = ("cac", "arpl")
METRICS = [
    ("AUROC", "auroc", "{:.4f}"),
    ("闭集ACC", "closed_set_acc", "{:.4f}"),
    ("未知拒绝率", "unknown_reject_rate", "{:.4f}"),
    ("FPR@TPR95", "fpr_at_tpr95", "{:.4f}"),
]


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
        runs.append({
            "layer": m.group("layer"),
            "seed": int(m.group("seed")),
            "method": m.group("method"),
            "dir": run_dir.name,
            "unknown_classes": d["unknown_classes"],
            "known_classes": d["known_classes"],
            "counts": d["counts"],
            "variants": {k: v["variants"] for k, v in d["results"].items()},
            "attribution": d["attribution"],
        })
    return runs


def mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def layer_sort_key(layer: str) -> float:
    try:
        return float(layer)
    except ValueError:
        return float("inf")  # '-1' 之类放到最后


def fmt_mean_std(vals: list[float], fmt: str) -> str:
    mu, sd = mean_std(vals)
    return f"{fmt.format(mu)}±{fmt.format(sd)}"


def pick(runs: list[dict], layer: str, method: str, variant: str, seed: int, field: str) -> float:
    for r in runs:
        if r["layer"] == layer and r["method"] == method and r["seed"] == seed:
            return float(r["variants"][method][variant][field])
    raise KeyError((layer, method, seed))


def build_report(runs: list[dict]) -> tuple[str, dict]:
    layers = sorted({r["layer"] for r in runs}, key=layer_sort_key)
    seeds = sorted({r["seed"] for r in runs})
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in runs)]
    L = []
    summary: dict = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "runs": len(runs), "layers": layers, "seeds": seeds, "methods": methods}

    L.append("# 合并类别 B（13→7）开集识别实验结果")
    L.append("")
    L.append(f"- 生成时间：{summary['generated_at']}")
    L.append(f"- 运行数：{len(runs)}（层 {layers} × seed {seeds} × 方法 {methods}）")
    L.append("- 脚本：`merge-b7/merge_taxonomy.py`（数据层合并）+ 未改动的 "
             "`openset-proto/proto_openset.py`")
    L.append("")

    # ---- 1. 设置
    L.append("## 1. 实验设置")
    L.append("")
    L.append("| 项目 | 内容 |")
    L.append("|---|---|")
    L.append("| 底座模型 | Qwen/Qwen3-VL-8B-Instruct（fp16，特征维度 4096） |")
    L.append("| 数据 | MM-SafetyBench SD_TYPO，1680 样本，13 类**合并为 7 类**（粒度 B） |")
    L.append("| 特征合并 | 只重映射 `unsafe_category`，features 原样保留 |")
    L.append(f"| 对比层 | decoder 第 {'、'.join(layers)} 层 |")
    L.append("| 未知类 | 每 seed 从 7 个合并类里**随机抽 2 个**（`--n-unknown 2`） |")
    L.append("| 方法 | " + "、".join(methods) + " |")
    L.append("| 阈值 tau | 各方法在每个 seed 下于验证集按 TPR 95% 校准 |")
    L.append("| 汇报口径 | B_matched（未知类下采样到与已知测试集 1:1）为主，A_full 作附表 |")
    L.append("")
    L.append("合并映射：")
    L.append("")
    L.append("| 合并类 | 含原类别 |")
    L.append("|---|---|")
    for name, members in [
        ("M1-Economic-Financial-Fraud", "05-EconomicHarm、06-Fraud、11-Financial_Advice"),
        ("M2-Legal-Opinion", "10-Legal_Opinion"),
        ("M3-Illegal-PhysicalHarm", "01-Illegal_Activity、04-Physical_Harm"),
        ("M4-Malware-PrivacyViolence", "03-Malware_Generation、09-Privacy_Violence"),
        ("M5-HateSpeech-Political", "02-HateSpeech、08-Political_Lobbying、13-Gov_Decision"),
        ("M6-Sex", "07-Sex"),
        ("M7-Health", "12-Health_Consultation"),
    ]:
        L.append(f"| {name} | {members} |")
    L.append("")

    # ---- 2. 未知类组合
    L.append("## 2. 每个 seed 抽中的未知类组合")
    L.append("")
    L.append("同一 seed 下 L24 与 L32 抽到的未知类相同（选择只取决于类别表与 seed）。")
    L.append("")
    L.append("| seed | 未知类（2 个） | 未知样本数 | 已知测试数 |")
    L.append("|---|---|---|---|")
    seen_seed = set()
    for r in runs:
        if r["seed"] in seen_seed:
            continue
        seen_seed.add(r["seed"])
        L.append("| {} | {} | {} | {} |".format(
            r["seed"], "、".join(r["unknown_classes"]),
            r["counts"]["unknown"], r["counts"]["known_test"]))
    L.append("")

    # ---- 3/4. 主表
    for variant, title, note in [
        ("B_matched", "3. 主结果 · 口径 B_matched（1:1）",
         "未知类下采样到与已知测试集等量；5 个 seed 的均值±样本标准差。"),
        ("A_full", "4. 附表 · 口径 A_full（未知类全部放入）",
         "该口径未知占绝大多数，「全判未知」的退化解也有很高的 ACC，请以 AUROC 为准。"),
    ]:
        L.append(f"## {title}")
        L.append("")
        L.append(note)
        L.append("")
        L.append("| 层 | 方法 | " + " | ".join(m[0] for m in METRICS) + " |")
        L.append("|---" * (2 + len(METRICS)) + "|")
        for layer in layers:
            for method in methods:
                cells = []
                for _, field, fmt in METRICS:
                    vals = [pick(runs, layer, method, variant, s, field) for s in seeds]
                    cells.append(fmt_mean_std(vals, fmt))
                L.append(f"| {layer} | {method} | " + " | ".join(cells) + " |")
        L.append("")
        summary[f"table_{variant}"] = {
            f"L{layer}_{method}": {
                field: mean_std([pick(runs, layer, method, variant, s, field) for s in seeds])[0]
                for _, field, _ in METRICS
            } for layer in layers for method in methods
        }

    # ---- 5. 误判方向（按未知类聚合）
    L.append("## 5. 未知类的误判方向（按未知类聚合，行内归一化）")
    L.append("")
    L.append("把「某个合并类被抽为未知」的那些 seed 的归属比例平均起来；"
             "`判为未知` = 被正确拒绝的比例，其余为被骗过阈值后归到的已知类。")
    L.append("")
    summary["attribution"] = {}
    for layer in layers:
        for method in methods:
            L.append(f"### 层 {layer} · {method}")
            L.append("")
            # 收集该 层×方法 下所有 run
            rs = [r for r in runs if r["layer"] == layer and r["method"] == method]
            per_cat: dict[str, list[dict]] = {}
            for r in rs:
                att = r["attribution"].get(method, {})
                for cat, row in att.items():
                    per_cat.setdefault(cat, []).append(row)
            if not per_cat:
                L.append("（无归属数据）")
                L.append("")
                continue
            L.append("| 未知类 | 被抽次数 | 判为未知 | 主要误判去向（top3） |")
            L.append("|---|---|---|---|")
            summary["attribution"][f"L{layer}_{method}"] = {}
            for cat in sorted(per_cat):
                rows = per_cat[cat]
                keys = {"判为未知"} | {k for row in rows for k in row if k != "判为未知"}
                avg = {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in keys}
                rej = avg.pop("判为未知", 0.0)
                sinks = sorted(avg.items(), key=lambda kv: -kv[1])[:3]
                detail = "、".join(f"{k} {v*100:.0f}%" for k, v in sinks if v > 0) or "无"
                L.append(f"| {cat} | {len(rows)} | {rej*100:.1f}% | {detail} |")
                summary["attribution"][f"L{layer}_{method}"][cat] = {
                    "n_drawn": len(rows), "reject_rate": rej,
                    "sinks": {k: v for k, v in sinks},
                }
            L.append("")

    # ---- 6. 注意事项
    L.append("## 6. 注意事项")
    L.append("")
    L.append("- 类别数从 13 变 7，**旧的 13 类数字不可直接对比**，只能定性参考"
             "（本次未跑同协议 13 类对照）。")
    L.append("- 合并后类大小很不均（M5=465、M1=443 vs M6=M7=109），"
             "而 7 类里随机抽 2 个未知 → 每次抽到的未知样本量差别很大，"
             "是单 seed 波动的主要来源。")
    L.append("- 只有 5 个已知类，已知测试集规模随抽法变化，指标粒度较粗。")
    L.append("- 脚本里『灰区类 10-13』的提示对合并后的类名不再生效（仅提示，无影响）。")
    L.append("")

    # ---- 7. 与旧 13 类结果指示性对比
    OLD13 = {  # 旧 13 类、5-seed 均值(B_matched)，来自
               # manual_log_collection/20260928-141645/不同层特征开放集对比报告_5seeds.md
        ("24", "cac"): {"auroc": 0.8468, "closed_set_acc": 0.8840,
                        "unknown_reject_rate": 0.3213, "fpr_at_tpr95": 0.6504},
        ("32", "cac"): {"auroc": 0.8574, "closed_set_acc": 0.8884,
                        "unknown_reject_rate": 0.3575, "fpr_at_tpr95": 0.6402},
        ("24", "arpl"): {"auroc": 0.8121, "closed_set_acc": 0.8611,
                         "unknown_reject_rate": 0.2393, "fpr_at_tpr95": 0.6743},
        ("32", "arpl"): {"auroc": 0.8446, "closed_set_acc": 0.8797,
                         "unknown_reject_rate": 0.3068, "fpr_at_tpr95": 0.6393},
    }
    L.append("## 7. 与旧 13 类结果的指示性对比")
    L.append("")
    L.append("**口径不同（旧 13 类抽 4 个未知 / 新 7 类抽 2 个未知），数字不可严格对比，"
             "仅供看方向。** 旧值取自 `manual_log_collection/20260928-141645/"
             "不同层特征开放集对比报告_5seeds.md`（同为 B_matched、5 seed 均值）。")
    L.append("")
    L.append("| 层 | 方法 | AUROC 13类→7类 | 闭集ACC 13类→7类 | 拒绝率 13类→7类 | FPR 13类→7类 |")
    L.append("|---|---|---|---|---|---|")
    summary["vs_old13"] = {}
    for layer in layers:
        for method in methods:
            if (layer, method) not in OLD13:
                continue
            cells, rec = [], {}
            for _, field, fmt in METRICS:
                old = OLD13[(layer, method)][field]
                new = summary["table_B_matched"][f"L{layer}_{method}"][field]
                cells.append(f"{fmt.format(old)} → {fmt.format(new)} ({new - old:+.4f})")
                rec[field] = {"old13": old, "new7": new, "delta": new - old}
            L.append(f"| {layer} | {method} | " + " | ".join(cells) + " |")
            summary["vs_old13"][f"L{layer}_{method}"] = rec
    L.append("")
    L.append("读法：**闭集 ACC 上升是合并的直接效果**（边界变少、每类样本变多）；"
             "若 AUROC / 拒绝率没有同步上升，说明合并**没有**解决未知类被误收的问题。")
    L.append("")

    L.append("## 8. 产物位置")
    L.append("")
    L.append("- 合并特征：`merge-b7/data/MMSB_sd_typo_mergeB7_layer{24,32}.npz`")
    L.append("- 每次运行：`merge-b7/outputs/<时间戳>_b7_L<层>_s<seed>_<方法>/`")
    L.append("- 汇总数据：`merge-b7/aggregate_summary.json`")
    L.append("")
    return "\n".join(L), summary


def main() -> int:
    runs = load_runs()
    if not runs:
        raise SystemExit(f"[错] {OUT_DIR} 下没找到任何 b7_ 运行结果，先跑 run_merge_b7.py")
    report, summary = build_report(runs)
    REPORT.write_text(report, encoding="utf-8")
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[已写入] {REPORT}")
    print(f"[已写入] {SUMMARY}")
    print(f"[覆盖] {len(runs)} 次运行，层 {summary['layers']}，seed {summary['seeds']}，方法 {summary['methods']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
