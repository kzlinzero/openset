"""汇总 merge-b7-mproto 的新结果与 merge-b7 的旧 CAC 结果，生成中文报告和 summary。

新结果：`outputs/*_mproto_{proto,k1..k4}_L{24,32}_s{42..46}`。
旧 CAC：`../merge-b7/outputs/*_b7_abtt1_L{24,32}_s{42..46}_cac`，只读。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "outputs"
OLD_OUT_DIR = REPO_ROOT / "merge-b7" / "outputs"
DATA_DIR = REPO_ROOT / "merge-b7" / "data"
DEFAULT_REPORT = SCRIPT_DIR / "多原型Proto_实验报告.md"
DEFAULT_SUMMARY = SCRIPT_DIR / "aggregate_mproto_summary.json"

LAYERS = ("24", "32")
SEEDS = (42, 43, 44, 45, 46)
METHOD_ORDER = ("cac", "proto", "k1", "k2", "k3", "k4")
METHOD_LABEL = {
    "cac": "cac（旧）",
    "proto": "proto（单原型）",
    "k1": "多原型 K=1",
    "k2": "多原型 K=2",
    "k3": "多原型 K=3",
    "k4": "多原型 K=4",
}
METRICS = [
    ("AUROC", "auroc", "{:.4f}"),
    ("闭集ACC", "closed_set_acc", "{:.4f}"),
    ("未知拒绝率", "unknown_reject_rate", "{:.4f}"),
    ("FPR@TPR95", "fpr_at_tpr95", "{:.4f}"),
]
EXPECTED_CAC = {
    ("24", "cac"): (0.9109, 0.0147),
    ("32", "cac"): (0.9056, 0.0193),
}

NEW_RE = re.compile(
    r"_(?P<name>mproto_(?P<kind>proto|k(?P<k>\d+))_L(?P<layer>-?\d+)_s(?P<seed>\d+))$")
OLD_RE = re.compile(
    r"_(?P<name>b7_abtt1_L(?P<layer>-?\d+)_s(?P<seed>\d+)_cac)$")


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), s


def fmt_ms(vals: list[float], fmt: str = "{:.4f}") -> str:
    if not vals:
        return "—"
    m, s = mean_std(vals)
    return f"{fmt.format(m)}±{fmt.format(s)}"


def _latest_metrics(run_dir: Path):
    hits = list(run_dir.glob("*.metrics.json"))
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def load_new_runs() -> list[dict]:
    runs: list[dict] = []
    for d in sorted(OUT_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = NEW_RE.search(d.name)
        if not m:
            continue
        mf = _latest_metrics(d)
        if mf is None:
            print(f"[警告] {d.name} 没有 metrics.json，跳过")
            continue
        data = json.loads(mf.read_text(encoding="utf-8"))
        kind = m.group("kind")
        method = "proto" if kind == "proto" else f"k{int(m.group('k'))}"
        diag_method = "proto" if method == "proto" else "proto-multi"
        res = next(iter(data["results"].values()))
        bm = res["variants"]["B_matched"]
        runs.append({
            "source": "new",
            "method": method,
            "layer": m.group("layer"),
            "seed": int(m.group("seed")),
            "run_dir": d,
            "n_protos": int(res.get("n_protos", 1)),
            "metrics": bm,
            "diag": data.get("diagnostics", {}).get(diag_method),
            "known_classes": data.get("known_classes", []),
            "unknown_classes": data.get("unknown_classes", []),
            "counts": data.get("counts", {}),
        })
    return runs


def load_old_cac() -> list[dict]:
    runs: list[dict] = []
    for d in sorted(OLD_OUT_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = OLD_RE.search(d.name)
        if not m:
            continue
        layer, seed = m.group("layer"), int(m.group("seed"))
        mf = d / f"proto_openset_seed{seed}_cac.metrics.json"
        if not mf.exists():
            mf = _latest_metrics(d)
        if mf is None:
            print(f"[警告] {d.name} 没有 metrics.json，跳过")
            continue
        data = json.loads(mf.read_text(encoding="utf-8"))
        bm = data["results"]["cac"]["variants"]["B_matched"]
        runs.append({
            "source": "old_cac",
            "method": "cac",
            "layer": layer,
            "seed": seed,
            "run_dir": d,
            "n_protos": 1,
            "metrics": bm,
            "diag": None,
            "known_classes": data.get("known_classes", []),
            "unknown_classes": data.get("unknown_classes", []),
            "counts": data.get("counts", {}),
        })
    return runs


def dedupe(runs: list[dict]) -> list[dict]:
    """同一 (method, layer, seed) 只保留最新的 run 目录（目录名带时间戳，排序即时间）。"""
    best: dict[tuple, dict] = {}
    for r in runs:
        best[(r["method"], r["layer"], r["seed"])] = r
    return list(best.values())


def check_complete(runs: list[dict]) -> None:
    present = {(r["method"], r["layer"], r["seed"]) for r in runs}
    expected = {(m, layer, seed) for m in METHOD_ORDER for layer in LAYERS for seed in SEEDS}
    missing = sorted(expected - present)
    if missing:
        lines = "\n".join(f"  - {m} L{layer} s{seed}" for m, layer, seed in missing)
        raise SystemExit(f"[错] 缺少 {len(missing)} 个运行，先补齐再汇总：\n{lines}")
    extra = sorted(present - expected)
    if extra:
        print(f"[警告] 有 {len(extra)} 个不在预期矩阵内的运行被忽略：{extra}")


def check_cac_baseline(runs: list[dict]) -> None:
    """旧 CAC 是本报告的唯一旧数据来源，先核对，防止读错目录。"""
    for layer in LAYERS:
        vals = [r["metrics"]["auroc"] for r in runs
                if r["method"] == "cac" and r["layer"] == layer]
        m, s = mean_std(vals)
        exp_m, exp_s = EXPECTED_CAC[(layer, "cac")]
        if abs(m - exp_m) > 1e-4 or abs(s - exp_s) > 1e-4:
            raise SystemExit(
                f"[错] L{layer} 旧 CAC 与旧报告不一致："
                f"实际 {m:.4f}±{s:.4f}，期望 {exp_m:.4f}±{exp_s:.4f}。"
                f"请检查 merge-b7/outputs 是否被改动。")


def metric_values(runs: list[dict], method: str, layer: str, field: str) -> list[float]:
    rs = [r for r in runs if r["method"] == method and r["layer"] == layer]
    rs.sort(key=lambda r: r["seed"])
    return [float(r["metrics"][field]) for r in rs]


def get_run(runs: list[dict], method: str, layer: str, seed: int) -> dict:
    for r in runs:
        if r["method"] == method and r["layer"] == layer and r["seed"] == seed:
            return r
    raise KeyError((method, layer, seed))


def diag_values(runs: list[dict], method: str, layer: str,
                split: str, field: str) -> list[float]:
    vals: list[float] = []
    for r in runs:
        if r["method"] != method or r["layer"] != layer or not r.get("diag"):
            continue
        v = r["diag"]["by_split"][split]["summary"].get(field)
        if v is not None:
            vals.append(float(v))
    return vals


def build_report(runs: list[dict]) -> tuple[str, dict]:
    new_runs = [r for r in runs if r["source"] == "new"]
    old_runs = [r for r in runs if r["source"] == "old_cac"]
    L: list[str] = []
    summary: dict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_runs": {"new": len(new_runs), "old_cac": len(old_runs), "total": len(runs)},
        "methods": list(METHOD_ORDER),
        "layers": list(LAYERS),
        "seeds": list(SEEDS),
        "metrics": {},
        "consistency": {},
        "diagnostics": {},
        "provenance": {
            "new_run_dirs": [str(r["run_dir"]) for r in sorted(
                new_runs, key=lambda r: (r["layer"], r["seed"], r["method"]))],
            "old_cac_dirs": [str(r["run_dir"]) for r in sorted(
                old_runs, key=lambda r: (r["layer"], r["seed"]))],
        },
    }

    L.append("# 合并 7 类 + abtt1 特征：多原型 Proto 开集识别实验")
    L.append("")
    L.append(f"- 生成时间：{summary['generated_at']}")
    L.append(f"- 运行数：{len(runs)}（新跑 {len(new_runs)}，复用旧 CAC {len(old_runs)}）")
    L.append(f"- 方法：{', '.join(METHOD_LABEL[m] for m in METHOD_ORDER)}")
    L.append(f"- 层：{', '.join(LAYERS)}；seed：{', '.join(map(str, SEEDS))}")
    L.append("- 新实验代码在 `merge-b7-mproto/`；`openset-proto/proto_openset.py` 与 `merge-b7/` 未修改。")
    L.append("")

    # ---------------- 1. 实验设置
    L.append("## 1. 实验设置")
    L.append("")
    L.append("| 项目 | 内容 |")
    L.append("|---|---|")
    L.append("| 数据 | MM-SafetyBench SD_TYPO，1680 条，13 类合并为 7 类 |")
    L.append("| 特征 | Qwen3-VL-8B-Instruct，4096 维，`center-abtt1`（去均值 + 去掉第一主方向） |")
    L.append("| 预处理拟合 | 均值和主方向用全部 1680 条估计（含未知类），沿用旧实验口径 |")
    L.append("| 层 | decoder 24、32 |")
    L.append("| 未知类 | 每 seed 随机抽 2 个合并类，`--n-unknown 2` |")
    L.append("| 切分 | 按 `question` 分组；已知类 90% 训练/验证、10% 测试，训练部分再切 10% 验证 |")
    L.append("| seed | 42–46（未知类组合与切分和旧报告一致） |")
    L.append("| 方法 | 单原型 proto、多原型 K=1–4；CAC 复用旧结果 |")
    L.append("| 多原型聚合 | 训练 `logsumexp_k(s/tau)`，打分 `max_{c,k} s` |")
    L.append("| 原型初始化 | `randn * 0.1`，不加多样性正则 |")
    L.append("| 超参数 | dim=256, tau=0.1, lr=1e-3, batch=64, weight_decay=1e-4, epochs=200, patience=50 |")
    L.append("| 阈值 | 验证集已知分数 95% 分位；测试集不参与 |")
    L.append("| 设备 | gpu0（旧 CAC 也是 gpu0） |")
    L.append("| 汇报口径 | 仅 B_matched（未知类下采样到与已知测试集 1:1） |")
    L.append("")
    L.append("**多原型定义**：`z = normalize(Wx)`，`P ∈ R^{C×K×d}` 每个原型 L2 归一化；")
    L.append("`s[c,k] = z·P[c,k]`，训练 logits 为 `l[c] = logsumexp_k(s[c,k]/tau)`，损失为交叉熵；")
    L.append("已知度分数 `max_{c,k} s[c,k]`，预测类 `argmax_c max_k s[c,k]`。K=1 时与单原型 Proto 等价。")
    L.append("")

    # ---------------- 2. 产物来源
    L.append("## 2. 产物来源")
    L.append("")
    L.append("| 项目 | 路径/说明 |")
    L.append("|---|---|")
    L.append(f"| 新实验脚本 | `{SCRIPT_DIR / 'mproto_openset.py'}` |")
    L.append(f"| 批量驱动 | `{SCRIPT_DIR / 'run_mproto.py'}` |")
    L.append(f"| 汇总脚本 | `{SCRIPT_DIR / 'aggregate_mproto.py'}` |")
    L.append(f"| 特征 | `{DATA_DIR / 'MMSB_sd_typo_mergeB7_layer{24,32}_center-abtt1.npz'}` |")
    L.append(f"| 新结果 | `{OUT_DIR}`（50 个 run 目录） |")
    L.append(f"| 旧 CAC | `{OLD_OUT_DIR}`（10 个 `*_b7_abtt1_L*_s*_cac` 目录） |")
    L.append("")
    L.append("单次新运行的命令模板：")
    L.append("")
    L.append("```bash")
    L.append("python mproto_openset.py \\")
    L.append("  --npz ../merge-b7/data/MMSB_sd_typo_mergeB7_layer<层>_center-abtt1.npz \\")
    L.append("  --method <proto|proto-multi> --n-protos <K> --n-unknown 2 --seed <seed> \\")
    L.append("  --device gpu0 --output-dir outputs --run-name mproto_<proto|kK>_L<层>_s<seed>")
    L.append("```")
    L.append("")

    # ---------------- 3. 主结果
    L.append("## 3. 主结果 · B_matched（5 个 seed 均值±标准差）")
    L.append("")
    for layer in LAYERS:
        L.append(f"### decoder layer {layer}")
        L.append("")
        L.append("| 方法 | " + " | ".join(label for label, _, _ in METRICS) + " |")
        L.append("|---|" + "---|" * len(METRICS))
        for method in METHOD_ORDER:
            cells = [METHOD_LABEL[method]]
            for _, field, fmt in METRICS:
                vals = metric_values(runs, method, layer, field)
                cells.append(fmt_ms(vals, fmt))
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

        layer_metrics: dict = {}
        for method in METHOD_ORDER:
            layer_metrics[method] = {}
            for _, field, _ in METRICS:
                vals = metric_values(runs, method, layer, field)
                m, s = mean_std(vals)
                layer_metrics[method][field] = {"mean": m, "std": s, "values": vals}
        summary["metrics"][layer] = layer_metrics

    L.append("说明：`cac（旧）` 直接读取 `merge-b7/outputs/` 的既有结果，未重跑；其 AUROC 均值/标准差")
    L.append("已与旧报告核对（L24 0.9109±0.0147，L32 0.9056±0.0193）。")
    L.append("")

    # ---------------- 4. K=1 一致性
    L.append("## 4. K=1 与单原型 proto 的一致性检查")
    L.append("")
    L.append("K=1 时多原型模型的损失和打分与单原型 Proto 数学等价，理应给出相同指标。")
    L.append("下表是同一层、同一 seed 下 `K=1 - proto` 的差值。")
    L.append("")
    consistency_rows = []
    max_abs_all = 0.0
    for layer in LAYERS:
        L.append(f"### decoder layer {layer}")
        L.append("")
        L.append("| seed | ΔAUROC | Δ闭集ACC | Δ未知拒绝率 | ΔFPR@TPR95 | max abs Δ |")
        L.append("|---|---|---|---|---|---|")
        for seed in SEEDS:
            p = get_run(runs, "proto", layer, seed)["metrics"]
            k1 = get_run(runs, "k1", layer, seed)["metrics"]
            diffs = {
                "auroc": k1["auroc"] - p["auroc"],
                "closed_set_acc": k1["closed_set_acc"] - p["closed_set_acc"],
                "unknown_reject_rate": k1["unknown_reject_rate"] - p["unknown_reject_rate"],
                "fpr_at_tpr95": k1["fpr_at_tpr95"] - p["fpr_at_tpr95"],
            }
            mx = max(abs(v) for v in diffs.values())
            max_abs_all = max(max_abs_all, mx)
            flag = " ⚠" if mx > 1e-4 else ""
            L.append(f"| {seed} | {diffs['auroc']:+.6f} | {diffs['closed_set_acc']:+.6f} | "
                     f"{diffs['unknown_reject_rate']:+.6f} | {diffs['fpr_at_tpr95']:+.6f} | "
                     f"{mx:.2e}{flag} |")
            consistency_rows.append({"layer": layer, "seed": seed, **diffs, "max_abs": mx})
        L.append("")
    verdict = "一致（≤1e-4）" if max_abs_all <= 1e-4 else "存在偏差，需排查实现"
    L.append(f"**结论：{verdict}**；所有层/seed 的 max|Δ| 为 {max_abs_all:.2e}。")
    L.append("")
    summary["consistency"] = {"max_abs_diff": max_abs_all, "rows": consistency_rows}

    # ---------------- 5. 原型诊断
    L.append("## 5. 原型诊断（仅已知类，5 个 seed 的均值±标准差）")
    L.append("")
    L.append("`active/K` 是每类“至少赢过一次样本”的原型数；`熵` 是样本对本类 K 个原型的")
    L.append("softmax 分配熵（按 log K 归一化到 [0,1]）；`类内余弦` 是同类原型两两余弦均值。")
    L.append("K=1 的熵为 0、类内余弦无定义。")
    L.append("")
    for layer in LAYERS:
        L.append(f"### decoder layer {layer}")
        L.append("")
        L.append("| 方法 | 划分 | active/K | 归一化熵 | 类内余弦均值 |")
        L.append("|---|---|---|---|---|")
        for method in ("k1", "k2", "k3", "k4"):
            k = int(method[1:])
            for split in ("train", "val", "test"):
                a = diag_values(runs, method, layer, split, "active_protos_mean")
                e = diag_values(runs, method, layer, split, "entropy_norm_mean")
                c = diag_values(runs, method, layer, split, "pairwise_cos_mean")
                a_txt = (f"{fmt_ms(a, '{:.2f}')}/{k}" if a else "—")
                e_txt = fmt_ms(e, "{:.4f}") if e else "—"
                c_txt = fmt_ms(c, "{:.4f}") if c else "N/A"
                L.append(f"| K={k} | {split} | {a_txt} | {e_txt} | {c_txt} |")
                summary["diagnostics"].setdefault(layer, {}).setdefault(method, {})[split] = {
                    "active_protos_mean": {"mean": mean_std(a)[0], "std": mean_std(a)[1]} if a else None,
                    "entropy_norm_mean": {"mean": mean_std(e)[0], "std": mean_std(e)[1]} if e else None,
                    "pairwise_cos_mean": {"mean": mean_std(c)[0], "std": mean_std(c)[1]} if c else None,
                }
        L.append("")
    L.append("注意：val/test 每个已知类只有约 10–25 条样本，诊断均值会随划分波动；逐类明细在")
    L.append("各 run 的 `metrics.json -> diagnostics` 中。")
    L.append("")

    # ---------------- 6. 自动摘要
    L.append("## 6. 自动摘要")
    L.append("")
    for layer in LAYERS:
        auc = {m: mean_std(metric_values(runs, m, layer, "auroc"))[0] for m in METHOD_ORDER}
        best = max(auc, key=auc.get)
        L.append(f"- **L{layer}**：AUROC 均值最高的是 `{METHOD_LABEL[best]}`（{auc[best]:.4f}）；"
                 f"K=3 为 {auc['k3']:.4f}，单原型 proto 为 {auc['proto']:.4f}，"
                 f"CAC 为 {auc['cac']:.4f}。")
        d_proto = auc["k3"] - auc["proto"]
        d_cac = auc["k3"] - auc["cac"]
        L.append(f"  K=3 相对单原型 ΔAUROC={d_proto:+.4f}，相对 CAC ΔAUROC={d_cac:+.4f}（边际均值差）。")
    L.append("")
    L.append("以上只描述本次数字，不对原因下结论；多原型是否真正有用，需要结合第 5 节的")
    L.append("active 原型数、分配熵和类内余弦一起看。")
    L.append("")

    # ---------------- 7. 注意事项
    L.append("## 7. 注意事项")
    L.append("")
    L.append("- **ABTT 口径**：均值和第一主方向用全部 1680 条（含未知类）估计，属于无标签的宽松口径；")
    L.append("  沿用旧实验是为了可比，不代表更严格的已知类拟合。")
    L.append("- **类别不均衡**：合并后 M5=465、M1=443，而 M6/M7=109；固定 K 对不同类的相对容量不同，")
    L.append("  拒绝率的标准差可能较大。")
    L.append("- **验证集双重用途**：验证集同时用于早停和阈值标定；K 固定为 3，K=1–4 只作敏感性，")
    L.append("  没有按测试集挑选 K。")
    L.append("- **单次初始化**：每个切分只训练一个模型初始化（seed 同时控制切分与初始化），")
    L.append("  没有单独估计训练方差。")
    L.append("- **口径**：只报 B_matched；未知类全量的 A_full 未计算。CAC 是旧结果，未在新目录重跑。")
    L.append("- **K=1 校验**：若第 4 节 max|Δ| 超过 1e-4，应先排查实现，而不是解读多原型结果。")
    L.append("")

    L.append("## 8. 产物")
    L.append("")
    L.append(f"- 报告：`{DEFAULT_REPORT}`")
    L.append(f"- 汇总：`{DEFAULT_SUMMARY}`")
    L.append(f"- 每次运行：`{OUT_DIR}/<时间戳>_mproto_*`")
    L.append("")
    return "\n".join(L) + "\n", summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", default=str(DEFAULT_REPORT), help="报告 Markdown 输出路径")
    p.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="汇总 JSON 输出路径")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = dedupe(load_new_runs() + load_old_cac())
    check_complete(runs)
    check_cac_baseline(runs)
    report, summary = build_report(runs)

    report_path = Path(args.report)
    summary_path = Path(args.summary)
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 报告：{report_path}")
    print(f"[完成] 汇总：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
