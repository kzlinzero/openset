"""汇总各层 proto_openset 运行的 B_matched 指标（多种子）。

自动发现 outputs/ 下 run-name 为 layerN（seed 42）或 layerN_sSEED 的运行目录，
按 层 x 方法 x 指标 聚合跨种子的均值与标准差（样本标准差，ddof=1）。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "openset-proto" / "outputs"

SEEDS = [42, 43, 44, 45, 46]
METRICS = ["auroc", "closed_set_acc", "unknown_reject_rate", "fpr_at_tpr95"]
METHODS = ["ncm", "proto", "cac", "arpl"]
RUN_NAME_RE = re.compile(r"^layer(-?\d+)(?:_s(\d+))?$")


def discover():
    """返回 {(layer, seed): (run_dir, metrics_dict)}"""
    found = {}
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_NAME_RE.match(d.name.split("_", 1)[1] if "_" in d.name else "")
        if not m:
            continue
        layer, seed = int(m.group(1)), int(m.group(2) or 42)
        mf = d / f"proto_openset_seed{seed}.metrics.json"
        if not mf.exists():
            print(f"[警告] 缺少 {mf}", file=sys.stderr)
            continue
        found[(layer, seed)] = (d, json.loads(mf.read_text(encoding="utf-8")))
    return found


def main():
    runs = discover()
    layers = sorted({ly for ly, _ in runs})
    seeds = sorted({sd for _, sd in runs})

    # 每个种子的未知类组合（按层应一致；逐层列出以防万一）
    unknown_by_layer = {}
    counts_by_layer = {}
    for ly in layers:
        combos = []
        for sd in seeds:
            if (ly, sd) in runs:
                combos.append((sd, runs[(ly, sd)][1]["unknown_classes"]))
        unknown_by_layer[ly] = combos
        c = runs[(ly, seeds[0])][1]["counts"] if (ly, seeds[0]) in runs else {}
        counts_by_layer[ly] = c

    # agg[layer][method][metric] = (mean, std)
    agg = {}
    for ly in layers:
        agg[ly] = {}
        for mth in METHODS:
            vals = {k: [] for k in METRICS}
            for sd in seeds:
                if (ly, sd) not in runs:
                    continue
                bm = runs[(ly, sd)][1]["results"][mth]["variants"]["B_matched"]
                for k in METRICS:
                    vals[k].append(bm[k])
            agg[ly][mth] = {
                k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                for k, v in vals.items()
            }

    print(f"发现 {len(runs)} 次运行，层 {layers}，种子 {seeds}\n")
    for ly in layers:
        print(f"layer {ly}: counts={counts_by_layer[ly]}")
        for sd, unk in unknown_by_layer[ly]:
            print(f"  seed {sd}: unknown={unk}")

    print("\n| 层 | 方法 | AUROC | 闭集ACC | 未知拒绝率 | FPR@TPR95 |")
    print("|---|---|---|---|---|---|")
    for ly in layers:
        for mth in METHODS:
            cells = []
            for k in METRICS:
                mu, sd_ = agg[ly][mth][k]
                cells.append(f"{mu:.4f}±{sd_:.4f}")
            print(f"| {ly} | {mth} | " + " | ".join(cells) + " |")

    payload = {
        "seeds": seeds,
        "layers": layers,
        "unknown_by_layer": {str(k): v for k, v in unknown_by_layer.items()},
        "counts_by_layer": {str(k): v for k, v in counts_by_layer.items()},
        "agg": {str(ly): {mth: {k: list(v) for k, v in agg[ly][mth].items()}
                          for mth in METHODS} for ly in layers},
        "n_runs": len(runs),
    }
    out_json = Path(__file__).with_suffix(".json")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
