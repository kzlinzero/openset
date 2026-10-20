"""批量驱动：2 层 × 5 seed × 5 个配置（单原型 + 多原型 K=1..4）= 50 次运行。

每次调用 `mproto_openset.py`，数据固定为 merge-b7 的 abtt1 特征；CAC 不在这里跑，
由汇总脚本直接读取 merge-b7/outputs/ 的旧结果。

用法：
  python run_mproto.py --dry-run
  python run_mproto.py --device gpu0
  python run_mproto.py --layers 24 --seeds 42 --force
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MPROTO = SCRIPT_DIR / "mproto_openset.py"
DATA_DIR = REPO_ROOT / "merge-b7" / "data"
OUT_DIR = SCRIPT_DIR / "outputs"
DRIVER_LOG_DIR = OUT_DIR / "_driver_logs"

DEFAULT_LAYERS = (24, 32)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
# (method, K, run-name label)
DEFAULT_CONFIGS = (
    ("proto", 1, "proto"),
    ("proto-multi", 1, "k1"),
    ("proto-multi", 2, "k2"),
    ("proto-multi", 3, "k3"),
    ("proto-multi", 4, "k4"),
)


def npz_for(layer: str) -> Path:
    p = DATA_DIR / f"MMSB_sd_typo_mergeB7_layer{layer}_center-abtt1.npz"
    if not p.exists():
        raise SystemExit(f"[错] 找不到特征 {p}；先确认 merge-b7/data/ 已生成 abtt1 特征")
    return p


def run_name(label: str, layer: str, seed: int) -> str:
    return f"mproto_{label}_L{layer}_s{seed}"


def existing_run_dir(name: str) -> Path | None:
    """只有包含 metrics.json 的目录才算完成，避免中断留下的空目录被当成已完成。"""
    hits = sorted(OUT_DIR.glob(f"*_{name}"))
    for d in reversed(hits):
        if any(d.glob("*.metrics.json")):
            return d
    return None


def summarize_run(run_dir: Path) -> str:
    mf = next(iter(run_dir.glob("*.metrics.json")), None)
    if mf is None:
        return "（没有 metrics.json）"
    d = json.loads(mf.read_text(encoding="utf-8"))
    method, res = next(iter(d["results"].items()))
    v = res["variants"]["B_matched"]
    return (f"{method} K={res['n_protos']}  AUROC={v['auroc']:.4f} "
            f"闭集={v['closed_set_acc']:.4f} 拒绝率={v['unknown_reject_rate']:.4f} "
            f"FPR={v['fpr_at_tpr95']:.4f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)),
                   help="逗号分隔的层号，如 24,32")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                   help="逗号分隔的 seed，如 42,43,44,45,46")
    p.add_argument("--device", default="gpu0", help="传给 mproto_openset.py 的 --device")
    p.add_argument("--python", default=sys.executable, help="用哪个解释器跑 mproto_openset.py")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不真的跑")
    p.add_argument("--force", action="store_true", help="即使已有完成的同名 run 也重跑")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    npzs = {layer: npz_for(layer) for layer in layers}

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        DRIVER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(layer, seed, method, k, label)
            for layer in layers for seed in seeds
            for method, k, label in DEFAULT_CONFIGS]
    print(f"[计划] {len(layers)} 层 × {len(seeds)} seed × {len(DEFAULT_CONFIGS)} 配置 "
          f"= {len(jobs)} 次运行")
    print(f"[解释器] {args.python}")
    print(f"[脚本] {MPROTO}")
    print(f"[设备] {args.device}\n")

    n_done = n_skip = n_fail = 0
    for i, (layer, seed, method, k, label) in enumerate(jobs, 1):
        name = run_name(label, layer, seed)
        existing = existing_run_dir(name)
        if existing is not None and not args.force:
            print(f"[{i:2d}/{len(jobs)}] 跳过（已存在 {existing.name}）")
            n_skip += 1
            continue

        cmd = [args.python, str(MPROTO),
               "--npz", str(npzs[layer]),
               "--method", method,
               "--n-protos", str(k),
               "--n-unknown", "2",
               "--seed", str(seed),
               "--device", args.device,
               "--output-dir", str(OUT_DIR),
               "--run-name", name]
        print(f"[{i:2d}/{len(jobs)}] {name}  [{method} K={k}]")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue

        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        log_text = proc.stdout or ""
        if proc.stderr:
            log_text += "\n[stderr]\n" + proc.stderr
        (DRIVER_LOG_DIR / f"{name}.stdout.log").write_text(log_text, encoding="utf-8")

        if proc.returncode != 0:
            n_fail += 1
            print(f"    [失败] 返回码 {proc.returncode}，输出末尾：")
            for line in log_text.splitlines()[-20:]:
                print("      " + line)
            print("\n[中止] 有一次运行失败，先排查再继续。")
            return 1

        done = existing_run_dir(name)
        if done is not None:
            print("    " + summarize_run(done))
        n_done += 1

    print(f"\n[完成] 新跑 {n_done} / 跳过 {n_skip} / 失败 {n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
