"""驱动器：在合并后的 7 类特征上重跑开集实验。

不改 openset-proto/proto_openset.py 一行——它只是被调用，参数由这里拼好：

    python ../openset-proto/proto_openset.py \
        --npz data/MMSB_sd_typo_mergeB7_layer{L}.npz \
        --method {cac|arpl} --n-unknown 2 --seed {S} --device gpu0 \
        --output-dir outputs --run-name b7_L{L}_s{S}_{method}

因为跑的是 7 类 npz，`--n-unknown 2` 就等于"从 7 个合并类里随机抽 2 个当未知类"。

用法：
  python run_merge_b7.py                     # 2 层 × 5 seed × {cac,arpl} = 20 次
  python run_merge_b7.py --layers 32 --seeds 42
  python run_merge_b7.py --dry-run           # 只打印命令，不执行
  python run_merge_b7.py --device cpu

已完成（outputs/ 下已有同名 run 目录）的会被跳过，方便中断后续跑。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTO = REPO_ROOT / "openset-proto" / "proto_openset.py"
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "outputs"
DRIVER_LOG_DIR = OUT_DIR / "_driver_logs"

DEFAULT_LAYERS = (24, 32)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_METHODS = ("cac", "arpl")
# none = 原始合并特征（基线，复用已有产物）；mean / abtt{k} = 去均值 / all-but-the-top 变体
DEFAULT_VARIANTS = ("none", "mean", "abtt1")


def npz_for(layer: str, variant: str = "none") -> Path:
    if variant == "none":
        p = DATA_DIR / f"MMSB_sd_typo_mergeB7_layer{layer}.npz"
    else:
        p = DATA_DIR / f"MMSB_sd_typo_mergeB7_layer{layer}_center-{variant}.npz"
    if not p.exists():
        raise SystemExit(
            f"[错] 找不到特征 {p}；基线先跑 merge_taxonomy.py，变体先跑 center_features.py")
    return p


def run_name(layer: str, seed: int, method: str, variant: str = "none") -> str:
    tok = "" if variant == "none" else f"{variant}_"
    return f"b7_{tok}L{layer}_s{seed}_{method}"


def existing_run_dir(name: str) -> Path | None:
    hits = sorted(OUT_DIR.glob(f"*_{name}"))
    return hits[-1] if hits else None


def summarize_run(run_dir: Path) -> str:
    """从一次运行的 metrics.json 里摘出关键信息，打印一行。"""
    mf = next(iter(run_dir.glob("*.metrics.json")), None)
    if mf is None:
        return "（没有 metrics.json）"
    d = json.loads(mf.read_text(encoding="utf-8"))
    out = []
    for method, res in d["results"].items():
        v = res["variants"]["B_matched"]
        out.append(f"{method}: AUROC={v['auroc']:.4f} 闭集={v['closed_set_acc']:.4f} "
                   f"拒绝率={v['unknown_reject_rate']:.4f} FPR={v['fpr_at_tpr95']:.4f}")
    return f"未知类={d['unknown_classes']}\n    " + "\n    ".join(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)),
                   help="逗号分隔的层号，如 24,32")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                   help="逗号分隔的 seed，如 42,43,44,45,46")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                   help="逗号分隔的方法，如 cac,arpl")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                   help="逗号分隔的预处理变体：none（原始）/ mean（去均值）/ abtt{k}（去均值+ABTT）")
    p.add_argument("--device", default="gpu0", help="传给 proto_openset.py 的 --device")
    p.add_argument("--python", default=sys.executable, help="用哪个解释器跑 proto_openset.py")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不真的跑")
    p.add_argument("--force", action="store_true", help="即使已有同名 run 目录也重跑")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    variants = [s.strip() for s in args.variants.split(",") if s.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DRIVER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(V, L, S, M) for V in variants for L in layers for S in seeds for M in methods]
    print(f"[计划] {len(variants)} 变体 × {len(layers)} 层 × {len(seeds)} seed × {len(methods)} 方法 "
          f"= {len(jobs)} 次运行")
    print(f"[解释器] {args.python}")
    print(f"[脚本] {PROTO}")
    print(f"[设备] {args.device}\n")

    n_done = n_skip = n_fail = 0
    for i, (V, L, S, M) in enumerate(jobs, 1):
        name = run_name(L, S, M, V)
        npz = npz_for(L, V)
        existing = existing_run_dir(name)
        if existing is not None and not args.force:
            print(f"[{i:2d}/{len(jobs)}] 跳过（已存在 {existing.name}）")
            n_skip += 1
            continue

        cmd = [args.python, str(PROTO),
               "--npz", str(npz),
               "--method", M,
               "--n-unknown", "2",
               "--seed", str(S),
               "--device", args.device,
               "--output-dir", str(OUT_DIR),
               "--run-name", name]
        print(f"[{i:2d}/{len(jobs)}] {name}  [{V}]")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue

        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        (DRIVER_LOG_DIR / f"{name}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            n_fail += 1
            print(f"    [失败] 返回码 {proc.returncode}，stdout 末尾：")
            for line in (proc.stdout or "").splitlines()[-15:]:
                print("      " + line)
            for line in (proc.stderr or "").splitlines()[-15:]:
                print("      " + line)
            print("\n[中止] 有一次运行失败，先排查再继续。")
            return 1

        done = existing_run_dir(name)
        if done is not None:
            print("    " + summarize_run(done).replace("\n", "\n    "))
        n_done += 1

    print(f"\n[完成] 新跑 {n_done} / 跳过 {n_skip} / 失败 {n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
