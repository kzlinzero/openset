"""单次开集实验：单原型 Proto 与多原型 Proto。

数据、切分、评估复用 `../openset-proto/proto_openset.py` 的公共函数（只读加载，不改旧文件），
训练循环在本文件内实现，以便接入新的多原型模型。

多原型模型：
  z = normalize(Wx)
  s[c,k] = z · P[c,k]
  l[c]   = logsumexp_k(s[c,k] / tau)
  loss   = CrossEntropy(l, y)
  score  = max_{c,k} s[c,k]          # 已知度分数，越大越像已知类
  pred   = argmax_c max_k s[c,k]

K=1 时与旧 ProtoModel 数学等价，用来校验实现。

用法：
  python mproto_openset.py --method proto --seed 42 --device gpu0
  python mproto_openset.py --method proto-multi --n-protos 3 --seed 42 --device gpu0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LEGACY_PATH = REPO_ROOT / "openset-proto" / "proto_openset.py"
DATA_DIR = REPO_ROOT / "merge-b7" / "data"
DEFAULT_NPZ = DATA_DIR / "MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz"
METHODS = ("proto", "proto-multi")


def _load_legacy():
    """目录名带连字符，不能普通 import；用文件路径加载旧模块，只读复用其函数。"""
    if not LEGACY_PATH.exists():
        raise SystemExit(f"[错] 找不到旧脚本：{LEGACY_PATH}")
    spec = importlib.util.spec_from_file_location("proto_openset_legacy", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[错] 无法加载旧脚本：{LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy()


# ---------------------------------------------------------------- 模型


class MultiProtoModel(legacy._TorchProtoBase):
    """投影层 + 每类 K 个可学习原型，类内 logsumexp 交叉熵。

    训练时同类所有原型通过 logsumexp 获得梯度；推理时用所有原型的最大余弦相似度作为
    已知度分数，与旧 ProtoModel 的口径一致。
    """

    def __init__(self, in_dim: int, dim: int, n_classes: int,
                 n_protos: int, tau: float) -> None:
        super().__init__(in_dim, dim, n_classes)
        if n_protos < 1:
            raise ValueError(f"n_protos 必须 >= 1，收到 {n_protos}")
        self.n_protos = int(n_protos)
        self.protos = nn.Parameter(torch.randn(n_classes, self.n_protos, dim) * 0.1)
        self.tau = float(tau)

    def sim(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        p = F.normalize(self.protos, dim=-1)
        return torch.einsum("bd,ckd->bck", z, p)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = self.sim(x)
        logits = torch.logsumexp(s / self.tau, dim=2)
        return F.cross_entropy(logits, y)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        s = self.sim(x).flatten(1)
        vals, idx = s.max(dim=1)
        pred = torch.div(idx, self.n_protos, rounding_mode="floor")
        return vals.cpu().numpy(), pred.cpu().numpy()


def build_model(method: str, in_dim: int, n_classes: int,
                args: argparse.Namespace) -> nn.Module:
    dim = max(args.dim, n_classes)
    if method == "proto":
        return legacy.ProtoModel(in_dim, dim, n_classes, args.tau)
    if method == "proto-multi":
        return MultiProtoModel(in_dim, dim, n_classes, args.n_protos, args.tau)
    raise SystemExit(f"[错] 未知方法：{method}（可用 {METHODS}）")


def effective_n_protos(method: str, args: argparse.Namespace) -> int:
    return 1 if method == "proto" else int(args.n_protos)


def train_model(method: str, x_train: np.ndarray, y_train: np.ndarray,
                x_val: np.ndarray, y_val: np.ndarray,
                args: argparse.Namespace, logger: "legacy.Logger",
                device: torch.device) -> tuple["legacy.TorchScorer", nn.Module, list[dict]]:
    """复制旧 run_torch_method 的训练流程，只把模型工厂换成本文件的 build_model。"""
    torch.manual_seed(args.seed)
    model = build_model(method, x_train.shape[1], int(y_train.max()) + 1, args).to(device)
    x_tr = torch.from_numpy(x_train).to(device)
    y_tr = torch.from_numpy(y_train).to(device)
    x_va = torch.from_numpy(x_val).to(device)
    y_va = torch.from_numpy(y_val).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(x_tr)
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history: list[dict] = []
    label = method if method == "proto" else f"{method}(K={args.n_protos})"

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = model.loss(x_tr[idx], y_tr[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = float(model.loss(x_va, y_va))
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1):
            logger(f"  {label} epoch {epoch:4d}  train_loss={total / n:.4f}  val_loss={val_loss:.4f}")
        if bad >= args.patience:
            logger(f"  {label} 验证集 loss 连续 {args.patience} 轮没降，提前停在第 {epoch} 轮")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger(f"  {label} 训练完成：最佳轮次 {best_epoch}，最佳 val_loss={best_val:.4f}")
    return legacy.TorchScorer(model, device), model, history


# ---------------------------------------------------------------- 诊断


@torch.no_grad()
def embed_numpy(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    model.eval()
    return model.embed(t).cpu().numpy()


@torch.no_grad()
def normalized_protos(model: nn.Module) -> np.ndarray:
    """返回 (C, K, d) 的归一化原型；单原型模型补一个 K=1 维。"""
    p = F.normalize(model.protos, dim=-1).detach().cpu().numpy()
    if p.ndim == 2:
        p = p[:, None, :]
    return p


def soft_entropy(s: np.ndarray, tau: float, n_protos: int) -> np.ndarray:
    """对 (n, K) 相似度做 softmax，返回归一化到 [0,1] 的熵；K=1 时为 0。"""
    if n_protos <= 1:
        return np.zeros(len(s), dtype=np.float64)
    logits = s / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    h = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    return h / np.log(n_protos)


def compute_diagnostics(model: nn.Module, method: str, n_protos: int, tau: float,
                        splits: dict[str, tuple[np.ndarray, np.ndarray]],
                        known: list[str], device: torch.device) -> dict:
    """只统计已知类；按 train/val/test 分开。"""
    protos = normalized_protos(model)
    out: dict = {"method": method, "n_protos": n_protos, "by_split": {}}
    for split_name, (x, y) in splits.items():
        z = embed_numpy(model, x, device)
        per_class: dict[str, dict] = {}
        active_vals: list[float] = []
        entropy_vals: list[float] = []
        cos_vals: list[float] = []

        for c, name in enumerate(known):
            m = y == c
            if not m.any():
                continue
            s = z[m] @ protos[c].T
            hard = s.argmax(axis=1)
            counts = np.bincount(hard, minlength=n_protos).astype(int)
            active = int((counts > 0).sum())
            ent = float(soft_entropy(s, tau, n_protos).mean())

            if n_protos > 1:
                gram = protos[c] @ protos[c].T
                iu = np.triu_indices(n_protos, k=1)
                cos = gram[iu]
                cos_stats = {"min": float(cos.min()), "mean": float(cos.mean()),
                             "max": float(cos.max())}
                cos_vals.append(cos_stats["mean"])
            else:
                cos_stats = {"min": None, "mean": None, "max": None}

            per_class[name] = {
                "n": int(m.sum()),
                "usage_counts": [int(v) for v in counts],
                "usage_ratio": [float(v / max(counts.sum(), 1)) for v in counts],
                "active_protos": active,
                "active_ratio": float(active / n_protos),
                "entropy_norm_mean": ent,
                "pairwise_cos": cos_stats,
            }
            active_vals.append(float(active))
            entropy_vals.append(ent)

        out["by_split"][split_name] = {
            "per_class": per_class,
            "summary": {
                "n_classes": len(per_class),
                "active_protos_mean": float(np.mean(active_vals)) if active_vals else None,
                "active_ratio_mean": float(np.mean([v / n_protos for v in active_vals]))
                if active_vals else None,
                "entropy_norm_mean": float(np.mean(entropy_vals)) if entropy_vals else None,
                "pairwise_cos_mean": float(np.mean(cos_vals)) if cos_vals else None,
            },
        }
    return out


def per_sample_assignment(model: nn.Module, n_protos: int, tau: float,
                          x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """返回每条样本的“预测类内最近原型编号”和预测类内 softmax 熵。"""
    z = embed_numpy(model, x, device)
    protos = normalized_protos(model)
    s = np.einsum("nd,ckd->nck", z, protos)
    flat = s.reshape(len(x), -1)
    idx = flat.argmax(axis=1)
    pred = idx // n_protos
    proto_id = idx % n_protos
    s_pred = s[np.arange(len(x)), pred, :]
    ent = soft_entropy(s_pred, tau, n_protos)
    return proto_id.astype(np.int64), ent.astype(np.float64)


# ---------------------------------------------------------------- 参数与主流程


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", default=str(DEFAULT_NPZ),
                        help="输入特征 npz（默认 merge-b7 的 abtt1 特征）")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs"),
                        help="日志与结果的输出目录")
    parser.add_argument("--unknown-categories", default=None,
                        help="指定未知类（逗号分隔）；不传则随机抽")
    parser.add_argument("--n-unknown", type=int, default=2, help="随机抽多少个未知类")
    parser.add_argument("--known-train-ratio", type=float, default=0.9,
                        help="已知类里分给训练（含验证）的比例")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="训练部分里再切给验证集的比例（只用于定阈值和早停）")
    parser.add_argument("--split-mode", choices=("group", "stratified"), default="group",
                        help="group = 按 question 分组切分；stratified = 普通分层切分")
    parser.add_argument("--method", choices=METHODS, default="proto-multi",
                        help="proto = 单原型；proto-multi = 每类 K 个原型")
    parser.add_argument("--n-protos", type=int, default=3, help="每类原型数 K（仅 proto-multi）")
    parser.add_argument("--dim", type=int, default=256, help="投影层输出维度")
    parser.add_argument("--epochs", type=int, default=200, help="最多训练多少轮")
    parser.add_argument("--patience", type=int, default=50, help="验证集 loss 多少轮不降就早停")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--tau", type=float, default=0.1, help="相似度 softmax 的温度")
    parser.add_argument("--target-tpr", type=float, default=0.95,
                        help="验证集上标定阈值时要求通过的已知样本比例")
    parser.add_argument("--pca-dim", type=int, default=0,
                        help="先做 PCA 降到多少维（0 表示不降维；只在训练集上 fit）")
    parser.add_argument("--log-every", type=int, default=20, help="每多少轮打一条训练日志（0 关闭）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cpu",
                        help="训练用哪块设备：cpu / gpu0 / gpu1（也接受 cuda:0）")
    parser.add_argument("--run-name", default="",
                        help="本次运行的目录名后缀；会在 outputs/ 下建带时间戳的子目录")
    parser.add_argument("--tag", default="", help="附加到输出文件名后面的标签")
    return parser.parse_args(argv)


def output_stem(args: argparse.Namespace) -> str:
    stem = f"mproto_openset_seed{args.seed}_{args.method}"
    if args.method == "proto-multi":
        stem += f"_k{args.n_protos}"
    if args.tag:
        stem += f"_{args.tag}"
    return stem


def run(args: argparse.Namespace) -> int:
    if args.n_protos < 1:
        raise SystemExit("[错] --n-protos 必须 >= 1")
    legacy.set_seed(args.seed)
    device = legacy.resolve_device(args.device)
    stem = output_stem(args)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{stamp}_{args.run_name or stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = legacy.Logger(run_dir / f"{stem}.log")
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_protos = effective_n_protos(args.method, args)

    logger(f"[运行] {started}")
    logger(f"[目录] {run_dir.resolve()}")
    logger(f"[设备] {legacy.describe_device(device)}")
    logger(f"[参数] {json.dumps(vars(args), ensure_ascii=False)}")
    logger(f"[旧脚本] {LEGACY_PATH}（只读复用，未修改）")

    # ---- 数据
    data = legacy.load_features(Path(args.npz))
    feats, cats, questions, index = (
        data["features"], data["category"], data["question"], data["index"]
    )
    logger.section("数据")
    logger(f"特征文件   : {Path(args.npz).resolve()}")
    logger(f"样本 / 维度: {feats.shape[0]} / {feats.shape[1]}")

    known, unknown = legacy.choose_unknown(
        cats, args.n_unknown, args.seed,
        args.unknown_categories.split(",") if args.unknown_categories else None)
    known = sorted(known)
    unknown = sorted(unknown)
    logger(f"已知类（{len(known)} 个）: {known}")
    logger(f"未知类（{len(unknown)} 个）: {unknown}")

    # ---- 划分
    def is_known(c: np.ndarray) -> np.ndarray:
        return np.isin(c, known)

    known_idx = np.where(is_known(cats))[0]
    unknown_idx = np.where(~is_known(cats))[0]
    trainval_idx, known_test_idx = legacy.split_once(
        known_idx, cats, questions, 1.0 - args.known_train_ratio, args.seed, args.split_mode)
    train_idx, val_idx = legacy.split_once(
        trainval_idx, cats, questions, args.val_ratio, args.seed + 1, args.split_mode)

    cls_of = {name: k for k, name in enumerate(known)}
    y_all = np.array([cls_of.get(c, -1) for c in cats], dtype=np.int64)
    x = feats.copy()
    if args.pca_dim > 0:
        from sklearn.decomposition import PCA

        k = min(args.pca_dim, len(train_idx) - 1, x.shape[1])
        pca = PCA(n_components=k, random_state=args.seed).fit(x[train_idx])
        x = pca.transform(x).astype(np.float32)
        logger(f"PCA 降维（只在训练集上 fit）: {feats.shape[1]} → {k}，"
               f"累计解释方差 {pca.explained_variance_ratio_.sum():.3f}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, 1e-12, None)

    x_train, y_train = x[train_idx], y_all[train_idx]
    x_val, y_val = x[val_idx], y_all[val_idx]
    x_ktest, y_ktest = x[known_test_idx], y_all[known_test_idx]
    x_utest = x[unknown_idx]

    per_class_split = {
        name: {
            "train": int((cats[train_idx] == name).sum()),
            "val": int((cats[val_idx] == name).sum()),
            "test": int((cats[known_test_idx] == name).sum()),
        }
        for name in known
    }

    logger.section("划分")
    logger(f"切分方式   : {args.split_mode}"
           + ("（按 question 分组，同一条问题不会跨划分）" if args.split_mode == "group" else ""))
    logger(f"训练 {len(train_idx)} 条 / 验证 {len(val_idx)} 条 / 已知测试 {len(known_test_idx)} 条"
           f" / 未知 {len(unknown_idx)} 条")
    rows = [[name, str(per_class_split[name]["train"]), str(per_class_split[name]["val"]),
             str(per_class_split[name]["test"])] for name in known]
    logger(legacy.render_table(["known class", "train", "val", "test"], rows))
    short = [name for name in known if per_class_split[name]["test"] == 0]
    if short:
        logger(f"[警告] 这些已知类在测试集里一条都没有：{short}（分组切分可能把整类都分走了）")
    logger("未知类样本数：" + "，".join(
        f"{name} {int((cats[unknown_idx] == name).sum())}" for name in unknown))

    # ---- 训练
    logger.section("方法")
    logger(f"[{args.method}] 每类原型数 K={n_protos}"
           + ("（K=1 应与单原型 proto 一致）" if args.method == "proto-multi" and n_protos == 1 else ""))
    scorer, model, history = train_model(
        args.method, x_train, y_train, x_val, y_val, args, logger, device)

    # ---- 阈值与评估
    score_val, _ = scorer.score(x_val)
    tau = float(np.percentile(score_val, (1.0 - args.target_tpr) * 100.0))
    logger(f"  阈值 tau = {tau:.4f}（验证集 {len(score_val)} 条已知样本的 "
           f"{(1 - args.target_tpr) * 100:.1f}% 分位，目标 TPR={args.target_tpr:.2f}）")

    score_ktest, pred_ktest = scorer.score(x_ktest)
    score_utest, pred_utest = scorer.score(x_utest)

    n_match = len(known_test_idx)
    matched_pos = np.searchsorted(
        unknown_idx, legacy.make_matched_unknown(unknown_idx, cats, n_match, args.seed + 2))
    result = legacy.evaluate(
        score_ktest, score_utest[matched_pos], pred_ktest, y_ktest,
        pred_utest[matched_pos], tau, len(known))
    result["description"] = f"未知类下采样到 {len(matched_pos)} 条与已知测试集一比一"

    logger.section("主结果 · 口径 B_matched")
    logger(f"已知测试 {result['n_known']} 条 / 未知 {result['n_unknown']} 条 "
           f"（未知占 {result['unknown_ratio'] * 100:.1f}%）")
    metric_defs = [
        ("AUROC", "auroc", "{:.4f}"),
        ("闭集ACC(仅已知)", "closed_set_acc", "{:.4f}"),
        ("已知接受率", "known_accept_rate", "{:.4f}"),
        ("未知拒绝率", "unknown_reject_rate", "{:.4f}"),
        ("FPR@TPR95", "fpr_at_tpr95", "{:.4f}"),
    ]
    logger(legacy.render_table(
        ["metric", args.method],
        [[label, fmt.format(result[field])] for label, field, fmt in metric_defs]))

    # ---- 未知类归属
    attr_lines, attr_full = legacy.attribution_table(
        cats[unknown_idx], pred_utest, score_utest, tau, known)
    logger.section("未知类的误判方向（按原型距离归属，行归一化）")
    logger(f"[{args.method}] tau={tau:.4f}")
    logger("\n".join(attr_lines))

    # ---- 原型诊断
    diagnostics = compute_diagnostics(
        model, args.method, n_protos, args.tau,
        {"train": (x_train, y_train), "val": (x_val, y_val), "test": (x_ktest, y_ktest)},
        known, device)
    logger.section("原型诊断（仅已知类）")
    for split_name, split_diag in diagnostics["by_split"].items():
        s = split_diag["summary"]
        logger(f"[{split_name}] 类均 active={s['active_protos_mean']:.2f}/{n_protos}  "
               f"熵={s['entropy_norm_mean']:.4f}  类内余弦="
               + ("N/A" if s["pairwise_cos_mean"] is None else f"{s['pairwise_cos_mean']:.4f}"))
        diag_rows = []
        for name, d in split_diag["per_class"].items():
            cos = d["pairwise_cos"]
            diag_rows.append([
                name, str(d["n"]), f"{d['active_protos']}/{n_protos}",
                f"{d['entropy_norm_mean']:.4f}",
                "N/A" if cos["mean"] is None else f"{cos['min']:.3f}/{cos['mean']:.3f}/{cos['max']:.3f}",
            ])
        logger(legacy.render_table(
            ["known class", "n", "active", "entropy", "cos min/mean/max"], diag_rows))

    # ---- 逐样本原型分配
    proto_id_ktest, ent_ktest = per_sample_assignment(model, n_protos, args.tau, x_ktest, device)
    proto_id_utest, ent_utest = per_sample_assignment(model, n_protos, args.tau, x_utest, device)

    # ---- 落盘
    scores_path = run_dir / f"{stem}.scores.csv"
    in_matched = np.zeros(len(unknown_idx), dtype=bool)
    in_matched[matched_pos] = True
    header = ["index", "category", "is_unknown", "in_matched_unknown",
              f"score_{args.method}", f"pred_{args.method}",
              f"proto_id_{args.method}", f"entropy_{args.method}"]
    lines = [",".join(header)]
    for pos in range(len(known_test_idx)):
        row = [
            str(index[known_test_idx[pos]]), cats[known_test_idx[pos]], "0", "0",
            f"{score_ktest[pos]:.6f}", known[pred_ktest[pos]],
            str(int(proto_id_ktest[pos])), f"{ent_ktest[pos]:.6f}",
        ]
        lines.append(",".join(row))
    for pos in range(len(unknown_idx)):
        row = [
            str(index[unknown_idx[pos]]), cats[unknown_idx[pos]], "1",
            "1" if in_matched[pos] else "0",
            f"{score_utest[pos]:.6f}", known[pred_utest[pos]],
            str(int(proto_id_utest[pos])), f"{ent_utest[pos]:.6f}",
        ]
        lines.append(",".join(row))
    scores_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics_path = run_dir / f"{stem}.metrics.json"
    metrics_path.write_text(json.dumps({
        "experiment": {
            "name": "merge-b7-mproto",
            "method": args.method,
            "n_protos": n_protos,
            "data_npz": str(Path(args.npz).resolve()),
        },
        "known_classes": known,
        "unknown_classes": unknown,
        "counts": {"train": len(train_idx), "val": len(val_idx),
                   "known_test": len(known_test_idx), "unknown": len(unknown_idx),
                   "matched_unknown": len(matched_pos)},
        "results": {
            args.method: {
                "tau": tau,
                "epochs_run": len(history),
                "last_epoch": history[-1]["epoch"] if history else None,
                "n_protos": n_protos,
                "variants": {"B_matched": result},
            }
        },
        "attribution": {args.method: attr_full},
        "diagnostics": {args.method: diagnostics},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = run_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps({
        "created_at": started,
        "run_dir": str(run_dir.resolve()),
        "args": vars(args),
        "device": {"requested": args.device, "resolved": str(device),
                   "name": legacy.describe_device(device)},
        "npz": str(Path(args.npz).resolve()),
        "legacy_script": str(LEGACY_PATH),
        "feature_dim_raw": int(feats.shape[1]),
        "feature_dim_used": int(x.shape[1]),
        "n_samples": int(feats.shape[0]),
        "method": args.method,
        "n_protos": n_protos,
        "known_classes": known,
        "unknown_classes": unknown,
        "methods": [args.method],
        "per_class_split": per_class_split,
        "unknown_counts": {name: int((cats[unknown_idx] == name).sum()) for name in unknown},
        "diagnostics_definition": {
            "usage": "已知类样本在本类 K 个原型中最近原型的硬分配计数/比例",
            "entropy": "softmax(s/ tau) 的归一化熵，K=1 时为 0",
            "pairwise_cos": "同类原型两两余弦的 min/mean/max，K=1 时为 N/A",
            "splits": ["train", "val", "test"],
        },
        "outputs": {
            "log": f"{stem}.log", "metrics": f"{stem}.metrics.json",
            "scores": f"{stem}.scores.csv", "manifest": f"{stem}.manifest.json",
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.section("产物")
    for p in (logger.path, metrics_path, scores_path, manifest_path):
        logger(f"  {p}")
    logger.close()
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
