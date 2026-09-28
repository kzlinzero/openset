"""开集识别第一版：冻结 VLM 特征 + 原型学习。

流程：
  读 get-embedding 抽好的样本级 feature（.npz）
    → 随机挑 4 个类当未知类，剩下 9 类当已知类
    → 已知类每类 90% 训练 / 10% 测试，训练集里再切 10% 当「纯已知」验证集
    → 在训练集上得到 9 个原型（四种方法 ncm / proto / cac / arpl）
    → 在验证集上定阈值 τ（默认让 95% 的已知样本通过）
    → 测试：已知类测试集 + 未知类，按两个配比口径各算一遍指标
    → 日志与结果落盘到 --output-dir

四种方法（推理时统一输出「已知度分数」，越大越像已知类，阈值逻辑因此完全一致）：

  ncm    不训练。原型 = 每类训练特征（L2 归一化后）的均值再归一化。
         分数 = 到最近原型的余弦相似度。
  proto  4096→dim 的线性投影层（输出 L2 归一化）+ 9 个可学习原型，
         损失 = 对余弦相似度做 softmax 交叉熵（ProtoNet 风格）。
         分数 = 到最近原型的余弦相似度。
  cac    投影层（输出 L2 归一化）+ 固定锚点（原型不学），用 Class Anchor
         Clustering（WACV 2021）的 anchor + dot 两个损失。
         分数 = 到最近锚点的负欧氏距离。
  arpl   投影层（输出 L2 归一化）+ 每类一个可学习「反向点」（代表「该类之外的
         空间」），用 Adversarial Reciprocal Points Learning（ECCV 2020）的
         正指数 softmax 做分类，另加一个 margin 正则项。
         分数 = 到最远反向点的平方欧氏距离（ARPL 的约定：离反向点越远越像已知类）。

本版刻意不做的简化（避免一上来就复杂）：
  * cac / arpl 只实现论文的核心损失项，不做伪未知样本生成
    （ARPL+CS 的混淆空间、PROSER 的 manifold mixup）；
  * cac 的锚点半径 alpha 因为特征被 L2 归一化，与论文默认值不同，用 --cac-alpha 调；
  * 不做白化 / PCA（--pca-dim 默认 0，只留了口子）；
  * 只跑一组类别划分、不画图、不微调 backbone。

用法：
  python proto_openset.py                     # 随机 4 类未知，四种方法全跑
  python proto_openset.py --method ncm --seed 7
  python proto_openset.py --unknown-categories 10-Legal_Opinion,11-Financial_Advice,12-Health_Consultation,13-Gov_Decision
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = REPO_ROOT / "get-embedding" / "outputs" / "MMSB_sd_typo_decoder[-1].npz"
ALL_METHODS = ("ncm", "proto", "cac", "arpl")


# ---------------------------------------------------------------- 基础设施


class Logger:
    """日志同时打到 stdout 和文件（仓库里没有 logging，统一用 print 风格）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fp = path.open("w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.fp.write(msg + "\n")
        self.fp.flush()

    def section(self, title: str) -> None:
        self(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    def close(self) -> None:
        self.fp.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    """把 --device 的写法解析成 torch.device。

    支持 cpu / gpu0 / gpu1 / cuda / cuda:0 / cuda:1。选到不存在的卡会直接报错，
    不会静默退回 CPU。
    """
    s = spec.strip().lower()
    if s in ("", "cpu"):
        return torch.device("cpu")
    if s.startswith("gpu"):
        s = "cuda" + (f":{s[3:]}" if s[3:] else "")
    if not s.startswith("cuda"):
        raise SystemExit(f"[错] 认不出的 --device：{spec}（可用 cpu / gpu0 / gpu1 / cuda:0）")
    if not torch.cuda.is_available():
        raise SystemExit(
            f"[错] --device {spec} 需要 CUDA，但当前 torch 看不到可用 GPU"
            f"（torch {torch.__version__}，编译时 CUDA {torch.version.cuda}）。改用 --device cpu 即可。")
    dev = torch.device(s)
    index = dev.index if dev.index is not None else 0
    if index >= torch.cuda.device_count():
        raise SystemExit(
            f"[错] --device {spec} 指向第 {index} 块卡，但机器上只有 "
            f"{torch.cuda.device_count()} 块（可用 gpu0..gpu{torch.cuda.device_count() - 1}）")
    return dev


def describe_device(device: torch.device) -> str:
    """给日志用的设备描述，带 GPU 型号。"""
    if device.type == "cpu":
        return "cpu"
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"{device}（{torch.cuda.get_device_name(index)}）"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """把 headers / rows 渲染成定宽对齐的文本表（表头用 ASCII，避免中文宽度问题）。"""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    out = [line, "-" * len(line)]
    for row in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(out)


# ---------------------------------------------------------------- 数据与划分


def load_features(npz_path: Path) -> dict[str, np.ndarray]:
    """读 .npz：features / unsafe_category / question / index。"""
    if not npz_path.exists():
        raise SystemExit(f"[错] 找不到特征文件：{npz_path}")
    with np.load(npz_path) as d:
        data = {
            "features": np.asarray(d["features"], dtype=np.float32),
            "category": np.asarray(d["unsafe_category"]).astype(str),
            "question": np.asarray(d["question"]).astype(str),
            "index": np.asarray(d["index"], dtype=np.int64),
        }
    if not np.isfinite(data["features"]).all():
        raise SystemExit("[错] features 里有 NaN / Inf，先回 get-embedding 重抽")
    return data


def choose_unknown(
    categories: np.ndarray, n_unknown: int, seed: int, explicit: list[str] | None
) -> tuple[list[str], list[str]]:
    """挑出未知类，返回 (known 类别名, unknown 类别名)。"""
    all_cats = sorted(set(categories.tolist()))
    if explicit:
        unknown = list(dict.fromkeys(explicit))
        missing = [c for c in unknown if c not in all_cats]
        if missing:
            raise SystemExit(
                f"[错] --unknown-categories 里有不存在的类别：{missing}\n可用类别：{all_cats}"
            )
        if len(unknown) >= len(all_cats):
            raise SystemExit("[错] 未知类不能把全部类别都占掉")
    else:
        if not 1 <= n_unknown < len(all_cats):
            raise SystemExit(f"[错] --n-unknown 必须在 1..{len(all_cats) - 1} 之间")
        unknown = sorted(random.Random(seed).sample(all_cats, n_unknown))
    known = [c for c in all_cats if c not in unknown]
    return known, unknown


def split_by_group(idx: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                   test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """类内按 group（question）切分，保证同一条 question 不跨切分，且各类测试比例一致。

    返回 (keep_idx, test_idx)。
    """
    rng = np.random.RandomState(seed)
    keep_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cls in np.unique(labels[idx]):
        cls_idx = idx[labels[idx] == cls]
        cls_groups = groups[cls_idx]
        uniq = np.unique(cls_groups)
        target = max(1, int(round(len(cls_idx) * test_ratio)))
        if len(uniq) < 2:
            # 该类所有样本共用一条 question，没法按组切，退化成随机抽
            picked = rng.choice(cls_idx, size=min(target, len(cls_idx)), replace=False)
        else:
            order = uniq.copy()
            rng.shuffle(order)
            picked_list: list[np.ndarray] = []
            n = 0
            for g in order:
                if n >= target or len(picked_list) == len(uniq) - 1:
                    break  # 至少留一组给 keep
                m = cls_idx[cls_groups == g]
                picked_list.append(m)
                n += len(m)
            picked = np.concatenate(picked_list)
        test_parts.append(picked)
        keep_parts.append(np.setdiff1d(cls_idx, picked))
    return (np.sort(np.concatenate(keep_parts)), np.sort(np.concatenate(test_parts)))


def split_by_stratified(idx: np.ndarray, labels: np.ndarray, test_ratio: float,
                        seed: int) -> tuple[np.ndarray, np.ndarray]:
    """普通分层切分（不按 question 分组），返回 (keep_idx, test_idx)。"""
    from sklearn.model_selection import train_test_split

    keep, test = train_test_split(
        idx, test_size=test_ratio, random_state=seed, stratify=labels[idx]
    )
    return np.sort(keep), np.sort(test)


def split_once(idx: np.ndarray, labels: np.ndarray, groups: np.ndarray,
               test_ratio: float, seed: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "group":
        return split_by_group(idx, labels, groups, test_ratio, seed)
    return split_by_stratified(idx, labels, test_ratio, seed)


def make_matched_unknown(unknown_idx: np.ndarray, labels: np.ndarray,
                         n_target: int, seed: int) -> np.ndarray:
    """把未知类下采样到 n_target 条，各类按原始占比分配，返回抽中的下标。"""
    rng = np.random.RandomState(seed)
    cls_list = np.unique(labels[unknown_idx])
    per_cls = {c: unknown_idx[labels[unknown_idx] == c] for c in cls_list}
    sizes = np.array([len(per_cls[c]) for c in cls_list], dtype=float)
    alloc = np.floor(n_target * sizes / sizes.sum()).astype(int)
    alloc = np.minimum(alloc, sizes.astype(int))
    # 补足取整误差：优先补给还有余量的类
    order = np.argsort(-sizes)
    k = 0
    while alloc.sum() < n_target:
        i = int(order[k % len(order)])
        if alloc[i] < len(per_cls[cls_list[i]]):
            alloc[i] += 1
        k += 1
        if k > 10 * len(order):
            break
    picked = [
        rng.choice(per_cls[c], size=int(k_), replace=False) if k_ > 0 else np.array([], dtype=int)
        for c, k_ in zip(cls_list, alloc)
    ]
    picked = [p for p in picked if len(p) > 0]
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)


# ---------------------------------------------------------------- 四种原型方法

# 每个方法返回 (score, pred)：
#   score - 已知度分数，越大越像已知类
#   pred  - 被归到的已知类下标


class NcmModel:
    """不训练：原型 = 每类训练特征（L2 归一化后）的均值再归一化。"""

    def __init__(self, dim: int, n_classes: int) -> None:
        self.dim = dim
        self.n_classes = n_classes

    def fit(self, x: np.ndarray, y: np.ndarray, logger: Logger, args: argparse.Namespace) -> list[dict]:
        protos = np.stack([x[y == c].mean(axis=0) if (y == c).any() else np.zeros(self.dim)
                           for c in range(self.n_classes)])
        self.protos = protos / np.clip(np.linalg.norm(protos, axis=1, keepdims=True), 1e-12, None)
        logger("  ncm：不训练，原型 = 各类训练特征（L2 归一化）的均值")
        return []

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sim = x @ self.protos.T
        return sim.max(axis=1), sim.argmax(axis=1)


class _TorchProtoBase(nn.Module):
    """投影层，输出 L2 归一化。子类各自定义损失与打分。"""

    def __init__(self, in_dim: int, dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, dim, bias=False)
        self.n_classes = n_classes

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x), dim=-1)


class ProtoModel(_TorchProtoBase):
    """投影层 + 可学习原型，对余弦相似度做 softmax 交叉熵。"""

    def __init__(self, in_dim: int, dim: int, n_classes: int, tau: float) -> None:
        super().__init__(in_dim, dim, n_classes)
        self.protos = nn.Parameter(torch.randn(n_classes, dim) * 0.1)
        self.tau = tau

    def sim(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) @ F.normalize(self.protos, dim=-1).T

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.sim(x) / self.tau, y)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        sim = self.sim(x)
        return sim.max(dim=1).values.cpu().numpy(), sim.argmax(dim=1).cpu().numpy()


class CacModel(_TorchProtoBase):
    """投影层 + 固定锚点（原型不学），Class Anchor Clustering 的 anchor + dot 损失。"""

    def __init__(self, in_dim: int, dim: int, n_classes: int, alpha: float, lam: float) -> None:
        super().__init__(in_dim, dim, n_classes)
        anchors = torch.zeros(n_classes, dim)
        for k in range(n_classes):
            anchors[k, k] = 1.0
        self.register_buffer("anchors", anchors * alpha)
        self.alpha = alpha
        self.lam = lam

    def dists(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cdist(self.embed(x), self.anchors)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = self.dists(x)
        d_y = d.gather(1, y[:, None]).squeeze(1)
        anchor_loss = (d_y / d.sum(dim=1).clamp_min(1e-8)).mean()
        dot_loss = F.relu(d_y - self.alpha).mean()
        return anchor_loss + self.lam * dot_loss

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        d = self.dists(x)
        return -d.min(dim=1).values.cpu().numpy(), d.argmin(dim=1).cpu().numpy()


class ArplModel(_TorchProtoBase):
    """投影层 + 每类一个反向点，ARPL 的正指数 softmax + margin 正则项。

    距离越大 → logit 越大 → 越可能是该类；分数取「到最远反向点的平方欧氏距离」。
    """

    def __init__(self, in_dim: int, dim: int, n_classes: int, tau: float, lam: float) -> None:
        super().__init__(in_dim, dim, n_classes)
        self.recip = nn.Parameter(torch.randn(n_classes, dim) * 0.05)
        self.margin = nn.Parameter(torch.tensor(1.0))
        self.tau = tau
        self.lam = lam

    def d2(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        return (
            z.pow(2).sum(dim=1, keepdim=True)
            + self.recip.pow(2).sum(dim=1)[None, :]
            - 2.0 * z @ self.recip.T
        ).clamp_min(0.0)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = self.d2(x)
        cls_loss = F.cross_entropy(d / self.tau, y)
        d_y = d.gather(1, y[:, None]).squeeze(1)
        margin = F.relu(self.margin).clamp_min(1e-3)
        return cls_loss + self.lam * ((d_y - margin) ** 2).mean()

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        d = self.d2(x)
        return d.max(dim=1).values.cpu().numpy(), d.argmax(dim=1).cpu().numpy()


class TorchScorer:
    """把 torch 模型的 score 包成「吃 numpy 出 numpy」，和 NcmModel 的接口保持一致。"""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        self.model.eval()
        return self.model.score(t)


def build_torch_model(method: str, in_dim: int, args: argparse.Namespace,
                      n_classes: int) -> ProtoModel | CacModel | ArplModel:
    dim = max(args.dim, n_classes)
    if method == "proto":
        return ProtoModel(in_dim, dim, n_classes, args.tau)
    if method == "cac":
        return CacModel(in_dim, dim, n_classes, args.cac_alpha, args.cac_lambda)
    return ArplModel(in_dim, dim, n_classes, args.tau, args.arpl_lambda)


def run_torch_method(method: str, x_train: np.ndarray, y_train: np.ndarray,
                     x_val: np.ndarray, y_val: np.ndarray,
                     args: argparse.Namespace, logger: Logger,
                     device: torch.device) -> tuple[object, list[dict]]:
    """训练一个 torch 原型方法，用验证集 loss 选最佳轮次（早停），返回模型与训练历史。"""
    torch.manual_seed(args.seed)
    model = build_torch_model(method, x_train.shape[1], args, int(y_train.max()) + 1).to(device)
    x_tr = torch.from_numpy(x_train).to(device)
    y_tr = torch.from_numpy(y_train).to(device)
    x_va = torch.from_numpy(x_val).to(device)
    y_va = torch.from_numpy(y_val).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(x_tr)
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history: list[dict] = []

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
            logger(f"  {method} epoch {epoch:4d}  train_loss={total / n:.4f}  val_loss={val_loss:.4f}")
        if bad >= args.patience:
            logger(f"  {method} 验证集 loss 连续 {args.patience} 轮没降，提前停在第 {epoch} 轮")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger(f"  {method} 训练完成：最佳轮次 {best_epoch}，最佳 val_loss={best_val:.4f}")
    return TorchScorer(model, device), history


# ---------------------------------------------------------------- 评估


def evaluate(score_known: np.ndarray, score_unknown: np.ndarray,
             pred_known: np.ndarray, y_known: np.ndarray,
             pred_unknown: np.ndarray, tau: float, n_known_classes: int) -> dict:
    """算一组指标。未知类统一当作第 n_known_classes 类（标签 -1）。"""
    y_all = np.concatenate([y_known, np.full(len(score_unknown), -1)])
    pred_all = np.where(
        np.concatenate([score_known, score_unknown]) >= tau,
        np.concatenate([pred_known, pred_unknown]),
        -1,
    )
    labels = list(range(n_known_classes)) + [-1]
    # 参考指标：把已知测试集自身的 5% 分位当门槛，与 val 标定的 tau 相互独立
    thr95 = float(np.percentile(score_known, 5.0))
    return {
        "n_known": int(len(score_known)),
        "n_unknown": int(len(score_unknown)),
        "unknown_ratio": float(len(score_unknown) / (len(score_known) + len(score_unknown))),
        "auroc": float(roc_auc_score(
            np.concatenate([np.ones(len(score_known)), np.zeros(len(score_unknown))]),
            np.concatenate([score_known, score_unknown]),
        )),
        "fpr_at_tpr95": float((score_unknown >= thr95).mean()),
        "tau": float(tau),
        "known_accept_rate": float((score_known >= tau).mean()),
        "unknown_reject_rate": float((score_unknown < tau).mean()),
        "closed_set_acc": float((pred_known == y_known).mean()),
        "open_set_acc": float((pred_all == y_all).mean()),
        "open_set_macro_f1": float(f1_score(y_all, pred_all, labels=labels, average="macro",
                                           zero_division=0)),
    }


def attribution_table(unknown_cat: np.ndarray, pred_unknown: np.ndarray,
                      score_unknown: np.ndarray, tau: float,
                      known: list[str], topk: int = 3) -> tuple[list[str], dict]:
    """每个未知类被判成了哪个已知类。

    先按阈值分两拨：分数低于 tau 的判为未知，其余（被骗过阈值的）再按原型距离归属到
    某个已知类。每行的「判为未知 + 各已知类占比」严格加起来等于 1。
    """
    lines: list[str] = []
    full: dict[str, dict[str, float]] = {}
    for cat in sorted(set(unknown_cat.tolist())):
        m = unknown_cat == cat
        total = int(m.sum())
        rejected = float((score_unknown[m] < tau).mean())
        counts: dict[str, float] = {}
        for k, name in enumerate(known):
            share = float(((pred_unknown[m] == k) & (score_unknown[m] >= tau)).mean())
            if share > 0:
                counts[name] = share
        ranking = sorted(counts.items(), key=lambda kv: -kv[1])[:topk]
        detail = "，".join(f"{name} {share * 100:.1f}%" for name, share in ranking) or "无"
        lines.append(f"  {cat:<26s} n={total:<4d} 判为未知 {rejected * 100:5.1f}%   "
                     f"其余 {100 * (1 - rejected):5.1f}% 归属：{detail}")
        full[cat] = {"判为未知": rejected, **counts}
    return lines, full


# ---------------------------------------------------------------- 主流程


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", default=str(DEFAULT_NPZ), help="get-embedding 产出的 .npz 特征文件")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs"),
                        help="日志与结果的输出目录")
    parser.add_argument("--unknown-categories", default=None,
                        help="指定未知类（逗号分隔）；不传则随机抽")
    parser.add_argument("--n-unknown", type=int, default=4, help="随机抽多少个未知类")
    parser.add_argument("--known-train-ratio", type=float, default=0.9,
                        help="已知类里分给训练（含验证）的比例")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="训练部分里再切给验证集的比例（只用于定阈值）")
    parser.add_argument("--split-mode", choices=("group", "stratified"), default="group",
                        help="group = 按 question 分组切分，stratified = 普通分层切分")
    parser.add_argument("--method", choices=("all",) + ALL_METHODS, default="all",
                        help="跑哪个方法，all 表示四种都跑并对比")
    parser.add_argument("--dim", type=int, default=256, help="投影层输出维度")
    parser.add_argument("--epochs", type=int, default=200, help="最多训练多少轮")
    parser.add_argument("--patience", type=int, default=50, help="验证集 loss 多少轮不降就早停")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--tau", type=float, default=0.1, help="距离/相似度 softmax 的温度")
    parser.add_argument("--cac-alpha", type=float, default=1.0, help="CAC 锚点半径（特征已 L2 归一化）")
    parser.add_argument("--cac-lambda", type=float, default=1e-3, help="CAC dot 损失权重")
    parser.add_argument("--arpl-lambda", type=float, default=1.0, help="ARPL margin 正则项权重")
    parser.add_argument("--target-tpr", type=float, default=0.95,
                        help="验证集上标定阈值时要求通过的已知样本比例")
    parser.add_argument("--pca-dim", type=int, default=0,
                        help="先做 PCA 降到多少维（0 表示不降维；只在训练集上 fit）")
    parser.add_argument("--log-every", type=int, default=20, help="每多少轮打一条训练日志（0 关闭）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cpu",
                        help="训练用哪块设备：cpu / gpu0 / gpu1（也接受 cuda:0 这种写法）")
    parser.add_argument("--run-name", default="",
                        help="本次运行的目录名后缀；会在 outputs/ 下建一个带时间戳的子目录")
    parser.add_argument("--tag", default="", help="附加到输出文件名后面的标签")
    return parser.parse_args(argv)


def output_stem(args: argparse.Namespace) -> str:
    stem = f"proto_openset_seed{args.seed}"
    if args.method != "all":
        stem += f"_{args.method}"
    if args.tag:
        stem += f"_{args.tag}"
    return stem


def run(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = resolve_device(args.device)
    stem = output_stem(args)
    # 每次运行单独一个带时间戳的子目录，避免多次实验的产出混在一起
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{stamp}_{args.run_name or stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / f"{stem}.log")
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger(f"[运行] {started}")
    logger(f"[目录] {run_dir.resolve()}")
    logger(f"[设备] {describe_device(device)}")
    logger(f"[参数] {json.dumps(vars(args), ensure_ascii=False)}")

    # ---- 数据
    data = load_features(Path(args.npz))
    feats, cats, questions, index = (
        data["features"], data["category"], data["question"], data["index"]
    )
    logger.section("数据")
    logger(f"特征文件   : {Path(args.npz).resolve()}")
    logger(f"样本 / 维度: {feats.shape[0]} / {feats.shape[1]}")

    known, unknown = choose_unknown(cats, args.n_unknown, args.seed, 
                                    args.unknown_categories.split(",") if args.unknown_categories else None)
    known = sorted(known)
    unknown = sorted(unknown)
    logger(f"已知类（{len(known)} 个）: {known}")
    logger(f"未知类（{len(unknown)} 个）: {unknown}")
    grey = [c for c in unknown if c.split("-")[0] in ("10", "11", "12", "13")]
    if grey:
        logger(f"[提示] 抽中的未知类含灰区类 {grey}，"
               f"这几类本身不显式含有害内容、彼此语义接近，难度会明显更高")

    # ---- 划分
    def is_known(c: np.ndarray) -> np.ndarray:
        return np.isin(c, known)

    known_idx = np.where(is_known(cats))[0]
    unknown_idx = np.where(~is_known(cats))[0]
    trainval_idx, known_test_idx = split_once(
        known_idx, cats, questions, 1.0 - args.known_train_ratio, args.seed, args.split_mode)
    train_idx, val_idx = split_once(
        trainval_idx, cats, questions, args.val_ratio, args.seed + 1, args.split_mode)

    cls_of = {name: k for k, name in enumerate(known)}
    y_all = np.array([cls_of.get(c, -1) for c in cats], dtype=np.int64)  # 未知类标 -1，不参与训练
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

    logger.section("划分")
    logger(f"切分方式   : {args.split_mode}"
           + ("（按 question 分组，同一条问题不会跨划分）" if args.split_mode == "group" else ""))
    logger(f"训练 {len(train_idx)} 条 / 验证 {len(val_idx)} 条 / 已知测试 {len(known_test_idx)} 条"
           f" / 未知 {len(unknown_idx)} 条")
    rows = []
    for name in known:
        rows.append([
            name,
            str(int((cats[train_idx] == name).sum())),
            str(int((cats[val_idx] == name).sum())),
            str(int((cats[known_test_idx] == name).sum())),
        ])
    logger(render_table(["known class", "train", "val", "test"], rows))
    short = [name for name, r in zip(known, rows) if r[3] == "0"]
    if short:
        logger(f"[警告] 这些已知类在测试集里一条都没有：{short}（分组切分可能把整类都分走了）")
    logger("未知类样本数：" + "，".join(
        f"{name} {int((cats[unknown_idx] == name).sum())}" for name in unknown))

    # ---- 两个测试口径（sel 是「未知类内部的位置下标」，不是全局下标）
    n_match = len(known_test_idx)
    matched_pos = np.searchsorted(
        unknown_idx, make_matched_unknown(unknown_idx, cats, n_match, args.seed + 2))
    variants = {
        "A_full": ("未知类全部放入", np.arange(len(unknown_idx))),
        "B_matched": (f"未知类下采样到 {len(matched_pos)} 条与已知测试集一比一", matched_pos),
    }

    # ---- 跑方法
    methods = list(ALL_METHODS) if args.method == "all" else [args.method]
    logger.section("方法")
    results: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    per_sample: dict[str, dict[str, np.ndarray]] = {}

    for method in methods:
        logger(f"[{method}]")
        if method == "ncm":
            model = NcmModel(x_train.shape[1], len(known))
            history = model.fit(x_train, y_train, logger, args)
        else:
            model, history = run_torch_method(
                method, x_train, y_train, x_val, y_val, args, logger, device)
        histories[method] = history

        score_val, _ = model.score(x_val)
        tau = float(np.percentile(score_val, (1.0 - args.target_tpr) * 100.0))
        logger(f"  阈值 tau = {tau:.4f}（验证集 {len(score_val)} 条已知样本的 "
               f"{(1 - args.target_tpr) * 100:.1f}% 分位，目标 TPR={args.target_tpr:.2f}）")

        score_ktest, pred_ktest = model.score(x_ktest)
        score_utest, pred_utest = model.score(x_utest)

        per_sample[method] = {
            "score_known_test": score_ktest,
            "pred_known_test": pred_ktest,
            "score_unknown": score_utest,
            "pred_unknown": pred_utest,
        }
        best_epoch = history[-1]["epoch"] if history else None
        results[method] = {"tau": tau, "epochs_run": len(history), "last_epoch": best_epoch,
                           "variants": {}}
        for key, (desc, sel) in variants.items():
            results[method]["variants"][key] = evaluate(
                score_ktest, score_utest[sel], pred_ktest, y_ktest, pred_utest[sel], tau, len(known))
            results[method]["variants"][key]["description"] = desc

    # ---- 主结果表
    metric_defs = [
        ("AUROC", "auroc", "{:.4f}"),
        ("ACC(含未知)", "open_set_acc", "{:.4f}"),
        ("macroF1(含未知)", "open_set_macro_f1", "{:.4f}"),
        ("闭集ACC(仅已知)", "closed_set_acc", "{:.4f}"),
        ("已知接受率", "known_accept_rate", "{:.4f}"),
        ("未知拒绝率", "unknown_reject_rate", "{:.4f}"),
        ("FPR@TPR95", "fpr_at_tpr95", "{:.4f}"),
    ]
    for key, (desc, _sel) in variants.items():
        n_k = results[methods[0]]["variants"][key]["n_known"]
        n_u = results[methods[0]]["variants"][key]["n_unknown"]
        logger.section(f"主结果 · 口径 {key} · {desc}")
        logger(f"该口径下 已知测试 {n_k} 条 / 未知 {n_u} 条（未知占 {n_u / (n_k + n_u) * 100:.1f}%）")
        rows = []
        for label, field, fmt in metric_defs:
            rows.append([label] + [fmt.format(results[m]["variants"][key][field]) for m in methods])
        logger(render_table(["metric"] + methods, rows))
        if key == "A_full":
            logger("（口径 A 里未知占绝大多数，「全判未知」的退化解也能拿到很高的 ACC，"
                   "看数请以 AUROC 为准）")

    # ---- 未知类被判成了谁
    logger.section("未知类的误判方向（按原型距离归属，行归一化）")
    attribution: dict[str, object] = {}
    for method in methods:
        ps = per_sample[method]
        tau = results[method]["tau"]
        lines, full = attribution_table(
            cats[unknown_idx], ps["pred_unknown"], ps["score_unknown"], tau, known)
        logger(f"[{method}] tau={tau:.4f}")
        logger("\n".join(lines))
        attribution[method] = full

    # ---- 落盘
    scores_path = run_dir / f"{stem}.scores.csv"
    in_matched = np.zeros(len(unknown_idx), dtype=bool)
    in_matched[matched_pos] = True
    header = ["index", "category", "is_unknown", "in_matched_unknown"]
    for method in methods:
        header += [f"score_{method}", f"pred_{method}"]
    lines = [",".join(header)]
    for pos in range(len(known_test_idx)):
        row = [str(index[known_test_idx[pos]]), cats[known_test_idx[pos]], "0", "0"]
        for method in methods:
            ps = per_sample[method]
            row += [f"{ps['score_known_test'][pos]:.6f}", known[ps["pred_known_test"][pos]]]
        lines.append(",".join(row))
    for pos in range(len(unknown_idx)):
        row = [str(index[unknown_idx[pos]]), cats[unknown_idx[pos]], "1",
               "1" if in_matched[pos] else "0"]
        for method in methods:
            ps = per_sample[method]
            row += [f"{ps['score_unknown'][pos]:.6f}", known[ps["pred_unknown"][pos]]]
        lines.append(",".join(row))
    scores_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics_path = run_dir / f"{stem}.metrics.json"
    metrics_path.write_text(json.dumps({
        "known_classes": known, "unknown_classes": unknown,
        "counts": {"train": len(train_idx), "val": len(val_idx),
                   "known_test": len(known_test_idx), "unknown": len(unknown_idx),
                   "matched_unknown": len(matched_pos)},
        "results": results, "attribution": attribution,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = run_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps({
        "created_at": started,
        "run_dir": str(run_dir.resolve()),
        "args": vars(args),
        "device": {"requested": args.device, "resolved": str(device),
                   "name": describe_device(device)},
        "npz": str(Path(args.npz).resolve()),
        "feature_dim_raw": int(feats.shape[1]),
        "feature_dim_used": int(x.shape[1]),
        "n_samples": int(feats.shape[0]),
        "known_classes": known,
        "unknown_classes": unknown,
        "methods": methods,
        "per_class_split": {name: r for name, r in zip(known, rows)},
        "unknown_counts": {name: int((cats[unknown_idx] == name).sum()) for name in unknown},
        "simplifications": [
            "cac / arpl 只实现了论文的核心损失项，未做伪未知样本生成",
            "cac 的 alpha 因特征 L2 归一化与论文默认值不同",
            "未做白化 / PCA（--pca-dim 默认 0）",
        ],
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
