# merge-b7-mproto

在 `merge-b7` 的 7 类合并、`abtt1` 特征基础上，比较**单原型 Proto** 与**多原型 Proto** 的开集识别效果。
CAC 结果直接复用 `merge-b7/outputs/` 里的旧产物，不重跑；本次只新建单原型和多原型实验。

## 数据与切分

- 特征：`../merge-b7/data/MMSB_sd_typo_mergeB7_layer{24,32}_center-abtt1.npz`（只读）。
- 7 个合并类，每 seed 随机抽 2 个未知类，`--n-unknown 2`。
- 按 `question` 分组切分（`--split-mode group`），seed 42–46 与旧报告一致。
- 已知类训练/验证/测试比例沿用 `0.9 / 0.1`；阈值在验证集按 TPR=95% 标定。

## 方法

投影层 `W: 4096 -> 256`，输出 L2 归一化得到 `z`；每类 K 个可学习原型 `P[c,k]`：

```
s[c,k] = z · P[c,k]
l[c]   = logsumexp_k(s[c,k] / tau)
loss   = CrossEntropy(l, y)
score  = max_{c,k} s[c,k]
pred   = argmax_c max_k s[c,k]
```

- 训练用类内 logsumexp（同类所有原型都能拿到梯度），推理用所有原型的最大余弦相似度。
- K=1 时与旧 `ProtoModel` 数学等价，用于实现一致性校验。
- 不加多样性正则；训练后在诊断里检查原型是否塌缩。
- 原型随机初始化 `randn * 0.1`，超参数沿用旧实验默认值。

## 运行

环境与旧实验相同（conda `llm2`）。先小规模验证，再跑全量：

```bash
cd merge-b7-mproto

# 单次：单原型
python mproto_openset.py \
  --npz ../merge-b7/data/MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz \
  --method proto --seed 42 --device cpu --epochs 5 --output-dir outputs/_smoke

# 单次：多原型 K=3
python mproto_openset.py \
  --npz ../merge-b7/data/MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz \
  --method proto-multi --n-protos 3 --seed 42 --device cpu --epochs 5 --output-dir outputs/_smoke

# 批量：2 层 × 5 seed × 5 配置（proto + K=1..4）= 50 次
python run_mproto.py --dry-run
python run_mproto.py --device gpu0

# 汇总（读取新结果 + merge-b7 旧 CAC）
python aggregate_mproto.py
```

## 产出

每次运行在 `outputs/<时间戳>_<run-name>/` 下生成：

| 文件 | 内容 |
|---|---|
| `*.log` | 运行日志，含参数、划分、训练曲线 |
| `*.metrics.json` | B_matched 指标、未知类归属、原型诊断 |
| `*.scores.csv` | 已知测试 + 未知样本的分数、预测、最近原型编号、软分配熵 |
| `*.manifest.json` | 参数、设备、数据路径、划分计数、版本 |

汇总生成：

- `多原型Proto_实验报告.md`
- `aggregate_mproto_summary.json`

## 诊断口径

只统计已知类，按 train / val / test 分开：

- **使用率**：样本在本类 K 个原型中最近原型的硬分配计数/比例；`active` 为使用率 > 0 的原型数。
- **软分配熵**：`p = softmax(s[c,:] / tau)`，`H_norm = -Σ p log p / log K`，按样本平均，K=1 时为 0。
- **类内余弦**：同类原型两两余弦的 min/mean/max，K=1 时为 N/A。

## 注意

- `abtt1` 的均值和第一主方向是用全部 1680 条（含未知类）估计的，沿用旧实验口径。
- 合并后类别不均衡（M5=465、M1=443，M6/M7=109），固定 K 对不同类别的相对容量不同。
- 验证集同时用于早停和阈值标定；K 固定为 3，K=1–4 只作敏感性，不按测试集挑选。
