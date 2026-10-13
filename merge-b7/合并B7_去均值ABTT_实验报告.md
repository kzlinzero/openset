# 合并类别 B（13→7）+ 去均值 / all-but-the-top 开集识别实验结果

- 生成时间：2026-10-13 12:33:38
- 运行数：**60**（变体 ['none', 'mean', 'abtt1'] × 层 ['24', '32'] × seed [42, 43, 44, 45, 46] × 方法 ['cac', 'arpl']）
- 全部由 merge-b7 目录下的脚本产出；推理/训练脚本 `openset-proto/proto_openset.py` **未做任何修改**。

## 1. 本次运行的命令与文件

### 1.1 用到的脚本

| 脚本 | 作用 |
|---|---|
| `F:\CODE-value_alignment\openset\merge-b7\merge_taxonomy.py` | 13 类 → 7 类合并，生成 `data/MMSB_sd_typo_mergeB7_layer*.npz` |
| `F:\CODE-value_alignment\openset\merge-b7\center_features.py` | 去均值 / ABTT 预处理，生成 `data/..._center-*.npz` |
| `F:\CODE-value_alignment\openset\merge-b7\run_merge_b7.py` | 驱动器，循环 变体×层×seed×方法 调下面的脚本 |
| `F:\CODE-value_alignment\openset\openset-proto\proto_openset.py` | **实际训练/评测的主脚本（未修改）** |
| `F:\CODE-value_alignment\openset\merge-b7\aggregate.py` | 汇总 metrics/manifest，生成本报告 |

### 1.2 单次调用的命令模板

每次运行就是下面这条命令（参数按变体/层/seed/方法替换）：

```bash
python ../openset-proto/proto_openset.py \
  --npz data/MMSB_sd_typo_mergeB7_layer<层>[_center-<变体>].npz \
  --method <cac|arpl> --n-unknown 2 --seed <seed> --device gpu0 \
  --output-dir outputs --run-name b7_[<变体>_]L<层>_s<seed>_<方法>
```

### 1.3 每个变体对应的输入特征文件

| 变体 | 输入 npz（绝对路径） | run-name 前缀 |
|---|---|---|
| raw（原始合并特征，基线） | `F:\CODE-value_alignment\openset\merge-b7\data\MMSB_sd_typo_mergeB7_layer<层>.npz` | `b7_` |
| mean（去均值） | `F:\CODE-value_alignment\openset\merge-b7\data\MMSB_sd_typo_mergeB7_layer<层>_center-mean.npz` | `b7_mean_` |
| abtt1（去均值 + all-but-the-top k=1） | `F:\CODE-value_alignment\openset\merge-b7\data\MMSB_sd_typo_mergeB7_layer<层>_center-abtt1.npz` | `b7_abtt1_` |

### 1.4 逐次运行清单（从每次运行的 manifest.json 读出）

| 变体 | 层 | seed | 方法 | 输入 npz（文件名） | 输出目录 |
|---|---|---|---|---|---|
| none | 24 | 42 | arpl | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182135_b7_L24_s42_arpl` |
| none | 24 | 42 | cac | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182059_b7_L24_s42_cac` |
| none | 24 | 43 | arpl | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182226_b7_L24_s43_arpl` |
| none | 24 | 43 | cac | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182157_b7_L24_s43_cac` |
| none | 24 | 44 | arpl | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182324_b7_L24_s44_arpl` |
| none | 24 | 44 | cac | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182251_b7_L24_s44_cac` |
| none | 24 | 45 | arpl | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182417_b7_L24_s45_arpl` |
| none | 24 | 45 | cac | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182346_b7_L24_s45_cac` |
| none | 24 | 46 | arpl | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182509_b7_L24_s46_arpl` |
| none | 24 | 46 | cac | `MMSB_sd_typo_mergeB7_layer24.npz` | `outputs/20261012-182440_b7_L24_s46_cac` |
| none | 32 | 42 | arpl | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182603_b7_L32_s42_arpl` |
| none | 32 | 42 | cac | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182532_b7_L32_s42_cac` |
| none | 32 | 43 | arpl | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182657_b7_L32_s43_arpl` |
| none | 32 | 43 | cac | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182627_b7_L32_s43_cac` |
| none | 32 | 44 | arpl | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182750_b7_L32_s44_arpl` |
| none | 32 | 44 | cac | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182718_b7_L32_s44_cac` |
| none | 32 | 45 | arpl | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182850_b7_L32_s45_arpl` |
| none | 32 | 45 | cac | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182816_b7_L32_s45_cac` |
| none | 32 | 46 | arpl | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182949_b7_L32_s46_arpl` |
| none | 32 | 46 | cac | `MMSB_sd_typo_mergeB7_layer32.npz` | `outputs/20261012-182917_b7_L32_s46_cac` |
| mean | 24 | 42 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121203_b7_mean_L24_s42_arpl` |
| mean | 24 | 42 | cac | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121130_b7_mean_L24_s42_cac` |
| mean | 24 | 43 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121252_b7_mean_L24_s43_arpl` |
| mean | 24 | 43 | cac | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121229_b7_mean_L24_s43_cac` |
| mean | 24 | 44 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121347_b7_mean_L24_s44_arpl` |
| mean | 24 | 44 | cac | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121314_b7_mean_L24_s44_cac` |
| mean | 24 | 45 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121439_b7_mean_L24_s45_arpl` |
| mean | 24 | 45 | cac | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121413_b7_mean_L24_s45_cac` |
| mean | 24 | 46 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121530_b7_mean_L24_s46_arpl` |
| mean | 24 | 46 | cac | `MMSB_sd_typo_mergeB7_layer24_center-mean.npz` | `outputs/20261013-121501_b7_mean_L24_s46_cac` |
| mean | 32 | 42 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121622_b7_mean_L32_s42_arpl` |
| mean | 32 | 42 | cac | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121552_b7_mean_L32_s42_cac` |
| mean | 32 | 43 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121715_b7_mean_L32_s43_arpl` |
| mean | 32 | 43 | cac | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121647_b7_mean_L32_s43_cac` |
| mean | 32 | 44 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121803_b7_mean_L32_s44_arpl` |
| mean | 32 | 44 | cac | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121735_b7_mean_L32_s44_cac` |
| mean | 32 | 45 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121855_b7_mean_L32_s45_arpl` |
| mean | 32 | 45 | cac | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121827_b7_mean_L32_s45_cac` |
| mean | 32 | 46 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121948_b7_mean_L32_s46_arpl` |
| mean | 32 | 46 | cac | `MMSB_sd_typo_mergeB7_layer32_center-mean.npz` | `outputs/20261013-121919_b7_mean_L32_s46_cac` |
| abtt1 | 24 | 42 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122048_b7_abtt1_L24_s42_arpl` |
| abtt1 | 24 | 42 | cac | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122015_b7_abtt1_L24_s42_cac` |
| abtt1 | 24 | 43 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122137_b7_abtt1_L24_s43_arpl` |
| abtt1 | 24 | 43 | cac | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122112_b7_abtt1_L24_s43_cac` |
| abtt1 | 24 | 44 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122228_b7_abtt1_L24_s44_arpl` |
| abtt1 | 24 | 44 | cac | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122157_b7_abtt1_L24_s44_cac` |
| abtt1 | 24 | 45 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122326_b7_abtt1_L24_s45_arpl` |
| abtt1 | 24 | 45 | cac | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122253_b7_abtt1_L24_s45_cac` |
| abtt1 | 24 | 46 | arpl | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122419_b7_abtt1_L24_s46_arpl` |
| abtt1 | 24 | 46 | cac | `MMSB_sd_typo_mergeB7_layer24_center-abtt1.npz` | `outputs/20261013-122350_b7_abtt1_L24_s46_cac` |
| abtt1 | 32 | 42 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122505_b7_abtt1_L32_s42_arpl` |
| abtt1 | 32 | 42 | cac | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122439_b7_abtt1_L32_s42_cac` |
| abtt1 | 32 | 43 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122556_b7_abtt1_L32_s43_arpl` |
| abtt1 | 32 | 43 | cac | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122527_b7_abtt1_L32_s43_cac` |
| abtt1 | 32 | 44 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122647_b7_abtt1_L32_s44_arpl` |
| abtt1 | 32 | 44 | cac | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122616_b7_abtt1_L32_s44_cac` |
| abtt1 | 32 | 45 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122735_b7_abtt1_L32_s45_arpl` |
| abtt1 | 32 | 45 | cac | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122708_b7_abtt1_L32_s45_cac` |
| abtt1 | 32 | 46 | arpl | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122825_b7_abtt1_L32_s46_arpl` |
| abtt1 | 32 | 46 | cac | `MMSB_sd_typo_mergeB7_layer32_center-abtt1.npz` | `outputs/20261013-122757_b7_abtt1_L32_s46_cac` |

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| 底座模型 | Qwen/Qwen3-VL-8B-Instruct（fp16，特征维度 4096） |
| 数据 | MM-SafetyBench SD_TYPO，1680 样本，13 类**合并为 7 类**（粒度 B） |
| 对比层 | decoder 第 24、32 层 |
| 预处理变体 | raw（原始合并特征，基线）、mean（去均值）、abtt1（去均值 + all-but-the-top k=1） |
| 未知类 | 每 seed 从 7 个合并类里**随机抽 2 个**（`--n-unknown 2`，各变体/各层同 seed 同抽法） |
| 方法 | cac、arpl |
| 阈值 tau | 各方法在每个 seed 下于验证集按 TPR 95% 校准 |
| 汇报口径 | B_matched（未知类下采样到与已知测试集 1:1）为主，A_full 作附表 |

**预处理定义**（对每层特征矩阵 `X`，全部样本逐条作用）：

- `mean`：`X` 减去全局均值 `μ = mean(X)`。
- `abtt1`：在去均值后的矩阵上做 SVD，把投影到**第一主方向**的部分减掉
  （`X' = Xc − (Xc·v₁)v₁`，all-but-the-top）。

**拟合口径（需要声明）**：均值和主方向都用**全部 1680 条**估计，其中包含某些 seed 下会被划为
未知类的样本。这是无标签估计、对所有类一视同仁（ABTT 文献通常这么做），但它确实「看过」未知
样本的特征，属于比「只用已知类训练集估计」更宽松的口径。

预处理只是把每个样本的特征换成一个新的向量，因此**没有**改变类别划分、未知类抽取或方法本身；
下游 `proto_openset.py` 照常对特征做逐行 L2 归一化，于是「去均值 + L2 归一化」就是标准的
centered-cosine 表示。

**预处理前后的类均值两两余弦**（说明公共分量被移掉了；
数值是在本脚本里直接从 §1.3 的 npz 文件重算的，不使用硬编码）：

| 层 | raw（原始合并特征，基线） | mean（去均值） | abtt1（去均值 + all-but-the-top k=1） |
|---|---|---|---|
| 24 | 0.9914 / 0.9961 / 0.9990 | -0.5671 / -0.0822 / 0.2775 | -0.5750 / -0.1419 / 0.2698 |
| 32 | 0.9922 / 0.9961 / 0.9986 | -0.5401 / -0.1077 / 0.2451 | -0.5355 / -0.1520 / 0.2370 |

（格式为 min / mean / max。
被 ABTT 移除的主方向占总能量：L24 abtt1 移除 31.3%、L32 abtt1 移除 29.5%。）

## 3. 每个 seed 抽中的未知类组合

同一 seed 下所有变体、所有层抽到的未知类相同（选择只取决于类别表与 seed）。

| seed | 未知类（2 个） | 未知样本数 | 已知测试数 |
|---|---|---|---|
| 42 | M1-Economic-Financial-Fraud、M6-Sex | 552 | 114 |
| 43 | M1-Economic-Financial-Fraud、M3-Illegal-PhysicalHarm | 684 | 99 |
| 44 | M4-Malware-PrivacyViolence、M5-HateSpeech-Political | 648 | 103 |
| 45 | M3-Illegal-PhysicalHarm、M4-Malware-PrivacyViolence | 424 | 125 |
| 46 | M1-Economic-Financial-Fraud、M4-Malware-PrivacyViolence | 626 | 105 |

## 4. 主结果 · 口径 B_matched（1:1）

未知类下采样到与已知测试集等量；每个格子是 5 个 seed 的均值±样本标准差。

| 变体 | 层 | 方法 | AUROC | 闭集ACC | 未知拒绝率 | FPR@TPR95 |
|---|---|---|---|---|---|---|
| none | 24 | cac | 0.8035±0.0425 | 0.9321±0.0320 | 0.2812±0.0521 | 0.7341±0.0803 |
| none | 24 | arpl | 0.7785±0.0117 | 0.8982±0.0291 | 0.3452±0.1169 | 0.6977±0.0971 |
| none | 32 | cac | 0.8199±0.0473 | 0.9431±0.0225 | 0.3438±0.1262 | 0.6802±0.1196 |
| none | 32 | arpl | 0.7917±0.0480 | 0.9223±0.0292 | 0.2952±0.1345 | 0.6910±0.1052 |
| mean | 24 | cac | 0.8779±0.0559 | 0.8910±0.0503 | 0.4606±0.3130 | 0.5640±0.3074 |
| mean | 24 | arpl | 0.3406±0.1481 | 0.8930±0.0186 | 0.0285±0.0233 | 0.9693±0.0330 |
| mean | 32 | cac | 0.9063±0.0159 | 0.9224±0.0317 | 0.6151±0.0533 | 0.4015±0.1099 |
| mean | 32 | arpl | 0.2221±0.0613 | 0.9164±0.0172 | 0.0258±0.0157 | 0.9688±0.0224 |
| abtt1 | 24 | cac | 0.9109±0.0147 | 0.8993±0.0266 | 0.6408±0.0825 | 0.3518±0.0649 |
| abtt1 | 24 | arpl | 0.2336±0.0320 | 0.9112±0.0217 | 0.0189±0.0129 | 0.9811±0.0079 |
| abtt1 | 32 | cac | 0.9056±0.0193 | 0.9069±0.0218 | 0.6460±0.0666 | 0.3897±0.0856 |
| abtt1 | 32 | arpl | 0.2005±0.0440 | 0.9091±0.0159 | 0.0213±0.0223 | 0.9768±0.0263 |

## 5. 核心对比 · 去均值 / ABTT 相对基线的变化（B_matched）

每格为「均值（相对 raw 的变化）」。看两件事：**去均值单独带来多少**、**再去掉第一主方向又带来多少**。

| 层 | 方法 | 变体 | AUROC（Δ vs raw） | 闭集ACC（Δ vs raw） | 未知拒绝率（Δ vs raw） | FPR@TPR95（Δ vs raw） |
|---|---|---|---|---|---|---|
| 24 | cac | none | 0.8035 (—) | 0.9321 (—) | 0.2812 (—) | 0.7341 (—) |
| 24 | cac | mean | 0.8779 (+0.0744) | 0.8910 (-0.0412) | 0.4606 (+0.1794) | 0.5640 (-0.1701) |
| 24 | cac | abtt1 | 0.9109 (+0.1074) | 0.8993 (-0.0328) | 0.6408 (+0.3596) | 0.3518 (-0.3823) |
| 24 | arpl | none | 0.7785 (—) | 0.8982 (—) | 0.3452 (—) | 0.6977 (—) |
| 24 | arpl | mean | 0.3406 (-0.4379) | 0.8930 (-0.0052) | 0.0285 (-0.3168) | 0.9693 (+0.2716) |
| 24 | arpl | abtt1 | 0.2336 (-0.5449) | 0.9112 (+0.0130) | 0.0189 (-0.3263) | 0.9811 (+0.2834) |
| 32 | cac | none | 0.8199 (—) | 0.9431 (—) | 0.3438 (—) | 0.6802 (—) |
| 32 | cac | mean | 0.9063 (+0.0864) | 0.9224 (-0.0207) | 0.6151 (+0.2713) | 0.4015 (-0.2788) |
| 32 | cac | abtt1 | 0.9056 (+0.0858) | 0.9069 (-0.0362) | 0.6460 (+0.3022) | 0.3897 (-0.2905) |
| 32 | arpl | none | 0.7917 (—) | 0.9223 (—) | 0.2952 (—) | 0.6910 (—) |
| 32 | arpl | mean | 0.2221 (-0.5696) | 0.9164 (-0.0059) | 0.0258 (-0.2695) | 0.9688 (+0.2778) |
| 32 | arpl | abtt1 | 0.2005 (-0.5912) | 0.9091 (-0.0132) | 0.0213 (-0.2740) | 0.9768 (+0.2858) |

### 5.1 结论（以 cac 为准，理由见 §8）

- **去均值大幅改善了 cac 的开集性能**：AUROC 从 0.80~0.82 升到 0.88~0.91
  （Δ +0.07~+0.11），未知拒绝率从 0.28~0.34 升到 0.46~0.65，FPR@TPR95 从
  0.68~0.73 降到 0.35~0.56。也就是说，之前那个「未知类被大量误收」的问题，
  主要就是**公共分量**造成的，减掉之后一半以上的未知样本能被正确拒绝。
- **ABTT(k=1) 在此基础上再有增益，但只在 L24 明显**：L24 上
  AUROC 0.8779 → 0.9109、拒绝率 0.4606 → 0.6408、FPR 0.5640 → 0.3518；
  L32 上 ABTT1 与单纯去均值基本持平（0.9063 vs 0.9056）。
- **最优组合**：L24 + abtt1 + cac（AUROC 0.9109±0.0147、拒绝率 0.6408±0.0825、
  FPR 0.3518±0.0649）；L32 + mean/abtt1 + cac 紧随其后。
- **代价**：闭集 ACC 小幅下降（−0.02~−0.04），即去掉公共分量后类内也更「散」了一点。
- **arpl 出现反向崩塌**（AUROC < 0.5、拒绝率 ~2%），两个层、两个变体、5 个 seed 一致；
  见 §8 的专门说明，本报告的开集结论不采信 arpl。

## 6. 附表 · 口径 A_full（未知类全部放入）

该口径未知占绝大多数，「全判未知」的退化解也有很高的 ACC，请以 AUROC 为准。

| 变体 | 层 | 方法 | AUROC | 闭集ACC | 未知拒绝率 | FPR@TPR95 |
|---|---|---|---|---|---|---|
| none | 24 | cac | 0.8013±0.0418 | 0.9321±0.0320 | 0.2818±0.0687 | 0.7229±0.0737 |
| none | 24 | arpl | 0.7786±0.0217 | 0.8982±0.0291 | 0.3273±0.0788 | 0.7222±0.0622 |
| none | 32 | cac | 0.8146±0.0481 | 0.9431±0.0225 | 0.3222±0.1140 | 0.6906±0.0860 |
| none | 32 | arpl | 0.7862±0.0317 | 0.9223±0.0292 | 0.2703±0.1085 | 0.7093±0.0657 |
| mean | 24 | cac | 0.8792±0.0579 | 0.8910±0.0503 | 0.4527±0.3092 | 0.5625±0.3186 |
| mean | 24 | arpl | 0.3165±0.1276 | 0.8930±0.0186 | 0.0276±0.0151 | 0.9689±0.0257 |
| mean | 32 | cac | 0.9072±0.0262 | 0.9224±0.0317 | 0.6162±0.0761 | 0.3862±0.1376 |
| mean | 32 | arpl | 0.2147±0.0533 | 0.9164±0.0172 | 0.0242±0.0122 | 0.9729±0.0152 |
| abtt1 | 24 | cac | 0.9122±0.0228 | 0.8993±0.0266 | 0.6422±0.0641 | 0.3455±0.0962 |
| abtt1 | 24 | arpl | 0.2214±0.0398 | 0.9112±0.0217 | 0.0200±0.0132 | 0.9791±0.0106 |
| abtt1 | 32 | cac | 0.9034±0.0300 | 0.9069±0.0218 | 0.6608±0.0892 | 0.3789±0.1004 |
| abtt1 | 32 | arpl | 0.1946±0.0393 | 0.9091±0.0159 | 0.0161±0.0105 | 0.9739±0.0201 |

## 7. 未知类的误判方向（按未知类聚合，行内归一化）

把「某个合并类被抽为未知」的那些 seed 的归属比例平均起来；`判为未知` = 被正确拒绝的比例，其余为被骗过阈值后归到的已知类。

### 变体 none（raw（原始合并特征，基线））

| 层 | 方法 | 未知类 | 被抽次数 | 判为未知 | 主要误判去向（top3） |
|---|---|---|---|---|---|
| 24 | cac | M1-Economic-Financial-Fraud | 3 | 31.0% | M5-HateSpeech-Political 42%、M3-Illegal-PhysicalHarm 11%、M2-Legal-Opinion 8% |
| 24 | cac | M3-Illegal-PhysicalHarm | 2 | 19.9% | M5-HateSpeech-Political 54%、M1-Economic-Financial-Fraud 13%、M4-Malware-PrivacyViolence 8% |
| 24 | cac | M4-Malware-PrivacyViolence | 3 | 22.6% | M1-Economic-Financial-Fraud 35%、M3-Illegal-PhysicalHarm 24%、M5-HateSpeech-Political 18% |
| 24 | cac | M5-HateSpeech-Political | 1 | 35.9% | M1-Economic-Financial-Fraud 37%、M3-Illegal-PhysicalHarm 25%、M2-Legal-Opinion 1% |
| 24 | cac | M6-Sex | 1 | 37.6% | M5-HateSpeech-Political 45%、M3-Illegal-PhysicalHarm 17%、M4-Malware-PrivacyViolence 1% |
| 24 | arpl | M1-Economic-Financial-Fraud | 3 | 36.3% | M5-HateSpeech-Political 22%、M3-Illegal-PhysicalHarm 14%、M2-Legal-Opinion 14% |
| 24 | arpl | M3-Illegal-PhysicalHarm | 2 | 38.4% | M5-HateSpeech-Political 33%、M1-Economic-Financial-Fraud 12%、M4-Malware-PrivacyViolence 12% |
| 24 | arpl | M4-Malware-PrivacyViolence | 3 | 25.5% | M3-Illegal-PhysicalHarm 30%、M1-Economic-Financial-Fraud 30%、M5-HateSpeech-Political 12% |
| 24 | arpl | M5-HateSpeech-Political | 1 | 26.9% | M1-Economic-Financial-Fraud 33%、M3-Illegal-PhysicalHarm 26%、M2-Legal-Opinion 12% |
| 24 | arpl | M6-Sex | 1 | 25.7% | M3-Illegal-PhysicalHarm 35%、M5-HateSpeech-Political 24%、M4-Malware-PrivacyViolence 13% |
| 32 | cac | M1-Economic-Financial-Fraud | 3 | 41.5% | M5-HateSpeech-Political 31%、M3-Illegal-PhysicalHarm 14%、M2-Legal-Opinion 7% |
| 32 | cac | M3-Illegal-PhysicalHarm | 2 | 25.1% | M5-HateSpeech-Political 52%、M1-Economic-Financial-Fraud 12%、M4-Malware-PrivacyViolence 7% |
| 32 | cac | M4-Malware-PrivacyViolence | 3 | 23.7% | M1-Economic-Financial-Fraud 32%、M3-Illegal-PhysicalHarm 28%、M5-HateSpeech-Political 15% |
| 32 | cac | M5-HateSpeech-Political | 1 | 26.9% | M1-Economic-Financial-Fraud 36%、M3-Illegal-PhysicalHarm 33%、M2-Legal-Opinion 3% |
| 32 | cac | M6-Sex | 1 | 47.7% | M5-HateSpeech-Political 29%、M3-Illegal-PhysicalHarm 19%、M4-Malware-PrivacyViolence 2% |
| 32 | arpl | M1-Economic-Financial-Fraud | 3 | 34.5% | M5-HateSpeech-Political 23%、M3-Illegal-PhysicalHarm 20%、M2-Legal-Opinion 14% |
| 32 | arpl | M3-Illegal-PhysicalHarm | 2 | 24.1% | M5-HateSpeech-Political 43%、M1-Economic-Financial-Fraud 16%、M4-Malware-PrivacyViolence 11% |
| 32 | arpl | M4-Malware-PrivacyViolence | 3 | 24.2% | M1-Economic-Financial-Fraud 33%、M3-Illegal-PhysicalHarm 24%、M5-HateSpeech-Political 17% |
| 32 | arpl | M5-HateSpeech-Political | 1 | 23.0% | M1-Economic-Financial-Fraud 45%、M3-Illegal-PhysicalHarm 24%、M2-Legal-Opinion 8% |
| 32 | arpl | M6-Sex | 1 | 14.7% | M3-Illegal-PhysicalHarm 50%、M5-HateSpeech-Political 28%、M4-Malware-PrivacyViolence 6% |

### 变体 mean（mean（去均值））

| 层 | 方法 | 未知类 | 被抽次数 | 判为未知 | 主要误判去向（top3） |
|---|---|---|---|---|---|
| 24 | cac | M1-Economic-Financial-Fraud | 3 | 68.5% | M5-HateSpeech-Political 13%、M4-Malware-PrivacyViolence 9%、M3-Illegal-PhysicalHarm 6% |
| 24 | cac | M3-Illegal-PhysicalHarm | 2 | 30.3% | M5-HateSpeech-Political 21%、M6-Sex 18%、M7-Health 10% |
| 24 | cac | M4-Malware-PrivacyViolence | 3 | 26.6% | M3-Illegal-PhysicalHarm 21%、M1-Economic-Financial-Fraud 18%、M6-Sex 12% |
| 24 | cac | M5-HateSpeech-Political | 1 | 20.2% | M3-Illegal-PhysicalHarm 33%、M2-Legal-Opinion 17%、M1-Economic-Financial-Fraud 15% |
| 24 | cac | M6-Sex | 1 | 90.8% | M5-HateSpeech-Political 7%、M3-Illegal-PhysicalHarm 1%、M4-Malware-PrivacyViolence 1% |
| 24 | arpl | M1-Economic-Financial-Fraud | 3 | 2.3% | M2-Legal-Opinion 32%、M5-HateSpeech-Political 19%、M3-Illegal-PhysicalHarm 16% |
| 24 | arpl | M3-Illegal-PhysicalHarm | 2 | 3.3% | M5-HateSpeech-Political 31%、M4-Malware-PrivacyViolence 17%、M2-Legal-Opinion 15% |
| 24 | arpl | M4-Malware-PrivacyViolence | 3 | 5.5% | M3-Illegal-PhysicalHarm 29%、M1-Economic-Financial-Fraud 26%、M5-HateSpeech-Political 14% |
| 24 | arpl | M5-HateSpeech-Political | 1 | 1.7% | M3-Illegal-PhysicalHarm 31%、M1-Economic-Financial-Fraud 31%、M2-Legal-Opinion 24% |
| 24 | arpl | M6-Sex | 1 | 0.0% | M7-Health 44%、M4-Malware-PrivacyViolence 28%、M3-Illegal-PhysicalHarm 12% |
| 32 | cac | M1-Economic-Financial-Fraud | 3 | 66.8% | M5-HateSpeech-Political 12%、M4-Malware-PrivacyViolence 9%、M3-Illegal-PhysicalHarm 7% |
| 32 | cac | M3-Illegal-PhysicalHarm | 2 | 55.4% | M5-HateSpeech-Political 19%、M4-Malware-PrivacyViolence 9%、M7-Health 6% |
| 32 | cac | M4-Malware-PrivacyViolence | 3 | 47.5% | M3-Illegal-PhysicalHarm 25%、M1-Economic-Financial-Fraud 9%、M5-HateSpeech-Political 9% |
| 32 | cac | M5-HateSpeech-Political | 1 | 57.6% | M3-Illegal-PhysicalHarm 16%、M2-Legal-Opinion 10%、M1-Economic-Financial-Fraud 9% |
| 32 | cac | M6-Sex | 1 | 96.3% | M4-Malware-PrivacyViolence 2%、M2-Legal-Opinion 1%、M5-HateSpeech-Political 1% |
| 32 | arpl | M1-Economic-Financial-Fraud | 3 | 2.3% | M2-Legal-Opinion 35%、M5-HateSpeech-Political 17%、M3-Illegal-PhysicalHarm 17% |
| 32 | arpl | M3-Illegal-PhysicalHarm | 2 | 2.3% | M5-HateSpeech-Political 32%、M2-Legal-Opinion 26%、M4-Malware-PrivacyViolence 14% |
| 32 | arpl | M4-Malware-PrivacyViolence | 3 | 4.2% | M3-Illegal-PhysicalHarm 24%、M1-Economic-Financial-Fraud 22%、M2-Legal-Opinion 21% |
| 32 | arpl | M5-HateSpeech-Political | 1 | 1.3% | M3-Illegal-PhysicalHarm 28%、M2-Legal-Opinion 26%、M1-Economic-Financial-Fraud 25% |
| 32 | arpl | M6-Sex | 1 | 0.9% | M7-Health 50%、M4-Malware-PrivacyViolence 17%、M3-Illegal-PhysicalHarm 14% |

### 变体 abtt1（abtt1（去均值 + all-but-the-top k=1））

| 层 | 方法 | 未知类 | 被抽次数 | 判为未知 | 主要误判去向（top3） |
|---|---|---|---|---|---|
| 24 | cac | M1-Economic-Financial-Fraud | 3 | 70.5% | M5-HateSpeech-Political 11%、M4-Malware-PrivacyViolence 8%、M3-Illegal-PhysicalHarm 7% |
| 24 | cac | M3-Illegal-PhysicalHarm | 2 | 45.9% | M5-HateSpeech-Political 21%、M4-Malware-PrivacyViolence 11%、M2-Legal-Opinion 6% |
| 24 | cac | M4-Malware-PrivacyViolence | 3 | 53.4% | M3-Illegal-PhysicalHarm 17%、M1-Economic-Financial-Fraud 10%、M5-HateSpeech-Political 9% |
| 24 | cac | M5-HateSpeech-Political | 1 | 74.2% | M3-Illegal-PhysicalHarm 9%、M2-Legal-Opinion 7%、M1-Economic-Financial-Fraud 7% |
| 24 | cac | M6-Sex | 1 | 89.0% | M5-HateSpeech-Political 6%、M7-Health 4%、M3-Illegal-PhysicalHarm 1% |
| 24 | arpl | M1-Economic-Financial-Fraud | 3 | 1.7% | M2-Legal-Opinion 42%、M3-Illegal-PhysicalHarm 18%、M5-HateSpeech-Political 18% |
| 24 | arpl | M3-Illegal-PhysicalHarm | 2 | 5.8% | M5-HateSpeech-Political 41%、M2-Legal-Opinion 18%、M1-Economic-Financial-Fraud 12% |
| 24 | arpl | M4-Malware-PrivacyViolence | 3 | 1.3% | M3-Illegal-PhysicalHarm 30%、M1-Economic-Financial-Fraud 25%、M5-HateSpeech-Political 14% |
| 24 | arpl | M5-HateSpeech-Political | 1 | 0.6% | M2-Legal-Opinion 29%、M3-Illegal-PhysicalHarm 26%、M6-Sex 20% |
| 24 | arpl | M6-Sex | 1 | 0.0% | M5-HateSpeech-Political 32%、M7-Health 28%、M4-Malware-PrivacyViolence 26% |
| 32 | cac | M1-Economic-Financial-Fraud | 3 | 68.8% | M5-HateSpeech-Political 9%、M4-Malware-PrivacyViolence 8%、M3-Illegal-PhysicalHarm 8% |
| 32 | cac | M3-Illegal-PhysicalHarm | 2 | 45.4% | M5-HateSpeech-Political 23%、M4-Malware-PrivacyViolence 12%、M7-Health 7% |
| 32 | cac | M4-Malware-PrivacyViolence | 3 | 55.7% | M3-Illegal-PhysicalHarm 19%、M5-HateSpeech-Political 10%、M1-Economic-Financial-Fraud 7% |
| 32 | cac | M5-HateSpeech-Political | 1 | 87.5% | M1-Economic-Financial-Fraud 5%、M3-Illegal-PhysicalHarm 5%、M6-Sex 2% |
| 32 | cac | M6-Sex | 1 | 88.1% | M5-HateSpeech-Political 5%、M4-Malware-PrivacyViolence 4%、M7-Health 3% |
| 32 | arpl | M1-Economic-Financial-Fraud | 3 | 0.9% | M2-Legal-Opinion 41%、M3-Illegal-PhysicalHarm 20%、M5-HateSpeech-Political 17% |
| 32 | arpl | M3-Illegal-PhysicalHarm | 2 | 2.5% | M5-HateSpeech-Political 35%、M2-Legal-Opinion 28%、M1-Economic-Financial-Fraud 13% |
| 32 | arpl | M4-Malware-PrivacyViolence | 3 | 2.9% | M1-Economic-Financial-Fraud 34%、M3-Illegal-PhysicalHarm 33%、M5-HateSpeech-Political 14% |
| 32 | arpl | M5-HateSpeech-Political | 1 | 1.7% | M3-Illegal-PhysicalHarm 45%、M1-Economic-Financial-Fraud 38%、M2-Legal-Opinion 10% |
| 32 | arpl | M6-Sex | 1 | 0.0% | M2-Legal-Opinion 31%、M5-HateSpeech-Political 25%、M7-Health 22% |

## 8. 注意事项

- **口径不同不可严格对比**：7 类里随机抽 2 个未知，与旧的 13 类结果不是同一口径；
  这里的主对比是**同一口径下不同预处理变体之间**。
- **方差主要来自「抽中哪 2 个未知类」**：合并后类大小很不均（M5=465、M1=443 vs M6=M7=109），
  拒绝率的标准差可达 ±0.10 以上；变体之间的小差距（<0.02）可能落在噪声内。
- **均值/主方向用全部样本估计**（含未知样本特征），是无标签的宽松口径，见 §2 声明。
- **arpl 在去均值/ABTT 特征上系统性失效（重要）**：AUROC 掉到 0.20~0.34，小于 0.5 说明它输出的
  「已知度」分数**方向反了**——未知样本反而得分更高，于是 95% 的未知类被当成已知接受（拒绝率 ~2%）。
  该现象在 L24/L32、mean/abtt1、5 个 seed 上完全一致（标准差仅 0.02~0.06），不是噪声。
  同时它的**闭集 ACC 仍保持 0.89~0.92**，说明分类没坏、坏的只是拒绝用的那个分数。
  原因方向：去均值让特征谱变平，而 ARPL 的「反向点 + margin」打分对特征尺度/分布很敏感，
  它的 margin、tau 等超参是按原始特征调的；要公平对比，需要**先在去均值特征上重新调 ARPL**。
  因此本报告的开集结论以 **cac** 为准，arpl 一栏只作「原样套用旧超参会崩」的记录。
- raw 变体的数字与上一份 `合并B7_实验报告.md` 完全一致（同一批运行的产物）。
- 只有 5 个已知类，已知测试集规模随抽法变化，指标粒度较粗。

## 9. 产物位置

- 合并特征：`merge-b7/data/MMSB_sd_typo_mergeB7_layer{24,32}.npz`
- 去均值/ABTT 特征：`merge-b7/data/MMSB_sd_typo_mergeB7_layer{24,32}_center-{mean,abtt1}.npz`
- 每次运行：`merge-b7/outputs/<时间戳>_b7_[<变体>_]L<层>_s<seed>_<方法>/`
