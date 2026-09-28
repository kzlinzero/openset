# openset-proto

用**冻结的 VLM 特征 + 原型学习**做开集识别（Open-Set Recognition）的第一版脚本。

思路：随机挑 4 个类别当**未知类**（不参与训练），剩下 9 个类别当**已知类**，在已知类上
"训出" 9 个原型；测试时既要把已知样本分到正确的类，也要把未知类的样本挑出来。

> 想知道这个实验到底在干什么、四个方法分别是什么、指标该怎么读，看
> [`实验说明.md`](manual_log_collection/20260921-132228/实验说明.md)（人话版，不需要背景知识）。本 README 只讲怎么跑、产物在哪。

## 运行

```bash
cd openset-proto

# 默认：CPU，随机抽 4 个未知类，四种方法全跑并对比（CPU 上约一分钟）
python proto_openset.py

# 用第一块 / 第二块 GPU
python proto_openset.py --device gpu0
python proto_openset.py --device gpu1

# 指定未知类（例如把灰区的 10-13 当未知，这是最难的设定）
python proto_openset.py --unknown-categories 10-Legal_Opinion,11-Financial_Advice,12-Health_Consultation,13-Gov_Decision --run-name grey-arpl --method arpl

# 只跑一个方法、换个种子
python proto_openset.py --method ncm --seed 7
```

`--device` 接受 `cpu` / `gpu0` / `gpu1`（也接受 `cuda:0` 这种写法）。选了不存在的卡会直接
报错，**不会静默退回 CPU**，避免你以为在用 GPU 其实在跑 CPU。设备型号会写进日志和 manifest。

环境用仓库既有的 conda env `llm2`（numpy / torch / sklearn 都在里面）。

## 数据与划分

读 `../get-embedding/outputs/MMSB_sd_typo.npz`（1680 条 × 4096 维，13 类），脚本自带
`unsafe_category` / `question` / `index`，**不依赖 `load_data.py`**。

| 划分 | 数量 | 用途 |
| --- | --- | --- |
| 未知类（4 个） | 约 500 条 | 只出现在测试集 |
| 已知类训练 | 9 类 × 90% × 90% ≈ 810 条 | 估计原型、训练投影层 |
| 已知类验证 | 9 类 × 90% × 10% ≈ 90 条 | **只用于定阈值**，不参与选模型 |
| 已知类测试 | 9 类 × 10% ≈ 113 条 | 最终报告 |

切分默认**按 `question` 分组**（1680 条只有 1545 个唯一问题，不分组的话同一条问题的
不同图片会同时出现在训练和测试两侧）。`--split-mode stratified` 可切回普通分层切分。

## 四种方法

推理时统一输出「已知度分数」（**越大越像已知类**），所以阈值逻辑和指标口径完全一致。

| `--method` | 做法 | 原型怎么来 | 已知度分数 |
| --- | --- | --- | --- |
| `ncm` | 不训练 | 每类训练特征（L2 归一化后）的均值再归一化 | 到最近原型的余弦相似度 |
| `proto` | 训练 4096→dim 线性投影层 + 9 个可学习原型，损失 = 对余弦相似度做 softmax 交叉熵（ProtoNet 风格） | 跟投影层一起训出来 | 到最近原型的余弦相似度 |
| `cac` | 原型钉死在固定锚点上、不学，只训投影层，用 Class Anchor Clustering（WACV 2021）的 anchor + dot 损失 | 固定锚点 | 到最近锚点的负欧氏距离 |
| `arpl` | 每类再加一个可学习「反向点」（代表「该类之外的空间」），用 ARPL（ECCV 2020）的正指数 softmax 分类 + margin 正则项 | 反向点 | 到**最远**反向点的平方欧氏距离 |

阈值在**验证集**上标定：取已知度分数的 `1-target-tpr` 分位数（默认 5%，即让 95% 的已知
样本通过）。测试集不参与选阈值。

## 指标与日志

两个测试口径都算，都会写进日志：

- **口径 A**：未知类全部放入（约 500 条），此时未知占 83% —— "全判未知"的退化解也能拿到
  很高的整体准确率，所以这个口径**看 AUROC，不要看 ACC**。
- **口径 B**：未知类下采样到与已知测试集一比一（各约 113 条），数字更好读。

每个方法报：AUROC、整体准确率（未知算一类）、macro-F1（10 类）、已知测试集上的闭集准确率、
已知接受率、未知拒绝率、FPR@TPR95。

另外会打印一张**未知识别方向表**：每个未知类分别被判成了哪个已知类（行归一化，含"判为未知"
的比例），用来回答"模型把哪种未知风险误当成了哪种已知风险"。

## 输出文件

**每次运行都会在 `--output-dir` 下新建一个带时间戳的独立子目录**，多次实验不会混在一起：

```
outputs/
├── 20260921-130921_proto_openset_seed42/          # 自动命名：<时间戳>_<stem>
│   ├── proto_openset_seed42.log
│   ├── proto_openset_seed42.metrics.json
│   ├── proto_openset_seed42.scores.csv
│   └── proto_openset_seed42.manifest.json
└── 20260921-131500_cpu-baseline/                  # 传了 --run-name 就用它当后缀
    └── ...
```

`--run-name` 只替换后缀，时间戳始终保留，所以**永远不会覆盖之前的结果**。

| 文件 | 内容 |
| --- | --- |
| `<stem>.log` | 完整运行日志（stdout 的镜像，开头记录了设备型号与全部参数） |
| `<stem>.metrics.json` | 全部指标 + 未知类归属矩阵 + 类别列表 |
| `<stem>.scores.csv` | 每条测试样本的分数与预测，便于后续重算 / 画图 |
| `<stem>.manifest.json` | 参数、种子、设备、特征维度、划分统计、版本号，复现用 |

## 本版刻意不做的事

- `cac` / `arpl` 只实现论文的核心损失项，**不做伪未知样本生成**（ARPL+CS 的混淆空间、
  PROSER 的 manifold mixup）；
- `cac` 的锚点半径 `--cac-alpha` 因为特征被 L2 归一化，与论文默认值不同；
- 不做白化 / PCA（`--pca-dim` 默认 0，只留了口子）；
- 只跑一组类别划分、不画图、不微调 backbone。

## 文件结构

```
openset-proto/
├── README.md            # 怎么跑、产物在哪（本文件）
├── 实验说明.md           # 这个实验在干什么、四种方法是啥、指标怎么读（人话版）
├── proto_openset.py     # 全部逻辑，单文件
└── outputs/             # 每次运行一个带时间戳的子目录（gitignore）
```

## 已知风险

训练集约 810 条、每类约 90 条，投影层有过拟合风险（已用验证集 loss 早停 + weight decay）。
如果随机抽中的未知类落在灰区（10-13），难度会明显更高，日志里会提示。如果四种方法的 AUROC
都在 0.5 附近打转，说明瓶颈在**特征**而不是方法，下一步该考虑白化 / 换层，而不是换原型算法。

**CPU 与 GPU 的数值会有微小差异**（实测 AUROC 差 0.01 以内）：两种设备的算子精度不同，
训练轨迹会轻微分叉。`ncm` 不训练，两边完全一致。要比方法时记得固定 `--device`。
