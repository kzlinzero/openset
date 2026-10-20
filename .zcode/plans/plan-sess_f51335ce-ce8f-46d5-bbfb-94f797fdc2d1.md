# 新实验实施计划：7 类合并 + abtt1 + 多原型 Proto

## 1. 已确认的实验设置

| 项目 | 设定 |
|---|---|
| 特征 | `merge-b7/data/MMSB_sd_typo_mergeB7_layer{24,32}_center-abtt1.npz`，只读复用 |
| 类别 | 合并后的 7 类 |
| 层 | decoder 24、32 |
| seed | 42–46（与旧报告相同的未知类组合和切分） |
| 未知类 | `--n-unknown 2`，切分口径 `group`（按 question 分组） |
| 对比方法 | 单原型 `proto`、多原型 `proto-multi`（K=1,2,3,4）、旧 CAC 结果 |
| 主表 K | 固定 K=3；K=1–4 全网格作敏感性 |
| 聚合方式 | 训练用类内 logsumexp，打分用所有原型的 max |
| 多样性正则 | 不加，训练后诊断 |
| 初始化 | `randn * 0.1`，与旧 Proto 一致 |
| 超参数 | 完全沿用旧默认：dim=256, tau=0.1, lr=1e-3, batch=64, weight_decay=1e-4, epochs=200, patience=50, target_tpr=0.95 |
| 设备 | gpu0 |
| 诊断 | 已知类，分 train/val/test；原型使用率、类内两两余弦、软分配熵 |
| 逐样本输出 | scores.csv 增加最近原型编号和熵 |
| 报告 | 中文 Markdown + summary JSON；只报 B_matched；均值±标准差 |
| K=1 校验 | 写入报告，核对与单原型 proto 是否一致 |
| 代码位置 | 根目录新建 `merge-b7-mproto/` |
| 旧代码 | 不修改 `openset-proto/proto_openset.py` 和 `merge-b7/*` |

## 2. 新目录结构

```
merge-b7-mproto/
├── README.md                     # 实验目的、公式、运行方式、产物说明
├── .gitignore                    # outputs/
├── mproto_openset.py             # 单次训练/评测脚本
├── run_mproto.py                 # 批量驱动（50 次运行）
├── aggregate_mproto.py           # 汇总新结果 + 旧 CAC，生成报告
├── 多原型Proto_实验报告.md        # 生成物，入库
├── aggregate_mproto_summary.json # 生成物，入库
└── outputs/                      # gitignore
```

数据只读引用 `../merge-b7/data/`；旧 CAC 只读引用 `../merge-b7/outputs/`。

## 3. `mproto_openset.py`

### 3.1 复用旧代码
目录名带连字符，不能普通 import。用 `importlib.util.spec_from_file_location` 加载
`../openset-proto/proto_openset.py`，只读复用：

`load_features`、`choose_unknown`、`split_once`、`make_matched_unknown`、`evaluate`、
`attribution_table`、`Logger`、`render_table`、`resolve_device`、`set_seed`、`describe_device`。

不复用 `run()` 和 `run_torch_method()`（前者写死方法列表和 A_full/B_matched 流程，后者写死
`build_torch_model`）。训练循环约 46 行，复制到新脚本，并把模型工厂换成本地工厂。

### 3.2 多原型模型
记 `z = normalize(Wx)`，原型 `P ∈ R^{C×K×d}`，每个原型也 L2 归一化：

```
s[c,k] = z · P[c,k]
l[c]   = logsumexp_k(s[c,k] / tau)
loss   = CrossEntropy(l, y)
score  = max_{c,k} s[c,k]          # 已知度分数，越大越像已知
pred   = argmax_c max_k s[c,k]     # 已知类索引
```

- `K=1` 时与旧 `ProtoModel` 数学等价，作为实现正确性校验。
- 不加入多样性正则；训练后在诊断里看同类原型余弦是否接近 1、实际使用原型数是否远小于 K。
- 原型用 `nn.Parameter(torch.randn(C, K, d) * 0.1)` 初始化。

### 3.3 训练与评估
- 数据、类别选择、group 切分、B_matched 下采样全部调用旧函数，保证与旧实验同口径。
- 模型训练循环复制旧版：`AdamW`、按 seed 重置 torch、验证集 loss 早停、保存最佳 state。
- 阈值 tau 仍在验证集已知分数上按 `1-target_tpr` 分位标定；测试集不参与。
- 只计算 B_matched 口径，不计算 A_full。
- `--method {proto,proto-multi}`；`--n-protos` 仅对多原型生效。
- `--n-unknown` 默认 2，`--split-mode` 默认 group，其余参数沿用旧默认值。

### 3.4 诊断
对每个划分（train/val/test）的每个已知类 c：

- **使用率**：该类样本在本类 K 个原型中最近原型的硬分配计数/比例；`active` 为使用率 > 0 的原型数。
- **软分配熵**：`p = softmax(s[c,:] / tau)`，`H_norm = -Σ p log p / log K`，按样本平均；K=1 时为 0。
- **类内余弦**：同类原型两两余弦的 min/mean/max；K=1 时记 N/A。

未知类样本不进入诊断汇总；K=1 的诊断值为平凡值，报告中注明。

### 3.5 输出
每次运行仍是独立时间戳目录，包含：

- `<stem>.log`
- `<stem>.metrics.json`：沿用旧字段（`known_classes/unknown_classes/counts/results/attribution`），
  `results[method]["variants"]["B_matched"]`；新增 `diagnostics` 和 `n_protos`。
- `<stem>.scores.csv`：旧列 `index,category,is_unknown,in_matched_unknown,score_<method>,pred_<method>`，
  追加 `proto_id_<method>`（预测类内最近原型编号）和 `entropy_<method>`（预测类内软分配熵）。
  范围仍是已知测试集 + 全部未知样本；训练/验证的逐样本分配只进诊断汇总，不写 csv。
- `<stem>.manifest.json`：参数、数据路径、设备、类别划分计数、版本信息、旧脚本路径。
  旧 manifest 的 `per_class_split` 被指标表覆盖的 bug 在新脚本中修正。

## 4. `run_mproto.py`

- 配置矩阵：2 层 × 5 seed × 5 个配置（proto + K=1,2,3,4）= **50 次新运行**。
- run-name：`mproto_proto_L24_s42`、`mproto_k1_L24_s42` … `mproto_k4_L32_s46`。
- 每次调用 `mproto_openset.py --method <proto|proto-multi> --n-protos <K> --n-unknown 2 --seed <s> --device gpu0`。
- 支持 `--dry-run`、`--force`、`--device`、`--python`；已有同名 run 目录默认跳过；driver 日志写
  `outputs/_driver_logs/<run_name>.stdout.log`；任一子进程失败立即中止。
- CAC 不重跑：旧结果由汇总脚本读取。

## 5. `aggregate_mproto.py`

- 扫描新目录 `outputs/*_mproto_{proto,k1..k4}_L{24,32}_s{42..46}`。
- 扫描旧 CAC：`../merge-b7/outputs/*_b7_abtt1_L{24,32}_s{42..46}_cac`，读取
  `proto_openset_seed<seed>_cac.metrics.json` 的 `B_matched`。
- 完整性检查：6 个方法配置（cac、proto、k1–k4）× 2 层 × 5 seed 必须齐全；缺任何一个直接报错，
  不生成缺行报告。
- 旧 CAC 数字与旧报告核对：L24 abtt1 AUROC 应为 0.9109±0.0147，L32 为 0.9056±0.0193；
  偏差超过报告的四位小数就停下排查。
- 报告 `多原型Proto_实验报告.md`，章节：
  1. 数据、特征、切分、方法、命令与产物来源（含旧 CAC 目录清单）
  2. 主结果 B_matched：AUROC、闭集 ACC、未知拒绝率、FPR@TPR95，5 seed 均值±标准差，按层分表
  3. K=1 与单原型 proto 的一致性检查（逐层、逐 seed 的指标差；超过 1e-4 标红说明）
  4. 原型诊断：train/val/test 分表，给类均 active 原型数、归一化熵、类内余弦
  5. 结论与注意事项（只基于本次数字，不预设多原型一定更好；注明 ABTT 全样本拟合口径、
     合并类不均衡、val 同时用于早停和阈值）
- 同时写 `aggregate_mproto_summary.json`，包含全部指标和诊断的机器可读版本。

## 6. 验证与执行顺序

1. 新建目录、README、`.gitignore`。
2. 实现 `mproto_openset.py`、`run_mproto.py`。
3. CPU 冒烟：L24/seed42 各跑一次 `proto` 和 `proto-multi K=3`，检查文件字段、CSV 列、诊断结构。
4. `run_mproto.py --dry-run` 检查 50 条命令和路径。
5. gpu0 跑满 50 次；预计约 1 小时。
6. 实现 `aggregate_mproto.py`，生成报告与 summary。
7. 校验 K=1 与 proto 一致、旧 CAC 数字与旧报告一致、所有 run 齐全。
8. 最后在根目录 `AGENTS.md` 增加 `merge-b7-mproto/` 的目录说明和运行命令（不改旧实验代码）。

## 7. 明确不做

- 不跑 ncm、arpl、mean/none 变体、A_full 附表。
- 不做多样性正则、不调超参、不做原始 13 类对齐分析、不做逐 seed 明细表/配对差值、不存 loss 曲线。
- 不修改 `openset-proto/proto_openset.py`、`merge-b7/run_merge_b7.py`、`merge-b7/aggregate.py`。

## 8. 风险

- 旧 CAC 结果依赖 `merge-b7/outputs/` 完整存在；缺失时汇总脚本直接失败。
- K=1 与 proto 可能因 `einsum` 与矩阵乘的浮点差异不是逐位相同，报告用容差 1e-4 判断。
- val/test 的已知类样本较少（约 10–25 条/类），诊断数字波动会较大，报告中同时给每类样本数。
- 50 次训练全部固定 gpu0；若显存/设备被占用，driver 会在第一次失败时中止，不会静默换设备。