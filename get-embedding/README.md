# get-embedding

对 MM-SafetyBench 的采样数据，逐条抽取**样本级 feature**（一个向量代表整条数据），
用于后续的开集 / 价值对齐分析（分类、聚类、可视化等）。

抽取方式参考 `../vlm-hidden-state-inspect-demo`：让模型看完「图片 + 问题」并生成回答，
然后把「输入 + 输出」的完整序列喂回去做一次前向拿到 hidden states，
**只截取输入部分的 token**（0 ~ `input_len - 1`，包含图片占位 token）做平均，
得到该条数据的 feature。

## 安装

用仓库已有的 conda 环境即可（`.idea/misc.xml` 里配置的 `D:\ProgramKZ\AnacondaKZ\envs\llm2`）：

```bash
conda activate llm2   # torch / transformers / numpy / tqdm
```

## 运行

```bash
cd get-embedding

# 全量 1680 条，Qwen3.5-0.8B 本地快照，最后一层
python build_embeddings.py

# 先小跑 200 条验证流程
python build_embeddings.py --sample-size 200

# 换中间层、换精度
python build_embeddings.py --layer -4 --dtype bf16 --tag layer4
```

`--help` 有全部参数。常用几个：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--dataset-path` | `../datasets/MM-SafetyBench` | 数据集根目录（含 `imgs/`、`processed_questions/`） |
| `--image-type` | `SD_TYPO` | `SD` 纯图 / `SD_TYPO` 图上叠字 / `TYPO` 纯文字图 |
| `--sample-size` | `None` | 按类别比例分层抽样的条数；不传则全量 |
| `--seed` | `42` | 抽样随机种子 |
| `--model-name` | `Qwen/Qwen3.5-0.8B` | HF Hub ID 或模型名 |
| `--local-model-dir` | 本地快照路径 | 存在则优先本地加载；传 `""` 强制走在线/缓存 |
| `--dtype` | `fp16` | `fp16` / `bf16` / `fp32`；fp16 出 NaN 时换 `bf16` |
| `--layer` | `-1` | 取 `hidden_states` 第几层，支持负数。该模型共 25 项（embedding 输出 + 24 层） |
| `--max-new-tokens` | `128` | 生成 response 的最大新 token 数 |
| `--output-dir` | `./outputs` | 输出目录 |
| `--tag` | `""` | 附加到输出文件名 stem 后的标签 |
| `--no-resume` | 关 | 清空检查点从头重跑 |

跑长任务时如果中断（或某几条失败），**直接重跑同一条命令**即可从检查点续跑，
已经成功的 index 不会重算。

## 输出文件

文件名 stem 默认是 `mm_safetybench_<image_type>`（例如 `mm_safetybench_sd_typo`），
加了 `--tag` 会再拼一段。三份产物都在 `--output-dir` 下：

### `<stem>.npz` —— 主产物（推荐加载这个）

自包含，一条数据一行，六个字段全在里面。字符串用 numpy 的 unicode dtype 存，
所以加载**不需要** `allow_pickle=True`：

```python
import numpy as np

d = np.load("outputs/mm_safetybench_sd_typo.npz")
features   = d["features"]              # (N, 1024) float32
image_path = d["image_path"]            # (N,) <U 绝对路径
question   = d["question"]              # (N,) <U
category   = d["unsafe_category"]       # (N,) <U 例如 "01-Illegal_Activity"
category_id= d["unsafe_category_id"]    # (N,) int64 例如 1
response   = d["response"]              # (N,) <U 模型生成的回答
index      = d["index"]                 # (N,) int64 数据集里的原始下标
```

### `<stem>.jsonl` —— 可读镜像

每行一条 JSON 记录，字段同上（`feature` 是 float 列表）。方便肉眼抽查、
或者直接用 `pandas.read_json(lines=True)` 读。

### `<stem>.manifest.json` —— 运行配置

模型、精度、取层、池化方式、抽样参数、样本数、feature 维度、
失败列表，以及 torch / transformers / numpy 版本。复现实验看这个。

### `<stem>.partial.jsonl` —— 增量检查点

每算完一条就 append 一行（带 `os.fsync`），用来续跑。跑完会保留，
删掉它等价于清空进度。

## 文件结构

```
get-embedding/
├── model_runner.py       # 模型层：加载 / 构造输入 / 生成 / 抽 hidden states / 池化
├── build_embeddings.py   # 主流程：加载数据集 → 逐条抽取 → 落盘
└── outputs/              # 产物（已加 .gitignore）
```

`model_runner.py` 是从 `../vlm-hidden-state-inspect-demo/model_runner.py` 改的：
去掉了展示层，把「取哪一层」变成参数，并新增 `mean_pool_input`。
没有直接 import 原文件，因为 `vlm-hidden-state-inspect-demo` 带连字符、不是合法模块名。

## 一处细节

池化是把 `hidden[0, 0:input_len, :]` 全部 token 求平均，其中 `input_len` 是
`processor` 展开后的输入序列长度 —— **包含图片占位 token**。一张图会被展开成几百个
token，所以 feature 里相当一部分信息来自图像内容。这对 MM-SafetyBench 是有意义的
（有害信息本身就藏在图片文字里）。

平均在 float32 下做（`hidden` 本身可能是 fp16），避免低精度累加的精度损失。
