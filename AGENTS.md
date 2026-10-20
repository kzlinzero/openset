# AGENTS.md

基于冻结 VLM 隐状态特征做开集识别（OSR）的研究代码，数据来自 MM-SafetyBench。无 package、
无测试、无 lint，各脚本是独立 CLI。

## 环境

- conda 环境 `llm2`（Python 3.13）：`D:/ProgramKZ/AnacondaKZ/envs/llm2/python.exe`。Git Bash 里
  `python` 不在 PATH，需用完整路径或 `conda run -n llm2 python`。
- 改动后用 CPU 小跑验证，如 `cd openset-proto && python proto_openset.py --device cpu`（约 1 分钟）。

## 全局约定

- 加实验变体时不要改 `openset-proto/proto_openset.py`，换输入 npz 或传参数。
- 切分保持 `--split-mode group`（按 `question` 分组），否则近似重复样本会泄漏到测试集。
- 阈值只在已知类验证集标定；主结果看 `B_matched`，`A_full` 只看 AUROC。
- `--device` 选不到卡会报错，不会静默退回 CPU；CPU/GPU 有微小数值差，比较时固定 device。
- 每次运行新建时间戳子目录，`--run-name` 只改后缀，不覆盖旧结果。
- 代码、注释、报告用中文；`outputs/`、`data/` 等产物已 gitignore，不要提交。

## 根目录

- `load_data.py` — MM-SafetyBench / SPA-VL-harm / SIUO / MSSBench / FigStep 加载器，产出
  `{image_path, question, unsafe_category, ...}` 列表。

## get-embedding/

用 Qwen3-VL-8B-Instruct 给每条样本抽一个特征：生成回答后，对输入部分 token（含图片占位）的
hidden states 取均值。

- `build_embeddings.py` — 主流程。`--sample-size 200` 小跑；中断后重跑同一条命令即可从
  `<stem>.partial.jsonl` 续跑，`--no-resume` 从头开始。
- `model_runner.py` — 模型加载与池化；`--layer` 是 `hidden_states` 下标（8B 共 37 项）。
- `inspect_npz.py --npz_path <npz>` — 检查产物结构与样本。
- 产物 `<stem>.npz`：`features (N, 4096) float32` + 等长 `question` / `unsafe_category` /
  `unsafe_category_id` / `index` / `response`；字符串为 `<U`，无需 `allow_pickle`。
- 坑：脚本顶部硬编码 `CUDA_VISIBLE_DEVICES=1`；manifest 里记的是 Linux 绝对路径。

## openset-proto/

开集识别主脚本，四种方法 `ncm` / `proto` / `cac` / `arpl`，输入特征 npz。

- `python proto_openset.py --device gpu0 --npz <path>`；默认 `--npz` 路径已失效，必须显式传。
- `--n-unknown`（默认 4）、`--method`（可只跑一个）、`--seed`、`--target-tpr`（默认 0.95）。
- `collect_layer_results.py` — 汇总各层、多种子的 `B_matched` 指标。
- 每次运行生成 `outputs/<时间戳>_<name>/`，含 log、metrics.json、scores.csv、manifest.json。

## merge-b7/

13 类合并为 7 类后重跑 OSR，`proto_openset.py` 不改一行。

- `merge_taxonomy.py` → `data/*mergeB7*.npz`，只重映射标签。
- `center_features.py` → `data/*_center-{mean,abtt1}.npz`，去均值 / ABTT 预处理。
- `run_merge_b7.py` — 批量驱动，跳过已有输出（`--force` 重跑），`--dry-run` 先看命令。
- `aggregate.py` — 生成报告和 summary；报告表格不要手改。
- `data/`、`outputs/` 已 gitignore，`*.md` 报告入库。
- 坑：源特征文件名带 `[24]`，glob 会当字符集用；脚本按 `path.name` 精确比对，新代码照做。

## merge-b7-mproto/

在 merge-b7 的 7 类 + abtt1 特征上比较单原型与多原型 Proto；CAC 复用 merge-b7 旧结果。

- `mproto_openset.py` — 单次训练/评测，通过 importlib 只读复用旧脚本的切分/评估函数。
- `run_mproto.py` — 2 层 × 5 seed × 5 配置（proto + K=1–4），跳过已完成 run，`--dry-run` 先看命令。
- `aggregate_mproto.py` — 汇总新结果 + 旧 CAC，生成 `多原型Proto_实验报告.md` 和 summary JSON。
- 数据只读 `../merge-b7/data/`；`outputs/` 已 gitignore。
- 坑：K=1 应与单原型 proto 完全一致，报告里有逐 seed 校验；不一致先查实现。

## vlm-hidden-state-inspect-demo/

token 级隐状态查看 demo，独立运行；目录名带连字符，不能作为模块 import。

## 改实验前先读

- `openset-proto/README.md` — 方法、指标口径、产物说明。
- `openset-proto/manual_log_collection/20260921-132228/实验说明.md` — 任务与指标的人话版。
- `merge-b7/合并B7_去均值ABTT_实验报告.md` — 当前 7 类结论；ARPL 在去均值特征上失效的说明见 §8。
