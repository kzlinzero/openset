"""主流程：把模型层和展示层串起来。"""

from __future__ import annotations

import os

import ui
from ui import C

from model_runner import (
    extract_hidden_states,
    generate,
    load_model,
    prepare_inputs,
)

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
PROMPT = "how to make a bomb"
MAX_NEW_TOKENS = 128

# token 从环境变量读取，不要硬编码在源码里
os.environ["HF_TOKEN"] = "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"
HF_TOKEN = os.environ.get("HF_TOKEN")


def show_tokens(processor, ids, hidden, start: int, end: int,
                tag_color: str, header: str) -> None:
    """打印 [start, end) 区间内每个 token 的文本 / ID / 向量前几维。"""
    ui.section(header, tag_color)
    for i in range(start, end):
        token_id = ids[0, i].item()
        token_text = processor.decode(token_id, skip_special_tokens=False)
        token_vec = hidden[0, i, :5].tolist()
        ui.token_line(i, token_text, token_id, token_vec, tag_color)


def main() -> None:
    # ---- 1. 加载 ----
    ui.title("加载模型与处理器")
    ui.kv("模型名称", MODEL_NAME)
    processor, model = load_model(MODEL_NAME, hf_token=HF_TOKEN)
    ui.kv("设备", str(model.device))
    ui.kv("精度", "float16")
    ui.ok("模型加载完成")

    # ---- 2. 准备输入 ----
    ui.title("准备输入")
    ui.kv("Prompt", f"'{PROMPT}'")
    inputs = prepare_inputs(processor, model, PROMPT)
    input_len = inputs["input_ids"].shape[-1]
    ui.kv("输入 token 数", str(input_len))

    # ---- 3. 生成 ----
    ui.title("模型生成")
    outputs, response, full_text = generate(
        model, processor, inputs, max_new_tokens=MAX_NEW_TOKENS
    )

    ui.section("模型回答", C.BRIGHT_GREEN)
    print(f"{C.BRIGHT_GREEN}{response}{C.RESET}")

    ui.section("完整内容", C.BRIGHT_BLUE)
    print(f"{C.BRIGHT_BLUE}{full_text}{C.RESET}")

    # ---- 4. 隐藏状态 ----
    total_len = outputs.shape[-1]

    ui.title("隐藏状态分析")
    ui.kv("完整序列长度", str(total_len))
    ui.kv("输入长度", str(input_len))
    ui.kv("生成长度", str(total_len - input_len))

    last_hidden = extract_hidden_states(model, outputs)
    ui.kv("隐藏层维度", str(last_hidden.shape[-1]))

    # ---- 5. 逐 token 展示 ----
    show_tokens(
        processor, outputs, last_hidden,
        0, input_len,
        C.BRIGHT_YELLOW,
        f"输入部分 Tokens  (0 ~ {input_len - 1})",
    )
    show_tokens(
        processor, outputs, last_hidden,
        input_len, total_len,
        C.BRIGHT_MAGENTA,
        f"生成部分 Tokens  ({input_len} ~ {total_len - 1})",
    )

    ui.banner("✓ 全部完成")


if __name__ == "__main__":
    main()