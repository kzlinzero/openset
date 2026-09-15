"""模型层：加载、构造输入、生成、抽取隐藏状态。不做任何打印。"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def load_model(model_name: str, hf_token: str | None = None, dtype=torch.float16):
    """加载 processor 与模型，返回 (processor, model)。"""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=True,
    )
    return processor, model


def prepare_inputs(processor, model, prompt: str):
    """把纯文本 prompt 走 chat template，变成可直接喂给模型的 inputs。"""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def generate(model, processor, inputs, max_new_tokens: int = 128):
    """生成，返回 (完整 ids, 新增部分文本, 完整文本)。"""
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    full_text = processor.decode(outputs[0], skip_special_tokens=True)
    return outputs, response, full_text


def extract_hidden_states(model, input_ids):
    """对完整序列做一次前向，返回最后一层的 hidden states。"""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[-1]