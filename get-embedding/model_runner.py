"""模型层：加载、构造多模态输入、生成、抽取隐藏状态、池化成 feature。

改写自 vlm-hidden-state-inspect-demo/model_runner.py，去掉了展示逻辑，
并把「取哪一层」变成参数，额外提供池化函数。不做任何打印。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def resolve_model_source(
    model_name: str,
    local_dir: str | None = None,
) -> tuple[str, bool]:
    """决定到底从哪里加载。

    规则：
      - local_dir 指定且是一个存在的目录 → 用 local_dir（本地加载）
      - 否则                              → 用 model_name（HF Hub ID 或缓存）

    返回 (实际加载路径, 是否来自本地目录)。
    """
    if local_dir and os.path.isdir(local_dir):
        return local_dir, True
    return model_name, False


def load_model(
    model_name: str,
    hf_token: str | None = None,
    dtype: torch.dtype = torch.float16,
    local_dir: str | None = None,
):
    """加载 processor 与模型。

    返回 (processor, model, source, is_local)。
    """
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    source, is_local = resolve_model_source(model_name, local_dir)

    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        source,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model, source, is_local


def load_image(image: str | Image.Image | None) -> Image.Image | None:
    """统一把 路径 / PIL.Image / None 转成 RGB 的 PIL.Image。"""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def prepare_inputs(processor, model, prompt: str, image: str | Image.Image | None = None):
    """把 prompt（可选图片）走 chat template，返回可直接喂给模型的 inputs。

    image 为 None 时退化为纯文本输入；否则拼成 [image, text] 的多模态消息。
    """
    pil_image = load_image(image)

    content: list[dict[str, Any]] = []
    if pil_image is not None:
        content.append({"type": "image", "image": pil_image})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # 多模态不要直接 tokenize=True，否则 image 占位符不会展开
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    if pil_image is not None:
        inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")

    return inputs.to(model.device)


def generate(model, processor, inputs, max_new_tokens: int = 128):
    """贪心生成，返回 (完整 ids, 新增部分文本, 完整文本)。"""
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    full_text = processor.decode(outputs[0], skip_special_tokens=True)
    return outputs, response, full_text


def extract_hidden_states(model, inputs, full_ids, layer: int = -1):
    """对「输入 + 生成」的完整序列做一次前向，返回指定层的 hidden states。

    多模态下不能只传 input_ids —— 需要保留 pixel_values / image_grid_thw /
    mm_token_type_ids 等，所以这里复用原始 inputs，但凡是跟序列长度相关的
    张量（input_ids / attention_mask / mm_token_type_ids）都要对齐到 full_ids 的长度。

    layer 支持负数索引；hidden_states 的第 0 项是 embedding 输出，
    第 i 项是第 i 层 transformer 的输出，共 num_layers + 1 项。
    """
    full_len = full_ids.shape[-1]

    forward_kwargs: dict[str, Any] = {}
    for k, v in inputs.items():
        if k == "input_ids":
            forward_kwargs[k] = full_ids
        elif k == "attention_mask":
            forward_kwargs[k] = torch.ones_like(full_ids)
        elif k == "mm_token_type_ids":
            # 生成部分是纯文本 → token type 补 0
            pad_len = full_len - v.shape[-1]
            if pad_len > 0:
                pad = torch.zeros(
                    (*v.shape[:-1], pad_len),
                    dtype=v.dtype,
                    device=v.device,
                )
                v = torch.cat([v, pad], dim=-1)
            forward_kwargs[k] = v
        else:
            forward_kwargs[k] = v

    with torch.no_grad():
        out = model(**forward_kwargs, output_hidden_states=True)

    hidden_states = out.hidden_states
    num_entries = len(hidden_states)
    if not -num_entries <= layer < num_entries:
        raise ValueError(
            f"layer={layer} 超出范围，该模型 hidden_states 共 {num_entries} 项 "
            f"(embedding 输出 + {num_entries - 1} 层)，合法范围 "
            f"[{-num_entries}, {num_entries - 1}]"
        )
    return hidden_states[layer], num_entries


def mean_pool_input(hidden, input_len: int) -> np.ndarray:
    """只取输入部分 [0, input_len) 的 token 向量做平均，返回 float32 的 numpy 向量。

    平均在 float32 下做，避免低精度 dtype 累加时的精度损失。
    """
    if input_len <= 0:
        raise ValueError(f"input_len 必须为正，实际为 {input_len}")
    if input_len > hidden.shape[1]:
        raise ValueError(
            f"input_len={input_len} 超过 hidden 的序列长度 {hidden.shape[1]}"
        )
    vecs = hidden[0, :input_len, :].to(torch.float32)
    return vecs.mean(dim=0).cpu().numpy().astype(np.float32)
