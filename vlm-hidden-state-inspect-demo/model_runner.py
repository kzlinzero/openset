"""模型层：加载、构造输入、生成、抽取隐藏状态。不做任何打印。"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any
from urllib.request import urlopen

import torch
from PIL import Image
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


def load_image(image: str | Image.Image | None) -> Image.Image | None:
    """统一把 路径 / URL / PIL.Image / None 转成 RGB 的 PIL.Image。"""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        with urlopen(image, timeout=10) as resp:
            return Image.open(BytesIO(resp.read())).convert("RGB")
    return Image.open(image).convert("RGB")


def prepare_inputs(
    processor,
    model,
    prompt: str,
    image: str | Image.Image | None = None,
):
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
    """生成，返回 (完整 ids, 新增部分文本, 完整文本)。"""
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    full_text = processor.decode(outputs[0], skip_special_tokens=True)
    return outputs, response, full_text


def extract_hidden_states(model, inputs, full_ids):
    """对完整序列做一次前向，返回最后一层的 hidden states。

    多模态下不能只传 input_ids —— 需要保留 pixel_values / image_grid_thw /
    mm_token_type_ids 等，所以这里复用原始 inputs，但凡是跟序列长度相关的
    张量（input_ids / attention_mask / mm_token_type_ids）都要对齐到 full_ids 的长度。
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
    return out.hidden_states[-1]