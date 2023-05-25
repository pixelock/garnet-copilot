# coding: utf-8

"""
@author: pixelock
@file: llm.py
@time: 2023/5/25 22:21
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel

from configs.cuda import DEVICE


class LLMEnum(str, Enum):
    ChatGLM = 'chatglm'


class LLMConfig(BaseModel):
    llm_type: str


class ChatGLMConfig(BaseModel):
    llm_type: str = LLMEnum.ChatGLM
    quantization: Optional[str] = 'int8'
    max_tokens: int = 2048
    temperature: float = 0.01
    top_p: float = 0.7
    device: str = DEVICE
    multi_gpu: bool = False
