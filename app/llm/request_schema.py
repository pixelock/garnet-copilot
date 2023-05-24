# coding: utf-8

"""
@author: pixelock
@file: request_schema.py
@time: 2023/5/24 21:08
"""

from typing import Optional, Union, List, Tuple

from app.request_schema import BaseRequest


class LLMRequest(BaseRequest):
    prompt: str


class ChatGLMRequest(LLMRequest):
    history: Optional[List[Union[Tuple[str, str], List[str, str]]]] = None
    maxLength: int = 2048
    topP: float = 0.7
    temperature: float = 0.95
