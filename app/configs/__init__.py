# coding: utf-8

"""
@author: pixelock
@file: __init__.py.py
@time: 2023/5/25 22:20
"""

from flask import Flask
from enum import Enum
from typing import Optional
from pydantic import BaseModel

from .llm import LLMEnum, LLMConfig, ChatGLMConfig


class ServiceEnum(str, Enum):
    LLM = 'llm'


class ServiceConfig(BaseModel):
    APP: ServiceEnum


class LLMServiceConfig(ServiceConfig):
    APP: ServiceEnum = ServiceEnum.LLM
    llm_config: Optional[LLMConfig] = ChatGLMConfig()


def register_config(app: Flask,
                    application=ServiceEnum.LLM):
    if application == ServiceEnum.LLM:
        config = LLMServiceConfig().dict()
        app.config.from_mapping(config)

    return app
