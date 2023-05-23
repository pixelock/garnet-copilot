# coding: utf-8

"""
@author: pixelock
@file: config.py
@time: 2023/5/21 21:01
"""

from pydantic import BaseModel


class BaseConfig(BaseModel):
    name: str
    debug: bool = True


class DevelopConfig(BaseConfig):
    name: str = 'dev'
    debug: bool = True


class ProductConfig(BaseConfig):
    name: str = 'prod'
    debug: bool = False
