# coding: utf-8

"""
@author: pixelock
@file: __init__.py.py
@time: 2023/5/25 23:02
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, validator


class BaseRequest(BaseModel):
    request_time: Optional[str] = None

    @validator('response_time', pre=True, always=True)
    def check_request_time(cls, request_time):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
