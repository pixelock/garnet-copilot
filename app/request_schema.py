# coding: utf-8

"""
@author: pixelock
@file: request_schema.py
@time: 2023/5/23 22:35
"""

from datetime import datetime
from pydantic import BaseModel, validator
from typing import Optional


class BaseRequest(BaseModel):
    request_time: Optional[str] = None

    @validator('response_time', pre=True, always=True)
    def check_request_time(cls, request_time):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
