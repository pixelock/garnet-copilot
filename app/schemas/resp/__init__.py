# coding: utf-8

"""
@author: pixelock
@file: __init__.py.py
@time: 2023/5/25 23:03
"""

from datetime import datetime
from pydantic import BaseModel, validator, root_validator
from pydantic.generics import GenericModel
from typing import Generic, TypeVar, Optional

DataT = TypeVar('DataT')
ErrorT = TypeVar('ErrorT')


class DataModel(BaseModel):
    request_time: Optional[str] = None
    response_time: Optional[str] = None
    cost: Optional[float] = None

    @validator('request_time', pre=True)
    def check_request_time(cls, request_time):
        if isinstance(request_time, datetime):
            return request_time.strftime('%Y-%m-%d %H:%M:%S.%f')
        elif isinstance(request_time, str):
            try:
                datetime.strptime(request_time, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError as e:
                raise ValueError(f'time data [{request_time}] can not be parsed')
        elif request_time is None:
            return request_time
        else:
            raise ValueError(
                f'time data must be a datetime or a string with `%Y-%m-%d %H:%M:%S.%f` format, got [{request_time}]'
            )

    @validator('response_time', pre=True, always=True)
    def check_response_time(cls, response_time):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    @root_validator
    def check_cost(cls, values):
        _cost = values['cost']
        _request_time = values['request_time']
        _response_time = values['response_time']
        if _cost is None and _request_time and _response_time:
            values['cost'] = (_response_time - _request_time).total_seconds() * 1000
        return values


class ErrorModel(BaseModel):
    code: int
    msg: str


class BaseResponse(GenericModel, Generic[DataT, ErrorT]):
    data: Optional[DataT]
    error: Optional[ErrorT]


Response = BaseResponse[DataModel, ErrorModel]
