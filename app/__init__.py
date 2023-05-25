# coding: utf-8

"""
@author: pixelock
@file: __init__.py
@time: 2023/5/11 22:42
"""

from flask import Flask

from app.configs import register_config
from app.configs import ServiceEnum


def create_app(application=ServiceEnum.LLM):
    app = Flask(__name__)

    register_config(
        app,
        application=application,
    )

    return app
