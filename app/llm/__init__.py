# coding: utf-8

"""
@author: pixelock
@file: __init__.py.py
@time: 2023/5/21 22:12
"""

from flask import Flask, current_app

from .manage import bp_llm


def init_llm(app: Flask):
    llm_type = app.config


def register_app(app: Flask):
    app.register_blueprint(bp_llm)
