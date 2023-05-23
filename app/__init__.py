# coding: utf-8

"""
@author: pixelock
@file: __init__.py
@time: 2023/5/11 22:42
"""

from flask import Flask


def create_app():
    app = Flask(__name__)

    return app
