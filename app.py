# coding: utf-8

"""
@author: pixelock
@file: app.py
@time: 2023/5/21 22:03
"""

import fire

from app import create_app


def main(host='0.0.0.0',
         port=5001,
         debug=True,
         application: str = 'llm',
         **kwargs):
    app = create_app(application=application)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    fire.Fire(main)
