# coding: utf-8
# @File: base.py
# @Author: pixelock
# @Time: 2023/4/25 23:58

import torch


class LLM(object):
    def __init__(self, model_id_or_dir: str, **kwargs):
        self.model_id_or_dir = model_id_or_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.init_model(**kwargs)

    def init_model(self, **kwargs):
        raise NotImplementedError
