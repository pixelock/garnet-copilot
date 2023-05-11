# coding: utf-8

"""
@author: pixelock
@file: cuda.py
@time: 2023/5/10 22:22
"""

import torch

__all__ = ['DEVICE', 'NUM_GPU']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPU = torch.cuda.device_count()
