# coding: utf-8
# @File: chatglm.py
# @Author: pixelock
# @Time: 2023/4/26 0:09

import os
from typing import Dict, Union, Optional
from torch.nn import Module
from transformers import AutoModel
from accelerate import dispatch_model

from .base import LLM


class ChatGLM(LLM):
    def __init__(self,
                 model_id_or_dir,
                 quantization: Optional[int] = None,
                 multi_gpu: bool = False,
                 num_gpu: Optional[int] = None,
                 device_map: Optional[Dict[str, int]] = None,
                 **kwargs):
        if multi_gpu:
            assert num_gpu is not None or device_map is not None, 'attempt to deploy model on multi gpu, either `num_gpu` or `device_map` must not be `None`'

        self.quantization = quantization
        self.multi_gpu = multi_gpu
        self.num_gpu = num_gpu
        self.device_map = device_map

        super().__init__(
            model_id_or_dir=model_id_or_dir,
            **kwargs,
        )

    def init_model(self, *args, **kwargs):
        model = AutoModel.from_pretrained(
            self.model_id_or_dir,
            trust_remote_code=True,
            **kwargs,
        )

        if self.quantization is not None:
            model.quantize(self.quantization)

        if self.device == 'cuda':
            model.half()

            if self.multi_gpu:
                assert self.device == 'cuda', f'multi gpu mode must be deployed in cuda device, but current device if {self.device}'

                if self.device_map is None:
                    self.device_map = self.auto_configure_device_map(self.num_gpu)
                model = dispatch_model(model, device_map=self.device_map)
            else:
                model.cuda()
        else:
            model.float()

        return model

    @staticmethod
    def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
        # transformer.word_embeddings 占用1层
        # transformer.final_layernorm 和 lm_head 占用1层
        # transformer.layers 占用 28 层
        # 总共30层分配到num_gpus张卡上
        num_trans_layers = 28
        per_gpu_layers = 30 / num_gpus

        # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
        # windows下 model.device 会被设置成 transformer.word_embeddings.device
        # linux下 model.device 会被设置成 lm_head.device
        # 在调用chat或者stream_chat时,input_ids会被放到model.device上
        # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
        # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
        device_map = {
            'transformer.word_embeddings': 0,
            'transformer.final_layernorm': 0,
            'lm_head': 0,
        }

        used = 2
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f'transformer.layers.{i}'] = gpu_target
            used += 1

        return device_map
