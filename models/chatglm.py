# coding: utf-8
# @File: chatglm.py
# @Author: pixelock
# @Time: 2023/4/26 0:09

from pydantic import root_validator
from typing import Dict, Optional, Any, List
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer
from accelerate import dispatch_model

from models.base import BaseLLM
from configs import DEVICE, NUM_GPU
from utils.cuda import check_gpu_status, fetch_available_gpus

CHATGLM_GPU_MEMORY_MAP = {
    'float16': 15000,
    'int8': 9000,
    'int4': 6000,
}


def load_model_on_multi_gpus(checkpoint_path: str,
                             quantization: str = 'int8',
                             device_map: Dict = None,
                             verbose: bool = True,
                             **kwargs) -> Module:
    available_gpus = fetch_available_gpus(
        threshold=CHATGLM_GPU_MEMORY_MAP.get(quantization, 0),
        devices=device_map,
        verbose=verbose
    )
    main_device = available_gpus[0]
    if verbose:
        print(f'available gpus: {available_gpus} | main device(with largest free memory): {main_device}')

    num_gpus = len(available_gpus)
    device_map = auto_configure_device_map(device_ids=available_gpus)
    if verbose:
        print(f'device map of model parameters:\n{device_map}')

    from accelerate import dispatch_model

    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()
    model = dispatch_model(model, device_map=device_map)
    if verbose:
        print('GPUs status after model been loaded:')
        check_gpu_status(verbose=verbose)
    return model


def auto_configure_device_map(num_gpus: int = None, device_ids: List[int] = None) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    device_ids = device_ids or list(range(num_gpus))
    target = device_ids.pop(0)

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {
        'transformer.word_embeddings': target,
        'transformer.final_layernorm': target,
        'lm_head': target,
    }

    used = 2
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            target += 1
            used = 0
        device_map[f'transformer.layers.{i}'] = target
        used += 1

    return device_map


class ChatGLM(BaseLLM):
    model_name_or_path: str
    quantization: Optional[str] = 'int8'
    device: str = DEVICE
    multi_gpu: bool = False
    tokenizer_kwargs: Dict = dict()
    model_kwargs: Dict = dict()

    model: Any = None
    tokenizer: Any = None

    @classmethod
    def load_tokenizer(cls, model_name_or_path, **kwargs):
        return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)

    @classmethod
    def load_model(cls, model_name_or_path, device='cuda', multi_gpu=False, quantization='int8', **kwargs):
        if device == 'cpu':
            # deploy on CPU
            model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float()
        elif multi_gpu and NUM_GPU > 1:
            # deploy on multi GPUs
            model = load_model_on_multi_gpus(
                checkpoint_path=model_name_or_path,
                quantization=quantization,
                verbose=True,
                **kwargs,
            )
        else:
            # deploy on single GPU
            model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().to(device)

        model.eval()
        return model

    @root_validator
    def validate_environment(cls, values: Dict) -> Dict:
        values['tokenizer'] = cls.load_tokenizer(values['model_name_or_path'], **values['tokenizer_kwargs'])
        values['model'] = cls.load_model(
            model_name_or_path=values['model_name_or_path'],
            device=values['device'],
            multi_gpu=values['multi_gpu'],
            quantization=values['quantization'],
            **values['model_kwargs'],
        )
        return values

    @property
    def llm_type(self) -> str:
        return 'ChatGLM'

    def _generate(self,
                  prompt,
                  history=None,
                  temperature=0.95,
                  top_p=0.7,
                  num_beams=1,
                  max_length=2048,):
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=history,
            max_length=max_length,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature
        )
        return response, history


if __name__ == '__main__':
    model = ChatGLM(model_name_or_path='THUDM/chatglm-6b', quantization='int4')
    print('Loaded ChatGLM')
