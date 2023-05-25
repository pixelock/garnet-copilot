# coding: utf-8

"""
@author: pixelock
@file: manage.py
@time: 2023/5/21 22:12
"""

import os
from flask import Blueprint, request, config

from models.chatglm import ChatGLM
from .request_schema import ChatGLMRequest

bp_llm = Blueprint('llm', __name__)


def create_model():
    model_name = os.environ.get('COPILOT_LLM_NAME', 'chatglm')
    if 'chatglm' in model_name:
        chatglm_quantization = os.environ.get('COPILOT_LLM_QUAN', 'int8')
        if chatglm_quantization == 'float16':
            model_ = ChatGLM(model_name_or_path='THUDM/chatglm-6b', quantization='float16')
        elif chatglm_quantization == 'int8':
            model_ = ChatGLM(model_name_or_path='THUDM/chatglm-6b-int8', quantization='int8')
        elif chatglm_quantization == 'int4':
            model_ = ChatGLM(model_name_or_path='THUDM/chatglm-6b-int4', quantization='int4')
        else:
            raise ValueError(
                f'ChatGLM only has `float16`, `int8`, `int4` format, {chatglm_quantization} format is not supported'
            )
    else:
        raise ValueError(f'{model_name} model is not supported')

    return model_


model = create_model()


@bp_llm.route('/', methods=['POST'])
def make_completion():
    input_json = request.json

    prompt = input_json['prompt']
    history = input_json.get('history', [])
    max_length = input_json.get('maxLength', 2048)
    top_p = input_json.get('topP', 0.7)
    temperature = input_json('temperature', 0.95)

    answer = model(
        prompt=prompt,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        # history=history,
    )
