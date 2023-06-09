# coding: utf-8
# @File: remote.py
# @Author: pixelock
# @Time: 2023/5/20 8:56

__all__ = ['RemoteLLM']

import requests
from typing import List, Dict, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class RemoteLLM(LLM):
    url: str
    prompt_key: str
    return_key: str
    headers: Optional[Dict] = None
    request_body: Optional[Dict] = None

    @property
    def _llm_type(self) -> str:
        return 'RestfulLLM'

    def _call(
        self,
        prompt: str,
        headers: Dict = None,
        request_body: Dict = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        headers = headers or self.headers
        request = request_body or self.request_body
        request[self.prompt_key] = prompt

        response = requests.post(
            url=self.url,
            headers=headers,
            json=request,
        )
        text = response.json()[self.return_key]
        return text
