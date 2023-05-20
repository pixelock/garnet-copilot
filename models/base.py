# coding: utf-8
# @File: base.py
# @Author: pixelock
# @Time: 2023/4/25 23:58

from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain.llms.base import LLM


class BaseLLM(LLM):
    @classmethod
    @abstractmethod
    def load_model(cls, *args, **kwargs):
        """Load and initial LLM."""

    @classmethod
    @abstractmethod
    def load_tokenizer(cls, *args, **kwargs):
        """Load and initial tokenizer."""

    @property
    def trainable(self):
        return False



# class BaseLLM(BaseModel, ABC):
#     @classmethod
#     @abstractmethod
#     def load_model(cls, *args, **kwargs):
#         """Load and initial LLM."""
#
#     @classmethod
#     @abstractmethod
#     def load_tokenizer(cls, *args, **kwargs):
#         """Load and initial tokenizer."""
#
#     @abstractmethod
#     def _generate(self, *args, **kwargs):
#         """Run the LLM on the given prompts."""
#
#     @property
#     @abstractmethod
#     def llm_type(self) -> str:
#         """Return type of llm."""
