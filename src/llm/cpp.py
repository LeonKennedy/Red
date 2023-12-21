#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: cpp.py
@time: 2023/12/15 15:16
@desc:
"""
from typing import List

from llama_cpp import Llama

from .config import zephyr_model

llm = Llama(model_path=zephyr_model, n_gpu_layers=0, chat_format="llama-2")


def llama_cpp_run(history: List):
    messages = [{
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    }] + history
    out = llm.create_chat_completion(messages=messages)
    choices = out["choices"][0]
    usage = out["usage"]
    return choices["message"], usage
