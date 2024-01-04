#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: llm.py
@time: 2024/1/2 18:03
@desc:
"""
from functools import lru_cache

from langchain_community.llms.llamacpp import LlamaCpp


def create_llm(model_path: str, temperature=0.75, layers=20, stop: str = "", verbose: bool = False):
    _stop = ["</s>", stop] if stop else ["</s>"]
    llm = LlamaCpp(
        temperature=temperature,
        n_gpu_layers=layers,
        n_ctx=4096,
        max_tokens=1024,
        n_batch=512,
        top_p=0.95,
        top_k=50,
        stop=_stop,
        model_path=model_path,
        verbose=verbose
    )
    print("created llm")
    return llm


def create_zephyr_llm():
    return create_llm(model_path="/mnt/data/ggml/llama.cpp/models/7B/zephyr-7b-beta.Q8_0.gguf")


@lru_cache
def create_mixtral_llm(verbose=False):
    return create_llm(model_path="/mnt/data/ggml/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_0.gguf",
                      layers=10,
                      verbose=verbose)
