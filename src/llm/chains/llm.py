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
from typing import List

from langchain_community.llms.llamacpp import LlamaCpp


def create_llm(model_path: str, stop: List, temperature=0.75, layers=20, verbose: bool = False) -> LlamaCpp:
    llm = LlamaCpp(
        temperature=temperature,
        n_gpu_layers=layers,
        n_ctx=4096,
        max_tokens=1024,
        n_batch=512,
        top_p=0.95,
        top_k=50,
        stop=stop,
        model_path=model_path,
        verbose=verbose
    )
    print("created llm, load:", model_path)
    return llm


def create_zephyr_llm(verbose=False):
    return create_llm(model_path="/mnt/data/ggml/llama.cpp/models/7B/zephyr-7b-beta.Q8_0.gguf",
                      stop=["</s>"], verbose=verbose)


def create_mixtral_llm(verbose=False):
    return create_llm(model_path="/mnt/data/ggml/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_0.gguf",
                      stop=[],
                      layers=12,
                      verbose=verbose)


def create_yi_llm(verbose=False):
    return create_llm("/mnt/data/ggml/llama.cpp/models/nous-hermes-2-yi-34b.Q4_0.gguf",
                      ["<|im_end|>"],
                      layers=16,
                      verbose=verbose)


_current_llm = None
_current_llm_name = None


def get_llm_by_name(name: str, verbose: bool = False):
    global _current_llm, _current_llm_name
    if name == "zephyr":
        if _current_llm_name != name:
            _current_llm = create_zephyr_llm(verbose=verbose)
    elif name == "mixtral":
        if _current_llm_name != name:
            _current_llm = create_mixtral_llm(verbose=verbose)
    elif name == "yi":
        if _current_llm_name != name:
            _current_llm = create_yi_llm(verbose=verbose)
    else:
        raise ValueError(f"Unknown llm name {name}")
    _current_llm_name = name
    return _current_llm
