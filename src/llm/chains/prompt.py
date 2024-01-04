#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: prompt.py
@time: 2024/1/4 11:49
@desc:
"""
from langchain_core.prompts import PromptTemplate

"""
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>
"""


def create_zephyr_prompt():
    prompt = PromptTemplate.from_template("<|system|>\n{system_prompt}</s>\n<|user|>\n{msg}</s>\n<|assistant|>\n")
    return prompt


''' MIXTRAL
<s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
'''


def create_mixtral_prompt():
    prompt = PromptTemplate.from_template("<s> {history}</s> [INST] {msg} [/INST]")
    return prompt


"""
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""


def create_yi_prompt():
    prompt = PromptTemplate.from_template(
        "<|im_start|>system\n{system_prompt}<|im_end|>\n{history}\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n")
    return prompt
