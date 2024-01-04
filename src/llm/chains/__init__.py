#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: __init__.py.py
@time: 2024/1/2 17:32
@desc:
"""
from langchain_core.prompts import HumanMessagePromptTemplate

from .zephyr_chat_prompt_template import ZephyrChatPromptTemplate
from .word_spell import get_spell_check_runnable, HalfJsonOutputParser
from .llm import create_llm, create_zephyr_llm, create_mixtral_llm
from .parser import RoleFilterOutputParser
from .mixtral_instruct_template import MixtralInstructPromptTemplate


def get_zephyr_runnable():
    pass


def get_mixtral_runnable(verbose=False):
    prompt = MixtralInstructPromptTemplate.from_template("<s> {history} </s> [INST] {instruction} [/INST]")
    llm = create_mixtral_llm(verbose=verbose)
    return prompt | llm
