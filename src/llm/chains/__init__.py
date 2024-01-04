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
from .zephyr_chat_prompt_template import ZephyrChatPromptTemplate
from .word_spell import get_spell_check_runnable, HalfJsonOutputParser
from .llm import create_zephyr_llm, create_mixtral_llm, create_yi_llm, get_llm_by_name
from .parser import RoleFilterOutputParser
from .prompt import create_mixtral_prompt, create_yi_prompt, create_zephyr_prompt


def get_zephyr_runnable(verbose=False):
    prompt = create_zephyr_prompt()
    llm = get_llm_by_name("zephyr", verbose=verbose)
    return prompt | llm


def get_mixtral_runnable(verbose=False):
    prompt = create_mixtral_prompt()
    llm = get_llm_by_name("mixtral", verbose=verbose)
    return prompt | llm


def get_yi_runnable(verbose=True):
    prompt = create_yi_prompt()
    llm = get_llm_by_name("yi", verbose=verbose)
    return prompt | llm
