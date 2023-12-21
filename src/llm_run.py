#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: llm_run.py
@time: 2023/12/18 11:58
@desc:
"""

from llm import llama_cpp_run


def run():
    while 1:
        msg = input()
        if msg == "q":
            break
        else:
            print(llama_cpp_run(msg))


if __name__ == '__main__':
    run()
