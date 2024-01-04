#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: parser.py
@time: 2024/1/2 17:41
@desc:
"""
import json
import re
from typing import List

from langchain_core.output_parsers import BaseOutputParser


class RoleFilterOutputParser(BaseOutputParser):
    def parse(self, text: str):
        pattern = ".?<\|\w+\|>[\n|:]+"
        # out = re.match(pattern, text, flags=0)
        out = re.sub(pattern, "", text, count=1)
        return out.strip()


if __name__ == '__main__':
    text = "\n<|assistant|>\n```json\n{}\n"
    p = RoleFilterOutputParser()
    out = p.parse(text)
