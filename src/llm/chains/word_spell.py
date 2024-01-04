#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: word_spell.py
@time: 2024/1/2 17:32
@desc:
"""
import json
from typing import List, Dict

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.base import T

from .llm import create_llm
from .zephyr_chat_prompt_template import ZephyrChatPromptTemplate
from .parser import RoleFilterOutputParser

_system_template = """You're an AI assistant to check spelling in English, provide accurate and precise corrections. 
User will give some English text, you need to check every word in the text, and return all spelling errors in a list of JSON format.
Return all spelling errors and nothing else.

<|user|>
i likes enlish
<|assistant|>
```json
{{"errors": [{{"origin":"likes", "fixed": "like"}}, {{"origin":"enlish", "fixed": "english"}}] }}
```

If there is no error, return empty json array.
```json
{{"errors": [] }}
```

"""


class HalfJsonOutputParser(BaseOutputParser):

    def parse(self, text: str) -> List:
        print(text)
        l = text.find('```json\n')
        rest_text = text[l + 8:]

        raw = json.loads(rest_text.strip())
        return self.filter_correct_word(raw)

    def filter_correct_word(self, raw: Dict) -> List:
        out = []
        if raw:
            for i in raw['errors']:
                o, f = i['origin'], i['fixed']
                if o != f and f != '':
                    out.append(i)
        return out


def get_chat_prompt():
    prompt = ZephyrChatPromptTemplate.from_messages(
        [
            ("system", _system_template),
            ("human", "check the below text spell:\n{sentence}"),
        ]
    )
    return prompt


def get_spell_check_runnable(llm=None):
    prompt = get_chat_prompt()
    if llm is None:
        llm = create_llm()
    return prompt | llm | RoleFilterOutputParser() | HalfJsonOutputParser()
