#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: gradio_ui_chat_langchain.py
@time: 2024/1/3 10:33
@desc:
"""
import json
from functools import lru_cache
from typing import List

from src.llm.chains import ZephyrChatPromptTemplate, RoleFilterOutputParser, HalfJsonOutputParser
from src.llm.chains import get_yi_runnable, get_mixtral_runnable, get_zephyr_runnable

DEFAULT_SYSTEM_TEMPLATE = """You're an AI assistant to check spelling in English, provide accurate and precise corrections. 
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

DEFAULT_USER_INPUT = "check the below text spell:\nKeyboard interuption in main thread"


def get_chat_prompt(system_template):
    prompt = ZephyrChatPromptTemplate.from_messages(
        [
            ("system", system_template),
        ]
    )
    return prompt


@lru_cache
def _get_llm():
    return create_llm(temperature=0.01, stop="```\n")


def chat(system_prompt: str, msg: str) -> str:
    prompt = get_chat_prompt(system_prompt)
    prompt.append(("human", msg))
    llm = _get_llm()
    runnable = prompt | llm | RoleFilterOutputParser() | HalfJsonOutputParser()
    out = runnable.invoke({})
    return list_to_json(out)


def list_to_json(list: List) -> str:
    o = json.dumps(list, ensure_ascii=False)
    return f"```json\n{o}\n```\n"


### above deprecated
"""
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>
"""


def zephyr_chat(system_prompt: str, history: List, msg: str) -> str:
    format_history = ""
    for u, a in history:
        tmp = f"<|user|>\n{u}</s>\n<|assistant|>\n{a}\n"
        format_history += tmp

    runnable = get_zephyr_runnable()
    yield from runnable.stream(input={"system_prompt": system_prompt, "history": format_history, "msg": msg})


''' MIXTRAL
<s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
'''


def mixtral_chat(history: List, msg: str) -> str:
    format_history = ""
    for u, a in history:
        tmp = f"[INST] {u} [/INST] {a}"
        format_history += tmp

    runnable = get_mixtral_runnable()
    yield from runnable.stream(input={"history": format_history, "msg": msg})


"""
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""


def yi_chat(system_prompt: str, history: List, msg: str) -> str:
    format_history = ""
    for u, a in history:
        tmp = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        format_history += tmp

    runnable = get_yi_runnable()
    yield from runnable.stream(input={"system_prompt": system_prompt, "history": format_history, "msg": msg})


if __name__ == '__main__':
    for i in yi_chat("hello, teacher! i am a student for learning english.", [],
                     "hello, teacher! i am a student for learning english."):
        print(i)
