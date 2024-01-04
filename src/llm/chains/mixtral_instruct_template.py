#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: mixtral_instruct_template.py
@time: 2024/1/3 17:35
@desc:
"""

from typing import Sequence, Any, List

from langchain_core.messages import get_buffer_string, BaseMessage, HumanMessage, AIMessage, SystemMessage, \
    FunctionMessage, ToolMessage, ChatMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, PipelinePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence

from .llm import create_mixtral_llm




def _get_buffer_string(messages: Sequence[BaseMessage]) -> str:
    pre_messages = []
    for m in messages[:-1]:
        if isinstance(m, HumanMessage):
            txt = f"[INST] {m.content} [INST]"
        elif isinstance(m, AIMessage):
            txt = m.content
        # elif isinstance(m, SystemMessage):
        #     role = "<|system|>"
        # elif isinstance(m, FunctionMessage):
        #     role = "Function"
        # elif isinstance(m, ToolMessage):
        #     role = "Tool"
        # elif isinstance(m, ChatMessage):
        #     role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        pre_messages.append(txt)
    pre_msg = " ".join(pre_messages)

    if pre_msg:
        return f"{pre_msg} [INST] {messages[-1].content} [/INST]"
    else:
        return f"[INST] {messages[-1].content} [/INST]"


class MixtralInstructPromptValue(ChatPromptValue):
    def to_string(self) -> str:
        return _get_buffer_string(self.messages)


class MixtralInstructPromptTemplate(ChatPromptTemplate):
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        messages = self.format_messages(**kwargs)
        return MixtralInstructPromptValue(messages=messages)


if __name__ == '__main__':
    template = MixtralInstructPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{instruction}"),
    ])
    o = template.invoke(input={"instruction": "你好"})
    print(o.to_string())
