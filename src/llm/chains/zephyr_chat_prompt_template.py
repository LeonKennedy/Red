#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: ZephyrChatPromptTemplate.py
@time: 2024/1/2 17:35
@desc:
"""
import asyncio
from typing import Sequence, Any, List

from langchain_core.messages import get_buffer_string, BaseMessage, HumanMessage, AIMessage, SystemMessage, \
    FunctionMessage, ToolMessage, ChatMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, PipelinePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence


def _get_buffer_string(
        messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "<|system|>"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ToolMessage):
            role = "Tool"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"{role}\n{m.content}</s>" if m.content else f"{role}\n"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)


class ZephyrChatPromptValue(ChatPromptValue):
    def to_string(self) -> str:
        """Return prompt as string."""
        return _get_buffer_string(self.messages, human_prefix="<|user|>", ai_prefix="<|assistant|>")


class ZephyrChatPromptTemplate(ChatPromptTemplate):
    def format_prompt(self, **kwargs: Any) -> ZephyrChatPromptValue:
        """
        Format prompt. Should return a PromptValue.
        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            PromptValue.
        """
        messages = self.format_messages(**kwargs)
        return ZephyrChatPromptValue(messages=messages)
