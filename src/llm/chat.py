#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: chat.py
@time: 2023/12/20 14:11
@desc:
"""
import asyncio
from typing import Sequence, Any, List

from langchain_core.messages import get_buffer_string, BaseMessage, HumanMessage, AIMessage, SystemMessage, \
    FunctionMessage, ToolMessage, ChatMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, PipelinePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from .chains import ZephyrChatPromptTemplate, create_llm


def create_prompt():
    prompt = ZephyrChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a friendly chatbot who always responds in the style of a primary school teacher."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
            AIMessage(content="")
        ])

    # full_prompt = PromptTemplate.from_template("""{input}\n<|assistant|>\n""")
    # pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=[{"input": prompt}])
    return prompt


_llm_Q = asyncio.Queue(maxsize=3)


def init_llm_queue() -> asyncio.Queue:
    _llm_Q.put_nowait(create_llm())
    print("[---]\n\n llm in queue \nTHis message only see once \n\n[---]")
    return _llm_Q


async def create_conversation_chain() -> RunnableSequence:
    prompt = create_prompt()
    llm = await _llm_Q.get()
    runnable = prompt | llm

    # runnable2 = LLMChain(llm=_llm, prompt=prompt, verbose=True, memory=memory)
    # prompt = PromptTemplate(template=_template, input_variables=["history", "input"])
    # conversation = ConversationChain(
    #     prompt=prompt, llm=_llm,
    #     memory=ZephyrMemory(human_prefix="<|user|>", ai_prefix="<|assistant|>", k=5)
    # )
    return runnable
