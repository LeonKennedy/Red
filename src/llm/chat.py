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
from typing import Sequence, Any, List

from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.messages import get_buffer_string, BaseMessage, HumanMessage, AIMessage, SystemMessage, \
    FunctionMessage, ToolMessage, ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, PipelinePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence


# _template = """<|system|>
# You are a friendly chatbot who always responds in the style of a primary school teacher.</s>
#
# {history}
#
# <|user|>
# {input}</s>
#
# <|assistant|>
# """


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
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """
        Format prompt. Should return a PromptValue.
        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            PromptValue.
        """
        messages = self.format_messages(**kwargs)
        return ZephyrChatPromptValue(messages=messages)


_llm = None


def create_prompt():
    prompt = ZephyrChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
            AIMessage(content="")
        ])

    # full_prompt = PromptTemplate.from_template("""{input}\n<|assistant|>\n""")
    # pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=[{"input": prompt}])
    return prompt


def _get_llm():
    global _llm
    if _llm is None:
        _llm = LlamaCpp(
            temperature=0.75,
            n_gpu_layers=20,
            n_ctx=4096,
            n_batch=512,
            top_p=0.95,
            top_k=50,
            stop=["</s>"],
            model_path="/mnt/data/ggml/llama.cpp/models/7B/zephyr-7b-beta.Q8_0.gguf",
            verbose=True
        )
    return _llm


def create_conversation_chain() -> RunnableSequence:
    prompt = create_prompt()
    runnable = prompt | _get_llm()

    # runnable2 = LLMChain(llm=_llm, prompt=prompt, verbose=True, memory=memory)
    # prompt = PromptTemplate(template=_template, input_variables=["history", "input"])
    # conversation = ConversationChain(
    #     prompt=prompt, llm=_llm,
    #     memory=ZephyrMemory(human_prefix="<|user|>", ai_prefix="<|assistant|>", k=5)
    # )
    return runnable
