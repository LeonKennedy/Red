#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: demo2.py
@time: 2023/12/22 16:02
@desc:
"""

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms.llamacpp import LlamaCpp

from src.llm import init_llm_queue, create_prompt

st.title('🦜🔗 Quickstart App')


@st.cache_resource
def create_llm():
    return LlamaCpp(
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


@st.cache_resource
def create_runnable():
    prompt = create_prompt()
    llm = create_llm()
    runnable = prompt | llm
    return runnable


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, k=5)

for message in st.session_state.memory.buffer:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    memory = st.session_state.memory

    runnable = create_runnable()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chuck in runnable.stream({"question": prompt, "history": memory.buffer}):
            full_response += chuck
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    memory.save_context({"input": prompt}, {"output": full_response})
