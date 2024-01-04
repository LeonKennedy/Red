#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: demo3.py
@time: 2024/1/3 10:00
@desc:
"""

import gradio as gr
import random
import time

from src.gradio_ui_chat_langchain import chat, DEFAULT_SYSTEM_TEMPLATE, DEFAULT_USER_INPUT, HalfJsonOutputParser

with gr.Blocks() as demo:
    system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_TEMPLATE, label="System Prompt")
    chatbot = gr.Chatbot(height=200, bubble_full_width=False)
    msg = gr.Textbox(value=DEFAULT_USER_INPUT)
    with gr.Row():
        clear = gr.ClearButton([msg, chatbot])
        sub = gr.Button("Submit")


    def respond(system_prompt, message, chat_history):
        bot_message = chat(system_prompt, message)
        chat_history.append((message, bot_message))
        return "", chat_history


    msg.submit(respond, [system_prompt, msg, chatbot], [msg, chatbot])
    sub.click(respond, [system_prompt, msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.queue().launch()
