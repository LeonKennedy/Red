#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: demo_multi_model.py
@time: 2024/1/3 17:27
@desc:
"""
from typing import List, Tuple

import gradio as gr
from enum import StrEnum

from src.llm.chains import get_mixtral_runnable
from src import yi_chat

_choices = [
    ("zephyr-7b-beta.Q8_0", "7B/zephyr-7b-beta.Q8_0.gguf"),
    ("mixtral-8x7b-instruct-v0.1.Q4_0", "mixtral-8x7b-instruct-v0.1.Q4_0.gguf"),
]


class SelectedModel(StrEnum):
    ZEPHYR = "zephyr-7b-beta.Q8_0"
    MIXTRAL = "mixtral-8x7b-instruct-v0.1.Q4_0"
    YI = "nous-hermes-2-yi-34b.Q4_0.gguf"


_model = SelectedModel.YI


def format_history(history: List[Tuple[str, str]]) -> str:
    out = []
    for u, a in history:
        tmp = f"[INST] {u} [/INST] {a}"
        out.append(tmp)
    return " ".join(out)


def chat(system_prompt: str, history: List[Tuple[str, str]]):
    if _model == SelectedModel.ZEPHYR:
        pass
    elif _model == SelectedModel.MIXTRAL:
        runnable = get_mixtral_runnable()
        yield from runnable.stream({"instruction": history[-1][0], "history": format_history(history[:-1])})
    elif _model == SelectedModel.YI:
        yield from yi_chat(system_prompt, history[:-1], history[-1][0])
    else:
        raise NotImplementedError("Unknown model")


with gr.Blocks() as demo:
    system_prompt = gr.Textbox(label="System Prompt", visible=_model != SelectedModel.MIXTRAL)
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(placeholder="Type a message...", scale=7)
        sub = gr.Button("Submit", variant="primary", scale=1)
    with gr.Row():
        clear = gr.ClearButton([msg, chatbot], value="🗑️  Clear", variant="secondary")


    def user(user_message, history):
        return "", history + [[user_message, None]]


    def bot(system_prompt, chat_history):
        chat_history[-1][1] = ""
        for character in chat(system_prompt, chat_history):
            chat_history[-1][1] += character
            yield chat_history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [system_prompt, chatbot], chatbot)
    sub.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [system_prompt, chatbot], chatbot)

if __name__ == "__main__":
    print("start use:", _model)
    demo.queue().launch()
