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

from src import yi_chat, mixtral_chat, zephyr_chat


class SelectedModel(StrEnum):
    ZEPHYR = "zephyr-7b-beta.Q8_0"
    MIXTRAL = "mixtral-8x7b-instruct-v0.1.Q4_0"
    YI = "nous-hermes-2-yi-34b.Q4_0"


def chat(select_model, system_prompt: str, history: List[Tuple[str, str]]):
    if select_model == SelectedModel.ZEPHYR:
        yield from zephyr_chat(system_prompt, history[:-1], history[-1][0])
    elif select_model == SelectedModel.MIXTRAL:
        yield from mixtral_chat(history[:-1], history[-1][0])
    elif select_model == SelectedModel.YI:
        yield from yi_chat(system_prompt, history[:-1], history[-1][0])
    else:
        raise NotImplementedError("Unknown model")


def clean_last_history(history):
    if history:
        return history[:-1], history[-1][0]
    else:
        return history, ""


CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

with (gr.Blocks(css=CSS) as demo):
    with gr.Row():
        selected = gr.Dropdown([
            ("ZEPHYR", SelectedModel.ZEPHYR),
            ("MIXTRAL", "mixtral-8x7b-instruct-v0.1.Q4_0"),
            ("YI", "nous-hermes-2-yi-34b.Q4_0")
        ], label="Model", info="current used model", scale=7)
        check = gr.Button("check", scale=1)
        check.click(lambda x: print(x, type(x)), selected)

    system_prompt = gr.Textbox(label="System Prompt", visible=selected != SelectedModel.MIXTRAL)


    def change_model(model):
        if model == SelectedModel.MIXTRAL:
            system_prompt.visible = False
        else:
            system_prompt.visible = True
        system_prompt.render()


    selected.change(change_model, selected)
    chatbot = gr.Chatbot(elem_id="chatbot")
    with gr.Row():
        msg = gr.Textbox(placeholder="Type a message...", scale=7)
        sub = gr.Button("Submit", variant="primary", scale=1)
    with gr.Row():
        clear = gr.ClearButton([msg, chatbot], value="🗑️  Clear", variant="secondary")
        undo = gr.Button("↩️ Undo", variant="secondary")


    def user(user_message, history):
        return "", history + [[user_message, None]]


    def bot(selected, system_prompt, chat_history):
        print(selected, type(selected))
        chat_history[-1][1] = ""
        for character in chat(selected, system_prompt, chat_history):
            chat_history[-1][1] += character
            yield chat_history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [selected, system_prompt, chatbot], chatbot)
    sub.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [selected, system_prompt, chatbot], chatbot)
    undo.click(clean_last_history, chatbot, [chatbot, msg])

if __name__ == "__main__":
    print("start use:", selected)
    demo.queue().launch()
    gr.ChatInterface()
