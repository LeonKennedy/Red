#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: develop.py
@time: 2023/12/18 10:32
@desc:
"""
import asyncio
from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate
import langchain as cl

from src.llm import create_conversation_chain, create_prompt
from src.tts import inference_to_file, split_inference_to_file
from src.sadtalk import get_avatar


def conversation_prompt():
    prompt = create_prompt()
    a = prompt.invoke(input={"history": [], "question": "hello, teacher"})
    print(type(prompt))


async def conversation_chain(msg: str):
    runnable = create_conversation_chain()
    async for chunk in runnable.astream(input={"history": [], "question": msg}):
        print(chunk)


def tts(content: str):
    # return inference_to_file(content)
    for chunk in split_inference_to_file(content):
        print(chunk)


def voice_to_video(path: str) -> str:
    avatar = get_avatar()
    video_path = avatar.run(path)
    print(video_path)


def run(msg):
    content = llm(msg)
    tmp_wav = tts(content)
    print("create voice:", tmp_wav.name)


if __name__ == '__main__':
    # run("hello, i like coffee")
    voice_to_video("/mnt/d4t/workspace/red/results/tts/tmp2wq385nb.wav")
    # conversation_chain("hello, teacher")
    # tts("Hello, student! How may I assist you today? Please let me know your historical inquiry and I will do my best to provide an insightful response. Remember to provide specific details and context so that I can better understand the question. Thank you for choosing me as your history tutor!")
    # asyncio.run(conversation_chain("hello, teacher"))
