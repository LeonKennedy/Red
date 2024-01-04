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
from src.llm.chains import get_spell_check_runnable
from src.llm.chains import get_mixtral_runnable, get_yi_runnable


def conversation_prompt():
    prompt = create_prompt()
    a = prompt.invoke(input={"history": [], "question": "hello, teacher"})
    print(type(prompt))


async def conversation_chain(msg: str):
    runnable = await create_conversation_chain()
    for chunk in runnable.stream(input={"history": [], "question": msg}):
        print(chunk)


def tts(content: str):
    # return inference_to_file(content)
    for chunk in split_inference_to_file(content):
        print(chunk)


def voice_to_video(path: str) -> str:
    avatar = get_avatar()
    video_path = avatar.run(path)
    print(video_path)


def word_spell_checker():
    examples = [
        "hello, teacher! i am a student for learning english.",
        "hello, teaher! i am a student for learing english."
    ]
    llm = create_llm(temperature=0.01, stop="```\n", verbose=True)
    runnable = get_spell_check_runnable(llm)
    for i in examples:
        print("[Origin]" + i)
        out = runnable.invoke({"sentence": i})
        print("[Result]", out)


if __name__ == '__main__':
    # run("hello, i like coffee")
    # voice_to_video("/mnt/d4t/workspace/red/results/tts/tmp2wq385nb.wav")
    # conversation_prompt()
    # tts("Hello, student! How may I assist you today? Please let me know your historical inquiry and I will do my best to provide an insightful response. Remember to provide specific details and context so that I can better understand the question. Thank you for choosing me as your history tutor!")
    # asyncio.run(conversation_chain("hello, teacher"))
    # word_spell_checker()

    runnable = get_yi_runnable(verbose=True)
    out = runnable.invoke({"msg": "hello, teacher? what is best food for dinner?", "history": []})
    print(1)
