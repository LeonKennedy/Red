#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: demo.py
@time: 2023/12/15 15:29
@desc:
"""
import os
import asyncio
import chainlit as cl
from chainlit.playground import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.llm import init_llm_queue, create_prompt
from src.tts import inference_to_file, split_inference_to_file
from src.sadtalk import get_avatar

_Q = init_llm_queue()


@cl.on_chat_start
def set_user_conversation_chain():
    print(cl.user_session.get("id"), "create history")
    # add_llm_provider(
    #     LangchainGenericProvider(id=chain.llm._llm_type, name="Llama-cpp", llm=chain.llm, is_chat=False)
    # )
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    cl.user_session.set("memory", memory)


async def protect(question: str, history, llm) -> cl.Message:
    prompt = create_prompt()
    msg = cl.Message(content="")
    runnable = prompt | llm
    for chunk in await cl.make_async(runnable.stream)(
            {"question": question, "history": history},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    _Q.put_nowait(llm)
    return msg


@cl.on_message
async def main(message: cl.Message):
    memory: ConversationBufferMemory = cl.user_session.get("memory")
    llm = await _Q.get()
    try:
        msg = await protect(message.content, memory.buffer, llm)
    except Exception as e:
        _Q.put_nowait(llm)
        raise e
    # avatar = get_avatar()
    memory.save_context({"input": message.content}, {"output": msg.content})
    try:
        for tmp_wav in split_inference_to_file(msg.content):
            cla = cl.Audio(name=os.path.basename(tmp_wav.name), path=tmp_wav.name, display="inline")
            await cla.send(for_id=msg.id)

            # video_path = avatar.run(tmp_wav.name)
            # clv = cl.Video(name=os.path.basename(video_path), path=video_path, display="inline")
            # await clv.send(for_id=msg.id)
            tmp_wav.close()
    except Exception as e:
        print(f"[ERROR] {e}")
        await msg.send()
