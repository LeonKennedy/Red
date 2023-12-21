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
import chainlit as cl
from chainlit.playground import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.llm import create_conversation_chain
from src.tts import inference_to_file, split_inference_to_file
from src.sadtalk import get_avatar


@cl.on_chat_start
def set_user_conversation_chain():
    print(cl.user_session.get("id"), "create history")

    # add_llm_provider(
    #     LangchainGenericProvider(id=chain.llm._llm_type, name="Llama-cpp", llm=chain.llm, is_chat=False)
    # )
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    cl.user_session.set("memory", memory)


@cl.on_message
async def main(message: cl.Message):
    memory: ConversationBufferMemory = cl.user_session.get("memory")
    runnable = create_conversation_chain()

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
            {"question": message.content, "history": memory.buffer},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    for tmp_wav in split_inference_to_file(msg.content):
        cla = cl.Audio(name=os.path.basename(tmp_wav.name), path=tmp_wav.name, display="inline")
        await cla.send(for_id=msg.id)
        tmp_wav.close()

    # tmp_wav = inference_to_file(msg.content)
    # msg.elements = [cl.Audio(name=os.path.basename(tmp_wav.name), path=tmp_wav.name, display="inline")]
    # await msg.send()
    # msg.elements.append(cl.Audio(name="dd", path=tmp_wav.name, display="inline"))

    memory.save_context({"input": message.content}, {"output": msg.content})

    # history.append(message)
    #
    # tmp_wav = inference_to_file(message["content"])
    # video_path = out_main(tmp_wav.name)
    # elements = [
    #     cl.Audio(name=os.path.basename(tmp_wav.name), path=tmp_wav.name, display="inline"),
    #     cl.Video(name=os.path.basename(video_path), path=video_path, display="inline")
    # ]
    # await cl.Message(content=message["content"], elements=elements).send()
    # print(cl.user_session.get("history"))
    # tmp_wav.close()
