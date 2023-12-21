#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: demo.py
@time: 2023/12/14 14:32
@desc:
"""

import streamlit as st
from st_audiorec import st_audiorec

st.title("Audio Recorder")
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
