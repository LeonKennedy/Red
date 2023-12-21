#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: coqui.py
@time: 2023/12/16 10:08
@desc:
"""

from TTS.api import TTS
from TTS.tts.utils.text.cleaners import english_cleaners
import tempfile

tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
_tts = None


def _get_tts():
    global _tts
    if _tts is None:
        _tts = TTS(model_name=tts_model, progress_bar=False)
    return _tts


# def inference(text: str):
#     return tts.tts(text, speaker=tts.speakers[0])


def inference_to_file(text: str) -> tempfile.NamedTemporaryFile:
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", dir="results/tts", delete=False)
    tts = _get_tts()
    tts.tts_to_file(text=english_cleaners(text), file_path=tmp_wav.name)
    return tmp_wav


def split_inference_to_file(text: str):
    tts = _get_tts()
    sens = tts.synthesizer.split_into_sentences(english_cleaners(text))
    for sen in sens:
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", dir="results/tts", delete=True)
        wav = tts.tts(
            text=sen,
            speaker=None,
            language=None,
            speaker_wav=None,
            split_sentences=False,
        )
        tts.synthesizer.save_wav(wav=wav, path=tmp_wav.name, pipe_out=None)
        yield tmp_wav


if __name__ == '__main__':
    inference_to_file("hello world")
