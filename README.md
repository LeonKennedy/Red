# Red
asr + llama + tts


# install
1. CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
2. pip install -r requirements.txt
3. .txt

# chainlit
github: https://github.com/Chainlit/chainlit

doc: https://docs.chainlit.io/get-started/overview


# talk-lamal
./talk-llama -mw ./models/ggml-base.en.bin -ml ../llama.cpp/models/7B/zephyr-7b-beta-pl.Q8_0.gguf -p "Georgi" -t 8

# SSL
openssl req -newkey rsa:2048 -nodes -keyout rsa_private.key -x509 -days 799 -out cert.crt

# model
## zephyr-7b-beta -> Mistral-7B-v0.1(4096)
zephyr-7b-beta -> Mistral-7B-v0.1(4096)


1. [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
2. [TheBloke/zephyr-7B-beta-AWQ](https://huggingface.co/TheBloke/zephyr-7B-beta-AWQ)

## Mixtral-8x7B-Instruct-v0.1
1. [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
2. [TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)


## Yi
1. [TheBloke/Nous-Hermes-2-Yi-34B-GGUF](https://huggingface.co/TheBloke/Nous-Hermes-2-Yi-34B-GGUF)

# Reference
1. [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF#provided-files)
2. [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
3. [openai/wishper](https://github.com/openai/whisper)
4. [llama.cpp](https://github.com/ggerganov/llama.cpp)
5. [Replicate](https://replicate.com/)