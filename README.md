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

# Reference & model
1. [TheBloke/Llama-2-7B-Chat-GGU](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF#provided-files)
2. [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
3. [TheBloke/zephyr-7B-beta-AWQ](https://huggingface.co/TheBloke/zephyr-7B-beta-AWQ)
4. [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
5. [openai/wishper](https://github.com/openai/whisper)
6. [llama.cpp](https://github.com/ggerganov/llama.cpp)