# Building-and-Evaluating-Advanced-RAG-Applications--Class-Notes

Video tutorial by Andrej Karpathy: https://www.youtube.com/watch?v=zjkBMFhNj_g

A Large Language Model (LLM) is just two files. For, example Meta's public llama-2-70b model has these two files:

- parameters (140GB): these are the weights of this neural network. Every parameter is stored as 2 bytes (float16)
- run.c (~500 lines of C code)

You can think of the parameters as a zip file of the internet.

LLM is essentially a next-word predictor. Given a context, what's the next word with the highest probability. A good predictor is thus a good compressor, as you need to store very little to reconstruct the larger body of text.
