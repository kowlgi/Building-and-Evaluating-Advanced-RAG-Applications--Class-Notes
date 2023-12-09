# Building-and-Evaluating-Advanced-RAG-Applications--Class-Notes

https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag

## Background info

Videos to watch

- A Survey of Techniques for Maximizing LLM Performance [video](https://www.youtube.com/watch?v=ahnGLM-RC1Y&t=3s)
- The New Stack and Ops for AI [video](https://www.youtube.com/watch?v=XGJNo8TpuVA)
- New Products - A Deep Dive [video](https://www.youtube.com/watch?v=pq34V_V5j18)
- Background on RAG [video](https://www.youtube.com/watch?v=Q-uEhJMu3ak)

## Introduction

What is Retrieval Augmented Generation (RAG)?

- a key method to get LLMs to answer questions over a user's own data
- helps give an LLM highly relevant context to generate an answer

When building a LLM app, it's important to have an evaluation system to iterate and improve the system, during

- initial development
- post-deployment maintenance

Two RAG methods help deliver significantly better context to LLMs by dynamically retrieving more coherent chunks of text, than simpler methods:

1. Sentence window retrieval
1. Auto-merging retrieval

What is Sentence window retrieval?

- technique to give LLM better context by retrieving not just the relevant sentence but the window of sentences that occur before and after it in the document

What is auto-merging retrieval?

- technique to organize document into tree-like structure. When a child node is considered relevant to a user's question, then the entire text of the parent node is provided as context for the LLM

To evaluate RAG-based systems, we use a triad of metrics for the three main steps of a RAG's execution:

1. Context Relevance:

- How relevant are the retrieved chunks of text to the user's question. Helps to identify and debug issues with how the system is retrieving context for the LLM

1. Groundedness
1. Answer Relevance

these metrics helps you identify which parts of your system are or are not working well.

The above evaluation metrics have similarities with Error Analysis in ML.

## Overview of retrieval techniques and evaluation metrics

Basic RAG pipeline components

- ingestion: comprises the following sequence of operations:

  1. load in documents
  1. each document is split into chunks
  1. for each chunk we generate an embedding, using an embedding model
  1. chunk + embedding are offloaded to an index, such as a vector db

- retrieval: is performed against that index
  1. launch a user query against the index
  1. select top K most similar chunks to the user query
- synthesis
  1. take the relevant chunks, combine with user query and put it into prompt window of the LLM
  1. generate the final response
