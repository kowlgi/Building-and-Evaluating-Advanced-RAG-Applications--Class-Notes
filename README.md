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

Some steps in creating a RAG system:

- merge all documents into a single one, this is necessary for retrieval methods
- index documents in a vector store
- instantiate a ServiceContext object, which contains:
  - llm we're going to use
  - embedding model

### Code walkthrough

- instantiate llm and service context with embedding model from huggingface
  `llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)`
  `service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")`

In instantiating the llm, we set a temperature parameters. towards 0 -> more stable answers. towards 1 -> more newness.

`VectorStoreIndex.from_documents(documents)`
This line of code covers 3 steps in the ingestion phase:

1. chunking
1. embedding
1. indexing

`query_engine = index.as_query_engine()`
This allows us to send user queries that do retrieval and synthesis against this data.

`response = query_engine.query("user's prompt")`
This executes a user query

### RAG triad

3 metrics for three stages of the system:

- Query -> context: Context relevance
  - is the retrieved context relevant to the query?
- Context -> Response: Groundedness
  - is the response supported by the context?
- Response -> Query: Answer relevance
  - is the response relevant to the query?

#### How to set up an evaluation system

1. come up with evaluation questions, to test the application
1. run the questions using an evaluation recorder like TruEra, to generate context relevance, groundedness and answer relevance scores

E.g.

#### Setting up Sentence Window Retrieval

`sentence_index = build_sentence_window_index(document, llm, embed_model="local:BAAI/bge-small-en-v1.5")`

`sentence_window_engine = get_sentence_window_query-engine(sentence_window_index)`

`window_response = sentence_window_engine.query("how do i get started on a personal project in AI")`

#### Setting up Auto-merging Retrieval

In this technique, we construct a hierarchy of larger parent nodes, with smaller child nodes that reference the parent node.

If a parent node has a majority of its children nodes retrieved, then the children nodes are replaced with the parent node i.e. hierarchically merged.

`automerging_index = build_automerging_index(documents, llm, embed_model="local:BAAI/bge-small-v1.5)`

`automerging_query_engine = getautomerging_query_engine(automerging_index)`

`response = automerging_query_engine.query('how to build an AI career')`

## Deep dive into RAG evaluation

RAG triad:

- context relevance
- groundedness
- answer relevance

`tru.reset_database()` before evaluation, we reset the db with recordings of evaluations

To build the index for retrieval, we create a single large document rather than multiple documents.
