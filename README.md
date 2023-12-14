# Building-and-Evaluating-Advanced-RAG-Applications--Class-Notes

https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag

## Background info

Videos to watch

- A Survey of Techniques for Maximizing LLM Performance [video](https://www.youtube.com/watch?v=ahnGLM-RC1Y&t=3s)
- The New Stack and Ops for AI [video](https://www.youtube.com/watch?v=XGJNo8TpuVA)
- New Products - A Deep Dive [video](https://www.youtube.com/watch?v=pq34V_V5j18)
- Background on RAG [video](https://www.youtube.com/watch?v=Q-uEhJMu3ak)

## Instructors

Jerry Liu - Llama Index
Anupam Datta - TruEra

## Introduction

What is Retrieval Augmented Generation (RAG)?

- a key method to get LLMs to answer questions over a user's own data
- helps give an LLM highly relevant context to generate an answer

When building a LLM app, it's important to have an evaluation system to iterate and improve the system, during

- initial development
- post-deployment maintenance

Two RAG methods help deliver significantly better context to LLMs by dynamically retrieving more coherent chunks of text, than simpler methods:

1. Sentence window retrieval
   - technique to give LLM better context by retrieving not just the relevant sentence but the window of sentences that occur before and after it in the document
1. Auto-merging retrieval
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

```python
  llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
  service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
```

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

A metric for each stage of the system:

1. Query -> context: Context relevance

   - is the retrieved context relevant to the query?

1. Context -> Response: Groundedness

   - is the response supported by the context?
   - it's called groundedness, because we're asking if the response is grounded in the context.

1. Response -> Query: Answer relevance

   - is the response relevant to the query?

#### How to set up an evaluation system

1. come up with evaluation questions, to test the application
1. run the questions using an evaluation recorder like TruEra, to generate context relevance, groundedness and answer relevance scores

#### Setting up Sentence Window Retrieval

```python
sentence_index = build_sentence_window_index(document, llm, embed_model="local:BAAI/bge-small-en-v1.5")

sentence_window_engine = get_sentence_window_query-engine(sentence_window_index)

window_response = sentence_window_engine.query("how do i get started on a personal project in AI")
```

#### Setting up Auto-merging Retrieval

In this technique, we construct a hierarchy of larger parent nodes, with smaller child nodes that reference the parent node.

If a parent node has a majority of its children nodes retrieved, then the children nodes are replaced with the parent node i.e. hierarchically merged.

```python
automerging_index = build_automerging_index(documents, llm, embed_model="local:BAAI/bge-small-v1.5")

automerging_query_engine = getautomerging_query_engine(automerging_index)

response = automerging_query_engine.query('how to build an AI career')
```

## Deep dive into RAG evaluation

RAG triad:

- context relevance
- groundedness
- answer relevance

`tru.reset_database()` before evaluation, we reset the db with recordings of evaluations

To build the index for retrieval, we create a single large document rather than multiple documents.

### Feedback functions

A feedback function provides a score after reviewing an LLM app's inputs, outputs and intermediate results.

`provider = fOpenAI()`
using openAI to evaluate retrieval system

We don't have to use an LLM to evaluate, we can use Burt models and other ways.

#### Answer relevance

We're checking: is the final response relevant to the query by the user?

Two evaluations for answer relevance:

- answer relevance score (0-1)
- supporting evidence for the score, such a chain-of-thought (cot) reasoning

The code structure:

- feedback function method
- pointer to user query
- pointer to app output

```python
from trulens_eval import OpenAI as fOpenAI
from trulens_eval import Feedback
provider = fOpenAI()
f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()
```

#### Context relevance

We're checking: how good is the retrieval?

The evaluation system looks at each piece of retrieved context and assesses how relevant that piece of context is to the question.

Similar in code structure to Answer relevance with some differences:

- as input, we also provide a pointer to the intermediate results i.e. context provided
- output aggregate and average scores across pieces of retrieved context

```python
f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)
```

#### Groundedness

We're checking: is the response supported by the context?

Similar in code structure to Context relevance.

```python
f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)
```

### Evaluate and Iterate

Often failure modes arise because context is too small. When we increase context, up to a certain point, we see improvements in context relevance.

When context relevance goes up, often we see improvements in groundedness as well, because the LLM in the completion step has enough relevance to complete the summary.

When the LLM does not have sufficient context, it tries to leverage its own internal knowledge from the pre-training dataset to try and fill those gaps, which results in a loss of groundedness.

Try different window sizes to see what window size results in the best evaluation metrics.

If the window size is too small:

- there may not be enough context.

If the window size is too big:

- irrelevant context can creep into the final response, which results in no so great scores in context relevance and groundedness.

#### Implementations of feedback functions

![Comparison of feedback functions](https://github.com/kowlgi/Building-and-Evaluating-Advanced-RAG-Applications--Class-Notes/blob/main/assets/img1.png "Comparison of feedback functions")

Difference between Human evals and Ground Truth evals:

- people in human evals may not be as much of an expert on the topic, as someone doing ground truth eval

Different people doing human eval match about 80%. LLM eval matches human eval by about 80-85%. This suggests LLM eval is quite comparable to human eval.

![Other types of evaluation functions](https://github.com/kowlgi/Building-and-Evaluating-Advanced-RAG-Applications--Class-Notes/blob/main/assets/img2.png "More types of evaluation")

## Deep dive into Sentence Window Retrieval

This method does two things:

- retrieve based on smaller sentences to better match relevant context
- synthesize based on extended context window around sentence

The standard RAG uses the same text trunk for both embedding and synthesis. The issue with this approach:

- embedding-based retrieval works well with smaller text chunks
- whereas the LLM needs more context and bigger chunks to synthesize a good answer

Sentence-window retrieval decouples embedding from retrieval.

- embeds smaller chunks or sentences and stores them in a vector database. Also, add context of sentences that occur before and after to each chunk
- during retrieval, we retrieve sentences that are most relevant to the question with a similarity search. Then replace the sentence with the full surrounding context.

TIP: When you have multiple documents, merge them into a single one, to improve text splitting accuracy when used with advanced retrieval techniques.

### How to set up

#### 1. Build the index

```python
# 1. instantiate a sentence window node parser
# window_size is the number of sentences on each side of a sentence to capture.
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# 2. set up an llm
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# 3. create the service context
from llama_index import ServiceContext
sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    # embed_model="local:BAAI/bge-large-en-v1.5"
    node_parser=node_parser,
)

# 4. set up a vector store index with the source document
from llama_index import VectorStoreIndex
sentence_index = VectorStoreIndex.from_documents(
    [document], service_context=sentence_context
)

```

##### Instantiate a sentence window node parser

The sentence window node parser is an object that does two things:

- splits document into single sentences, which are considered nodes
- augments each chunk with a surrounding context around that sentence

##### Create service context

The service context is a wrapper object for all the things needed for indexing:

1. the llm
1. the embedding model
   - huggingface bge-small-en-v1.5 is compact, small and fast for its size
1. node parser

##### Create vector store index

Creation of the vector store index comprises the following:

1. splitting document into sentences
1. augmenting each sentence with context
1. embedding into vector store

Save the index to disk, so you can load it later without rebuilding it.

```python
sentence_index.storage_context.persist(persist_dir="./sentence_index")
```

##### Putting it together

Put all of the above steps together into a single function `build_sentence_window_index`:

- inputs:
  - documents: object
  - llm: object
  - embed_model: string
  - sentence_window_size: number # The number of sentences on each side of a sentence to capture.
  - saved_index_dir: string
- output:
  - sentence_index: object

#### 2. Set up the query engine

```python
# 1. Define a metadata replacement post-processor.
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor

postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)

# 2. Add the sentence transformer re-rank model.
from llama_index.indices.postprocessor import SentenceTransformerRerank

# BAAI/bge-reranker-base
# link: https://huggingface.co/BAAI/bge-reranker-base
rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)
```

##### Define a metadata replacement post-processor

This takes a value stored in the metadata and replaces the node text with that value.

- remember, we stored a value called ['window'](#1-build-the-index) in the metadata, which is the window surrounding the sentence.
- we do this post-processing step after retrieving a node from the index and before sending the value to the LLM.

##### Define a sentence transformer re-rank model

This step does the following:

- reduces the LLM token usage
- Takes the nodes and re-ranks them using a specialized model for the task.
- Generally, we retrieve a larger Top K, run the re-ranker and then keep the Top N where N < K. For example K = 6 and N = 2
- the bge-reranker-base is a hugging-face reranker model

##### Putting it together

Put all of the above together into a single function `get_sentence_window_query_engine`:

- inputs:
  - sentence_vector_index: object
  - similarity_top_k: number
  - rerank_top_n: number
- output:
  - sentence_window_engine: object
    - This object is created from the [sentence_index](#putting-it-together)

### Evaluating sentence-window retrieval

Evaluation setup:

- gradually increase the sentence window size starting with 1
- evaluate app versions with the RAG triad
- track experiments to pick the best sentence window size
- tradeoff between token usage/cost and context relevance
- relationship between context relevance and groundedness
  - increasing window size increases context relevance of course, but also increases groundedness. Because with low context, the LLM uses its pre-existing knowledge to fill in the gaps in retrieved context. Therefore, more retrieved context also increases groundedness.
  - as we increase window size, context relevance and groundedness increase up to a point, and then they start to decrease. This is because the LLM could get overwhelmed with context that's too large, and fall back on pre-existing knowledge.

As sentence window size in increased, the total tokens goes up and this could have an impact on the cost if we were to increase the number of records.

## Deep dive into Auto-merging Retrieval

An issue with the naive approach is we're getting a bunch of fragmented chunks to put into the LLM's context window. If the fragmentation is worse, the smaller the chunk size.

![Where auto-merging helps](https://github.com/kowlgi/Building-and-Evaluating-Advanced-RAG-Applications--Class-Notes/blob/main/assets/img3.png "Where auto-merging helps")

The problem:

- In the Top K retrieved chunks step above, the chunks could be from different sections of text, which might not coherent, thus hampering the LLM's ability to synthesize within its context window.

The solution:

- What auto-merging retrieval does is merge smaller child chunks into a bigger parent chunk to ensure more coherent context.
- How does it do it?
  - define a hierarchy of smaller chunks linked to parent chunks
  - if the set of smaller chunks linking to a parent chunk exceeds some threshold, then "merge" smaller chunks into the bigger parent chunk
  - retrieve the parent chunk instead to help ensure more coherent contxt

![How smaller chunks are merged into a parent chunk](https://github.com/kowlgi/Building-and-Evaluating-Advanced-RAG-Applications--Class-Notes/blob/main/assets/img4.png "How smaller chunks are merged into a parent chunk")

### How to set up

#### 1. Build the index

```python
# 1. load documents
from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

# 2. instantiate the hierarchical node parser
from llama_index.node_parser import HierarchicalNodeParser

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

# 3. instantiate LLM
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

from llama_index import ServiceContext

# 4. create service context
auto_merging_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=node_parser,
)

# 5. store parent nodes in doc store
from llama_index import VectorStoreIndex, StorageContext

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


# 6. create automerging index by embedding leaf nodes in vector db
automerging_index = VectorStoreIndex(
    leaf_nodes, storage_context=storage_context, service_context=auto_merging_context
)

# 7. persist the vector index (embedded leaf nodes) and sotrage context (parent nodes) to storage
automerging_index.storage_context.persist(persist_dir="./merging_index")
```

The `chunk_sizes` setting for the hierarchical node parser can be set to any decreasing set of sizes.

- size refers to the # of tokens.
- [2048, 512, 128] is three layers of nodes. [2048, 512] defines two layers
- fewer layers means it's easier to build index as well as easier retrieval, as more layers means more checks.
- if fewer layers works well, we'd obviously prefer it as we want to work with a simpler structure

Note: the vector store index only embeds the leaf node. The parent node is stored in an in-memory doc store. In the Top K retrieval step, we retrieve leaf nodes.

##### Putting it together

Put all of the above together into a single function `build_automerging_index`:

- inputs
  - documents: object
  - llm: object
  - embed_model: string
  - save_index_dir: string
  - chunk_sizes: Array[number]
- output
  - automerging_index: object

#### 2. Create auto-merging retriever

```python
# 1. create auto-merging retriever

from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine

automerging_retriever = automerging_index.as_retriever(
    similarity_top_k=12
)

retriever = AutoMergingRetriever(
    automerging_retriever,
    automerging_index.storage_context,
    verbose=True
)

# 2. set up re-ranker: to reduce token usage

rerank = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-base")

auto_merging_engine = RetrieverQueryEngine.from_args(
    automerging_retriever, node_postprocessors=[rerank]
)
```

For automerging retriever to work well, we set a large Top K, for e.g. 12. Remember that the embedded leaf notes are small size i.e. 128 tokens

##### Putting it together

Put all of the above together into a single function `get_automerging_query_engine`:

- inputs
  - automerging_index: object
  - similarity_top_k: number // 12
  - rerank_top_n: number // 6
- output
  - automerging_enging: object

### Evaluating auto-merging retrieval

Evaluation setup

- iterate with different hierachical structurs: number of levels, children and chunk sizes
- gain intuition about hyperparameters that work best with certain doc types (e.g. employment contracts vs invoices)
- auto-merging is complementary to sentence-window retrieval.
  - if say child 1 and child 4 of parent are relevant context, auto-merging will merge into parent node
  - in contrast, sentence-window will not do this kind of merging, as the sections are not in a contiguous section of text.

## Conclusion

Reducing LLM hallucination is going to be the top priority for all developers.

These RAG techniques are just the tip of the iceberg.

More ways to improve RAG performance:

- understand data pipeline
- retrieval strategy
- LLM prompts

Other things to look into for performance:

- Chunk sizes
- Retrieval techniques like hybrid search
- LLM-based reasoning like Chain-of-thought

Other things to look into for evaluation:

- model confidence
- calibration
- explanability
- fairness
- toxicity
