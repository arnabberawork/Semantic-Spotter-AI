# Semantic-Spotter-AI

## 1. Background

This project demonstrate "Build a RAG System" in insurance domain
using  [LangChain](https://python.langchain.com/docs/introduction/).

## 2. Problem Statement

The goal of the project is to build a robust generative search system capable of effectively and accurately
answering questions from a bunch of policy documents.

## 3. Document

1. The policy documents can be found [here](https://github.com/arnabberawork/Semantic-Spotter-AI/tree/main/data)


## 4. Approach 

LangChain is a framework that simplifies the development of LLM applications LangChain offers a suite of tools,
components, and interfaces that simplify the construction of LLM-centric applications. LangChain enables developers to
build applications that can generate creative and contextually relevant content LangChain provides an LLM class designed
for interfacing with various language model providers, such as OpenAI, Cohere, and Hugging Face.

- LangChain's versatility and flexibility enable seamless integration with various data sources, making it a comprehensive
solution for creating advanced language model-powered applications.

LangChain framework consists of the following:

- *Components: Modular abstractions for different components required for language models.

- *Use-Case Specific Chains: Prebuilt chains tailored to specific applications.

- *Model I/O: Interfaces with language models, prompts, and output parsers.

- *Retrieval: Handles application-specific data, including document loaders, transformers, embeddings, vector stores, and retrievers.

- *Chains: Constructs sequences of LLM calls.

- *Memory: Persists application state.

- *Agents: Selects appropriate tools dynamically.

- *Callbacks: Logs and streams intermediate steps.

## 5. System Layers

**Reading & Processing PDF Files:** [PyPDFDirectoryLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html)

- Used PyPDFDirectoryLoader from LangChain to process PDFs.

- Implemented page-wise splitting in addition to text splitting to enhance retrieval granularity.


**Document Chunking:**  [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/)

Utilized RecursiveCharacterTextSplitter to ensure semantically meaningful text chunks.

Applied customized chunking strategies based on document structure.


**Generating Embeddings:** 

- Leveraged `OpenAIEmbeddings` from LangChain to generate embeddings for the policy documents.  
- Added support for caching embeddings to optimize performance and reduce redundant computations.  
- Integrated `CacheBackedEmbeddings` from LangChain to store and reuse embeddings efficiently.  
- Enhanced embedding generation by applying customized preprocessing steps to ensure semantic relevance.


**Retrievers:** [VectoreStoreRetriever](https://api.python.langchain.com/en/latest/langchain_api_reference.html#module-langchain.retrievers).

- Implemented retrieval from `ChromaDB`'s VectorStoreRetriever.
- Retrievers provide Easy way to combine documents with language models.A retriever is an interface that returns documents given an unstructured query.

**Re-Ranking with a Cross Encoder:**

- Integrated HuggingFaceCrossEncoder (model BAAI/bge-reranker-base) for re-ranking retrieved results.
- with [HuggingFaceCrossEncoder](https://python.langchain.com/api_reference/community/cross_encoders/langchain_community.cross_encoders.huggingface.HuggingFaceCrossEncoder.html)

**Enhancements in Query Handling:**
- Added new question sets to improve model performance.
- Improved the ranking logic to ensure more relevant results appear higher.

## Chains:

 - we can create a chain that takes user input, formats it with a PromptTemplate, and then passes the formatted response to an LLM.
- Leveraged the rlm/rag-promp from LangChain Hub.Constructed a custom RAG chain for better integration with Chroma DB and cosine similarity,retreivalQA 
- [reference](https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/)



## 6. System Architecture

![](./Images/arch1.png) 
![](./Images/arch2.png)

## 7. Prerequisites

- Python 3.7+
- langchain 0.3.13
- Please ensure that you add your OpenAI API key in .env file

## 8. Running

- Clone the github repository
  ```shell
  $ git clone https://github.com/arnabberawork/Semantic-Spotter-AI.git
  ```
- Open
  the [notebook](https://github.com/arnabberawork/Semantic-Spotter-AI/blob/main/Semantic_Spotter_AI_Langchain.ipynb)
  in jupyter and run all cells.
