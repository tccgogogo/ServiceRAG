# ServiceRAG: Service Recommendation As a Retrieval-Augmented Generation Problem

## Project Overview

Service recommendation systems have become increasingly crucial in today’s digital ecosystem, as developers face an overwhelming array of rapidly evolving choices across cloud
computing, web services, and API marketplaces. Traditional approaches like collaborative filtering, content-based filtering,and graph neural networks face inherent limitations in addressing
service selection for mashup development, including cold start problems, difficulty identifying authentic service dependencies,and challenges adapting to rapidly changing service ecosystems.
We propose ServiceRAG, a novel framework that reformulates service recommendation as a retrieval-augmented generation problem. ServiceRAG enhances service recommendation through three key components: 
- a semantic enrichment mechanism that generates expanded representations of both mashups and APIs to facilitate precise semantic matching
- a compatibility-aware retrieval system that captures genuine service dependencies through a hybrid retrieval approach
- a chain-of-thought recommendation generation process that leverages large language models to produce contextually appropriate service compositions.

## Installation Steps

1. Clone the project code

```bash
git clone <repository-url>
cd graph-rag
```

2. Configure the environment

Edit the `config.py` file to set the following configurations:
- Model paths (EMBED_MODEL_PATH, RERANK_MODEL_PATH, LLM_MODEL_PATH)
- Dataset paths (TRAIN_MASHUP_PATH, TEST_MASHUP_PATH)
- API configurations (OPENAI_BASE_URL, OPENAI_API_KEY)
- Retrieval parameters (RETRIEVER_K, BM25_K1, BM25_B, etc.)

## Usage

### Execute the semantic enrichment module for data rewriting

```bash
python semantic_enrichment.py
```

### Running the Evaluation Process

```bash
python main.py
```

## Project Structure

```
ServiceRAG/
├── config.py                     # Configuration file
├── main.py                       # Main program entry
├── semantic_enrichment.py        # semantic enrichment module
├── rag.py                        # RAG system implementation
├── trainer.py                    # Training and evaluation module
└── utils.py                      # Utility function collection
```

