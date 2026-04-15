# Customer Content QA System (Multi-Tenant RAG with LangGraph)

## Overview

This project implements a multi-tenant, multilingual Retrieval-Augmented Generation (RAG) system for a B2B retail platform. It allows users to ask natural language questions and receive grounded answers with ~~verifiable citations~~.

The system enforces strict country and language isolation, ensuring no cross-tenant data leakage.

## Architecture

### High-Level Flow

User Query → Query Validation → Language Check → Retrieval based on Country and Language → Further Filtering → Answer Generation → Citation Extraction → ~~Validation~~ → Response

---

## Components

### 1. Ingestion

* Parses documents
* Performs semantic chunking using ollama local model
* Stores embeddings in ChromaDB
* Separate collections per (country, language)

### 2. Retrieval

* Query-specific collection lookup
* Top-K similarity search
* Distance-based filtering

### 3. LangGraph Agent

Pipeline includes:

* Input validation
* Language support check
* Retrieval based on country and language
* Relevance filtering
* Answer generation (LLM)
* Citation extraction (with excerpts)
* ~~Citation validation~~ Currently not working (debugging)
* Retry loop on failure

### 4. API Layer

* FastAPI-based service
* Endpoint: `POST /ask`

---

## API Usage

### Endpoint

POST /ask

### Request

```json
{
  "question": "What is return policy?",
  "country": "A",
  "language": "en"
}
```

### Response

```json
{
  "answer": "...",
  "citations": [...],
  "is_valid": true,
  "latency_ms": 120.5
}
```

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Set environment variables

Create `.env` file:

```
OPENAI_API_KEY=your_key_here
```

### 3. Single setup file

```bash
uv run setup.py
```

### 4. Re running the server after the initial setup

```bash
uvicorn api:app --reload
```

## Evaluation

Run evaluation harness:

```bash
uv run test.py
```

This tests:

* Retrieval correctness
* Answer grounding
---

## Design Decisions

### Multi-Tenant Isolation

Separate ChromaDB collections per country-language pair to prevent cross-tenant leakage.

### ~~Citation Enforcement~~ Issue: debugging

Answers are validated against retrieved documents using both structural and LLM-based checks.

### LangGraph Usage

Used for:

* Explicit control flow
* Retry logic
* Modular pipeline design

---

## Future Improvements

* Hybrid search (BM25 + vector)
* Cross-encoder reranking
* Streaming responses
* Caching layer
* Fine-grained evaluation metrics
* Expand validation by incorporating a broader set of multilingual test cases

---

## Author

Tushar Saini
