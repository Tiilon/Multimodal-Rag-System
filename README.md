# Multimodal RAG System

An asynchronous Retrieval-Augmented Generation (RAG) pipeline for multimodal document ingestion and retrieval.

The project ingests documents, extracts text + tables + images, enriches visual content with captions, stores embeddings in a vector database, and answers questions with an LLM using retrieved context.

## What This Project Does

- Parses PDF, DOCX, Excel, CSV, and Markdown documents with Docling.
- Chunks text content for semantic retrieval.
- Converts tables to text-friendly records.
- Captions images using a vision-capable model.
- Stores all content in a vector store (Qdrant or Chroma).
- Supports:
	- general semantic search
	- content-type search (text/table/image)
	- metadata-filtered search
	- page-aware retrieval when queries include phrases like "page 5"

## Key Features

- Async ingestion pipeline end-to-end for better throughput.
- Concurrent image captioning during visual extraction.
- Page metadata for text, tables, and images.
- Unique IDs and filenames for visual artifacts to avoid overwrites.
- Modular architecture for models and vector stores.
- Config-driven behavior via environment variables.

## Tech Stack

- Python 3.12+
- Docling
- LangChain
- Qdrant (default) or Chroma
- Ollama embeddings
- Groq or Ollama for LLM/Vision
- Sentence Transformers tokenizer for chunking

## Project Structure

```text
src/
	core/
		config.py            # Settings loaded via env vars (prefix: RAG_)
		pipeline.py          # Main async ingestion/search/RAG orchestration
	document_processing/
		parser.py            # Doc conversion, visual extraction, image captioning
		chunker.py           # Text chunking
	models/
		factory.py           # LLM / embedding / vision model selection
		groq.py
		ollama.py
	vector_stores/
		factory.py           # Vector store selection
		qdrant_store.py
		chroma_store.py
main.py                  # Entry point
data/                    # Input documents
extracted_elements/      # Saved extracted visual artifacts
```

## Prerequisites

1. Python 3.12+
2. One of the following model setups:
	 - Groq API key for LLM/Vision, and a running Ollama instance for embeddings, or
	 - Fully Ollama-based setup (LLM, Vision, Embeddings)
3. Vector store:
	 - Qdrant server (default is `http://localhost:6333`), or
	 - Local Chroma persistence

## Installation

### Option A: Using uv

```bash
uv sync
```

### Option B: Using venv + pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Configuration

Settings are defined in `src/core/config.py` and loaded from environment variables prefixed with `RAG_`.

Create a `.env` file in the project root (example):

```env
# Required when using Groq models
GROQ_API_KEY=your_groq_api_key

# Vector store
RAG_VECTOR_STORE_TYPE=qdrant
RAG_QDRANT_URL=http://localhost:6333
RAG_QDRANT_API_KEY=
RAG_COLLECTION_NAME=documents

# Embeddings
RAG_EMBEDDING_TYPE=ollama
RAG_EMBEDDING_MODEL=nomic-embed-text
RAG_OLLAMA_BASE_URL=http://localhost:11434

# LLM
RAG_LLM_TYPE=groq
RAG_LLM_MODEL=llama-3.3-70b-versatile
RAG_LLM_TEMPERATURE=0.1

# Vision model
RAG_VISION_TYPE=groq
RAG_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
RAG_VISION_TEMPERATURE=0.1
```

## Quick Start

### 1. Add documents

Put your files in `data/`.

Supported formats include:
- PDF (`.pdf`)
- Word (`.docx`)
- Excel (`.xlsx`)
- CSV (`.csv`)
- Markdown (`.md`)

Note: `main.py` currently has `pdf_test.pdf` enabled by default in the sample `potential_files` list. Uncomment or add additional filenames there for your own ingestion runs.

### 2. Ingest documents

In `main.py`, uncomment this line if you want to run ingestion:

```python
# asyncio.run(process_documents())
```

Then run:

```bash
python main.py
```

This will:
- parse and chunk documents
- extract/caption visual elements
- write vectors to the configured store
- write summary metadata to `document_store_metadata.json`

### 3. Query the system

`main.py` currently runs an example query:

```python
answer = await rag.answer_query("Explain the table in page 5")
```

You can replace that string with your own prompt.

## Retrieval Behavior

- If a query contains `page N` (for example, "What does page 3 say?"), the pipeline attempts page-filtered retrieval first.
- If page-filtered results are empty, it falls back to standard semantic search.
- Content types are tagged in metadata as `text`, `table`, or `image`.

## Generated Outputs

- `document_store_metadata.json`
	- Per-document summary (pages, table count, image count, chunk count)
- `extracted_elements/<document_name>/`
	- extracted table CSVs
	- extracted table images
	- extracted image files
	- `tables_metadata.json`

## Notes on Page Metadata

- Text page metadata is derived from chunk provenance in Docling `doc_items`.
- Visual metadata stores both:
	- `page_number` (primary page), and
	- `page_numbers` (all pages represented by the item)

If you changed metadata logic recently, re-ingest documents to refresh stored vectors and payload metadata.

## Common Troubleshooting

### No results found for page queries

- Ensure documents were ingested after page metadata fixes.
- Re-run ingestion to rebuild vector payloads.

### Groq API key errors

- Make sure `GROQ_API_KEY` is set in your environment or `.env`.

### Qdrant connection issues

- Verify `RAG_QDRANT_URL` and that Qdrant is running.
- For local file-based use, set `RAG_QDRANT_URL` empty and use `RAG_QDRANT_PATH`.

### Ollama model not found

- Confirm Ollama is running and required models are pulled.

## Development Tips

- Switch vector store via `RAG_VECTOR_STORE_TYPE` (`qdrant` or `chroma`).
- Tune batch sizes in vector store implementations for throughput.
- Tune chunking strategy in `src/document_processing/chunker.py` if retrieval granularity is off.

## License

MIT License

Copyright (c) 2026 [Tiilonleeb Konlan]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
