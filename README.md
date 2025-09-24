# Cryptocurrency Policy Analyzer

A tool for analyzing and querying cryptocurrency regulation and policy documents using a simple and robust Retrieval-Augmented Generation (RAG) pipeline. The system ingests web documents, splits them into chunks, indexes them in Elasticsearch for both vector and BM25 retrieval, and answers queries by combining the most relevant passages. If an OpenAI API key is not provided, the app still runs and returns top passages instead of an LLM‑generated answer.

## Overview

- **Purpose:**
  - Ingest and convert policy documents (HTML/PDF) from URLs into Markdown using Docling.
  - Split documents into meaningful chunks using Markdown header splitting.
  - Optionally annotate chunks with structured metadata via the OpenAI API (if key provided).
  - Index chunks in Elasticsearch with both vector embeddings and BM25 text for hybrid retrieval.
  - Retrieve and merge results from semantic and BM25 searches and answer queries with an LLM if available; otherwise return the most relevant passages.

---

## 1. Data Ingestion and Document Conversion

- **DataProcessor Class:**
  - Automates the lifecycle from URL ingestion to Elasticsearch indexing.
  - Loads links from `policy_analyzer/data_processor/data_links` (default).
  - Uses Docling to fetch HTML/PDF and convert to Markdown.
  - Generates metadata (source URL, domain, url_id, extraction_time, title).
  - Splits text by Markdown headers (`#`, `##`, ...).
  - Optionally calls OpenAI to extract structured metadata per chunk.
  - Writes debug files in the persistence directory and indexes chunks to Elasticsearch.

---

## 2. Indexing in Elasticsearch

- **Vector Store:**
  - Uses `ElasticsearchStore` (LangChain) pointed at `ES_URL`.
  - Auth via `ES_API_KEY` or `ES_USER`/`ES_PASSWORD`.
  - Default index name: `policy-index`.
  - Embeddings: `text-embedding-3-small` if `OPENAI_API_KEY` present; otherwise a deterministic `SimpleEmbeddings` fallback.
- **Batch Indexing:**
  - Chunks are indexed in batches (batch size: 20) using:
    - `vector_store.add_documents(batch)`
  - Uses a `ThreadPoolExecutor` for parallelized indexing.
- **BM25 Field:**
  - Raw chunk text is stored in `_source.text` for keyword search (BM25).
  - BM25 queries use the modern `query=` style: `es_client.search(index, query=..., size=k)`.

---

## 3. Retrieval and Hybrid RAG

- **WebAnalyzer Class:**
  - Orchestrates hybrid retrieval with two methods:
    - Semantic similarity using the vector store retriever.
    - BM25 keyword search against `_source.text`.
  - Combines results using weighted reciprocal rank (semantic weight 0.6, BM25 weight 0.25), deduplicates, and feeds the top-k to an LLM for final answers when available.
  - If no `OPENAI_API_KEY` is configured, the API returns the top relevant passages (no model call).

---

## 4. Simplified Design (No Graph/DSPy)

- This repository intentionally removes experimental GraphRAG and DSPy multi-hop reasoning to improve reliability and footprint. The current design focuses on hybrid retrieval (semantic + BM25) and a straightforward RAG prompt.

---

## 5. End-to-End Pipeline Summary

- **Data Ingestion:**
  - Load links → Convert to Markdown (using Docling) → Generate metadata → Split into chunks (Markdown and SemanticChunker) → GPT-based annotation.
- **Indexing:**
  - Index chunks into Elasticsearch with:
    - Vector embeddings for semantic search.
    - BM25 searchable text for keyword-based IR.
    - Enriched metadata (if annotated).
- **Retrieval:**
  - **WebAnalyzer:** executes semantic and BM25 search; merges and scores results via weighted reciprocal rank.

---

## 6. Techniques

- **Embeddings:**
  - Uses `text-embedding-3-small` or fallback deterministic `SimpleEmbeddings` when OpenAI key is absent.
- **BM25 Information Retrieval:**
  - Implements classic BM25 scoring based on term frequency, inverse document frequency, and length normalization.
- **Hybrid Weighted Scoring:**
  - Merges results from semantic and BM25 using weighted reciprocal rank.
- **GPT-Based Annotation:**
  - Utilizes the OpenAI Chat API with function calling to extract structured metadata from document chunks.
  

---

## 7. Tools and Technologies

- **Docling:**
  - Converts URLs (HTML/PDF) to Markdown.
  - Integrated in `DataProcessor.process_url`.
- **LangChain Components:**
  - `MarkdownHeaderTextSplitter`: Splits documents by Markdown headers.
  - `SemanticChunker`: Uses semantic embeddings to further divide text.
  - `OpenAIEmbeddings`: Generates vector embeddings for indexing and similarity search.
- **Elasticsearch & ElasticsearchStore:**
  - Stores document chunks with both vector embeddings and BM25 searchable text.
- **OpenAI Chat API:**
  - Used for function calling to annotate chunks and as an LLM for query answering.
- **DSPy:**
  - Provides typed function signatures (`GenerateSubQuery` and `SynthesizeAnswer`) for orchestrating multi-hop reasoning.
  


## Setup Instructions

### Prerequisites

- Python 3.9+
- Elasticsearch instance (local or cloud)
- OpenAI API key (for embeddings and annotations)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto_policy_analyzer.git
   cd crypto_policy_analyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```
   
5. Edit the `.env` file with your configuration details:
   - Add your Elasticsearch connection details
   - Add your OpenAI API key
   - Configure other settings as needed

### Local Elasticsearch (Quickstart)

If you don’t already have Elasticsearch and Kibana, the quickest local setup is:

```
curl -fsSL https://elastic.co/start-local | sh
```

This launches a single-node Elastic Stack locally (via Docker). After it starts:

- Elasticsearch: `http://localhost:9200`
- Kibana: `http://localhost:5601`

The script prints credentials or enrollment details. Configure one of the following in your `.env` based on what it outputs:

- `ES_API_KEY=...`
- `ES_USER=elastic` and `ES_PASSWORD=...`

Then set `ES_URL=http://localhost:9200` and `ES_INDEX_NAME=policy-index` (or your chosen name).

### Running the Application

1. Process documents and build the vector store:
   ```bash
   # URLs file location (default): policy_analyzer/data_processor/data_links
   python -m policy_analyzer.data_processor.data_processor
   ```

2. Start the web application:
   ```bash
   python app.py
   ```

3. Access the web interface at `http://localhost:5001`

Notes:
- If `OPENAI_API_KEY` is set, answers are generated with an LLM (`gpt-4o`). If not, the service returns the top relevant passages.
- Elasticsearch TLS verification is enabled by default. Configure certificates appropriately for cloud endpoints.


## Usage

### Adding Document Sources

Add URLs to policy documents in the `policy_analyzer/data_processor/data_links` file, one URL per line:

```
https://example.com/policy-document1.pdf
https://example.com/policy-document2.html
```

### Query Examples

The system can answer questions like:

- "What are the current SEC regulations for stablecoins?"
- "How does the Infrastructure Investment and Jobs Act affect crypto tax reporting?"
- "What was the reasoning in the SEC vs Ripple case?"

## Project Structure

```
crypto_policy_analyzer/
├── app.py                     # Flask web application
├── data_links                 # (Deprecated for ingestion) example list – ingestion uses package path
├── policy_analyzer/           # Main package
│   ├── data_processor/        # Ingestion, chunking, and indexing
│   │   ├── data_processor.py  # Main ingestion/indexing script (module-run)
│   │   ├── document_chunker.py
│   │   ├── semantic_metadata.py
│   │   └── data_links         # URL list used by the processor
│   ├── web_analyzer.py        # Hybrid retrieval and RAG
│   ├── models.py              # (Optional) data models
│   ├── templates/             # Web UI templates
│   └── static/                # Web UI static assets
├── crypto_policy_index/       # Storage for processed documents
├── .env.example               # Environment variable template
└── requirements.txt           # Python dependencies
```

## Configuration Options

### Elasticsearch

- `ES_URL`: Elasticsearch endpoint URL (default: `http://localhost:9200`)
- `ES_USER` and `ES_PASSWORD`: Basic auth credentials
- `ES_API_KEY`: API key for Elasticsearch (alternative to username/password)
- `ES_INDEX_NAME`: Name of the index to use

### OpenAI

- `OPENAI_API_KEY`: Your OpenAI API key

### Document Processing

- `URL_TIMEOUT`: Timeout in seconds for URL fetching
- `CHUNK_SIZE`: Target size of document chunks
- `CHUNK_OVERLAP`: Overlap between chunks
