# Cryptocurrency Policy Analyzer

A tool for analyzing and querying cryptocurrency regulation and policy documents using advanced NLP techniques including RAG (Retrieval Augmented Generation), semantic search, and graph-based retrieval.

## Overview

- **Purpose:**
  - Ingest and convert policy documents (HTML/PDF) from URLs into Markdown.
  - Split documents into meaningful chunks (using both markdown header splitting and semantic chunking).
  - Optionally annotate chunks with structured metadata using the OpenAI Chat API.
  - Index chunks in Elasticsearch using both embedding-based (vector) and BM25 (keyword) approaches.
  - Retrieve and merge results from semantic, BM25, and (optionally) graph-based searches.
  - Perform multi-hop reasoning (chain-of-thought) with DSPy for complex queries.

---

## 1. Data Ingestion and Document Conversion

- **DataProcessor Class:**
  - **Objective:** Automate the full lifecycle from document URL ingestion to Elasticsearch indexing.
  - **Key Steps:**
    - **Load Links:**
      - Reads a file (e.g., `data_links`) to obtain document URLs.
    - **Fetch and Convert:**
      - Uses **Docling (DocumentConverter)** to fetch HTML/PDF and convert to Markdown.
    - **Document Metadata Generation:**
      - Creates a LangChain Document with metadata:
        - `source URL`
        - `domain` (derived from `urlparse`)
        - `url_id` (hash or provided ID)
        - `extraction_time`
        - `title` (parsed from the first Markdown header or line)
    - **Text Chunking:**
      - **MarkdownHeaderTextSplitter:**
        - Splits large Markdown documents based on headers (`#`, `##`, `###`, etc.).
      - **SemanticChunker:**
        - Further splits each chunk semantically using embeddings.
      - **Fallback Mechanism:**
        - If semantic chunking fails, it falls back to the raw split.
    - **Annotation (Optional):**
      - **Method:** `situate_and_annotate_chunk`
      - **Mechanism:** Calls the OpenAI Chat API with a function schema to extract structured fields such as:
        - Document type, summary, laws, organizations, etc.
      - **Outcome:** Enriches each chunk’s metadata with the extracted details.
    - **Persistence:**
      - Optionally writes chunks with metadata to a local file (e.g., `latest_chunks_with_metadata.txt`) for debugging.
    - **Indexing:**
      - Prepared chunks are later indexed into Elasticsearch using the `build_vector_store` method.

---

## 2. Indexing in Elasticsearch

- **Vector Store Initialization:**
  - **Tool:** `ElasticsearchStore` from `langchain_elasticsearch`.
  - **Authentication:**
    - Uses an API key if provided; otherwise, falls back to username/password.
  - **Index Configuration:**
    - Default index names: `policy-project` or `crypto-policy`.
  - **Embeddings:**
    - Uses `OpenAIEmbeddings` when an OpenAI API key is present.
    - Falls back to a `SimpleEmbeddings` class generating deterministic random vectors if no key is provided.
- **Batch Indexing:**
  - Chunks are indexed in batches (batch size: 20) using:
    - `vector_store.add_documents(batch)`
  - Uses a `ThreadPoolExecutor` for parallelized indexing.
- **BM25 Fields:**
  - In addition to vector embeddings, raw text is stored in a field (e.g., `"text"`) analyzed with standard Lucene indexing.
  - BM25 queries are executed using:
    - `es_client.search(index=INDEX_NAME, body=search_body)`

---

## 3. Retrieval and Hybrid RAG

- **WebAnalyzer Class:**
  - **Purpose:** Orchestrates different retrieval strategies.
  - **Retrieval Methods:**
    - **Semantic Retrieval:**
      - Uses `self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15})` for embedding-based search.
    - **BM25 Retrieval:**
      - Constructs and executes BM25 queries over textual fields.
      - Returns results as LangChain Document objects (content + metadata).
    - **Graph-Based Retrieval:**
      - If GraphRAG is installed:
        - Extracts entities from the query using a custom prompt and LLM.
        - Performs entity-based search in a knowledge graph to enhance retrieval.
  - **Hybrid/Enhanced Retrieval:**
    - **Hybrid Retrieval:**
      - Combines semantic and BM25 search results.
    - **Enhanced Retrieval:**
      - Merges semantic, BM25, and graph-based search results.
    - **Weighted Scoring Scheme:**
      - Semantic weight: 0.6
      - BM25 weight: 0.25
      - Graph weight: 0.15
    - **Process:**
      - Deduplicates documents by content hash.
      - Aggregates reciprocal rank scores from each retrieval method.
      - Sorts documents by final weighted score and returns the top-k results.

---

## 4. Multi-Hop Chain-of-Thought with DSPy

- **DSPy Signatures:**
  - **GenerateSubQuery (dspy.Signature):**
    - **Inputs:**
      - `question`: The original complex query.
      - `context`: The current retrieved context.
    - **Output:**
      - `sub_query`: A refined, more focused query.
  - **SynthesizeAnswer (dspy.Signature):**
    - **Inputs:**
      - `question`: The original complex question.
      - `sub_queries`: List of generated sub-questions.
      - `contexts`: Retrieved contexts for each sub-query.
    - **Output:**
      - `answer`: The final consolidated answer.
- **MultiHopChainOfThoughtRAG Class:**
  - **LLM Integration:**
    - Accepts an LLM instance (e.g., OpenAI Chat) and a `retriever_func` for fetching passages.
  - **Process Workflow:**
    - **Initialization:**
      - Starts with the user’s original complex question.
    - **Iterative Hops:**
      - For each hop:
        - Retrieves new passages using `retriever_func(current_query, k=self.passages_per_hop)`.
        - Accumulates retrieved context.
        - Generates a refined sub-query with `self.generate_sub_query(...)` (except on the final hop).
    - **Final Synthesis:**
      - Compiles all sub-queries and contexts using `self.synthesize_answer(...)` to produce the comprehensive answer.
  - **Chain-of-Thought:**
    - Each hop refines the query and builds upon the previous context, culminating in a multi-step reasoning process.

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
  - **WebAnalyzer:**
    - Executes semantic retrieval, BM25 search, and graph-based search.
    - Merges and scores results using a weighted reciprocal rank scheme.
- **Multi-Hop Reasoning:**
  - **MultiHopChainOfThoughtRAG:**
    - Iteratively refines the query and synthesizes a final answer via DSPy’s chain-of-thought mechanism.

---

## 6. Techniques

- **Embeddings:**
  - Uses transformer-based embeddings (OpenAI’s `text-embedding-ada-002`) or fallback simple embeddings.
- **BM25 Information Retrieval:**
  - Implements classic BM25 scoring based on term frequency, inverse document frequency, and length normalization.
- **Hybrid Weighted Scoring:**
  - Merges results from semantic, BM25, and (optionally) graph-based retrieval using weighted reciprocal rank.
- **GPT-Based Annotation:**
  - Utilizes the OpenAI Chat API with function calling to extract structured metadata from document chunks.
- **Chain-of-Thought Reasoning:**
  - Uses DSPy to facilitate multi-hop reasoning by breaking down complex questions into sub-queries and synthesizing a final answer.

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
- **GraphRAG From Ragflow**
  - Extends retrieval capabilities with entity extraction and graph-based search.


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

### Running the Application

1. Process documents and build the vector store:
   ```bash
   python -m policy_analyzer.data_processor
   ```

2. Start the web application:
   ```bash
   python app.py
   ```

3. Access the web interface at `http://localhost:5001`

## Usage

### Adding Document Sources

Add URLs to policy documents in the `data_links` file, one URL per line:

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
├── data_links                 # List of document URLs to process
├── policy_analyzer/           # Main package
│   ├── data_processor.py      # Document processing and indexing
│   ├── web_analyzer.py        # Query processing and retrieval
│   ├── query_analyze_prompt.py # Prompts for query analysis
│   ├── models.py              # Data models
│   ├── graphrag_core/         # Graph-based RAG components
│   ├── templates/             # Web UI templates
│   └── static/                # Web UI static assets
├── crypto_policy_index/       # Storage for processed documents
├── .env.example               # Environment variable template
└── requirements.txt           # Python dependencies
```

## Configuration Options

### Elasticsearch

- `ES_URL`: Elasticsearch endpoint URL
- `ES_USER` and `ES_PASSWORD`: Basic auth credentials
- `ES_API_KEY`: API key for Elasticsearch (alternative to username/password)
- `ES_INDEX_NAME`: Name of the index to use

### OpenAI

- `OPENAI_API_KEY`: Your OpenAI API key

### Document Processing

- `URL_TIMEOUT`: Timeout in seconds for URL fetching
- `CHUNK_SIZE`: Target size of document chunks
- `CHUNK_OVERLAP`: Overlap between chunks

