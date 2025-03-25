# Cryptocurrency Policy Analyzer

A tool for analyzing and querying cryptocurrency regulation and policy documents using advanced NLP techniques including RAG (Retrieval Augmented Generation), semantic search, and graph-based retrieval.


## Architecture Components

### Web Application (app.py)
- **Framework**: Flask-based web application.
- **Purpose**: Serves as the primary user interface and provides RESTful APIs for policy information queries.
- **Key Features**:
  - Endpoint `/api/query` to receive user queries and return relevant policy insights.
  - Health check endpoint `/api/health` for system monitoring.
  - Integration with the WebAnalyzer module to perform search and retrieval.
  - Logging of incoming requests and outgoing responses to support auditability.

### Data Processing System (data_processor.py)
- **Purpose**: Ingests, processes, and indexes policy documents.
- **Key Capabilities**:
  - **Document Loading**: Fetches content from URLs listed in a data_links file.
  - **Document Conversion & Extraction**: Converts documents to Markdown for consistent text handling.
  - **Markdown + Semantic Chunking**: First chunk documents with Markdowns then if the chunk is over 1500 chars, chunk again semantically.
  - **Metadata Extraction**: Uses GPT-4 structured outputs to extract metadata - keywords and entities, summaries.
  - **Vectorization & Indexing**:
    - Integrates with OpenAI or a fallback SimpleEmbeddings model.
    - Indexes content in Elasticsearch, preserving metadata for downstream retrieval.
  - **Batch Processing & Parallelization**: Employs ThreadPoolExecutor for efficient handling of large document sets.

### Web Analyzer (web_analyzer.py)
- **Purpose**: Retrievs most relevant documents from the indexed data.
- **Key Features**:
  - **Hybrid Retrieval**:
    - Semantic (vector) search for contextual matching.
    - BM25 (keyword) search for exact term matching + Graph-based entity retrieval (using GraphRAG from Ragflow) to discover documents linked by named entities.
  - **Result Ranking & Deduplication**: Scores and merges results from different retrieval methods to avoid redundancy.
  - **Health Check**: Verifies connectivity and availability of embeddings, Elasticsearch, and GraphRAG subsystems.

## Data Pipeline

### Data Collection
- Retrieves document URLs from a data_links configuration file.
- Converts each document to a standardized Markdown format.

### Text Processing
- **Markdown-Aware Chunking**: Divides each document at logical Markdown headings.
- **Semantic Chunking**: Further splits text based on semantic boundaries to ensure each chunk is contextually coherent.

### Vectorization & Indexing
- Extracts embeddings for each chunk using OpenAI or SimpleEmbeddings (fallback).
- Stores chunk vectors and metadata in Elasticsearch.
- Metadata includes document source, headings, and extracted entities.

### Batch Processing
- Uses parallelism (via ThreadPoolExecutor) to optimize document ingestion.
- Ensures resilient processing; errors are logged and fallback strategies are employed if services are unavailable.

## Query System

### Query Processing
- Combines multiple retrieval strategies:
  - Semantic (Vector) Search for contextual similarity.
  - BM25 (Keyword) Search for lexical matching.
  - Graph-Based Entity Search to leverage relationships among entities.
- Applies a weighted ranking algorithm to merge results from different retrieval methods.

### Contextual Compression
- Filters and refines retrieved chunks to discard irrelevant data.
- Strives to maintain only the most pertinent information for the final answer.

### Response Generation
- Final stage uses an LLM to produce a coherent, user-friendly response.
- Incorporates retrieved context and metadata to support accuracy.
- Ensures answers cite sources, including metadata such as document URLs or headings.


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

