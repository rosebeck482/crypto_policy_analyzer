# Cryptocurrency Policy Analyzer

A tool for analyzing and querying cryptocurrency regulation and policy documents using advanced NLP techniques including RAG (Retrieval Augmented Generation), semantic search, and graph-based retrieval.

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

