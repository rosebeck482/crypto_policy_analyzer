# Cryptocurrency Policy Analyzer

A simple application that uses a Retrieval Augmented Generation (RAG) system to answer questions about cryptocurrency and digital asset policies and regulations.

## Features

- Document ingestion of cryptocurrency policy papers, articles, and regulations
- AI-powered question answering based on the provided documents
- Simple web interface for interaction
- Command-line interface for quick queries

## Installation

1. Clone this repository:
```bash
git clone https://github.com/rosebeck482/crypto_policy_analyzer.git
cd crypto-policy-analyzer
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

4. Create a .env file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
SECRET_KEY=your_secret_key_here
```

## Document Storage

The application uses two directories for document storage:
- `documents/`: User-uploaded documents through the web interface
- `data/`: Pre-loaded documents with cryptocurrency regulations and policy information

## Usage

### Web Interface

1. Start the web application:
```bash
python app.py
```

2. Open your browser and go to `http://localhost:5000`

3.  Ask questions about cryptocurrency policy via the web interface

