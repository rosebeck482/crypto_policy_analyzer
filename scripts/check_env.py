#!/usr/bin/env python
"""
Environment variable checker script.
Run this to verify your environment setup before running the application.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import from the root package
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Required environment variables
REQUIRED_VARS = {
    "ES_URL": "Elasticsearch URL (e.g., http://localhost:9200)",
    "ES_INDEX_NAME": "Elasticsearch index name (e.g., policy-index)",
    "OPENAI_API_KEY": "OpenAI API key for embeddings and annotations"
}

# Optional variables with default values
OPTIONAL_VARS = {
    "ES_USER": "Elasticsearch username (if not using API key)",
    "ES_PASSWORD": "Elasticsearch password (if not using API key)",
    "ES_API_KEY": "Elasticsearch API key (alternative to username/password)",
    "FLASK_SECRET_KEY": "Secret key for Flask sessions (auto-generated if not set)",
    "URL_TIMEOUT": "Timeout for URL fetching in seconds (default: 30)",
    "CHUNK_SIZE": "Size of document chunks (default: 1000)",
    "CHUNK_OVERLAP": "Overlap between document chunks (default: 200)"
}

# Check required variables
missing_vars = []
for var, description in REQUIRED_VARS.items():
    if not os.environ.get(var):
        missing_vars.append(f"{var} - {description}")

# Print results
if missing_vars:
    print("❌ ERROR: Missing required environment variables:")
    for var in missing_vars:
        print(f"  - {var}")
    print("\nPlease set these variables in your .env file or environment.")
    print("You can use .env.example as a template.")
    sys.exit(1)
else:
    print("✅ All required environment variables are set!")

# Check optional variables
missing_optional = []
for var, description in OPTIONAL_VARS.items():
    if not os.environ.get(var):
        missing_optional.append(f"{var} - {description}")

if missing_optional:
    print("\n⚠️  Missing optional environment variables (defaults will be used):")
    for var in missing_optional:
        print(f"  - {var}")

# Check Elasticsearch connection if ES_URL is set
if os.environ.get("ES_URL"):
    try:
        from elasticsearch import Elasticsearch
        
        es_url = os.environ.get("ES_URL")
        es_api_key = os.environ.get("ES_API_KEY")
        es_user = os.environ.get("ES_USER")
        es_password = os.environ.get("ES_PASSWORD")
        
        if es_api_key:
            es_client = Elasticsearch(
                es_url,
                api_key=es_api_key,
                verify_certs=False
            )
        elif es_user and es_password:
            es_client = Elasticsearch(
                es_url,
                basic_auth=(es_user, es_password),
                verify_certs=False
            )
        else:
            es_client = Elasticsearch(es_url, verify_certs=False)
        
        health = es_client.cluster.health()
        print(f"\n✅ Successfully connected to Elasticsearch cluster: {health.get('cluster_name')}")
        print(f"   Status: {health.get('status')}")
        print(f"   Nodes: {health.get('number_of_nodes')}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to Elasticsearch: {str(e)}")
        print("   Please check your Elasticsearch configuration.")

# Check OpenAI connection if OPENAI_API_KEY is set
if os.environ.get("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        models = client.models.list()
        print("\n✅ Successfully connected to OpenAI API")
        print(f"   Available models: {len(models.data)}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to OpenAI API: {str(e)}")
        print("   Please check your OpenAI API key.")

print("\nEnvironment check completed.") 