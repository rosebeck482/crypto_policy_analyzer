# Data Processor for Cryptocurrency Policy Analyzer
# Handles loading data, processing content, generating embeddings, and storing in Elasticsearch

import os
import logging
import sys
import time
import copy
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv, dotenv_values
from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain.schema.document import Document as LangchainDocument
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import time

# Import our custom document chunker
from document_chunker import process_url_to_chunks, fetch_document_content
# Import our semantic metadata module
import semantic_metadata

# Constants
ES_URL = ""
ES_USER = ""  
ES_PASSWORD = ""  
ES_API_KEY = ""  
INDEX_NAME = ""
URL_TIMEOUT = 30  # seconds
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OPENAI_API_KEY = ""  

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import json
from dotenv import dotenv_values
import numpy as np
from langchain.embeddings.base import Embeddings

# Use the hardcoded OpenAI API key first, then check environment
api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY not found in environment or hardcoded")
else:
    logger.info("OPENAI_API_KEY found - using it for embeddings and completions")


# Simple embeddings class that doesn't require API calls
class SimpleEmbeddings(Embeddings):
    
    def __init__(self, vector_size=1536):
        # Initialize with vector size (1536 is the same as OpenAI embeddings)
        self.vector_size = vector_size
        logger.info(f"Using SimpleEmbeddings with vector_size={vector_size}")
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Return embeddings for a list of documents
        # Generate deterministic embeddings based on text hash
        return [self._get_embedding_for_text(text) for text in texts]
    
    def embed_query(self, text: str) -> list[float]:
        # Return embeddings for a query
        return self._get_embedding_for_text(text)
    
    def _get_embedding_for_text(self, text: str) -> list[float]:
        # Generate a deterministic embedding based on text hash
        # Use hash of text as seed for random number generator
        np.random.seed(hash(text) % 2**32)
        vector = np.random.normal(0, 1, self.vector_size)
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()

# If we have an API key, use OpenAI, otherwise use SimpleEmbeddings
if api_key:
    logger.info("Using OpenAI embeddings")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    client = OpenAI(api_key=api_key)
else:
    logger.info("Using SimpleEmbeddings due to missing API key")
    embeddings = SimpleEmbeddings()
    client = None

# Determine if we'll use OpenAI API for annotations
OPENAI_AVAILABLE = api_key is not None

class DataProcessor:
    def __init__(self, links_file="data_links", persistence_dir="crypto_policy_index"):
        # Locate data_links file - check both in current directory and project root
        self.links_file = links_file
        current_dir = Path(__file__).parent
        # Look for data_links in the current directory, not the parent directory
        self.root_links_file = current_dir / links_file
        
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        self.links = self._load_links()
        logger.info(f"Links loading took: {time.time() - start_time:.2f} seconds")
    
        self.documents = []
        self.chunks = []
        self._add_start_index = True
    
    def _load_links(self):
        with open(self.root_links_file, 'r') as f:
            links = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(links)} links from {self.root_links_file}")
        return links

    
    def initialize_elasticsearch_index(self):
        logger.info(f"Initializing Elasticsearch index at {ES_URL}")
        try:
            es_api_key = os.environ.get("ES_API_KEY", ES_API_KEY)
            logger.info(f"Elasticsearch API key found: {'Yes' if es_api_key else 'No'}")
            
            from elasticsearch import Elasticsearch
            
            # Choose embedding model based on OpenAI availability
            if OPENAI_AVAILABLE and api_key:
                logger.info("Using OpenAI embeddings for vector store")
                embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
            else:
                logger.info("Using SimpleEmbeddings for vector store")
                embedding_model = SimpleEmbeddings()

            if es_api_key:
                logger.info("Using API key authentication for Elasticsearch")
                es_client = Elasticsearch(
                    ES_URL,
                    api_key=es_api_key
                )
                vector_store = ElasticsearchStore(
                    index_name=INDEX_NAME,
                    es_connection=es_client,
                    embedding=embedding_model
                )
            else:
                logger.info("Using username/password authentication for Elasticsearch")
                es_client = Elasticsearch(
                    ES_URL,
                    basic_auth=(ES_USER, ES_PASSWORD)
                )
                vector_store = ElasticsearchStore(
                    index_name=INDEX_NAME,
                    es_connection=es_client,
                    embedding=embedding_model
                )
            
            logger.info(f"Successfully connected to Elasticsearch index: {INDEX_NAME}")
            return vector_store
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {str(e)}")
            raise
    
    def split_text(self, text: str) -> List[str]:
        return [text]
    
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    
    # Process a URL and generate document chunks (now using our modular approach)
    def process_url(self, url, url_id=None):
        try:
            logger.info(f"Processing URL: {url}")
            
            # Use our document_chunker module to fetch and chunk the document
            # No longer passing embeddings for semantic chunking since that's been removed
            chunks = process_url_to_chunks(
                url, 
                self.persistence_dir
                # No longer passing embeddings parameter
                # No longer passing max_chunk_size parameter
            )
            
            # If we have chunks and OPENAI_AVAILABLE, enhance them with semantic metadata
            if chunks and OPENAI_AVAILABLE and client:
                # Get the full document markdown for context (using the first chunk's source)
                full_document_markdown = ""
                if chunks:
                    for chunk in chunks:
                        if 'source' in chunk.metadata:
                            full_document_markdown = fetch_document_content(chunk.metadata['source'])
                            break
                
                if full_document_markdown:
                    logger.info(f"Enhancing {len(chunks)} chunks with semantic metadata")
                    chunks = semantic_metadata.process_chunks_with_semantic_metadata(chunks, full_document_markdown)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return []
    
    # Process all URLs in the links file and split into chunks
    def process_all_urls(self):
        logger.info("Processing all URLs...")
        
        if not self.links:
            logger.warning("No links to process")
            return False
        
        # Reset documents and chunks lists
        self.documents = []
        self.chunks = []
        
        # Clear the latest_chunks_with_metadata.txt file
        latest_chunks_file_path = self.persistence_dir / "latest_chunks_with_metadata.txt"
        logger.info(f"Clearing previous content from {latest_chunks_file_path}")
        with open(latest_chunks_file_path, 'w', encoding='utf-8') as f:
            f.write(f"NEW RUN - Started at {datetime.now().isoformat()}\n\n")
        
        # Process each URL
        start_time = time.time()
        logger.info(f"Starting to process {len(self.links)} URLs at: {time.strftime('%H:%M:%S')}")
        
        processed_count = 0
        for index, url in enumerate(self.links):
            logger.info(f"Processing URL {index+1}/{len(self.links)}")
            
            url_start_time = time.time()
            url_chunks = self.process_url(url)
            
            if url_chunks:
                self.chunks.extend(url_chunks)
                processed_count += 1
                logger.info(f"URL {index+1} processed successfully, got {len(url_chunks)} chunks")
            else:
                logger.error(f"Failed to process URL {index+1}: {url}")
            
            url_processing_time = time.time() - url_start_time
            logger.info(f"URL {index+1} processing completed in {url_processing_time:.2f} seconds")
            
            if index < len(self.links) - 1:
                logger.info(f"Waiting 1 second before processing next URL...")
                time.sleep(1)
        
        total_time = time.time() - start_time
        logger.info(f"Successfully processed {processed_count}/{len(self.links)} URLs")
        logger.info(f"Generated {len(self.documents)} documents and {len(self.chunks)} chunks")
        logger.info(f"Total processing time for all URLs: {total_time:.2f} seconds")
        
        # Save chunks with their metadata to a text file for inspection
        if self.chunks:
            chunks_file_path = self.persistence_dir / "chunks_with_metadata.txt"
            logger.info(f"Saving chunks with metadata to {chunks_file_path}")
            
            with open(chunks_file_path, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(self.chunks):
                    f.write(f"{'='*80}\n")
                    f.write(f"CHUNK {i+1}/{len(self.chunks)}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    # Write chunk content
                    f.write("CONTENT:\n")
                    f.write(f"{chunk.page_content}\n\n")
                    
                    # Write chunk metadata
                    f.write("METADATA:\n")
                    for key, value in chunk.metadata.items():
                        # Display full value without truncation
                        f.write(f"  {key}: {value}\n")
                    
                    f.write("\n\n")
            
            logger.info(f"Saved chunks with metadata to {chunks_file_path}")
        
        if not self.chunks:
            logger.error("No chunks were generated. Check the URLs and try again.")
            return False
        
        return True
    
    # Build the Elasticsearch vector store from chunks
    def build_vector_store(self, max_workers: int = 4):
        if not self.chunks:
            logger.warning("No chunks available to build vector store. Run process_all_urls() first.")
            return None
            
        try:
            es_store = self.initialize_elasticsearch_index()
            index_start = time.time()
            logger.info(f"Starting vector store build at: {time.strftime('%H:%M:%S')}")
            logger.info(f"Building vector store with {len(self.chunks)} chunks")
            
            batch_size = 20
            total_chunks = len(self.chunks)
            total_batches = (total_chunks + batch_size - 1) // batch_size

            # Process a batch of chunks for indexing
            def index_batch(batch, batch_num):
                try:
                    logger.info(f"[Thread] Indexing batch {batch_num}/{total_batches} with {len(batch)} chunks")
                    es_store.add_documents(batch)
                    logger.info(f"[Thread] Completed batch {batch_num}")
                    return batch_num
                except Exception as e:
                    logger.error(f"[Thread] Failed batch {batch_num}: {str(e)}")
                    return None

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(0, total_chunks, batch_size):
                    batch = self.chunks[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    futures.append(executor.submit(index_batch, batch, batch_num))

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        logger.info(f"[Main] Batch {result} successfully indexed")
                    else:
                        logger.warning(f"[Main] One batch failed to index")
            
            build_time = time.time() - index_start
            logger.info(f"Vector store building took: {build_time:.2f} seconds")
            
            with open(self.persistence_dir / "elasticsearch_index.txt", "w") as f:
                f.write(f"Elasticsearch index: {INDEX_NAME}\n")
                f.write(f"Chunks count: {len(self.chunks)}\n")
                f.write(f"Built at: {datetime.now().isoformat()}\n")
                f.write(f"URL: {ES_URL}\n")
                f.write(f"Chunking method: semantic\n")
            
            logger.info(f"Saved index reference to {self.persistence_dir}")
            return es_store

        except Exception as e:
            logger.error(f"Error building vector store: {str(e)}")
            return None
    
    # Extract title from markdown content
    def _extract_title(self, content: str) -> Optional[str]:
        # Reusing existing function from document_chunker
        from document_chunker import extract_title
        return extract_title(content)

# Main function to run the processor
def main():
    logger.info("Starting data processor")
    processor = DataProcessor()
    
    # Process URLs and generate chunks
    logger.info("Processing URLs")
    if processor.process_all_urls():
        logger.info("URL processing completed successfully")
        
        # Build the vector store with chunks
        logger.info("Building vector store")
        es_store = processor.build_vector_store()
        
        if es_store:
            logger.info("Vector store built successfully")
            return 0
        else:
            logger.error("Failed to build vector store")
            return 1
    else:
        logger.error("Failed to process URLs")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 