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
from langchain_experimental.text_splitter import SemanticChunker

import json
import time

# Load environment variables
load_dotenv()

# Constants - load from environment variables with fallbacks
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
ES_USER = os.environ.get("ES_USER", "elastic")
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")
ES_API_KEY = os.environ.get("ES_API_KEY", "")
INDEX_NAME = os.environ.get("ES_INDEX_NAME", "policy-index")
URL_TIMEOUT = int(os.environ.get("URL_TIMEOUT", "30"))  # seconds
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the environment variables 
logger.info(f"Using ES_URL: {ES_URL}")
logger.info(f"ES_API_KEY present: {'Yes' if ES_API_KEY else 'No'}")
logger.info(f"ES_INDEX_NAME: {INDEX_NAME}")


from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import json
from dotenv import dotenv_values
import numpy as np
from langchain.embeddings.base import Embeddings

# Load API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY not found in environment")
else:
    logger.info("OPENAI_API_KEY found in environment")
    # Don't log any part of the API key for security

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
    client = None  # No OpenAI client available

# Determine if we'll use OpenAI API for annotations
OPENAI_AVAILABLE = api_key is not None

class DataProcessor:
    def __init__(self, links_file="data_links", persistence_dir="crypto_policy_index"):
        # Locate data_links file - check both in current directory and project root
        self.links_file = links_file
        current_dir = Path(__file__).parent
        root_dir = current_dir.parent
        self.root_links_file = root_dir / links_file
        
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
    
    # Process a URL and generate document chunks
    def process_url(self, url, url_id=None):
        try:
            logger.info(f"Processing URL: {url}")
            
            # Extract domain for metadata
            domain = urlparse(url).netloc
            if not domain:
                domain = "file"  # For local files
            
            # Ensure we have an ID for tracking/metadata
            if url_id is None:
                url_id = str(hash(url))
            
            # Use DocumentConverter to fetch and convert the content
            converter = DocumentConverter()
            content = converter.url_to_markdown(url, timeout=URL_TIMEOUT)
            
            if not content:
                logger.error(f"Failed to extract content from {url}")
                return []
            
            # Basic document metadata
            metadata = {
                "source": url,
                "domain": domain,
                "url_id": url_id,
                "extraction_time": datetime.now().isoformat(),
                "title": self._extract_title(content) or domain
            }
            logger.info(f"Extracted {len(content)} characters of content")
            
            # Create a document from the content
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            chunks = self.split_documents_Markdown([doc])
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return []
    
    # Extract structured data from document chunks
    def situate_and_annotate_chunk(self, full_document_markdown: str, chunk_markdown: str, metadata: dict) -> dict:
        # If OpenAI is not available, return basic metadata
        if not OPENAI_AVAILABLE or client is None:
            logger.info("OpenAI API not available - skipping structured annotation")
            return {
                "document_title": metadata.get("title", ""),
                "document_type_purpose": "Policy document",
                "summary": "",
                "law_or_regulation": [],
                "legislation": [],
                "organizations": [],
                "case_law": [],
                "entities": [],
                "technology_terms": [],
                "financial_concepts": [],
                "government_actions": [],
                "dates_and_times": [],
                "locations": [],
                "context_and_implications": "",
                "outcomes": ""
            }

        try:
            # Add a delay before API call to avoid rate limiting
            logger.info("Adding delay before OpenAI API call to avoid rate limiting...")
            time.sleep(3)

            function_schema = {
                "type": "function",
                "function": {
                    "name": "extract_document_information",
                    "description": "Extract structured and typed information from a document chunk for retrieval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_title": {
                                "type": "string",
                                "description": "Title of the document."
                            },
                            "document_type_purpose": {
                                "type": "string",
                                "description": "Type and purpose of the document."
                            },
                            "summary": {
                                "type": "string",
                                "description": "Concise summary of the document chunk in 1-3 sentences."
                            },
                            "law_or_regulation": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific laws or regulations mentioned (e.g., securities law, AML, estate law)."
                            },
                            "legislation": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Legislative acts or bills discussed (e.g., RFIA, DCCPA, Toomey Bill)."
                            },
                            "organizations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Organizations or regulatory bodies involved (e.g., SEC, CFTC, IRS)."
                            },
                            "case_law": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Legal cases referenced (e.g., SEC v. Ripple, Terraform ruling)."
                            },
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Named entities (companies, DAOs, exchanges, platforms, etc.)."
                            },
                            "technology_terms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Blockchain and crypto technologies mentioned (e.g., stablecoins, PoS, DeFi)."
                            },
                            "financial_concepts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Financial terms like staking, token issuance, capital gains, etc."
                            },
                            "government_actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Actions taken by government entities (e.g., sanctions, hearings, executive orders)."
                            },
                            "dates_and_times": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Important dates or time periods referenced (e.g., March 2023, January 2024)."
                            },
                            "locations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Jurisdictions or regions mentioned (e.g., Wyoming, New York)."
                            },
                            "context_and_implications": {
                                "type": "string",
                                "description": "Context and implications of the document's content."
                            },
                            "outcomes": {
                                "type": "string",
                                "description": "Outcomes or results described in the document."
                            }
                        },
                        "required": [
                            "document_title",
                            "document_type_purpose",
                            "summary"
                        ]
                    }
                }
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that extracts structured summaries from document chunks. "
                        "Use the provided function to extract the required information."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"<document>\n{full_document_markdown}\n</document>\n\n"
                        f"<chunk>\n{chunk_markdown}\n</chunk>\n\n"
                        f"Existing metadata: {json.dumps(metadata, ensure_ascii=False)}"
                    )
                }
            ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[function_schema],
                tool_choice={"type": "function", "function": {"name": "extract_document_information"}},
                temperature=0.2
            )

            try:
                tool_call = response.choices[0].message.tool_calls[0]
                function_args = tool_call.function.arguments

                if isinstance(function_args, str):
                    structured_data = json.loads(function_args)
                else:
                    structured_data = function_args

                return structured_data

            except Exception as parse_error:
                logger.error(f"Error parsing function call arguments: {parse_error}")
                return {
                    "document_title": "",
                    "document_type_purpose": "",
                    "summary": "",
                    "law_or_regulation": [],
                    "legislation": [],
                    "organizations": [],
                    "case_law": [],
                    "entities": [],
                    "technology_terms": [],
                    "financial_concepts": [],
                    "government_actions": [],
                    "dates_and_times": [],
                    "locations": [],
                    "context_and_implications": "",
                    "outcomes": ""
                }

        except Exception as e:
            logger.error(f"Error with structured call: {str(e)}")
            # Check if it's a rate limit error and retry with backoff
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning("Rate limit hit, backing off and retrying once...")
                try:
                    # Longer backoff for rate limit (10 seconds)
                    time.sleep(10)
                    # Try again with the same function schema
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        tools=[function_schema],
                        tool_choice={"type": "function", "function": {"name": "extract_document_information"}},
                        temperature=0.2
                    )
                    
                    tool_call = response.choices[0].message.tool_calls[0]
                    function_args = tool_call.function.arguments
                    
                    if isinstance(function_args, str):
                        structured_data = json.loads(function_args)
                    else:
                        structured_data = function_args
                    
                    return structured_data
                    
                except Exception as retry_error:
                    logger.error(f"Error on retry: {str(retry_error)}")
            
            # Return empty metadata structure
            return {
                "document_title": metadata.get("title", ""),
                "document_type_purpose": "Policy document",
                "summary": "",
                "law_or_regulation": [],
                "legislation": [],
                "organizations": [],
                "case_law": [],
                "entities": [],
                "technology_terms": [],
                "financial_concepts": [],
                "government_actions": [],
                "dates_and_times": [],
                "locations": [],
                "context_and_implications": "",
                "outcomes": ""
            }

    # Split documents using Markdown headers
    def split_documents_Markdown(self, documents: List[Document]) -> List[Document]:
        try:
            # Get the splitters
            markdown_splitter = self.get_markdown_splitter()
            semantic_chunker = self.get_semantic_chunker()
            
            chunks = []
            
            for document in documents:
                logger.info(f"Splitting document with {len(document.page_content)} characters")
                
                # Use Markdown header text splitter first to get rough chunks
                header_chunks = markdown_splitter.split_text(document.page_content)
                
                logger.info(f"Markdown header splitter created {len(header_chunks)} chunks")
                
                # Process each header chunk with LangChain's Document format
                for i, header_chunk in enumerate(header_chunks):
                    # Create LangchainDocument for the semantic chunker
                    lc_document = LangchainDocument(
                        page_content=header_chunk,
                        metadata=copy.deepcopy(document.metadata)
                    )
                    
                    # Use semantic chunker for more meaningful boundaries
                    try:
                        semantic_chunks = semantic_chunker.split_documents([lc_document])
                        logger.info(f"  Header chunk {i+1} split into {len(semantic_chunks)} semantic chunks")
                        
                        # Process each semantic chunk
                        for j, semantic_chunk in enumerate(semantic_chunks):
                            # Update metadata with the chunk information
                            self.enrich_chunk_metadata(semantic_chunk, document.metadata, i * 100 + j)
                            
                            # Annotate chunk with structured information
                            self.annotate_chunk(document.page_content, semantic_chunk)
                            
                            # Save chunk for debugging/inspection
                            self.persist_chunk(semantic_chunk, f"DOC{i+1}_CHUNK{j+1}")
                            
                            chunks.append(semantic_chunk)
                    except Exception as chunk_error:
                        logger.error(f"Error in semantic chunking: {str(chunk_error)}")
                        # If semantic chunking fails, use the header chunk directly
                        header_doc = Document(
                            page_content=header_chunk,
                            metadata=copy.deepcopy(document.metadata)
                        )
                        self.enrich_chunk_metadata(header_doc, document.metadata, i)
                        self.annotate_chunk(document.page_content, header_doc)
                        self.persist_chunk(header_doc, f"DOC{i+1}_FALLBACK")
                        chunks.append(header_doc)
                    
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error splitting documents with MarkdownHeaderTextSplitter: {str(e)}")
            return []

    # Initialize and return a Markdown header text splitter
    def get_markdown_splitter(self) -> MarkdownHeaderTextSplitter:
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5")
            ],
            strip_headers=False
        )

    # Initialize and return a semantic chunker with embeddings
    def get_semantic_chunker(self) -> SemanticChunker:
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )

    # Merge document metadata into chunk and add index
    def enrich_chunk_metadata(self, chunk: Document, metadata: dict, index: int) -> None:
        chunk.metadata.update(metadata)
        if self._add_start_index:
            chunk.metadata["start_index"] = index

    # Obtain structured data and update chunk metadata
    def annotate_chunk(self, full_text: str, chunk: Document) -> None:
        structured_data = self.situate_and_annotate_chunk(full_text, chunk.page_content, chunk.metadata)
        chunk.metadata.update(structured_data)

    # Save chunk content and metadata to a file
    def persist_chunk(self, chunk: Document, label: str) -> None:
        chunks_file_path = self.persistence_dir / "latest_chunks_with_metadata.txt"
        logger.info(f"Saving chunk to {chunks_file_path}")
        with open(chunks_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{label}\n")
            f.write(f"{'='*80}\n\n")
            f.write("CONTENT:\n")
            f.write(f"{chunk.page_content}\n\n")
            f.write("METADATA:\n")
            for key, value in chunk.metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n\n")

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
        # Try to find a # header at the start of the document
        lines = content.strip().split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.startswith('# '):
                return line.replace('# ', '').strip()
        
        # If not found, try other approaches like first non-empty line
        for line in lines:
            if line.strip():
                # Return at most 80 characters to avoid overly long titles
                return line.strip()[:80]
        
        return None

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