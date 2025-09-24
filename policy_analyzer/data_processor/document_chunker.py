# handles document fetching from URLs and chunking the content
import os
import logging
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.embeddings.base import Embeddings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
URL_TIMEOUT = 30  # seconds

#Convert a URL to Markdown format using a DocumentConverter.
def url_to_markdown(converter, url, timeout=30):
    try:
        # Docling does not expose timeout directly; ensure upstream HTTP client is configured.
        # Here we call convert and rely on external network timeouts.
        result = converter.convert(url)
        return result.document.export_to_markdown()
    except Exception as e:
        logger.error(f"Error in url_to_markdown: {str(e)}")
        return ""

# Fetch and convert a document from a URL.
def fetch_document_from_url(url: str, url_id: Optional[str] = None) -> Optional[Document]:
    try:
        logger.info(f"Fetching document from URL: {url}")
        
        # Extract domain for metadata
        domain = urlparse(url).netloc
        if not domain:
            domain = "file"  # For local files
        
        # Ensure we have an ID for tracking/metadata
        if url_id is None:
            url_id = str(hash(url))
        
        # Use DocumentConverter to fetch and convert the content
        converter = DocumentConverter()
        content = url_to_markdown(converter, url, timeout=URL_TIMEOUT)
        
        if not content:
            logger.error(f"Failed to extract content from {url}")
            return None
        
        # Basic document metadata
        metadata = {
            "source": url,
            "domain": domain,
            "url_id": url_id,
            "extraction_time": datetime.now().isoformat(),
            "title": extract_title(content) or domain
        }
        logger.info(f"Extracted {len(content)} characters of content")
        
        # Create a document from the content
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        
        return doc
        
    except Exception as e:
        logger.error(f"Error fetching document from URL {url}: {str(e)}")
        return None

# Extract title from markdown content.
def extract_title(content: str) -> Optional[str]:
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

# Split a document into chunks based on Markdown headers.
def split_document_by_headers(document: Document, 
                             persistence_dir: Optional[Path] = None,
                             embeddings: Optional[Embeddings] = None,
                             max_chunk_size: int = 2000) -> List[Document]:
    try:
        logger.info(f"Splitting document with {len(document.page_content)} characters")
        chunks = []
        
        # Create splitter
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5")
            ],
            strip_headers=False
        )
        
        # Split text into header-based chunks
        try:
            # Make sure we're passing a string, not a Document object
            page_content = document.page_content
            if not isinstance(page_content, str):
                logger.warning(f"Document page_content is not a string, it's a {type(page_content)}. Converting to string.")
                page_content = str(page_content)
                
            # The split_text method returns a list of either strings or Documents
            header_chunks = splitter.split_text(page_content)
            logger.info(f"Markdown header splitter created {len(header_chunks)} chunks")
            
            # Process each header chunk
            for i, header_chunk in enumerate(header_chunks):
                # Check if header_chunk is already a Document object
                if isinstance(header_chunk, Document):
                    # Use the page_content from the Document
                    chunk_content = header_chunk.page_content
                    # Merge the header metadata with our document metadata
                    chunk_metadata = copy.deepcopy(document.metadata)
                    chunk_metadata.update(header_chunk.metadata)
                else:
                    # If it's a string, use it directly
                    chunk_content = header_chunk
                    chunk_metadata = copy.deepcopy(document.metadata)
                
                # Create a new document with the header chunk content
                chunk_doc = Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                )
                
                # Add index to metadata
                chunk_doc.metadata['chunk_index'] = i
                
                # Save chunk to file for inspection
                if persistence_dir:
                    save_chunk_to_file(persistence_dir, chunk_doc)
                
                # Add to chunks
                chunks.append(chunk_doc)
            
            logger.info(f"Split document into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting document by headers: {str(e)}")
            # Return document as a single chunk if splitting fails
            document.metadata['chunk_index'] = 0
            return [document]
        
    except Exception as e:
        logger.error(f"Error in split_document_by_headers: {str(e)}")
        return [document]

# Fetch the content from a URL and return it as markdown.
def fetch_document_content(url: str) -> str:
    try:
        converter = DocumentConverter()
        content = url_to_markdown(converter, url, timeout=URL_TIMEOUT)
        return content
    except Exception as e:
        logger.error(f"Error fetching document content from {url}: {str(e)}")
        return ""

# Save a chunk to a file in the persistence directory.
def save_chunk_to_file(persistence_dir: Path, chunk: Document):
    try:
        # Create latest_chunks_with_metadata.txt file if it doesn't exist
        chunks_file = persistence_dir / "latest_chunks_with_metadata.txt"
        
        with open(chunks_file, 'a', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write("CHUNK\n")
            f.write(f"{'='*80}\n\n")
            
            # Write chunk content
            f.write("CONTENT:\n")
            f.write(f"{chunk.page_content}\n\n")
            
            # Write chunk metadata
            f.write("METADATA:\n")
            for key, value in chunk.metadata.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n\n")
    except Exception as e:
        logger.error(f"Error saving chunk to file: {str(e)}")

#Process a URL to fetch content and split into document chunks.
def process_url_to_chunks(url: str, 
                         persistence_dir: Optional[Path] = None,
                         embeddings: Optional[Embeddings] = None,
                         max_chunk_size: int = 2000) -> List[Document]:
    try:
        # Fetch document
        document = fetch_document_from_url(url)
        if not document:
            logger.error(f"Failed to fetch document from {url}")
            return []
        
        # Split document by headers
        return split_document_by_headers(document, persistence_dir)
    except Exception as e:
        logger.error(f"Error processing URL to chunks: {str(e)}")
        return [] 
