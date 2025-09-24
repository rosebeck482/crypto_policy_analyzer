# Web Analyzer Module for Cryptocurrency Policy Analysis
# Queries policy documents with Elasticsearch vector store and graph-based search

import os
import logging
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
import hashlib

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from elasticsearch import Elasticsearch

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# GraphRAG and DSPy functionality removed for stability and simplicity

# Load environment variables
load_dotenv()

# Initialize API keys from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Elasticsearch credentials from environment variables
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
ES_USER = os.environ.get("ES_USER", "elastic")
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")
ES_INDEX_NAME = os.environ.get("ES_INDEX_NAME", "policy-index")

# Log API key availability (safely). If missing, we will fall back to a non-LLM mode.
if openai_api_key:
    logger.info(f"OPENAI_API_KEY found with length: {len(openai_api_key)}")
else:
    logger.warning("OPENAI_API_KEY not found; LLM features will be disabled, using SimpleEmbeddings.")

# Reload dotenv to ensure we have the latest values
logger.info("Reloading environment variables to ensure latest values...")
load_dotenv(override=True)

# Log Elasticsearch configuration
logger.info(f"Elasticsearch URL: {ES_URL}")
logger.info(f"Elasticsearch index: {ES_INDEX_NAME}")

INDEX_NAME = ES_INDEX_NAME


# Analyzes web content for cryptocurrency policy information
class WebAnalyzer:
    
    # Initialize WebAnalyzer with Elasticsearch connection
    def __init__(self, links_file=None, persistence_dir=None, semantic_weight=0.6, bm25_weight=0.25, graph_weight=0.15):
        
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.graph_weight = graph_weight
        
        # Initialize embeddings: OpenAI if available, otherwise local SimpleEmbeddings
        if openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=openai_api_key
            )
        else:
            from langchain.embeddings.base import Embeddings
            import numpy as np

            class SimpleEmbeddings(Embeddings):
                def __init__(self, vector_size: int = 1536):
                    self.vector_size = vector_size
                def _vec(self, text: str):
                    np.random.seed(hash(text) % (2**32))
                    v = np.random.normal(0, 1, self.vector_size)
                    v = v / np.linalg.norm(v)
                    return v.tolist()
                def embed_documents(self, texts):
                    return [self._vec(t) for t in texts]
                def embed_query(self, text):
                    return self._vec(text)

            self.embeddings = SimpleEmbeddings()
        
        # Connect to Elasticsearch
        logger.info(f"Connecting to Elasticsearch index: {ES_INDEX_NAME}")
        try:
            # Use API key if available
            es_api_key = os.environ.get("ES_API_KEY")
            if es_api_key:
                logger.info("Using API key authentication for Elasticsearch")
                # Connect to Elasticsearch for vector search with API key
                self.vectorstore = ElasticsearchStore(
                    es_url=ES_URL,
                    index_name=ES_INDEX_NAME,
                    embedding=self.embeddings,
                    es_api_key=es_api_key
                )
                
                # Initialize Elasticsearch client for direct BM25 queries
                self.es_client = Elasticsearch(
                    ES_URL,
                    api_key=es_api_key,
                    verify_certs=True
                )
            else:
                logger.info("Using username/password authentication for Elasticsearch")
                # Connect to Elasticsearch for vector search
                self.vectorstore = ElasticsearchStore(
                    es_url=ES_URL,
                    index_name=ES_INDEX_NAME,
                    embedding=self.embeddings,
                    es_user=ES_USER,
                    es_password=ES_PASSWORD
                )
                
                # Initialize Elasticsearch client for direct BM25 queries
                self.es_client = Elasticsearch(
                    ES_URL,
                    basic_auth=(ES_USER, ES_PASSWORD),
                    verify_certs=True
                )
                
            logger.info(f"Successfully connected to Elasticsearch index: {ES_INDEX_NAME}")
            
            # Create a retriever with similarity search
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 15}  # Get more results for hybrid reranking
            )
            
            # Set up embeddings filter for contextual compression
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.76  # Set threshold for relevance
            )
            
            # Create compression retriever to filter out irrelevant results
            self.compression_retriever = ContextualCompressionRetriever(
                base_retriever=self.retriever,
                base_compressor=embeddings_filter
            )
            
            # Initialize the LLM only if API key available; otherwise, use non-LLM fallback
            self.llm = None
            if openai_api_key:
                try:
                    self.llm = ChatOpenAI(
                        model="gpt-4o",
                        temperature=0,
                        openai_api_key=openai_api_key
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize ChatOpenAI; non-LLM fallback will be used. Error: {e}")
            
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {str(e)}")
            raise
    
    # Format a list of documents into a single string
    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Perform BM25 search against Elasticsearch
    def bm25_search(self, query: str, k: int = 15) -> List[Document]:
        """
        Perform a BM25 search directly against the Elasticsearch index.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Documents
        """
        # Use the same index as the vector store but with a BM25 query
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"],  # Adjust field name if different in your index
                    "type": "best_fields"
                }
            },
            "size": k
        }
        
        try:
            response = self.es_client.search(index=ES_INDEX_NAME, query=search_body["query"], size=k)
            
            # Convert Elasticsearch hits to Document objects
            documents = []
            for hit in response["hits"]["hits"]:
                # Extract content and metadata, adjusting field names as needed for your index
                content = hit["_source"].get("text", "")
                metadata = {k: v for k, v in hit["_source"].items() if k != "text"}
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def chunk_to_content(self, doc: Document) -> str:
        """
        Format a document for hybrid retrieval by combining original content with contextualized content if available.
        
        Args:
            doc: Document to format
            
        Returns:
            Formatted content string
        """
        if "original_content" in doc.metadata and "contextualized_content" in doc.metadata:
            return f"{doc.metadata['original_content']}\n\nContext: {doc.metadata['contextualized_content']}"
        elif "contextualized_content" in doc.metadata:
            return f"{doc.page_content}\n\nContext: {doc.metadata['contextualized_content']}"
        else:
            return doc.page_content
    
    # Graph-based retrieval removed; using hybrid (semantic + BM25) only
    
    def hybrid_retrieval(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform hybrid retrieval (semantic + BM25) with weighted scoring.
        
        Args:
            query: The search query
            k: Number of final documents to retrieve
            
        Returns:
            List of retrieved and ranked documents
        """
        # This method is kept unchanged for backward compatibility
        # Get semantic search results
        candidate_k = min(50, k * 5)  # Get more candidates for scoring
        
        try:
            semantic_docs = self.retriever.get_relevant_documents(query, k=candidate_k)
            logger.info(f"Retrieved {len(semantic_docs)} documents from semantic search")
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            semantic_docs = []
        
        # Get BM25 search results
        try:
            bm25_docs = self.bm25_search(query, k=candidate_k)
            logger.info(f"Retrieved {len(bm25_docs)} documents from BM25 search")
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            bm25_docs = []
        
        # If one method fails completely, use results from the other
        if not semantic_docs and not bm25_docs:
            logger.error("Both retrieval methods failed. No documents to return.")
            return []
        
        # Combine and deduplicate results
        unique_docs = {}
        def _stable_id(text: str) -> str:
            return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()
        
        # Add all documents to unique_docs dictionary using content hash as key
        for doc in semantic_docs + bm25_docs:
            doc_id = _stable_id(doc.page_content)
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
        
        combined_docs = list(unique_docs.values())
        logger.info(f"Combined {len(combined_docs)} unique documents for ranking")
        
        # Score and rank documents using the combined reciprocal rank method
        doc_scores = {}
        
        # Score semantic docs
        for i, doc in enumerate(semantic_docs):
            doc_id = _stable_id(doc.page_content)
            score = self.semantic_weight * (1.0 / (i + 1))  # Reciprocal rank
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        
        # Score BM25 docs
        for i, doc in enumerate(bm25_docs):
            doc_id = _stable_id(doc.page_content)
            score = self.bm25_weight * (1.0 / (i + 1))  # Reciprocal rank
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        
        # Sort by score
        sorted_docs = sorted(
            combined_docs,
            key=lambda doc: doc_scores.get(_stable_id(doc.page_content), 0),
            reverse=True
        )
        
        return sorted_docs[:k]
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a query against the vector store using standard RAG (semantic + BM25 retrieval).
        
        Args:
            question: Query string
        
        Returns:
            Dictionary with query results and analysis
        """
        logger.info(f"Processing query with standard RAG: {question}")
        return self._process_query_standard(question)
    
    def _process_query_standard(self, query: str) -> Dict[str, Any]:
        """
        Process a query against the vector store using hybrid retrieval (semantic + BM25).
        
        Args:
            query: Query string
        
        Returns:
            Dictionary with query results and analysis
        """
        logger.info(f"Processing query: {query}")
        
        # RAG prompt template
        prompt_template = """You are an expert in cryptocurrency regulations and policy.
        Answer the question based ONLY on the provided context. If you can't find the answer 
        in the context, say "I don't have enough information about that." Do not make up or 
        infer information that's not explicitly stated in the context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create RAG chain with cached document retrieval
        def retrieve_and_cache(q):
            nonlocal cached_docs
            cached_docs = self.hybrid_retrieval(q)
            logger.info("Using hybrid retrieval (semantic + BM25)")
            return cached_docs
        
        # Cache for storing retrieved documents
        cached_docs = []
        
        # Build chain when LLM is available; otherwise, fall back to a deterministic answer
        if self.llm is not None:
            rag_chain = (
                {"context": RunnableLambda(retrieve_and_cache) | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            rag_chain = None
        
        # Process the query
        try:
            if rag_chain is not None:
                result = rag_chain.invoke(query)
            else:
                # Fallback: return top passage snippets
                if not cached_docs:
                    result = "No relevant documents found and no LLM available."
                else:
                    snippet = "\n\n".join([d.page_content[:400] for d in cached_docs[:2]])
                    result = f"LLM unavailable. Top passages:\n\n{snippet}"
            logger.info("Successfully processed query")
            
            # The documents are already in cached_docs from the chain execution
            relevant_docs = cached_docs
            
            # Format the response
            response = {
                "query": query,
                "answer": result,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "source_count": len(relevant_docs)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "source_count": 0
            }
    
    def process_query_classic(self, query: str) -> Dict[str, Any]:
        """
        Process a query using only semantic search (original method).
        For backward compatibility.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with query results and analysis
        """
        logger.info(f"Processing query with classic method: {query}")
        
        # RAG prompt template
        prompt_template = """You are an expert in cryptocurrency regulations and policy.
        Answer the question based ONLY on the provided context. If you can't find the answer 
        in the context, say "I don't have enough information about that." Do not make up or 
        infer information that's not explicitly stated in the context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create RAG chain (original method)
        if self.llm is not None:
            rag_chain = (
                {"context": self.compression_retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            rag_chain = None
        
        # Process the query
        try:
            if rag_chain is not None:
                result = rag_chain.invoke(query)
            else:
                docs = self.compression_retriever.get_relevant_documents(query)
                if not docs:
                    result = "No relevant documents found and no LLM available."
                else:
                    snippet = "\n\n".join([d.page_content[:400] for d in docs[:2]])
                    result = f"LLM unavailable. Top passages:\n\n{snippet}"
            logger.info("Successfully processed query")
            
            # Get the relevant documents for transparency
            relevant_docs = self.compression_retriever.get_relevant_documents(query)
            
            # Format the response
            response = {
                "query": query,
                "answer": result,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "source_count": len(relevant_docs)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "source_count": 0
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the analyzer.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Simple query to check if Elasticsearch is working
            test_result = self.vectorstore.similarity_search("test", k=1)
            
            # Try a BM25 query
            bm25_working = True
            try:
                bm25_docs = self.bm25_search("test", k=1)
            except Exception:
                bm25_working = False
            
            status = {
                "status": "healthy",
                "elasticsearch": "connected",
                "vector_search": "operational",
                "bm25_search": "operational" if bm25_working else "unavailable",
                "index": ES_INDEX_NAME,
                "message": "WebAnalyzer is operational with hybrid retrieval"
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "elasticsearch": "disconnected",
                "index": ES_INDEX_NAME,
                "message": f"Error: {str(e)}"
            }
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query wrapper for backward compatibility with the Flask application.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with answer, documents, and metadata
        """
        logger.info(f"Query received: {question}")
        
        try:
            # Process the query using the hybrid approach
            result = self.process_query(question)
            
            # Format the response for the Flask app
            return {
                "answer": result["answer"],
                "documents": [source["content"] for source in result["sources"]],
                "metadata": [source["metadata"] for source in result["sources"]]
            }
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "documents": [],
                "metadata": []
            }
    
    def find_document_by_content(self, passage_text: str) -> Optional[Document]:
        """
        Find a document by matching its content to a given passage of text.
        
        Args:
            passage_text: A string passage of text to find in the document collection
            
        Returns:
            A Document object if found, None otherwise
        """
        if not passage_text or len(passage_text) < 20:
            logger.warning("Passage text too short for reliable matching")
            return None
            
        try:
            logger.info(f"Searching for document with passage text: {passage_text[:50]}...")
            
            # Create a match phrase query to search for the text within first 200 chars
            match_query = {
                "match_phrase": {
                    "text": {
                        "query": passage_text[:200],
                        "slop": 5
                    }
                }
            }
            
            # Execute the search
            response = self.es_client.search(
                index=INDEX_NAME,
                query=match_query,
                size=1  # Just get the top result
            )
            
            # Check if we found a match
            if response["hits"]["total"]["value"] > 0:
                hit = response["hits"]["hits"][0]
                logger.info(f"Found matching document with score: {hit['_score']}")
                
                # Extract the content and metadata
                content = hit["_source"].get("text", "")
                metadata = {k: v for k, v in hit["_source"].items() if k != "text"}
                
                # Create a Document object
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                return doc
            else:
                logger.info("No matching document found for the passage")
                return None
                
        except Exception as e:
            logger.error(f"Error finding document by content: {str(e)}")
            return None
