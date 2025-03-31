"""
Semantic Metadata Extractor

This module handles the extraction of semantic metadata from document chunks
using OpenAI's API to enhance document understanding and searchability.
"""

import json
import time
import logging
import os
from typing import Dict, Any, Optional

from openai import OpenAI

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables")
    OPENAI_AVAILABLE = False
    client = None
else:
    logger.info("OPENAI_API_KEY found - using it for metadata extraction")
    OPENAI_AVAILABLE = True
    client = OpenAI(api_key=api_key)

# Extract structured and semantic metadata from a document chunk using OpenAI API.
def extract_semantic_metadata(full_document_markdown: str, chunk_markdown: str, metadata: dict) -> dict:
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

def process_chunks_with_semantic_metadata(chunks, full_document_markdown):
    """
    Process a list of document chunks to add semantic metadata using OpenAI.
    
    Args:
        chunks (list): List of document chunks (Document objects with page_content and metadata)
        full_document_markdown (str): The full document content for context
        
    Returns:
        list: The same chunks with enhanced metadata
    """
    logger.info(f"Enhancing {len(chunks)} chunks with semantic metadata")
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        semantic_data = extract_semantic_metadata(full_document_markdown, chunk.page_content, chunk.metadata)
        chunk.metadata.update(semantic_data)
        
        # Add a small delay between API calls to avoid rate limits
        if i < len(chunks) - 1:
            time.sleep(1)
    
    logger.info(f"Completed semantic metadata extraction for {len(chunks)} chunks")
    return chunks 