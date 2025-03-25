# Licensed under the MIT License
# Reference:
# - LightRag (https://github.com/HKUDS/LightRAG)
# - MiniRAG (https://github.com/HKUDS/MiniRAG)

import os
import logging
import json
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPTS = {}

PROMPTS["Refactored_minirag_query2kwd"] = """---Role---

You are a helpful assistant tasked with identifying both answer-type and low-level keywords in a user's query related to cryptocurrency and blockchain regulation.

---Goal---

Given the query, list both answer-type and low-level keywords.

- `answer_type_keywords` focus on the *type* of answer the user is likely looking for (e.g., a law, regulatory body, case, or technology). These must be selected from the *Answer type pool* below.
- `entities_from_query` are specific entities or terms explicitly mentioned in the query (e.g., SEC, Ripple, staking rewards, July 2023).

---Instructions---

- Output the keywords in **JSON** format with exactly **two keys**:
  - `"answer_type_keywords"`: A list of up to 3 high-likelihood answer types from the Answer Type Pool.
  - `"entities_from_query"`: A list of specific low-level entities or terms from the query.

---Answer Type Pool---

Use this updated answer type pool to guide your categorization:

{{
  "LAW_OR_REGULATION": ["SECURITIES LAW", "TAX LAW", "ANTI-MONEY LAUNDERING", "MONEY TRANSMISSION LAW", "ESTATE LAW", "REPORTING REQUIREMENT"],
  "ORGANIZATION": ["SEC", "CFTC", "FINCEN", "IRS", "OCC", "OFAC", "DOJ", "TREASURY"],
  "LEGISLATION": ["RESPONSIBLE FINANCIAL INNOVATION ACT", "DIGITAL COMMODITIES CONSUMER PROTECTION ACT", "TOOMEY STABLECOIN BILL", "MCHENRY-THOMPSON BILL"],
  "CASE_LAW": ["SEC v. RIPPLE", "SEC v. TELEGRAM", "SEC v. KIK", "SEC v. TERRAFORM LABS"],
  "ENTITY": ["RIPPLE", "COINBASE", "TERRAFORM LABS", "PROMETHEUM", "KRAKEN", "NEXO"],
  "TECHNOLOGY": ["STABLECOINS", "DECENTRALIZED FINANCE", "SMART CONTRACTS", "PROOF-OF-STAKE", "CBDC", "WALLET"],
  "FINANCIAL_CONCEPT": ["TOKEN OFFERING", "STAKING", "CAPITAL GAINS", "INCOME REPORTING", "LIKE-KIND EXCHANGE"],
  "GOVERNMENT_ACTION": ["EXECUTIVE ORDER", "ENFORCEMENT ACTION", "SANCTION", "SETTLEMENT", "HEARING", "REGULATORY CLARITY"],
  "DATE AND TIME": ["JANUARY 2024", "JULY 13, 2023", "2022", "MARCH 2023"],
  "LOCATION": ["WYOMING", "NEW YORK", "CALIFORNIA", "FLORIDA", "UTAH"],
  "PLAN": ["REGULATORY FRAMEWORK", "COMPLIANCE PERIOD"],
  "BEHAVIOR": ["MARKET MANIPULATION", "INSIDER TRADING"]
}}

---Examples---

Example 1:

Query: "What was the rationale used by the federal court to deny Terraform Labs' motion to dismiss based on the Ripple ruling?"

Output:
{{
  "answer_type_keywords": ["CASE_LAW", "LAW_OR_REGULATION", "ENTITY"],
  "entities_from_query": ["Terraform Labs", "Ripple", "motion to dismiss", "Southern District of New York", "Howey Test"]
}}

---

Example 2:

Query: "What obligations do stablecoin issuers have under the Toomey Stablecoin Bill?"

Output:
{{
  "answer_type_keywords": ["LEGISLATION", "LAW_OR_REGULATION", "TECHNOLOGY"],
  "entities_from_query": ["Toomey Stablecoin Bill", "stablecoin issuers", "payment stablecoins", "regulatory obligations"]
}}

---

Example 3:

Query: "How did the SEC justify its enforcement action against Ripple in 2020?"

Output:
{{
  "answer_type_keywords": ["GOVERNMENT_ACTION", "CASE_LAW", "ORGANIZATION"],
  "entities_from_query": ["SEC", "Ripple", "enforcement action", "unregistered securities", "2020"]
}}

---

Example 4:

Query: "What does the Infrastructure Investment and Jobs Act require from digital asset brokers?"

Output:
{{
  "answer_type_keywords": ["LEGISLATION", "LAW_OR_REGULATION"],
  "entities_from_query": ["Infrastructure Investment and Jobs Act", "digital asset brokers", "reporting requirement", "IRS"]
}}

---

Example 5:

Query: "What is the IRS's position on staking rewards?"

Output:
{{
  "answer_type_keywords": ["LAW_OR_REGULATION", "FINANCIAL_CONCEPT", "ORGANIZATION"],
  "entities_from_query": ["IRS", "staking rewards", "Rev. Rul. 2023-14", "gross income"]
}}

---

-Real Data-
######################
Query: {query}
Answer type pool: {{Answer Type Pool}}
######################
Output:
"""


PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
]

# Class to handle advanced prompting techniques for the WebAnalyzer
class WebAnalyzerPrompts:
    
    # Initialize the WebAnalyzerPrompts class
    def __init__(self):
        from web_analyzer import WebAnalyzer
        
        # Initialize the WebAnalyzer for retrieval operations
        self.analyzer = WebAnalyzer()
        
        # Get the OpenAI API key from the environment
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OpenAI API key not found in environment variables")
    
    def generate_next_hop_query(self, original_question, current_context, hop_number):
        """Generate the next hop query using both context and keyword extraction."""
        # First, extract keywords from the original question
        keyword_prompt = PROMPTS["keywords_extraction"].format(
            examples="\n".join(PROMPTS["keywords_extraction_examples"]),
            query=original_question
        )
        
        # Use the analyzer's LLM to generate the keywords
        keywords_response = self.analyzer.llm.invoke(keyword_prompt).content
        
        try:
            keywords = json.loads(keywords_response)
            high_level_keywords = keywords.get("high_level_keywords", [])
            low_level_keywords = keywords.get("low_level_keywords", [])
        except Exception as e:
            logger.error(f"Error parsing keywords JSON: {str(e)}")
            high_level_keywords = []
            low_level_keywords = []
        
        # Now, generate the sub-query with awareness of keywords
        sub_query_prompt = (
            "You are an expert in cryptocurrency policy analysis. "
            "Based on the current context and the original question, "
            "suggest a refined sub-query to further explore details that might help answer the overall question.\n\n"
            f"Original Question: {original_question}\n"
            f"Context so far: {current_context}\n"
            f"High-level concepts from original question: {', '.join(high_level_keywords)}\n"
            f"Specific entities from original question: {', '.join(low_level_keywords)}\n"
            f"Current hop number: {hop_number + 1}\n\n"
            "Create a focused sub-query that explores an aspect of the original question "
            "not fully addressed by the current context. The sub-query should be clear and specific."
            "Sub-Query:"
        )
        
        return self.analyzer.llm.invoke(sub_query_prompt).content.strip()
    
    def deduplicate_documents(self, documents):
        """Remove duplicate documents from a list based on content similarity."""
        if not documents:
            return []
        
        unique_docs = []
        seen_contents = set()
        
        for doc in documents:
            # Create a simplified representation of the document content
            content_hash = hash(doc.page_content[:100])
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def process_query_advanced(self, question):
        """Process a query using DSPy Multi-Hop RAG for advanced analysis"""
        from dspy_multihop import MultiHopChainOfThoughtRAG
        import dspy
        
        # Create a DSPy-compatible LLM adapter
        dspy_llm = dspy.OpenAI(model="gpt-4o-mini", api_key=self.openai_api_key)
        
        # Define the retriever function that DSPy will use
        def retriever_func(query, k):
            # Use the existing hybrid retrieval function
            docs = self.analyzer.hybrid_rerank_retrieve(query, k=k)
            # Return just the text content for DSPy
            return [doc.page_content for doc in docs]
        
        # Create the multi-hop RAG processor
        multihop_rag = MultiHopChainOfThoughtRAG(
            llm=dspy_llm,
            retriever_func=retriever_func,
            passages_per_hop=4,
            max_hops=2
        )
        
        # Process the query with the DSPy multi-hop module
        result = multihop_rag.process(question)
        
        # Convert the result back to the standard format
        # Map the context back to document objects with metadata
        docs = []
        for passage in result["all_passages"]:
            # Find the matching document for this passage
            matching_doc = self.analyzer.find_document_by_content(passage)
            if matching_doc:
                docs.append(matching_doc)
        
        # Remove duplicates
        unique_docs = []
        doc_ids = set()
        for doc in docs:
            doc_id = hash(doc.page_content[:100])
            if doc_id not in doc_ids:
                doc_ids.add(doc_id)
                unique_docs.append(doc)
        
        # Format the response to match the expected structure
        response = {
            "query": question,
            "sub_queries": result["sub_queries"],
            "answer": result["answer"],
            "sources": [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in unique_docs
            ],
            "source_count": len(unique_docs)
        }
        
        return response

def process_query(self, question, advanced_mode=False):
    """
    Process a user query using either standard or advanced RAG
    
    Args:
        question: The user's question
        advanced_mode: Whether to use multi-hop RAG (True) or standard RAG (False)
    
    Returns:
        Dict containing the answer and sources
    """
    if advanced_mode:
        return self.process_query_advanced(question)
    else:
        # Your existing query processing logic
        return self.existing_process_query(question)
