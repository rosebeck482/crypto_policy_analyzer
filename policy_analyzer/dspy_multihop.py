#Multi-hop Chain-of-Thought RAG implementation using DSPy.


import logging
from typing import List, Dict, Any, Callable, Optional, Union

import dspy
from dspy.signatures import SignatureRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define signatures for the multi-hop RAG components
class GenerateSubQuery(dspy.Signature):
    """Generate a sub-query based on the original question and current context."""
    
    question: str = dspy.InputField(desc="The original complex question")
    context: str = dspy.InputField(desc="The current context and previously retrieved information")
    
    sub_query: str = dspy.OutputField(desc="A focused sub-query that helps answer part of the original question")
    
class SynthesizeAnswer(dspy.Signature):
    """Synthesize a final answer from all retrieved information."""
    
    question: str = dspy.InputField(desc="The original complex question")
    sub_queries: List[str] = dspy.InputField(desc="The list of sub-queries that were asked")
    contexts: List[str] = dspy.InputField(desc="The list of contexts retrieved for each sub-query")
    
    answer: str = dspy.OutputField(desc="A comprehensive answer to the original question based on all contexts")

class MultiHopChainOfThoughtRAG:
    def __init__(self, 
                 llm: Any, 
                 retriever_func: Callable,
                 passages_per_hop: int = 3,
                 max_hops: int = 2):
        self.llm = llm
        self.retriever_func = retriever_func
        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops
        
        # Initialize DSPy modules
        self.generate_sub_query = dspy.Predict(GenerateSubQuery, llm=self.llm)
        self.synthesize_answer = dspy.Predict(SynthesizeAnswer, llm=self.llm)
        
        logger.info(f"Initialized MultiHopChainOfThoughtRAG with max_hops={max_hops}, passages_per_hop={passages_per_hop}")
    
    def process(self, question: str) -> Dict[str, Any]:
        logger.info(f"Processing question with multi-hop RAG: {question}")
        
        # Initialize tracking variables
        all_sub_queries = []
        all_contexts = []
        all_passages = []
        accumulated_context = ""
        
        # First hop is directly from the question
        current_query = question
        
        # Perform multi-hop reasoning
        for hop in range(self.max_hops):
            logger.info(f"Hop {hop+1}: Query = {current_query}")
            
            # Retrieve documents for the current query
            retrieved_passages = self.retriever_func(current_query, k=self.passages_per_hop)
            
            # Update tracking
            if hop > 0:  # First hop isn't a sub-query
                all_sub_queries.append(current_query)
            
            # Combine retrieved passages into a context
            current_context = "\n\n".join(retrieved_passages)
            all_contexts.append(current_context)
            all_passages.extend(retrieved_passages)
            
            # Update accumulated context
            if accumulated_context:
                accumulated_context += f"\n\nNew Information:\n{current_context}"
            else:
                accumulated_context = current_context
            
            # For the last hop, don't generate a new sub-query
            if hop == self.max_hops - 1:
                break
                
            # Generate the next sub-query based on accumulated context
            try:
                sub_query_result = self.generate_sub_query(
                    question=question,
                    context=accumulated_context
                )
                current_query = sub_query_result.sub_query
                logger.info(f"Generated sub-query: {current_query}")
            except Exception as e:
                logger.error(f"Error generating sub-query: {str(e)}")
                break
        
        # Synthesize the final answer
        try:
            synthesis_result = self.synthesize_answer(
                question=question,
                sub_queries=all_sub_queries, 
                contexts=all_contexts
            )
            answer = synthesis_result.answer
        except Exception as e:
            logger.error(f"Error synthesizing answer: {str(e)}")
            answer = "I don't have enough information to answer that question completely based on the available documents."
        
        # Return the complete result
        return {
            "query": question,
            "sub_queries": all_sub_queries,
            "answer": answer,
            "all_contexts": all_contexts,
            "all_passages": all_passages
        } 
