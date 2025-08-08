"""
Ultra-lightweight RAG Pipeline for deployment on Render
Uses simple TF-IDF instead of heavy ML embeddings
"""

import re
import os
import logging
import math
from typing import List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv
from collections import Counter

logger = logging.getLogger(__name__)
load_dotenv()

class LightRAGPipeline:
    """Ultra-lightweight RAG using keyword matching and simple scoring."""
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Groq client."""
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.documents = []
        self.doc_word_counts = []
        self.vocabulary = set()
        
    def simple_text_splitter(self, text: str, chunk_size: int = 600, overlap: int = 50) -> List[str]:
        """Simple text splitter optimized for memory usage."""
        # Clean text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Skip very short or empty sentences
            if len(sentence.strip()) < 10:
                continue
                
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Limit total chunks to prevent memory issues
        if len(chunks) > 50:  # Limit to 50 chunks max
            chunks = chunks[:50]
            
        return chunks
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization without external libraries."""
        # Convert to lowercase and split on non-alphanumeric
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter out very short words and common stop words
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                     'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they', 'have'}
        return [w for w in words if len(w) > 2 and w not in stop_words]
    
    def calculate_tf_idf(self, query_words: List[str]) -> List[float]:
        """Simple TF-IDF calculation without scikit-learn."""
        scores = []
        total_docs = len(self.documents)
        
        for doc_words in self.doc_word_counts:
            score = 0
            doc_length = len(doc_words)
            
            if doc_length == 0:
                scores.append(0)
                continue
                
            for word in query_words:
                if word in doc_words:
                    # Term frequency
                    tf = doc_words[word] / doc_length
                    
                    # Document frequency
                    df = sum(1 for doc in self.doc_word_counts if word in doc)
                    
                    # IDF
                    idf = math.log(total_docs / max(1, df))
                    
                    # TF-IDF score
                    score += tf * idf
            
            scores.append(score)
        
        return scores
    
    def process_documents(self, pdf_texts: List[str]) -> None:
        """Process documents using simple keyword indexing."""
        logger.info("Processing documents with ultra-lightweight pipeline...")
        
        # Simple text splitting
        all_chunks = []
        for text in pdf_texts:
            chunks = self.simple_text_splitter(text)
            all_chunks.extend(chunks)
        
        self.documents = all_chunks
        logger.info(f"Created {len(all_chunks)} text chunks")
        
        # Create word count vectors
        self.doc_word_counts = []
        for doc in self.documents:
            words = self.simple_tokenize(doc)
            word_count = Counter(words)
            self.doc_word_counts.append(word_count)
            self.vocabulary.update(words)
        
        logger.info("Created keyword index for document retrieval")
    
    def retrieve_documents(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve documents using simple TF-IDF scoring with memory optimization."""
        if not self.documents:
            return []
        
        # Tokenize query
        query_words = self.simple_tokenize(query)
        
        if not query_words:
            return self.documents[:min(n_results, 2)]  # Return fewer if no valid query words
        
        # Calculate scores
        scores = self.calculate_tf_idf(query_words)
        
        # Get top documents
        doc_scores = list(zip(scores, self.documents))
        doc_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Filter out very low scores and return documents
        relevant_docs = []
        for score, doc in doc_scores[:n_results]:
            if score > 0.01:  # Threshold for relevance
                relevant_docs.append(doc)
        
        # If no relevant docs found, return top documents anyway (but fewer)
        if not relevant_docs:
            relevant_docs = [doc for _, doc in doc_scores[:min(2, len(doc_scores))]]
        
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        return relevant_docs
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Groq API with optimized prompt."""
        if not context.strip():
            return "No relevant information found in the documents."
        
        # Keep prompt short and focused
        prompt = f"""Answer based on the context below:

Question: {query}

Context: {context}

Answer (max 100 words):"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Fastest model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,  # Reduced from 150
                temperature=0.3  # Lower temperature for consistency
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error: Unable to generate answer"
    
    def answer_query(self, query: str, n_results: int = 3) -> Dict[str, str]:
        """Complete pipeline to answer a query with memory optimization."""
        logger.info(f"Processing query: {query[:50]}...")
        
        # Retrieve relevant documents (reduced from 5 to 3 for memory)
        docs = self.retrieve_documents(query, n_results)
        
        # Limit context size to prevent memory issues
        context_parts = []
        total_length = 0
        max_context_length = 2000  # Limit context size
        
        for doc in docs:
            if total_length + len(doc) > max_context_length:
                # Truncate the last document if needed
                remaining = max_context_length - total_length
                if remaining > 100:  # Only add if meaningful length remains
                    context_parts.append(doc[:remaining] + "...")
                break
            context_parts.append(doc)
            total_length += len(doc)
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        return {
            "query": query,
            "answer": answer,
            "num_docs_retrieved": len(docs),
            "context_length": len(context)
        }
    
    def process_multiple_queries(self, queries: List[str]) -> List[Dict[str, str]]:
        """Process multiple queries efficiently."""
        results = []
        for query in queries:
            result = self.answer_query(query)
            results.append(result)
        return results
