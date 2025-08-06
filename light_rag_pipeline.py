"""
Ultra-lightweight RAG Pipeline for deployment on Render
Uses simple TF-IDF instead of heavy ML embeddings
"""

import re
import os
import logging
from typing import List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

logger = logging.getLogger(__name__)
load_dotenv()

class LightRAGPipeline:
    """Ultra-lightweight RAG using TF-IDF instead of embeddings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Groq client."""
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit features for memory
            stop_words='english',
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        self.doc_vectors = None
        
    def simple_text_splitter(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Simple text splitter without heavy dependencies."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def process_documents(self, pdf_texts: List[str]) -> None:
        """Process documents using lightweight text splitting and TF-IDF."""
        logger.info("Processing documents with lightweight pipeline...")
        
        # Simple text splitting
        all_chunks = []
        for text in pdf_texts:
            chunks = self.simple_text_splitter(text)
            all_chunks.extend(chunks)
        
        self.documents = all_chunks
        logger.info(f"Created {len(all_chunks)} text chunks")
        
        # Create TF-IDF vectors
        if self.documents:
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
            logger.info("Created TF-IDF vectors for document retrieval")
    
    def retrieve_documents(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieve documents using TF-IDF similarity."""
        if not self.documents or self.doc_vectors is None:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top documents
        top_indices = similarities.argsort()[-n_results:][::-1]
        
        # Filter out very low similarity scores
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                relevant_docs.append(self.documents[idx])
        
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        return relevant_docs
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Groq API."""
        if not context.strip():
            return "No relevant information found in the documents."
        
        prompt = f"""You are an expert insurance policy analyst. Answer the question based ONLY on the provided context.

Question: {query}

Context from policy documents:
{context}

Instructions:
- Extract specific details, numbers, percentages, time periods directly from the context
- Be precise and concise (maximum 2 sentences)
- If information is not in the context, say "Information not available in provided documents"

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Faster model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150  # Limit response length
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def answer_query(self, query: str, n_results: int = 5) -> Dict[str, str]:
        """Complete pipeline to answer a query."""
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        docs = self.retrieve_documents(query, n_results)
        context = "\n\n".join(docs)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        return {
            "query": query,
            "answer": answer,
            "num_docs_retrieved": len(docs)
        }
    
    def process_multiple_queries(self, queries: List[str]) -> List[Dict[str, str]]:
        """Process multiple queries efficiently."""
        results = []
        for query in queries:
            result = self.answer_query(query)
            results.append(result)
        return results
