from pypdf import PdfReader
import os
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class RAGPipeline:
    """
    A comprehensive RAG (Retrieval-Augmented Generation) pipeline for PDF document analysis.
    
    This pipeline processes PDFs, creates vector embeddings, and answers queries using
    advanced techniques like query expansion with hypothetical answer generation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the RAG pipeline with Groq client and embedding function."""
        self.client = Groq(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
        )
        self.chroma_collection = None
        self.embedding_function = None
        
    def _setup_embeddings(self):
        """Set up ChromaDB and embedding function."""
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        
        self.embedding_function = SentenceTransformerEmbeddingFunction()
        chroma_client = chromadb.Client()
        
        # Create a unique collection name to avoid conflicts
        collection_name = f"document-collection-{hash(str(os.getcwd())) % 10000}"
        
        try:
            self.chroma_collection = chroma_client.create_collection(
                collection_name, embedding_function=self.embedding_function
            )
        except Exception as e:
            # If collection exists, get it
            self.chroma_collection = chroma_client.get_collection(
                collection_name
            )
            logger.info(f"Using existing collection: {collection_name}")
    
    def load_pdfs_from_folder(self, folder_path: str) -> List[str]:
        """
        Load and extract text from all PDF files in a folder.
        
        Args:
            folder_path: Path to folder containing PDF files
            
        Returns:
            List of extracted text from all pages
        """
        pdf_texts = []
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {folder_path}")
            
        logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        for filename in pdf_files:
            pdf_path = os.path.join(folder_path, filename)
            logger.info(f"Processing: {filename}")
            
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pdf_texts.append(text.strip())
                        
                logger.info(f"Extracted {len(reader.pages)} pages from {filename}")
                        
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
                
        return pdf_texts
    
    def process_documents(self, pdf_texts: List[str]) -> None:
        """
        Process PDF texts into chunks and create embeddings.
        
        Args:
            pdf_texts: List of text extracted from PDFs
        """
        from langchain.text_splitter import (
            RecursiveCharacterTextSplitter,
            SentenceTransformersTokenTextSplitter,
        )
        
        # Setup embeddings if not already done
        if self.chroma_collection is None:
            self._setup_embeddings()
        
        # Split text into character-based chunks
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""], 
            chunk_size=1000, 
            chunk_overlap=100  # Add some overlap for better context
        )
        character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
        
        # Further split into token-based chunks
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=256
        )
        
        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)
        
        logger.info(f"Created {len(token_split_texts)} text chunks")
        
        # Add to vector database
        ids = [str(i) for i in range(len(token_split_texts))]
        self.chroma_collection.add(ids=ids, documents=token_split_texts)
        
        logger.info(f"Added {len(token_split_texts)} documents to vector database")
    
    def generate_hypothetical_answer(self, query: str, model: str = "llama-3.3-70b-versatile") -> str:
        """
        Generate a hypothetical answer for query expansion technique.
        
        Args:
            query: The original user query
            model: The Groq model to use
            
        Returns:
            Generated hypothetical answer
        """
        prompt = """You are an expert insurance policy analyst. 
        You are an insurance policy analyzer. 
        Extract only the key factual data points from the context given, focusing on:

        - Coverage limits or percentages
        - Deductibles or co-pays
        - Time periods, waiting periods, renewal periods
        - Important deadlines
        - Exclusions if mentioned

        
"""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating hypothetical answer: {e}")
            return ""
    
    def retrieve_documents(self, query: str, use_query_expansion: bool = True, n_results: int = 5) -> List[str]:
        """
        Retrieve relevant documents using vector similarity search.
        
        Args:
            query: The user query
            use_query_expansion: Whether to use hypothetical answer generation
            n_results: Number of documents to retrieve
            
        Returns:
            List of retrieved document texts
        """
        if self.chroma_collection is None:
            raise ValueError("No documents loaded. Call process_documents first.")
        
        search_query = query
        
        if use_query_expansion:
            hypothetical_answer = self.generate_hypothetical_answer(query)
            search_query = f"{query} {hypothetical_answer}"
            logger.info(f"Using expanded query for retrieval")
        
        results = self.chroma_collection.query(
            query_texts=[search_query], 
            n_results=n_results, 
            include=["documents"]
        )
        
        retrieved_documents = results["documents"][0]
        logger.info(f"Retrieved {len(retrieved_documents)} relevant documents")
        
        return retrieved_documents
    
    def generate_answer(self, query: str, context: str, model: str = "llama-3.3-70b-versatile") -> str:
        """
        Generate final answer based on retrieved context.
        
        Args:
            query: The user query
            context: Retrieved document context
            model: The Groq model to use
            
        Returns:
            Generated answer
        """
        final_prompt = f"""You are an expert insurance document analyst. Based on the following context from insurance policy documents, provide a comprehensive and accurate answer to the question.

Question: {query}

Context from policy documents:
{context}

Instructions:
- Extract specific details, numbers, percentages, time periods, and conditions directly from the context
- Include exact waiting periods, coverage limits, percentages, and specific conditions mentioned
- Use precise terminology from the policy documents
- If there are specific dollar amounts, percentages, or time periods, include them exactly as stated
- Only use information that is explicitly stated in the provided context
- If the answer requires specific policy details that aren't in the context, state that the information is not available
- No explanations or commentary. Keep the answer as short as possible. DO NOT give more than 2 sentences.
Provide a detailed, accurate answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {e}"
    
    def answer_query(self, query: str, use_query_expansion: bool = True, n_results: int = 8) -> Dict[str, str]:
        """
        Complete pipeline to answer a single query.
        
        Args:
            query: The user query
            use_query_expansion: Whether to use query expansion
            n_results: Number of documents to retrieve
            
        Returns:
            Dictionary with query, retrieved context, and answer
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        retrieved_documents = self.retrieve_documents(query, use_query_expansion, n_results)
        context = "\n\n".join(retrieved_documents)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        return {
            "query": query,
            "context": context,
            "answer": answer,
            "num_docs_retrieved": len(retrieved_documents)
        }
    
    def process_multiple_queries(self, queries: List[str], use_query_expansion: bool = True) -> List[Dict[str, str]]:
        """
        Process multiple queries and return answers.
        
        Args:
            queries: List of user queries
            use_query_expansion: Whether to use query expansion
            
        Returns:
            List of answer dictionaries
        """
        results = []
        
        logger.info(f"Processing {len(queries)} queries")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.answer_query(query, use_query_expansion)
            results.append(result)
        
        return results

def process_pdf_and_queries(folder_path: str, queries: List[str], use_query_expansion: bool = True) -> List[str]:
    """
    Main function to process PDFs and answer queries.
    
    Args:
        folder_path: Path to folder containing PDF files
        queries: List of queries to answer
        use_query_expansion: Whether to use query expansion technique
        
    Returns:
        List of answers
    """
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Load and process documents
    pdf_texts = pipeline.load_pdfs_from_folder(folder_path)
    pipeline.process_documents(pdf_texts)
    
    # Process queries
    results = pipeline.process_multiple_queries(queries, use_query_expansion)
    
    # Extract just the answers for backward compatibility
    answers = [result["answer"] for result in results]
    
    # Print results
    print("\n" + "="*80)
    print("RAG PIPELINE RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Documents retrieved: {result['num_docs_retrieved']}")
        print("-" * 60)
    
    return answers

if __name__ == "__main__":
    # Example usage with medical insurance policy queries
    pdf_path = "./data"
    queries = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    try:
        answers = process_pdf_and_queries(pdf_path, queries, use_query_expansion=True)
        print(f"\nProcessed {len(answers)} queries successfully!")
        print("=" * 80)
        for answer in answers:
            print(answer)
            print("=" * 80)
        
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        print(f"Error: {e}")
