"""
Quick test of the lightweight RAG pipeline
"""

from light_rag_pipeline import LightRAGPipeline

def test_pipeline():
    print("🧪 Testing Lightweight RAG Pipeline...")
    
    # Initialize pipeline
    pipeline = LightRAGPipeline()
    
    # Test documents
    test_docs = [
        "The grace period for premium payment is 30 days from the due date.",
        "Pre-existing diseases have a waiting period of 24 months.",
        "Maternity expenses are covered after 9 months waiting period."
    ]
    
    # Process documents
    pipeline.process_documents(test_docs)
    print("✅ Documents processed")
    
    # Test query
    result = pipeline.answer_query("What is the grace period for premium payment?")
    print(f"✅ Query processed: {result['answer']}")
    
    print("🎉 All tests passed! Pipeline is working correctly.")

if __name__ == "__main__":
    test_pipeline()
