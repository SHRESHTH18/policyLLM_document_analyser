"""
Flask API server for RAG Pipeline with blob URL support
Optimized for Render deployment with lightweight dependencies
"""

from flask import Flask, request, jsonify
import requests
import tempfile
import os
import logging
import gc  # Garbage collection for memory management
from io import BytesIO
from pypdf import PdfReader
from light_rag_pipeline import LightRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration for memory optimization
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max (reduced from 16MB)

from flask import Flask, request, jsonify
import requests
import tempfile
import os
import logging
from io import BytesIO
from pypdf import PdfReader
from light_rag_pipeline import LightRAGPipeline

# Try to import CORS, make it optional for development
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS if available
if CORS_AVAILABLE:
    CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def extract_text_from_pdf_url(url: str) -> list:
    """
    Extract text directly from PDF URL without downloading to disk.
    
    Args:
        url: The blob URL to download PDF from
        
    Returns:
        List of extracted text from all pages
    """
    try:
        logger.info(f"Extracting text directly from PDF URL: {url}")
        
        # Download PDF content into memory
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower():
            logger.warning(f"Content type is {content_type}, but proceeding anyway")
        
        # Read PDF content into BytesIO buffer
        pdf_buffer = BytesIO(response.content)
        
        # Extract text using PdfReader
        reader = PdfReader(pdf_buffer)
        pdf_texts = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pdf_texts.append(text.strip())
        
        logger.info(f"Successfully extracted text from {len(reader.pages)} pages")
        return pdf_texts
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "message": "HackRX RAG API is running!",
        "status": "healthy",
        "endpoint": "/hackrx/run",
        "version": "1.0.0"
    })

@app.route('/health', methods=['GET'])
def health():
    """Additional health check endpoint for monitoring."""
    return jsonify({"status": "ok"}), 200

@app.route('/hackrx/run', methods=['POST'])
def process_document_and_queries():
    """
    Process PDF from blob URL and answer queries.
    
    Expected JSON format:
    {
        "documents": "https://blob.url/file.pdf",
        "questions": ["Question 1?", "Question 2?"]
    }
    
    Returns:
    {
        "answers": ["Answer 1", "Answer 2"]
    }
    """
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'documents' not in data:
            return jsonify({"error": "Missing 'documents' field"}), 400
            
        if 'questions' not in data:
            return jsonify({"error": "Missing 'questions' field"}), 400
        
        blob_url = data['documents']
        questions = data['questions']
        
        if not isinstance(questions, list):
            return jsonify({"error": "'questions' must be a list"}), 400
        
        if len(questions) == 0:
            return jsonify({"error": "At least one question is required"}), 400
        
        # Limit questions to prevent memory issues
        if len(questions) > 10:
            return jsonify({"error": "Maximum 10 questions allowed per request"}), 400
        
        logger.info(f"Processing request with {len(questions)} questions")
        
        # Step 1: Extract text directly from PDF URL (no download needed!)
        pdf_texts = extract_text_from_pdf_url(blob_url)
        
        # Step 2: Initialize lightweight RAG pipeline
        pipeline = LightRAGPipeline()
        
        # Step 3: Process documents directly with extracted text
        pipeline.process_documents(pdf_texts)
        
        logger.info(f"Processed {len(pdf_texts)} pages from PDF")
        
        # Step 4: Process queries one by one to avoid memory issues
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Process single query
                result = pipeline.answer_query(question)
                answers.append(result["answer"])
                
                # Force garbage collection after each query to free memory
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        # Clean up pipeline memory
        del pipeline
        gc.collect()
        
        logger.info(f"Successfully processed {len(answers)} queries")
        
        # Step 5: Return response in required format
        return jsonify({"answers": answers})
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting HackRX RAG Flask API Server")
    print("=" * 50)
    print("📍 Endpoint: POST /hackrx/run")
    print("📚 Health: http://localhost:5001/")
    print("=" * 50)
    
    # Development server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=False)
