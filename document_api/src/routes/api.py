from flask import Blueprint, request, jsonify
from services.llm_service import LLMService
from document_api.src.services.document_service import PDFService
from services.retrieval_service import RetrievalService

api = Blueprint('api', __name__)

@api.route('/process-query', methods=['POST'])
def process_query():
    try:
        pdf_file = request.files.get('pdf_file')
        query = request.form.get('query')

        if not pdf_file or not query:
            return jsonify({'error': 'PDF file and query are required.'}), 400

        # Initialize services
        pdf_service = PDFService()
        document_processor = pdf_service.extract_text(pdf_file)
        
        llm_service = LLMService()
        parsed_query = llm_service.process_query(query," ")
        
        # retrieval_service = RetrievalService()
        # decision, amount, justification = retrieval_service.retrieve_information(document_processor, parsed_query)

        response = {
            'decision': parsed_query['decision'],
            'amount': parsed_query['amount'],
            'justification': parsed_query['justification']
        }

        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500