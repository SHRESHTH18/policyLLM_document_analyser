from flask import Flask, request, jsonify
from flask_cors import CORS
from services.document_service import DocumentService
import io
import os
import logging

# Setup Flask
app = Flask(__name__)
CORS(app)

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Service
doc_service = DocumentService()

@app.route("/extract", methods=["POST"])
def extract_text():
    if "file" not in request.files:
        logger.warning("No file part in the request.")
        return jsonify({"error": "No file provided", "success": False}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        logger.warning("Uploaded file has an empty filename.")
        return jsonify({"error": "Empty filename", "success": False}), 400

    try:
        # Read file into memory
        file_stream = io.BytesIO(uploaded_file.read())

        # Try to extract file extension, defaulting if missing
        original_filename = uploaded_file.filename or "upload.pdf"
        ext = os.path.splitext(original_filename)[1].lower().lstrip(".")

        # Assign a fallback filename to in-memory stream
        file_stream.filename = f"upload.{ext}"
        logger.info(f"Processing file: {file_stream.filename}")

        # Process the document
        result = doc_service.process_document(file_stream)

        return jsonify({
            "text": result.get("text", ""),  # Limit response size
            "metadata": result.get("metadata", {}),
            "success": True
        }), 200

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve), "success": False}), 415  # Unsupported Media Type
    except Exception as e:
        logger.exception("Unexpected error occurred during file processing.")
        return jsonify({"error": f"Internal server error: {str(e)}", "success": False}), 500


if __name__ == "__main__":
    # Run on custom port to avoid conflicts
    app.run(debug=False, host="0.0.0.0", port=5050)
