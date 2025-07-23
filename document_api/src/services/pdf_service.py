import pypdf
import io
class PDFService:
    def __init__(self):
        pass

    def extract_text(self, pdf_file):
        """Extract text from uploaded PDF file object"""
        text = ""
        
        # If it's a file path (string), open it
        if isinstance(pdf_file, str):
            with open(pdf_file, "rb") as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        else:
            # If it's a file object from Flask request.files
            pdf_file.seek(0)  # Reset file pointer
            reader = pypdf.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
        return text.strip() if text else None

    def extract_metadata(self, pdf_path):
        metadata = {}
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            metadata = reader.metadata
        return metadata if metadata else None

    def process_pdf(self, pdf_path):
        text = self.extract_text(pdf_path)
        metadata = self.extract_metadata(pdf_path)
        return {
            "text": text,
            "metadata": metadata
        }