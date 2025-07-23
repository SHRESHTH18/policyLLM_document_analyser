import os

class Config:
    DEBUG = os.getenv('DEBUG', False)
    TESTING = os.getenv('TESTING', False)
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')
    
    # LLM Configuration
    LLM_API_KEY = os.getenv('LLM_API_KEY', 'your_llm_api_key_here')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
    
    # Document Processing Configuration
    PDF_EXTRACTOR_PATH = os.getenv('PDF_EXTRACTOR_PATH', '/path/to/pdf/extractor')
    
    # Other configurations can be added as needed
    MAX_QUERY_LENGTH = 512
    RESPONSE_FORMAT = 'json'