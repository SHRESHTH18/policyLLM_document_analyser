"""
WSGI entry point for lightweight Flask application
Optimized for Render deployment
"""

import os
import gc
from flask_api import app

# Memory optimization
gc.set_threshold(700, 10, 10)  # More aggressive garbage collection

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
