from flask import Flask, request, jsonify
from routes.api import api

app = Flask(__name__)

# Register blueprints
app.register_blueprint(api)

@app.route('/')
def home():
    return "Welcome to the Document API"

if __name__ == '__main__':
    app.run(debug=True)