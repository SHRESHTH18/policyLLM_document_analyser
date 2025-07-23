# Document API

This project provides a system that utilizes Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails.

## Objective

The main goal of this API is to accept a query related to a document (e.g., insurance policy) and a document (in PDF format), process the query, and return a structured JSON response containing decisions based on the content of the document.

## Features

- **Query Parsing**: The system can parse and structure queries to identify key details such as age, procedure, location, and policy duration.
- **Document Processing**: Supports extraction of text from various document formats, including PDFs and Word files.
- **Semantic Search**: Utilizes LLMs to search and retrieve relevant clauses or rules from documents based on the processed queries.
- **Decision Evaluation**: Evaluates retrieved information to determine decisions such as approval status or payout amounts.
- **Structured Response**: Returns a JSON response that includes the decision, amount (if applicable), and justification with references to the specific clauses used.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/document-api.git
   cd document-api
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/app.py
```

The API will be available at `http://localhost:5000`.

## API Endpoints

### POST /api/query

This endpoint accepts a PDF document and a natural language query.

**Request Body:**
```json
{
  "document": "<base64_encoded_pdf>",
  "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
}
```

**Response:**
```json
{
  "decision": "approved",
  "amount": 1000,
  "justification": {
    "clauses": [
      "Clause 1: Coverage for knee surgery",
      "Clause 2: Policy duration and eligibility"
    ]
  }
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.