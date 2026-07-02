# SHL Assessment Recommender

A FastAPI-based conversational assistant for recommending SHL assessment products based on hiring context such as role, skills, seniority, and assessment preferences.

## What this project does

This application accepts chat-style requests and returns:

- SHL assessment recommendations
- Catalog-based comparisons
- Basic clarification when the request is incomplete

It uses a lightweight rule-based agent pipeline with catalog-backed retrieval so it can work even when prebuilt indexes are not present.

## Project structure

- app/main.py — FastAPI application entry point
- app/api/ — API routes and dependencies
- app/agent/ — intent detection, safety checks, clarification, recommendation, and comparison logic
- app/catalog/ — catalog parsing and scraping support
- app/retrieval/ — retrieval engine and indexing helpers
- app/models/ — request, response, and assessment schemas
- data/ — sample SHL catalog data
- tests/ — automated tests

## Requirements

- Python 3.10
- pip

## Setup

1. Open a terminal in the project folder:

   ```bash
   cd "C:\Users\S VENKATESHWARLU\OneDrive\Desktop\SHL Assesment"
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the application

Start the FastAPI server with:

```bash
uvicorn app.main:app --reload
```

Then open:

- Health check: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs

## Example API request

### POST /chat

```json
{
  "messages": [
    {
      "role": "user",
      "content": "We need a Python backend developer for a junior role. Recommend SHL assessments."
    }
  ]
}
```

Example response:

```json
{
  "reply": "I found 5 SHL assessment recommendation(s) for ...",
  "recommendations": [
    {
      "name": "SHL Technical Skills Assessment",
      "url": "https://www.shl.com/products/technical-skills-assessment",
      "test_type": "Technical"
    }
  ],
  "end_of_conversation": false
}
```

## Run tests

```bash
pytest -q
```

## Notes

- The app includes a fallback path that uses the local catalog JSON when BM25/FAISS indexes are not available.
- If you want to use the scraper or build indexes later, the project structure already supports those modules.
