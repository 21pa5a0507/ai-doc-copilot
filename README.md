# HexPilot AI

HexPilot AI is a RAG-based document assistant for querying Hexnode documentation and Keka policy documents.

It combines a FastAPI backend, React frontend, hybrid retrieval, ONNX reranking, Gemini answer generation, and a small evaluation workflow into one practical GenAI project.

## Core Features

- Query Hexnode documentation
- Query Keka policy documents
- Select source: Hexnode, Keka, or both
- Hybrid retrieval using FAISS and BM25
- ONNX-based reranking
- Gemini answer generation with model fallback handling
- Source-backed answers with retrieved evidence chunks
- Evaluation script for routing, retrieval, and answer checks
- Docker support for the backend

## Tech Stack

### Frontend

- React
- Vite
- Tailwind CSS
- Axios

### Backend

- FastAPI
- Uvicorn
- LangChain
- LangGraph
- Google Gemini
- FAISS
- BM25
- ONNX Runtime

### Data Processing

- crawl4ai
- BeautifulSoup
- PyPDFLoader
- RecursiveCharacterTextSplitter

## How It Works

The React frontend sends a question and selected source to the FastAPI backend.

The backend:

- routes the request to Hexnode, Keka, or both sources
- retrieves relevant chunks using vector search and keyword search
- reranks the retrieved chunks with an ONNX reranker
- sends the final evidence to Gemini
- returns a grounded answer with supporting sources

For `both` source mode, the backend retrieves from both Hexnode and Keka, then uses a combined LangGraph flow to synthesize one answer.

Indexes are stored under `storage/` so the backend can load existing indexes on startup.

## Project Structure

```txt
backend/
frontend/
evaluation/
keka_data/
storage/
Dockerfile
.github/workflows/
```

## Installation

### Backend

Run from the project root:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The backend starts at:

```txt
http://localhost:8000
```

### Frontend

Run from the project root in another terminal:

```bash
cd frontend/ragline-ui
npm install
npm run dev
```

## API Endpoints

```txt
GET  /health
POST /ask
```

Example request:

```json
{
  "question": "How to enroll a Windows device?",
  "source": "default"
}
```

Source values:

- `default` - Hexnode documentation
- `keka` - Keka policy documents
- `both` - combined Hexnode and Keka answer

## Docker Usage

The Dockerfile runs the FastAPI backend only.

```bash
docker build -t ai-doc-copilot .
docker run -p 8000:8000 --env GOOGLE_API_KEY=your_google_api_key ai-doc-copilot
```

## Deployment

The repository includes GitHub Actions for:

- frontend lint and build checks
- backend Python syntax validation
- backend Docker image build
- Docker Hub publish
- Google Cloud Run deployment for the backend
- deployed backend health check

The React frontend is separate from the backend container and can be deployed independently using `VITE_API_URL` to point to the backend.

## Evaluation

The evaluation script tests the running backend API.

It checks:

- source routing
- expected document matches
- retrieval hits
- retrieval relevance using expected keywords
- answer keyword coverage
- fallback behavior
- simple hallucination-risk signals

Start the backend first, then run from the project root:

```bash
python evaluation/scripts/run_evaluation.py
```
