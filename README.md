# HexPilot AI

HexPilot AI is a RAG-based document assistant for querying Hexnode documentation, Keka policy documents, and support workflows.

It combines a FastAPI backend, React frontend, hybrid retrieval, ONNX reranking, Gemini answer generation, LangGraph workflow tooling, Jira integration, and a small evaluation workflow into one practical GenAI project.

## Core Features

- Query Hexnode documentation
- Query Keka policy documents
- Select source: Hexnode, Keka, both, or Support Workflow
- Troubleshoot support issues using Hexnode documentation
- Create, close, and request approved deletion for Jira tickets
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
- Jira Cloud REST API
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

- routes the request to Hexnode, Keka, both sources, or Support Workflow
- retrieves relevant chunks using vector search and keyword search
- reranks the retrieved chunks with an ONNX reranker
- sends the final evidence to Gemini
- returns a grounded answer with supporting sources

For `both` source mode, the backend retrieves from both Hexnode and Keka, then uses a combined LangGraph flow to synthesize one answer.

For `workflow` source mode, the backend runs a LangGraph support workflow agent. It can troubleshoot using Hexnode documentation, create Jira tickets, close fixed tickets, and request Jira-approved deletion.

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
POST /workflow/process-delete-approvals
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
- `workflow` - support workflow agent for troubleshooting and Jira ticket actions

The workflow source accepts recent chat history from the frontend so follow-up requests like "create a ticket for this issue" can use the previous support context.

### Workflow Approval Endpoint

`POST /workflow/process-delete-approvals` is intended for Cloud Scheduler. It checks Jira tickets marked with `delete-requested` and deletes only the tickets that also have `delete-approved`.

When `WORKFLOW_SCHEDULER_TOKEN` is set, callers must send:

```txt
X-Workflow-Scheduler-Token: your-token
```

## Environment Variables

The backend uses these environment variables depending on enabled features:

```env
GOOGLE_API_KEY=your_google_api_key
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your_email@example.com
JIRA_API_TOKEN=your_jira_api_token
JIRA_PROJECT_KEY=KAN
JIRA_ISSUE_TYPE=Task
WORKFLOW_SCHEDULER_TOKEN=your_scheduler_token
```

## Docker Usage

The Dockerfile runs the FastAPI backend only.

```bash
docker build -t ai-doc-copilot .
docker run -p 8000:8000 \
  --env GOOGLE_API_KEY=your_google_api_key \
  --env JIRA_BASE_URL=https://your-domain.atlassian.net \
  --env JIRA_EMAIL=your_email@example.com \
  --env JIRA_API_TOKEN=your_jira_api_token \
  --env JIRA_PROJECT_KEY=KAN \
  --env JIRA_ISSUE_TYPE=Task \
  --env WORKFLOW_SCHEDULER_TOKEN=your_scheduler_token \
  ai-doc-copilot
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

For Jira delete approvals in production, configure Cloud Scheduler to call:

```txt
POST https://your-backend-url/workflow/process-delete-approvals
```

every 15 minutes with the `X-Workflow-Scheduler-Token` header. Approval happens in Jira by adding the `delete-approved` label to a ticket that already has `delete-requested`.

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
- workflow response checks
- average latency timings returned by the backend

Start the backend first, then run from the project root:

```bash
python evaluation/scripts/run_evaluation.py
```
