# HexPilot AI Architecture

HexPilot AI is a multi-source RAG assistant for answering questions from Hexnode documentation and Keka policy documents. A React chat UI sends questions to a FastAPI backend. The backend routes each question to the right knowledge source, retrieves relevant evidence, reranks it, and uses Gemini to generate a grounded answer.

The project is structured like a small deployable GenAI service. The frontend, API layer, retrieval pipelines, answer generation, evaluation, and deployment setup are kept separate so each part can be improved without rewriting the whole system.

---

## 1. High-Level Architecture

```text
User
  |
  v
React + Vite Frontend
  - Chat interface
  - Source selector: Hexnode Docs / Keka Policies / Both Sources
  - Session-only chat history
  - Sources panel
  |
  v
FastAPI Backend
  - POST /ask
  - GET /health
  - Source-based routing
  |
  +-------------------------------+
  |                               |
  v                               v
Hexnode Knowledge Pipeline      Keka Knowledge Pipeline
  - Hexnode help docs             - Local Keka policy PDFs
  - cleaned chunks                - PDF chunks
  - FAISS vector index            - LangChain FAISS index
  - BM25 retrieval                - BM25 retrieval
  - hybrid search                 - MMR vector retrieval
  - ONNX reranking                - ONNX reranking
  |                               |
  +---------------+---------------+
                  |
                  v
Answer Generation / Combined Graph
  - Direct RAG for single-source questions
  - LangGraph combined flow for both sources
  - Optional agent/graph paths behind environment flags
                  |
                  v
Gemini LLM
  - Grounded answer generation
  - Model fallback stack
                  |
                  v
Answer + Supporting Sources
```

---

## 2. Frontend Structure

The frontend is a React + Vite application focused on asking questions and reviewing retrieved sources.

```text
frontend/ragline-ui
  ├── src
  │   ├── App.jsx
  │   ├── api/ragApi.js
  │   ├── components/ChatBox.jsx
  │   ├── components/SourcesPanel.jsx
  │   └── components/Message.jsx
  └── package.json
```

### Main responsibilities

- `App.jsx` owns theme state, sidebar state, session chat history, active messages, and retrieved sources.
- `ChatBox.jsx` handles chat input, source selection, suggested prompts, loading state, and typing animation.
- `SourcesPanel.jsx` displays the retrieved evidence chunks returned by the backend.
- `ragApi.js` sends `POST /ask` requests to the backend using `VITE_API_URL`, with `http://localhost:8000` as the default.

The frontend does not know how retrieval works. It only sends:

```json
{
  "question": "user question",
  "source": "default | keka | both"
}
```

This keeps the UI simple and keeps retrieval logic inside the backend.

---

## 3. Backend API Design

The backend is a FastAPI service with two main endpoints.

```text
GET  /health
POST /ask
```

### `/health`

Returns readiness status for the Hexnode, Keka, and combined graph pipelines.

### `/ask`

Accepts a user question and a source selector.

```json
{
  "question": "How to enroll a Windows device?",
  "source": "default"
}
```

Source routing is handled in `backend/main.py`.

```text
source = default  -> Hexnode docs pipeline
source = keka     -> Keka policy pipeline
source = both     -> Combined Hexnode + Keka graph
```

The response contains the original question, generated answer, retrieved chunks, selected source, and tool metadata when available.

---

## 4. Hexnode RAG Pipeline

The Hexnode pipeline uses cached/indexed Hexnode help documentation when available. If no index is found, the backend can crawl Hexnode Windows documentation, clean the content, chunk it, embed it, and save a FAISS index.

```text
Hexnode cached docs / sitemap crawl
  -> clean documentation content
  -> chunk text
  -> generate ONNX embeddings
  -> store chunks in FAISS
  -> build BM25 index
  -> retrieve using vector + keyword search
  -> normalize and combine results
  -> deduplicate chunks
  -> rerank with ONNX reranker
  -> generate answer with Gemini
```

### Retrieval design

The Hexnode retriever combines:

1. **FAISS vector search** for semantic similarity.
2. **BM25 keyword search** for exact product terms, feature names, and UI wording.

The two result sets are normalized, combined, deduplicated, and reranked before answer generation. This is useful for product documentation because users may ask semantically, but the best evidence often depends on exact terms from the docs.

---

## 5. Keka RAG Pipeline

The Keka pipeline is built around local policy PDFs stored in `keka_data`.

```text
Keka policy PDFs
  -> load PDF pages
  -> split into chunks
  -> generate ONNX embeddings
  -> store chunks in LangChain FAISS
  -> retrieve using BM25 + MMR vector search
  -> deduplicate chunks
  -> rerank with ONNX reranker
  -> generate grounded answer with Gemini
```

### Retrieval design

The Keka retriever uses:

- **BM25** for policy names, exact terms, limits, and clauses.
- **MMR vector retrieval** for semantic coverage and diversity.
- **Deduplication** to remove repeated chunks.
- **ONNX reranking** to select the final top documents.

This works well for HR policy questions because policy answers often require both semantic matching and exact document evidence.

---

## 6. Routing, Optional Agents, and Combined Graph

The default single-source flow is intentionally predictable. For Hexnode and Keka requests, the backend uses direct routing logic to decide whether the question is a normal search, a list request, or a process/setup-style request.

### Hexnode routing

Hexnode questions can use:

```text
search_hexnode_docs
list_hexnode_topics
get_hexnode_setup_steps
```

The direct Hexnode path is used by default. A LangGraph tool flow also exists and can be enabled as a fallback path through the `USE_HEXNODE_GRAPH_FALLBACK` environment flag.

### Keka routing

Keka questions can use:

```text
search_keka_policies
list_keka_policies
get_keka_process_steps
```

The direct Keka path is used by default. A LangChain agent also exists and can be enabled through the `USE_KEKA_AGENT_ROUTER` environment flag.

### Combined source flow

When the user selects `Both Sources`, the backend runs a combined LangGraph flow.

```text
Question
  -> retrieve from Hexnode
  -> retrieve from Keka
  -> tag chunks by source
  -> synthesize one answer with Gemini
  -> return combined answer + source-specific evidence
```

This is useful when a question needs both product documentation and internal policy context.

---

## 7. LLM Integration

Gemini is used for answer generation. Model access is centralized in `backend/rag/gemini_models.py`.

The backend defines a model stack:

```text
primary model
  -> fallback model 1
  -> fallback model 2
  -> fallback model 3
  -> fallback model 4
```

If a retryable model error happens, the backend tries the next model in the stack. This prevents one temporary model failure from immediately breaking the answer flow.

The prompts follow a grounded pattern:

```text
Use only the provided context.
If the answer is not present, say that directly.
Do not invent policy details, links, or setup steps.
```

This keeps the assistant safer for product documentation and HR policy use cases.

---

## 8. Core Request Flow

```text
1. User asks a question in the React UI.
2. User selects a source: Hexnode, Keka, or Both.
3. Frontend sends POST /ask to FastAPI.
4. Backend routes the request based on source.
5. The selected retriever fetches candidate chunks.
6. Hybrid retrieval improves recall.
7. ONNX reranking improves final evidence precision.
8. Gemini generates a grounded answer from the retrieved evidence.
9. Backend returns answer and source chunks.
10. Frontend displays the answer and supporting sources.
```

For `both` source mode, the retrieval step runs against both Hexnode and Keka, then the combined LangGraph flow synthesizes one response.

---

## 9. Evaluation System

The repository includes an evaluation folder with a question dataset and a script for testing the RAG API.

```text
evaluation
  ├── dataset/questions.json
  └── scripts/run_evaluation.py
```

The evaluation script calls the running backend at `http://localhost:8000/ask` and checks:

- expected source selection
- expected document matches
- retrieval hit rate
- retrieval relevance through expected keywords
- answer keyword coverage
- fallback behavior
- simple hallucination-risk signals

This is a practical evaluation setup. It does not prove perfect factual correctness, but it helps catch wrong routing, weak retrieval, missing evidence, and unsupported answers.

---

## 10. Deployment Architecture

The backend is containerized with Docker.

```text
Dockerfile
  -> Python 3.11 slim image
  -> install backend serving requirements
  -> copy backend code
  -> copy storage files
  -> run FastAPI with Uvicorn on port 8000
```

GitHub Actions is configured for:

- frontend lint/build checks
- backend Python syntax validation
- backend Docker image build
- Docker Hub publish on main branch pushes
- Cloud Run deployment for the backend
- deployed backend health check

The React frontend is a separate Vite app. It can be deployed independently and configured with `VITE_API_URL` to call the backend.

---

## 11. Why This Architecture Is Practical

This architecture is useful because it keeps the system understandable and debuggable.

- The frontend handles the user experience.
- The backend owns routing, retrieval, reranking, and answer generation.
- Hexnode and Keka pipelines stay separate instead of mixing all data blindly.
- Hybrid retrieval improves recall across semantic and keyword-heavy questions.
- ONNX reranking improves the final evidence sent to Gemini.
- Combined-source mode supports questions that need both documentation and policy context.
- Supporting chunks are returned to the UI, so answers can be inspected.
- Evaluation checks the parts of a RAG system that usually fail first: routing, retrieval, evidence quality, and fallback behavior.

The main design choice is reliability. The default single-source paths use direct routing and RAG, while optional agent/graph paths are available without making the whole system depend on them.
