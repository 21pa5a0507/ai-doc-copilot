# HexPilot AI Architecture

HexPilot AI is a multi-source GenAI assistant for Hexnode documentation, Keka HR policies, combined source questions, and support workflow automation with Jira.

## Complete Workflow Diagram

```text
User
  |
  v
React + Vite Chat UI
  - Source selector: Hexnode / Keka / Both / Workflow
  - Chat history for workflow follow-ups
  - Source panel for retrieved evidence
  |
  v
Frontend API Client
  - ragApi.js
  - POST /ask { question, source, history }
  |
  v
FastAPI Backend
  - POST /ask
  - GET /health
  - POST /workflow/process-delete-approvals
  - Startup initializes Hexnode, Keka, and combined graph runtimes
  |
  v
Assistant Router
  - source=default  -> Hexnode RAG
  - source=keka     -> Keka RAG
  - source=both     -> Combined Graph
  - source=workflow -> Support Workflow Agent
  |
  +---------------------------+---------------------------+---------------------------+---------------------------+
  |                           |                           |                           |
  v                           v                           v                           v
Hexnode RAG                Keka RAG                    Combined Graph              Support Workflow Agent
Product docs               HR policy PDFs              LangGraph flow              LangGraph ReAct tool agent
hexnode_tools.py           keka_rag/tools.py           combined_graph.py           workflow_agent/main.py
  |                           |                           |                           |
  |                           |                           |                           +--> Troubleshoot issue
  |                           |                           |                           |    using Hexnode docs
  |                           |                           |                           |
  |                           |                           |                           +--> Create Jira ticket
  |                           |                           |                           |
  |                           |                           |                           +--> Close Jira ticket
  |                           |                           |                           |
  |                           |                           |                           +--> Request delete approval
  |                           |                           |                                and scheduled deletion
  |                           |                           |
  v                           v                           v
Hybrid Retrieval            Hybrid Retrieval            Parallel Retrieval
FAISS + BM25                FAISS MMR + BM25            Hexnode + Keka
vector_store.py             keka_rag/retriever.py       source-labelled evidence
  |                           |                           |
  v                           v                           v
Deduplication               Deduplication               Gemini synthesis
  |                           |                           |
  v                           v                           |
ONNX Reranker               ONNX Reranker                |
onnx_reranker.py            onnx_reranker.py             |
  |                           |                           |
  +-------------+-------------+---------------------------+
                |
                v
Prompt Construction + Context Injection
  - Hexnode answer_generator.py
  - Keka rag_chain.py
  - Combined graph synthesis prompt
  - Workflow agent final response after tool execution
                |
                v
Gemini LLM Layer
  - Primary model: gemini-2.5-flash-lite
  - Fallback model stack for retryable failures
  - Grounded answer generation from retrieved/tool context
                |
                v
API Response
  - question
  - answer
  - chunks
  - source
  - tool_calls
  - timings
                |
                v
React Rendering
  - Assistant message
  - Markdown-style formatting
  - Source cards with title, URL/source, excerpt, relevance
```

## Knowledge Processing Flows

```text
Hexnode Knowledge Flow
Hexnode sitemap/cache
  |
  v
scraper.py
  - Crawl Windows-focused Hexnode docs when cache is missing
  - Extract main/article content
  - Split content by headings
  |
  v
chunker.py
  - Clean text
  - Chunk with overlap
  - Filter short/noisy chunks
  |
  v
ONNX Embeddings
  - backend/models/hexnode_embedding
  - all-MiniLM-L6-v2 wrapper
  |
  v
FAISS IndexFlatIP + Metadata Cache + BM25 Tokens
  - storage/hexnode/index/vector_store_emb.index
  - storage/hexnode/index/vector_store_meta.pkl
```

```text
Keka Knowledge Flow
keka_data/*.pdf
  |
  v
PyPDFLoader
  - Load policy PDFs
  - Preserve file_name metadata
  |
  v
RecursiveCharacterTextSplitter
  - chunk_size=800
  - chunk_overlap=100
  |
  v
ONNX Embeddings
  - LangChain Embeddings adapter
  - backend/models/hexnode_embedding
  |
  v
LangChain FAISS Index + Cached Chunks
  - storage/keka/index/faiss_emb
  - storage/keka/cache/docs_chunk.pkl
```

## Evaluation And Deployment

```text
Evaluation
evaluation/dataset/questions.json
  |
  v
evaluation/scripts/run_evaluation.py
  - Calls live FastAPI /ask endpoint
  - Checks source accuracy
  - Checks expected document matches
  - Checks retrieval relevance
  - Checks answer keyword coverage
  - Flags fallback and hallucination risk
  - Saves summary metrics
```

```text
Deployment
GitHub Actions
  |
  +--> Frontend lint + production build
  |
  +--> Backend Python syntax validation
  |
  +--> Docker image build
  |
  +--> Docker Hub publish on main branch
  |
  v
Google Cloud Run
  - Deploy backend container
  - Verify /health after deployment
```

## Layer-by-Layer Explanation

**Client Layer**  
React/Vite provides the chat UI, source selector, local chat sessions, workflow history handoff, and source evidence panel.

**API Layer**  
FastAPI exposes `/ask`, `/health`, and the workflow scheduler endpoint. It initializes Hexnode, Keka, and combined graph runtime objects at startup.

**Assistant Routing Layer**  
`main.py` routes requests based on `source`. The implemented routes are Hexnode, Keka, Both, and Workflow.

**Hexnode RAG**  
Uses keyword-based routing for topics, setup steps, and general search. Retrieval uses FAISS + BM25, then ONNX reranking.

**Keka RAG**  
Uses keyword-based routing for policy lists, process steps, and policy search. Retrieval uses FAISS MMR + BM25, then ONNX reranking.

**Combined Graph**  
A LangGraph flow retrieves from Hexnode and Keka in parallel, labels evidence by source, and asks Gemini to synthesize one answer.

**Support Workflow Agent**  
A LangGraph ReAct agent lets Gemini select tools for troubleshooting, Jira creation, Jira closure, and delete approval handling.

**Retrieval And Ranking**  
Both knowledge bases generate lexical and semantic candidates, deduplicate them, and use the local ONNX reranker for final context selection.

**LLM Layer**  
Gemini receives prompts with retrieved context or tool outputs. The code uses a primary Gemini model plus fallback models for retryable failures.

**Response Layer**  
The backend returns answer text, evidence chunks, source, tool calls, and timings. The frontend renders the answer and source evidence.

**Evaluation Layer**  
The evaluation script tests the live backend for source accuracy, document match, retrieval relevance, answer quality, fallback behavior, hallucination risk, and timings.

**Deployment Layer**  
Docker packages the backend with stored indexes. GitHub Actions validates, builds, publishes, deploys to Cloud Run, and checks `/health`.

## Why This Architecture Is Strong

- Clear source routing keeps Hexnode, Keka, combined RAG, and workflow automation understandable.
- Hybrid retrieval improves recall by combining semantic and lexical search.
- ONNX embeddings and reranking keep retrieval quality local and predictable.
- The workflow agent is isolated from normal RAG paths, so Jira actions stay explicit.

## Why It Is Practical For Product Integration

- The frontend uses one `/ask` API contract for all assistant modes.
- The backend returns sources and timings, which helps debugging and user trust.
- Docker, GitHub Actions, and Cloud Run deployment are already represented in the repo.

## Key Engineering Decisions

- Use deterministic routing for normal RAG flows and LangGraph only where multi-step execution is needed.
- Use separate Hexnode and Keka indexes because their data sources and chunking pipelines are different.
- Use combined-source synthesis after retrieval instead of merging both knowledge bases into one undifferentiated index.
- Use Jira delete approval labels instead of immediate destructive deletion.
