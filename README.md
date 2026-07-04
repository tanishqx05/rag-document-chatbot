# RAG Document Chatbot

An AI-powered document chatbot that lets you upload any PDF and ask questions about it using Retrieval-Augmented Generation (RAG). Runs entirely locally — no API keys, no internet required, no quota limits.

## How It Works

```
PDF → extract text → split into chunks → embed with nomic-embed-text
Question → embed → search FAISS → retrieve top 4 chunks → send to llama3.2 → answer
```

1. **Upload a PDF** — text is extracted, split into 1000-character chunks with 200-character overlap
2. **Chunks are embedded** using `nomic-embed-text` (local Ollama model) and stored in a FAISS vector store
3. **Ask a question** — it's embedded and compared against all chunks using cosine similarity
4. **Top 4 most relevant chunks** are retrieved and passed as context to `llama3.2`
5. **llama3.2 generates an answer** grounded in the document context only

## Tech Stack

Python · LangChain · FAISS · Ollama · llama3.2 · nomic-embed-text · FastAPI · Docker

## Features

- 📄 Upload any PDF (up to 10MB)
- 💬 Real-time chat interface
- 🔒 100% local — no data leaves your machine
- 🧠 Context-aware answers grounded in the document
- 🗑️ Clear session and upload a new document anytime
- 🐳 Fully containerized with Docker

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Ollama](https://ollama.com)
- Models pulled locally:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

## Running Locally

**Step 1: Make sure Ollama is running**
```bash
# Ollama starts automatically on install
# Verify with:
ollama list
```

**Step 2: Build and run the Docker container**
```bash
cd app
docker build -t rag-chatbot .
docker run -p 8000:8000 rag-chatbot
```

**Step 3: Open in browser**
```
http://localhost:8000
```

## Project Structure

```
rag_chatbot/
├── app/
│   ├── main.py           # FastAPI app - upload, chat, clear endpoints
│   ├── rag_pipeline.py   # Core RAG logic - PDF processing, embeddings, retrieval
│   ├── index.html        # Chat UI - single page frontend
│   ├── requirements.txt
│   └── Dockerfile
└── README.md
```

## API Endpoints

| Method | Endpoint  | Description                        |
|--------|-----------|------------------------------------|
| GET    | /         | Serves the chat UI                 |
| GET    | /health   | Health check                       |
| POST   | /upload   | Upload and process a PDF           |
| POST   | /chat     | Ask a question about the document  |
| GET    | /history  | Get chat history                   |
| POST   | /clear    | Clear session and vector store     |

## Example

**Upload a PDF**, then ask:
```json
POST /chat
{
  "question": "What is this document about?"
}
```

**Response:**
```json
{
  "answer": "The document is about..."
}
```

## Why RAG?

Standard LLMs hallucinate when asked about specific documents they haven't seen. RAG fixes this by:
- Only answering from retrieved document context
- Saying "I couldn't find that information" when the answer isn't in the document
- Making answers traceable back to the source

## Author

Tejasv Rathore — [GitHub](https://github.com/Tejasv1910)
