# Chat with your PDFs

This project lets you ingest PDF documents and chat with them using a Retrieval-Augmented Generation (RAG) workflow built with LangChain, FAISS, and Streamlit.

## Features
- Upload one or more PDFs and turn them into retrievable chunks.
- FAISS vector store built from text embeddings (OpenAI by default).
- Conversational retrieval chain keeps chat history grounded in documents.
- Streamlit UI for document upload and interactive Q&A.

## Quick Start

1. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure secrets**
   - Copy `.env.example` to `.env` and fill in the missing values.
   - The default pipeline uses OpenAI embeddings and chat models. Set `OPENAI_API_KEY` in your environment or `.env` file.

4. **Launch Streamlit**
   ```bash
   streamlit run app.py
   ```

When the app starts you can upload PDFs and begin chatting.

## Configuration

Environment variables allow you to switch embedding or chat models without editing code:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Required when using OpenAI for embeddings or chat. |
| `OPENAI_MODEL_NAME` | Chat completion model, defaults to `gpt-3.5-turbo`. |
| `OPENAI_EMBEDDING_MODEL` | Embedding model, defaults to `text-embedding-3-small`. |
| `EMBEDDING_PROVIDER` | `openai` (default) or `huggingface`. |
| `LLM_PROVIDER` | `openai` (default), `ollama`, or `huggingface`. |
| `HUGGINGFACE_API_TOKEN` | Required when `LLM_PROVIDER` or `EMBEDDING_PROVIDER` is `huggingface`. |
| `HUGGINGFACE_EMBEDDING_MODEL` | Hugging Face embedding model id, defaults to `sentence-transformers/all-MiniLM-L6-v2`. |
| `HUGGINGFACE_CHAT_MODEL` | Hugging Face chat/inference endpoint id (required when `LLM_PROVIDER=huggingface`). |
| `OLLAMA_MODEL` | Model name served by Ollama when `LLM_PROVIDER=ollama` (defaults to `llama3`). |

## Project Structure

```
.
├── app.py                  # Streamlit entry point
├── rag
│   ├── __init__.py
│   ├── chat.py             # Conversational retrieval chain
│   ├── config.py           # Settings helpers
│   ├── embeddings.py       # Embedding provider selection
│   ├── ingestion.py        # PDF loading and chunking
│   └── vectorstore.py      # Vector-store helpers
├── requirements.txt
└── README.md
```

## Next Steps
- Add persistence for vector stores (disk cache) once the prototype works for you.
- Swap in another embeddings provider (Hugging Face, local) if you want a fully local stack.
- Install [Ollama](https://ollama.com) if you want a completely local, free chat model (`LLM_PROVIDER=ollama`).
- Add tests around ingestion to guard against regressions for large PDFs.
# chatwithpdf
