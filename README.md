# 📚 RAG Pipeline — Retrieval-Augmented Generation for PDF Documents

A modular, CLI-based **Retrieval-Augmented Generation (RAG)** system that lets you ingest PDF documents, embed them into a vector database, and ask natural-language questions answered by Google's Gemini LLM — grounded in your own data.

---

## ✨ Features

| Feature | Description |
|---|---|
| **PDF Ingestion** | Recursively loads all `.pdf` files from `data/raw/` using LangChain's `PyPDFLoader`. |
| **Recursive Text Splitting** | Splits documents into overlapping 1000-character chunks with 200-character overlap for better context retention. |
| **Gemini Embeddings** | Generates vector embeddings using Google's `gemini-embedding-2-preview` model via `langchain-google-genai`. |
| **ChromaDB Vector Store** | Stores and retrieves embeddings locally using ChromaDB with persistent on-disk storage. |
| **RAG QA Generation** | Retrieves the top-k most relevant chunks and passes them as context to `gemini-2.5-flash` for answer generation. |
| **Source Citations** | Every response includes the source PDF filename and page number for traceability. |
| **Interactive CLI** | An interactive terminal session for asking multiple questions in a loop. |
| **Single Query Mode** | Run a one-off query directly from the command line via `--query`. |
| **Auto-Sync & Re-Ingestion** | Automatically detects file additions, modifications, or deletions in `data/raw/` and re-ingests when needed using a file manifest. |
| **Robust Embedding Wrapper** | Handles the 1-to-1 embedding mismatch between Google's embedding model and ChromaDB's expected format. |
| **Environment-Based Config** | All keys, model names, and paths are configurable via a `.env` file. |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌──────────┐    similarity    ┌──────────────┐
│ Embedding │───────────────▶│  ChromaDB     │
│  (Query)  │    search       │  Vector Store │
└──────────┘                  └──────┬───────┘
                                     │ top-k docs
                                     ▼
                              ┌──────────────┐
                              │  RAG Prompt   │
                              │  (context +   │
                              │   question)   │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │  Gemini LLM   │
                              │  (gemini-2.5  │
                              │   -flash)     │
                              └──────┬───────┘
                                     │
                                     ▼
                              Generated Answer
                              + Source Citations
```

### Pipeline Stages

1. **Load** — `PyPDFLoader` + `DirectoryLoader` reads all PDFs from `data/raw/`.
2. **Split** — `RecursiveCharacterTextSplitter` breaks pages into 1 000-char chunks with 200-char overlap.
3. **Embed** — Each chunk is embedded individually via `GoogleGenerativeAIEmbeddings` (wrapped by `RobustEmbeddings`).
4. **Store** — Embeddings are persisted to disk in a ChromaDB collection.
5. **Retrieve** — At query time, the user's question is embedded and the top 4 similar chunks are fetched.
6. **Generate** — Retrieved chunks are injected into a system prompt and sent to Gemini for answer generation.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.12+ | Core language |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) | Fast Python package management & virtual environments |
| **LLM** | Google Gemini 2.5 Flash | Answer generation |
| **Embeddings** | Google Gemini Embedding 2 Preview | Vector embeddings for semantic search |
| **Vector Database** | ChromaDB | Local, persistent vector storage and similarity search |
| **Search Strategy** | Cosine Similarity (Top-K) | Dense vector nearest-neighbor retrieval |
| **Orchestration** | LangChain | Document loading, splitting, retrieval, prompt chaining (LCEL) |
| **PDF Parsing** | PyPDF | Extracting text from PDF documents |
| **Config** | python-dotenv | Environment variable management |

---

## 🔍 Search Strategy — How Retrieval Works

This pipeline uses **dense vector similarity search** (cosine similarity) via ChromaDB. Here is how a query flows through the retrieval system:

```
User Question: "Who is Aritra?"
         │
         ▼
┌─────────────────────────┐
│  1. Embed the Query      │  → Google Gemini Embedding model converts the
│     into a vector        │     question into a 768-dimensional vector
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  2. Cosine Similarity    │  → ChromaDB compares the query vector against
│     Search in ChromaDB   │     all stored document chunk vectors
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  3. Return Top-K (k=4)  │  → The 4 most semantically similar chunks
│     Matching Chunks      │     are returned, ranked by similarity score
└─────────────────────────┘
```

### Configuration

| Parameter | Value | Location |
|---|---|---|
| Search type | `similarity` (cosine distance) | `src/retrieval/retriever.py` |
| Top-K results | `4` | `src/retrieval/retriever.py` |

### What This Means

- **Semantic, not keyword-based** — The search understands meaning. Asking "developer skills" will match chunks containing "programming experience" even without shared words.
- **Dense retrieval** — Every chunk and query is represented as a dense floating-point vector. Similarity is measured by the angle between vectors (cosine similarity).
- **No exact-match fallback** — There is no keyword/BM25 search layer. Exact terms (like IDs, names, or codes) may not rank highest if semantically dissimilar to the surrounding context.

## 📂 Project Structure

```
Rag/
├── .env                          # API keys & configuration (git-ignored)
├── .gitignore
├── pyproject.toml                # Dependencies & project metadata
├── uv.lock                       # Locked dependency versions
├── README.md
├── learning.md                   # Tech decisions, alternatives & known flaws
│
├── data/
│   └── raw/                      # Drop your PDF files here
│       └── *.pdf
│
├── chroma_db/                    # Persisted vector database (git-ignored)
│   ├── <collection-uuid>/
│   ├── chroma.sqlite3
│   └── ingest_manifest.json      # File manifest for change detection
│
└── src/
    ├── __init__.py
    ├── main.py                   # CLI entrypoint, auto-sync logic, manifest tracking
    │
    ├── config/
    │   └── settings.py           # Loads .env, exposes settings singleton
    │
    ├── ingestion/
    │   ├── loader.py             # PDF loading via DirectoryLoader + PyPDFLoader
    │   ├── splitter.py           # RecursiveCharacterTextSplitter (1000/200)
    │   └── embedder.py           # GoogleGenerativeAIEmbeddings factory
    │
    ├── retrieval/
    │   ├── vector_store.py       # ChromaDB create/load + RobustEmbeddings wrapper
    │   └── retriever.py          # Similarity retriever (top-k=4)
    │
    └── generation/
        └── qa_pipeline.py        # RAG chain: retrieve → prompt → Gemini → parse
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** package manager installed
- A **Google AI API Key** with access to Gemini models

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Rag
```

### 2. Create `.env` File

```env
GOOGLE_API_KEY=your_google_api_key_here
MODEL_NAME=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-2-preview
CHROMA_DB_DIR=chroma_db
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Add Your PDFs

Place your PDF files in the `data/raw/` directory:

```bash
mkdir -p data/raw
cp /path/to/your/documents/*.pdf data/raw/
```

### 5. Run the Pipeline

```bash
# Default: auto-ingests if needed, then starts interactive QA
uv run src/main.py

# Force re-ingestion
uv run src/main.py --ingest

# Single query
uv run src/main.py --query "What is the candidate's name?"

# Interactive session
uv run src/main.py --interactive
```

---

## 💡 Usage Examples

### Interactive Mode

```
=======================================================
   Interactive RAG QA Session (Type 'exit' or 'q' to quit)
=======================================================
Loading vector database...
System ready. Ask any question!

Ask a question > who is aritra?

--- Response ---
Aritra Dhank is a Full Stack Developer with experience in
delivering production features during a software engineering internship.
----------------

Sources:
- Aritra Dhank_CSE-AIML.pdf (Page 1)

=======================================================
```

### Auto-Sync Detection

When you add, remove, or modify PDFs in `data/raw/`, the system automatically detects the change on next run:

```
Detected changes in data/raw or database. Running ingestion pipeline...
--- RAG Ingestion Pipeline ---
Successfully loaded 2 documents from data/raw
Successfully split into 15 chunks.
Clearing old vector database at chroma_db...
Old vector database cleared.
Creating vector store from 15 chunks...
Vector store created successfully.
Ingestion completed. Starting interactive QA mode...
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Your Google AI API key |
| `MODEL_NAME` | `gemini-1.5-flash` | The Gemini model used for answer generation |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | The embedding model for vectorization |
| `CHROMA_DB_DIR` | `chroma_db` | Directory path for the persistent ChromaDB store |

---

## 📄 License

This project is for educational and personal use.
