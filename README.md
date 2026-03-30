# Debate Research Assistant

An AI-powered debate preparation tool that generates structured debate cases for **British Parliamentary (BP)** and **Asian Parliamentary (AP)** formats. Built with LangChain, Mistral AI, and Streamlit.

## Features

- **Multi-source RAG pipeline** — retrieves context from PDFs, JSONL speech transcripts, curated web pages, and live web search (Tavily) to produce well-grounded arguments.
- **Format-aware generation** — supports British Parliamentary and Asian Parliamentary debate formats with team and speaker-role selection.
- **Structured output** — responses follow a consistent flow: Direct Answer → Arguments → Examples → Counter-arguments → Conclusion.
- **Automatic fallback** — when local context is insufficient, the system augments with real-time web search results.
- **Persistent vector store** — ChromaDB index is built once and reused across sessions, avoiding redundant re-indexing.
- **Streamlit UI** — clean, interactive interface for entering motions and configuring debate parameters.

## Project Structure

```
Debate Research Assistant/
├── debate_back.py          # Backend: data loading, vector store, LLM pipeline
├── debate_ui.py            # Frontend: Streamlit interface
├── requirements.txt        # Python dependencies
├── .env                    # API keys (create manually — not committed)
├── README.md
├── RAG resources/          # Local reference documents
│   ├── speeches.jsonl      # JSONL corpus of real debate speeches
│   ├── Asian Parliamentary Style Guide.pdf
│   ├── British_Parlamentary_Style.pdf
│   └── PARLIAMENTARY DEBATE.pdf
├── Test Run/               # Demo screenshots and recording
│   ├── interphase.png
│   ├── output.png
│   └── testRun_recording.mov
└── debate_db/              # ChromaDB persistent store (auto-generated on first run)
```

## Prerequisites

- Python 3.10+
- API keys for the following services:

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| [Mistral AI](https://console.mistral.ai/) | `MISTRAL_API_KEY` | LLM for generating debate cases |
| [Tavily](https://tavily.com/) | `TAVILY_API_KEY` | Web search fallback |
| [HuggingFace](https://huggingface.co/settings/tokens) | `HUGGINGFACEHUB_API_TOKEN` | Embedding model access |

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/SahilBhatti09/Debate-Research-Assistant
   cd "Debate Research Assistant"
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```
   MISTRAL_API_KEY=your_mistral_key
   TAVILY_API_KEY=your_tavily_key
   HUGGINGFACEHUB_API_TOKEN=your_hf_token
   ```

5. **First run** — the application will automatically download PDFs from Google Drive, parse the JSONL corpus, and build the ChromaDB vector store. Subsequent runs load the existing store.

## Usage

### Streamlit App (recommended)

```bash
streamlit run debate_ui.py
```

This launches the web interface where you can:

1. Enter a debate motion.
2. Select the debate format (BP or AP).
3. Choose your team (Government / Opposition).
4. Pick a speaker role.
5. Click **Generate Debate Case** to receive a structured response.

### Command-Line Mode

```bash
python debate_back.py
```

Starts an interactive terminal session. Type your question or motion and receive AI-generated debate content. Enter `0` to exit.

## How It Works

```
┌──────────────┐
│  User Query   │
└──────┬───────┘
       ▼
┌──────────────────────────────────┐
│  Retriever (ChromaDB, k=10)      │◄── JSONL speeches + PDFs + Web pages
└──────┬───────────────────────────┘
       ▼
┌──────────────────────────────────┐
│  Context sufficient (≥300 chars)?│
│  No → Tavily web search fallback │
└──────┬───────────────────────────┘
       ▼
┌──────────────────────────────────┐
│  Mistral AI (mistral-small-2506) │◄── System prompt with debate rules
└──────┬───────────────────────────┘
       ▼
┌──────────────┐
│   Response    │
└──────────────┘
```

### Data Sources (priority order)

1. **JSONL speech transcripts** — real debate speeches with motion, role, and team metadata.
2. **PDF documents** — hosted on Google Drive, downloaded and parsed at build time.
3. **Curated web pages** — debate format guides scraped via `WebBaseLoader`.
4. **Live web search** — Tavily results used as a fallback when retrieved context is thin.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Mistral AI (`mistral-small-2506`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Web Search | Tavily |
| Frontend | Streamlit |
| Document Parsing | PyMuPDF, WebBaseLoader |

## Learning Outcomes

Building and working with this project provides hands-on experience with the following concepts:

### 1. Retrieval-Augmented Generation (RAG)
- Designing an end-to-end RAG pipeline that combines retrieval with generative AI to produce grounded, evidence-backed responses.
- Understanding how vector similarity search retrieves relevant context from a large corpus before generation.

### 2. LangChain Framework
- Using LangChain components — document loaders, text splitters, embeddings, vector stores, retrievers, and prompt templates — to build a modular AI application.
- Composing a multi-step pipeline where each stage (retrieval → context evaluation → generation) feeds into the next.

### 3. Vector Databases & Embeddings
- Converting unstructured text (PDFs, JSONL, web pages) into vector embeddings using sentence-transformers (`all-MiniLM-L6-v2`).
- Storing and querying embeddings with ChromaDB, including persistence across sessions and tuning retrieval parameters (e.g., `k=10`).

### 4. Multi-Source Data Ingestion
- Loading and normalising documents from diverse formats: JSONL speech transcripts, PDF files (via PyMuPDF), and live web pages (via WebBaseLoader).
- Downloading remote files from Google Drive programmatically.
- Splitting large documents into overlapping chunks for effective retrieval.

### 5. Large Language Model Integration
- Integrating a cloud-hosted LLM (Mistral AI) through LangChain's chat model interface.
- Crafting system prompts that enforce structured, domain-specific output (debate format, tone, argument flow).
- Controlling generation behaviour with parameters like `temperature`.

### 6. Fallback & Resilience Patterns
- Implementing a graceful fallback mechanism: when local retrieval context is insufficient (< 300 characters), the system dynamically augments with live Tavily web search results.
- Handling multiple result formats (list, dict, string) from external APIs.

### 7. Streamlit for Rapid Prototyping
- Building an interactive web UI with Streamlit — form inputs, radio buttons, dropdowns, spinners, and markdown rendering.
- Connecting a frontend to a Python backend module with a simple function import.

### 8. Prompt Engineering
- Designing effective system prompts that set priorities, structure, and tone for LLM output.
- Separating system-level instructions from user-level context injection in `ChatPromptTemplate`.

### 9. Environment & Dependency Management
- Managing API keys securely with `.env` files and `python-dotenv`.
- Tracking project dependencies in `requirements.txt`.

---

Developed by **Sahil Bhatti** | Powered by Mistral AI & LangChain
