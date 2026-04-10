# RAG Workshop: From Zero to Production

> **Build a Retrieval-Augmented Generation system from scratch using LangChain, OpenAI, and ChromaDB.**

A 3-hour hands-on workshop covering the full RAG pipeline — from loading documents to evaluating answer quality.

---

## What You'll Learn

| Module | What You'll Build |
|--------|-------------------|
| **Foundations** | Understand embeddings, vector search, and chunking |
| **Build Your First RAG** | End-to-end pipeline: PDF → Chunks → Embeddings → Q&A |
| **Advanced RAG** | Multi-query retrieval, conversational memory, evaluation |
| **Production & Future** | Agentic RAG, Graph RAG, modern stack architecture |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/raviranjan0309/mit-rag-2026.git
cd mit-rag-2026
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OR export it directly:
export OPENAI_API_KEY="sk-your-key-here"
```

### 4. Download the Sample Document

Download the "Attention Is All You Need" paper (or use any PDF):

```bash
# Download from arXiv
curl -L -o examples/attention_is_all_you_need.pdf https://arxiv.org/pdf/1706.03762
```

### 5. Run the Examples

```bash
cd examples

# Start here — builds the full pipeline
python 01_basic_rag.py

# Then explore advanced techniques
python 02_multi_query_rag.py
python 03_conversational_rag.py
python 04_rag_vs_no_rag.py
python 05_ragas_eval.py
python 06_metadata_filtering.py
```

---

## Repository Structure

```
rag-workshop/
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── .env.example                 ← Template for API keys
├── .gitignore
│
├── examples/                    ← Workshop code (run in order)
│   ├── 01_basic_rag.py          ← Complete RAG pipeline from scratch
│   ├── 02_multi_query_rag.py    ← Multi-query retrieval for better recall
│   ├── 03_conversational_rag.py ← Chat memory for follow-up questions
│   ├── 04_rag_vs_no_rag.py      ← Side-by-side hallucination comparison
│   ├── 05_ragas_eval.py         ← Evaluate RAG quality with RAGAS
│   └── 06_metadata_filtering.py ← Scope retrieval by page/source/date
│
├── notebooks/                   ← Jupyter notebook (Google Colab ready)
│   └── rag_workshop.ipynb       ← All examples in one interactive notebook
│
└── docs/                        ← Reference material
    ├── session_plan.md          ← Full 3-hour session plan
    └── architecture.md          ← RAG architecture reference
```

---

## Examples Overview

### 01 — Basic RAG Pipeline
Build the complete pipeline: load a PDF, chunk it, embed with OpenAI, store in ChromaDB, and answer questions with grounded responses.

```python
# The core loop in ~10 lines
loader = PyPDFLoader("attention_is_all_you_need.pdf")
pages = loader.load()
chunks = splitter.split_documents(pages)
vectorstore = Chroma.from_documents(chunks, embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
result = qa_chain.invoke({"query": "What is multi-head attention?"})
```

### 02 — Multi-Query RAG
Generate multiple query variations to retrieve more relevant chunks. Watch the logs to see how the LLM rewrites your question.

### 03 — Conversational RAG
Add memory so follow-up questions work naturally. "How many layers does **it** use?" correctly resolves to the Transformer.

### 04 — RAG vs No-RAG Comparison
The most compelling demo — same question, side-by-side. See hallucination prevention in action.

### 05 — RAGAS Evaluation
Measure your RAG system with Faithfulness, Answer Relevancy, and Context Precision scores.


---

## The RAG Pipeline

```
INDEXING (done once)                    QUERYING (per question)
──────────────────                      ───────────────────────

 ┌──────────────┐                       ┌──────────────┐
 │  PDF / Docs  │                       │  User Query  │
 └──────┬───────┘                       └──────┬───────┘
        │                                      │
        ▼                                      ▼
 ┌──────────────┐                       ┌──────────────┐
 │   Chunking   │                       │  Embed Query │
 └──────┬───────┘                       └──────┬───────┘
        │                                      │
        ▼                                      ▼
 ┌──────────────┐    similarity    ┌──────────────────┐
 │  Embed Chunks├────search───────►│ Retrieve Top-K   │
 └──────┬───────┘                  └──────┬───────────┘
        │                                 │
        ▼                                 ▼
 ┌──────────────┐                  ┌──────────────────┐
 │ Vector Store │                  │  LLM + Context   │
 │  (ChromaDB)  │                  │  → Grounded      │
 └──────────────┘                  │    Answer         │
                                   └──────────────────┘
```

---

## Key Concepts

| Concept | One-Line Explanation |
|---------|---------------------|
| **RAG** | Give the LLM the right context at question time, not training time |
| **Embedding** | Convert text to a numeric vector that captures meaning |
| **Vector Store** | Database optimized for finding similar vectors fast |
| **Chunking** | Split documents into pieces small enough to be useful context |
| **Retriever** | Finds the most relevant chunks for a given query |
| **Re-ranking** | Second-pass scoring for higher precision retrieval |
| **RAGAS** | Framework to measure RAG quality without human evaluation |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OPENAI_API_KEY not set` | `export OPENAI_API_KEY="sk-..."` or add to `.env` |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| PDF not found | Download: `curl -L -o examples/attention_is_all_you_need.pdf https://arxiv.org/pdf/1706.03762` |
| ChromaDB errors | Delete `chroma_db/` folder and re-run `01_basic_rag.py` |
| Rate limit errors | Wait 60 seconds, or use a different API key |
| Slow embedding | Normal — first run embeds all chunks. Subsequent runs load from disk. |

---

## Going Further

- Build RAG on your own documents (project docs, textbook, dataset).
- Add hybrid search + re-ranking. Measure improvement with RAGAS.
- Replace the chain with a LangGraph agent. Add retrieval decision nodes.
- Explore Graph RAG — build a knowledge graph, run hybrid queries.

### Recommended Resources

| Resource | Link |
|----------|------|
| LangChain Docs | [python.langchain.com/docs](https://python.langchain.com/docs/) |
| LangChain RAG Tutorial | [python.langchain.com/docs/tutorials/rag](https://python.langchain.com/docs/tutorials/rag/) |
| OpenAI Embeddings Guide | [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings) |
| ChromaDB Docs | [docs.trychroma.com](https://docs.trychroma.com/) |
| RAGAS Docs | [docs.ragas.io](https://docs.ragas.io/) |
| Original RAG Paper | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) |
| Graph RAG Paper | [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130) |

---

## Tech Stack

- **Python 3.9+**
- **LangChain** — Orchestration framework
- **OpenAI** — Embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
- **ChromaDB** — Local vector store
- **RAGAS** — RAG evaluation framework

---

## License

MIT — Use freely for teaching, learning, and building.
