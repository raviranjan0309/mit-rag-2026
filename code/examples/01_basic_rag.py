"""
============================================================
RAG Workshop — Example 01: Basic RAG Pipeline
============================================================
Build a complete Retrieval-Augmented Generation pipeline
from scratch using LangChain, OpenAI, and ChromaDB.

Steps:
  1. Load a PDF document
  2. Split it into chunks
  3. Embed chunks and store in ChromaDB
  4. Build a retrieval-augmented Q&A chain
  5. Ask questions and get grounded answers

Usage:
  python 01_basic_rag.py

Prerequisites:
  - pip install langchain langchain-openai langchain-community chromadb pypdf tiktoken
  - export OPENAI_API_KEY="sk-..."
  - Place a PDF file in the same directory (default: attention_is_all_you_need.pdf)
============================================================
"""

import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
PDF_PATH = os.getenv("RAG_PDF_PATH", "attention_is_all_you_need.pdf")
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    print("Run: export OPENAI_API_KEY='sk-...'")
    sys.exit(1)


# ──────────────────────────────────────────────────────────
# STEP 1: LOAD THE DOCUMENT
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading PDF document...")
print("=" * 60)

if not os.path.exists(PDF_PATH):
    print(f"\nERROR: PDF not found at '{PDF_PATH}'")
    print("Download the 'Attention Is All You Need' paper:")
    print("  https://arxiv.org/pdf/1706.03762")
    print(f"\nOr set a custom path: RAG_PDF_PATH=/path/to/your.pdf python {sys.argv[0]}")
    sys.exit(1)

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"\nLoaded {len(pages)} pages from '{PDF_PATH}'")
print(f"\nFirst 500 characters of page 1:\n")
print(pages[0].page_content[:500])
print(f"\nMetadata: {pages[0].metadata}")

# Key concept: Each page is a `Document` object with:
#   - .page_content  → the text
#   - .metadata      → dict with source, page number, etc.


# ──────────────────────────────────────────────────────────
# STEP 2: CHUNK THE DOCUMENTS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Splitting into chunks...")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,          # Max characters per chunk
    chunk_overlap=CHUNK_OVERLAP,    # Overlap to preserve context at boundaries
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]  # Try these in order
)

chunks = splitter.split_documents(pages)

print(f"\nOriginal pages: {len(pages)}")
print(f"After chunking: {len(chunks)} chunks")
print(f"Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
print(f"\nSample chunk (chunk #5):\n{chunks[5].page_content}")
print(f"\nChunk metadata: {chunks[5].metadata}")

# Try it: change CHUNK_SIZE to 200 or 2000 and see the difference!


# ──────────────────────────────────────────────────────────
# STEP 3: CREATE EMBEDDINGS + STORE IN CHROMADB
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Embedding chunks and storing in ChromaDB...")
print("=" * 60)

# Initialize the embedding model
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL  # 1536 dimensions, fast and affordable
)

# Quick demo: what does an embedding look like?
sample_text = "Attention mechanisms allow models to focus on relevant parts of the input."
sample_vector = embeddings.embed_query(sample_text)
print(f"\nEmbedding model: {EMBEDDING_MODEL}")
print(f"Embedding dimensions: {len(sample_vector)}")
print(f"First 5 values: {sample_vector[:5]}")
print(f"Type: list of {type(sample_vector[0]).__name__}")

# Create ChromaDB vector store from chunks
# This embeds ALL chunks and stores them — may take 30-60 seconds
print(f"\nEmbedding {len(chunks)} chunks... (this may take a moment)")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR  # Saves to disk so we don't re-embed every run
)

print(f"Vector store created with {vectorstore._collection.count()} vectors")
print(f"Persisted to: {CHROMA_DIR}")


# ──────────────────────────────────────────────────────────
# DEMO: RAW SIMILARITY SEARCH (before the LLM)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEMO: Similarity search in vector store...")
print("=" * 60)

query = "What is the multi-head attention mechanism?"
results = vectorstore.similarity_search_with_score(query, k=3)

print(f"\nTop 3 results for: '{query}'\n")
for i, (doc, score) in enumerate(results):
    print(f"--- Result {i+1} (distance: {score:.4f} — lower is more similar) ---")
    print(f"Source: Page {doc.metadata.get('page', 'N/A')}")
    print(f"Content: {doc.page_content[:200]}...")
    print()


# ──────────────────────────────────────────────────────────
# STEP 4: BUILD THE RAG CHAIN
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Building the RAG chain...")
print("=" * 60)

# Initialize the LLM
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0  # 0 = deterministic, good for factual tasks
)

# Create a custom prompt that forces grounded answers
rag_prompt = PromptTemplate(
    template="""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the provided documents."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# Create the retriever (top 4 most relevant chunks)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Build the chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" = put all retrieved docs into one prompt
    retriever=retriever,
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True  # Show which chunks were used
)

print(f"RAG chain built: {LLM_MODEL} + {EMBEDDING_MODEL} + ChromaDB")


# ──────────────────────────────────────────────────────────
# STEP 5: ASK QUESTIONS!
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Asking questions with RAG...")
print("=" * 60)


def ask(question: str) -> None:
    """Ask a question using the RAG chain and display the answer with sources."""
    print(f"\n{'─' * 60}")
    print(f"Q: {question}")
    print(f"{'─' * 60}")
    result = qa_chain.invoke({"query": question})
    print(f"\nA: {result['result']}")
    print(f"\nSources used ({len(result['source_documents'])} chunks):")
    for i, doc in enumerate(result["source_documents"]):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i+1}] Page {page}: {preview}...")


# Test questions
ask("What is the architecture of the Transformer model?")
ask("How many attention heads did the authors use?")
ask("What datasets were used for training?")

# This should say "not enough information" — it's not in the paper!
ask("Who invented the iPhone?")

print("\n" + "=" * 60)
print("Done! You've built a working RAG pipeline.")
print("=" * 60)
