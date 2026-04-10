"""
============================================================
RAG Workshop — Example 06 (Bonus): Metadata Filtering
============================================================
Restrict retrieval to specific pages, sources, or date ranges.

Real-world use cases:
  - "Search only in legal documents from 2024"
  - "Find info from the methodology section only"
  - "Retrieve from department X's knowledge base"

Usage:
  python 06_metadata_filtering.py

Prerequisites:
  - Run 01_basic_rag.py first (to create the ChromaDB store)
  - export OPENAI_API_KEY="sk-..."
============================================================
"""

import os
import sys

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────
# LOAD VECTOR STORE
# ──────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

total = vectorstore._collection.count()
print(f"Loaded {total} vectors from ChromaDB\n")


# ──────────────────────────────────────────────────────────
# UNFILTERED SEARCH (baseline)
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("UNFILTERED: Search across all chunks")
print("=" * 60)

query = "attention mechanism"
results = vectorstore.similarity_search(query, k=4)

print(f"\nQuery: '{query}'  |  Results: {len(results)}\n")
for doc in results:
    page = doc.metadata.get("page", "?")
    print(f"  Page {page}: {doc.page_content[:100]}...")


# ──────────────────────────────────────────────────────────
# FILTERED: Only specific pages
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FILTERED: Only pages 3-6 (methodology section)")
print("=" * 60)

# ChromaDB filter syntax uses $gte (>=) and $lte (<=)
filtered_results = vectorstore.similarity_search(
    query="attention mechanism",
    k=4,
    filter={"page": {"$gte": 3, "$lte": 6}},
)

print(f"\nQuery: '{query}'  |  Filter: pages 3-6  |  Results: {len(filtered_results)}\n")
for doc in filtered_results:
    page = doc.metadata.get("page", "?")
    print(f"  Page {page}: {doc.page_content[:100]}...")


# ──────────────────────────────────────────────────────────
# FILTERED: Single page
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FILTERED: Only page 1 (abstract and introduction)")
print("=" * 60)

page1_results = vectorstore.similarity_search(
    query="model performance results",
    k=4,
    filter={"page": 0},  # Page indexing starts at 0
)

print(f"\nQuery: 'model performance results'  |  Filter: page 0  |  Results: {len(page1_results)}\n")
for doc in page1_results:
    page = doc.metadata.get("page", "?")
    print(f"  Page {page}: {doc.page_content[:100]}...")


# ──────────────────────────────────────────────────────────
# SHOW ALL AVAILABLE METADATA
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Available metadata keys in your chunks:")
print("=" * 60)

# Get a sample to show metadata structure
sample = vectorstore.similarity_search("test", k=1)
if sample:
    print(f"\n  Metadata keys: {list(sample[0].metadata.keys())}")
    print(f"  Sample:        {sample[0].metadata}")

print("""
  In production, you would add richer metadata during ingestion:
    - source:     "legal_docs" | "hr_policies" | "engineering"
    - author:     "alice" | "bob"
    - date:       "2024-01-15"
    - department: "finance" | "legal"
    - access:     "public" | "confidential"

  Then filter at retrieval time:
    filter={
        "$and": [
            {"source": "legal_docs"},
            {"date": {"$gte": "2024-01-01"}},
            {"access": "public"}
        ]
    }
""")

print("=" * 60)
print("Key takeaway: Metadata filtering is how you implement")
print("access control and scoping in production RAG systems.")
print("=" * 60)
