"""
============================================================
RAG Workshop — Example 02: Multi-Query RAG
============================================================
Improve retrieval by generating multiple query variations.
Instead of one search, the LLM rewrites your question into
3-5 variants, retrieves for each, and deduplicates results.

Why this works:
  - A single query might miss relevant chunks due to wording
  - "How does the model learn word relationships?" vs
    "What mechanism captures token dependencies?"
  - Same intent, different embeddings → different results

Usage:
  python 02_multi_query_rag.py

Prerequisites:
  - Run 01_basic_rag.py first (to create the ChromaDB store)
  - export OPENAI_API_KEY="sk-..."
============================================================
"""

import os
import sys
import logging

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

# Enable logging to SEE the generated query variations (great for demos!)
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────
# LOAD EXISTING VECTOR STORE
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("Loading existing vector store from ChromaDB...")
print("=" * 60)

if not os.path.exists(CHROMA_DIR):
    print(f"\nERROR: ChromaDB directory not found at '{CHROMA_DIR}'")
    print("Run 01_basic_rag.py first to create the vector store.")
    sys.exit(1)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

print(f"Loaded {vectorstore._collection.count()} vectors from ChromaDB")


# ──────────────────────────────────────────────────────────
# STANDARD RETRIEVAL (baseline)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BASELINE: Standard single-query retrieval")
print("=" * 60)

query = "How does the model learn relationships between words?"

standard_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
standard_docs = standard_retriever.invoke(query)

print(f"\nQuery: '{query}'")
print(f"Retrieved: {len(standard_docs)} chunks\n")
for i, doc in enumerate(standard_docs):
    print(f"  [{i+1}] Page {doc.metadata.get('page', '?')}: {doc.page_content[:120]}...")


# ──────────────────────────────────────────────────────────
# MULTI-QUERY RETRIEVAL
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("IMPROVED: Multi-query retrieval")
print("=" * 60)

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.3  # Slightly creative for generating query variations
)

# MultiQueryRetriever generates 3 query variations automatically
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
)

print(f"\nOriginal query: '{query}'")
print("\nThe retriever will generate variations like:")
print("  → 'What mechanism captures word dependencies in Transformers?'")
print("  → 'How are token relationships modeled in the attention architecture?'")
print("  → 'What allows the model to understand context across a sentence?'")
print("\nWatch the logs below for actual generated queries...\n")

# Retrieve — the logs will show the generated query variations
multi_docs = multi_query_retriever.invoke(query)

print(f"\nResults: {len(multi_docs)} unique chunks (vs {len(standard_docs)} with single query)")


# ──────────────────────────────────────────────────────────
# COMPARE RESULTS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"\nSingle-query retrieval: {len(standard_docs)} chunks")
print(f"Multi-query retrieval:  {len(multi_docs)} chunks")
print(f"Additional chunks found: {len(multi_docs) - len(standard_docs)}")

# Show the multi-query results
print("\nMulti-query chunks:")
for i, doc in enumerate(multi_docs):
    print(f"\n  [{i+1}] Page {doc.metadata.get('page', '?')}:")
    print(f"      {doc.page_content[:200]}...")


# ──────────────────────────────────────────────────────────
# USE IN A FULL RAG CHAIN
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Full RAG chain with multi-query retrieval")
print("=" * 60)

rag_prompt = PromptTemplate(
    template="""Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL, temperature=0),
    chain_type="stuff",
    retriever=multi_query_retriever,
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True,
)

result = qa_chain.invoke({"query": query})
print(f"\nQ: {query}")
print(f"\nA: {result['result']}")
print(f"\nUsed {len(result['source_documents'])} source chunks")

print("\n" + "=" * 60)
print("Key takeaway: Multi-query retrieval finds more relevant")
print("chunks by approaching the same question from multiple angles.")
print("=" * 60)
