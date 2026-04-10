"""
============================================================
RAG Workshop — Example 04: With RAG vs Without RAG
============================================================
Side-by-side comparison showing how RAG prevents hallucination.

The same question is asked:
  1. Without RAG → LLM relies on parametric memory (may hallucinate)
  2. With RAG    → LLM uses retrieved context (grounded answers)

This is the most compelling "aha moment" in the workshop.

Usage:
  python 04_rag_vs_no_rag.py

Prerequisites:
  - Run 01_basic_rag.py first (to create the ChromaDB store)
  - export OPENAI_API_KEY="sk-..."
============================================================
"""

import os
import sys
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage


# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────
print("=" * 70)
print("  RAG vs NO-RAG: Side-by-Side Comparison")
print("=" * 70)

# Load vector store
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# RAG chain
rag_prompt = PromptTemplate(
    template="""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough information in the provided documents."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True,
)


# ──────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────
def ask_without_rag(question: str) -> str:
    """Plain LLM — no retrieval, just the model's parametric memory."""
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer the question directly and concisely."),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content


def ask_with_rag(question: str) -> dict:
    """RAG — retrieves relevant chunks first, then generates grounded answer."""
    return qa_chain.invoke({"query": question})


def compare(question: str) -> None:
    """Run the same question through both approaches and display results."""
    print(f"\n{'═' * 70}")
    print(f"  QUESTION: {question}")
    print(f"{'═' * 70}")

    # Without RAG
    print(f"\n  {'─' * 30}")
    print(f"  WITHOUT RAG (Pure LLM)")
    print(f"  {'─' * 30}")
    start = time.time()
    plain_answer = ask_without_rag(question)
    plain_time = time.time() - start
    print(f"\n  {plain_answer}")
    print(f"\n  ⏱ {plain_time:.2f}s | No sources (model's training data only)")

    # With RAG
    print(f"\n  {'─' * 30}")
    print(f"  WITH RAG (Retrieved + Grounded)")
    print(f"  {'─' * 30}")
    start = time.time()
    rag_result = ask_with_rag(question)
    rag_time = time.time() - start
    print(f"\n  {rag_result['result']}")
    print(f"\n  ⏱ {rag_time:.2f}s | Sources:")
    for doc in rag_result["source_documents"]:
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    → Page {page}: {preview}...")

    print()


# ──────────────────────────────────────────────────────────
# TEST 1: Specific factual question about the paper
# ──────────────────────────────────────────────────────────
compare(
    "What is the exact number of encoder and decoder layers in the base Transformer model?"
)

# ──────────────────────────────────────────────────────────
# TEST 2: Detailed technical question
# ──────────────────────────────────────────────────────────
compare(
    "What value of dropout rate did the authors use in their model?"
)

# ──────────────────────────────────────────────────────────
# TEST 3: Question NOT in the document
# ──────────────────────────────────────────────────────────
compare(
    "What programming language was used to implement the Transformer?"
)

# ──────────────────────────────────────────────────────────
# TEST 4: Multi-hop question
# ──────────────────────────────────────────────────────────
compare(
    "How does the positional encoding work and why is it necessary?"
)


# ──────────────────────────────────────────────────────────
# KEY INSIGHTS
# ──────────────────────────────────────────────────────────
print("═" * 70)
print("  KEY INSIGHTS")
print("═" * 70)
print("""
  1. The LLM without RAG may sound confident but could be wrong.
     It relies entirely on what it learned during training.

  2. The RAG answer cites exact values from the paper.
     Every claim is traceable to a specific source chunk.

  3. When the answer ISN'T in the document, a well-prompted RAG
     system says "I don't have enough information" instead of guessing.

  4. RAG adds latency (retrieval step) but dramatically improves
     accuracy for domain-specific questions.
""")
print("═" * 70)
