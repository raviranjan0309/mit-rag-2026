"""
============================================================
RAG Workshop — Example 03: Conversational RAG with Memory
============================================================
Add chat history so follow-up questions work naturally.

Problem without memory:
  User: "What is the Transformer architecture?"
  AI:   "The Transformer uses self-attention and..."
  User: "How many layers does it use?"
  AI:   "I don't know what 'it' refers to."  ← FAIL

With conversational memory, "it" resolves to "Transformer"
because the chain reformulates the question using history.

Usage:
  python 03_conversational_rag.py

Prerequisites:
  - Run 01_basic_rag.py first (to create the ChromaDB store)
  - export OPENAI_API_KEY="sk-..."
============================================================
"""

import os
import sys

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory


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
# LOAD EXISTING VECTOR STORE
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("Loading vector store...")
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

print(f"Loaded {vectorstore._collection.count()} vectors")


# ──────────────────────────────────────────────────────────
# BUILD CONVERSATIONAL RAG CHAIN
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Building conversational RAG chain with memory...")
print("=" * 60)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Memory: keeps last 5 conversation turns (10 messages: 5 human + 5 AI)
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

# Conversational RAG chain
# Under the hood, this:
#   1. Takes the new question + chat history
#   2. Reformulates the question to be standalone
#      e.g., "How many layers?" → "How many layers does the Transformer use?"
#   3. Uses the standalone question for retrieval
#   4. Generates an answer from retrieved context
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    return_source_documents=True,
    verbose=False,
)

print("Chain ready. Memory window: last 5 turns.\n")


# ──────────────────────────────────────────────────────────
# HELPER FUNCTION
# ──────────────────────────────────────────────────────────
def chat(user_input: str) -> None:
    """Send a message and display the response."""
    print(f"\n{'─' * 50}")
    print(f"  You: {user_input}")
    print(f"{'─' * 50}")
    result = conv_chain.invoke({"question": user_input})
    print(f"\n  AI:  {result['answer']}")
    print(f"\n  [Retrieved {len(result['source_documents'])} chunks]")


# ──────────────────────────────────────────────────────────
# DEMO CONVERSATION
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("DEMO: Conversational RAG in action")
print("=" * 60)
print("\nWatch how follow-up questions resolve correctly:\n")

# Turn 1: Establish the topic
chat("What is the Transformer architecture?")

# Turn 2: "it" refers to Transformer (requires memory)
chat("How many layers does it use?")

# Turn 3: "the attention heads" builds on previous context
chat("What about the attention heads?")

# Turn 4: Comparative question that references earlier context
chat("Can you compare the base and large model configurations?")


# ──────────────────────────────────────────────────────────
# SHOW MEMORY CONTENTS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Current conversation memory:")
print("=" * 60)

for msg in memory.chat_memory.messages:
    role = "You" if msg.type == "human" else "AI "
    content = msg.content[:120].replace("\n", " ")
    print(f"  {role}: {content}...")


# ──────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("INTERACTIVE MODE — Ask your own questions!")
print("Type 'quit' to exit, 'memory' to see chat history.")
print("=" * 60)

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nGoodbye!")
        break

    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nGoodbye!")
        break
    if user_input.lower() == "memory":
        print("\n--- Chat Memory ---")
        for msg in memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "AI "
            print(f"  {role}: {msg.content[:150]}...")
        continue

    chat(user_input)
