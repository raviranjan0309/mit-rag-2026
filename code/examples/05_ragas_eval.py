"""
============================================================
RAG Workshop — Example 05: RAG Evaluation with RAGAS
============================================================
Measure your RAG system's quality using three key metrics:

  - Faithfulness:       Does the answer only use facts from the context?
  - Answer Relevancy:   Does the answer actually address the question?
  - Context Precision:  Were the retrieved chunks useful?

Why evaluate?
  "A RAG that feels good can still be faithless."
  Systematic measurement catches problems human review misses.

Usage:
  python 05_ragas_eval.py

Prerequisites:
  - Run 01_basic_rag.py first (to create the ChromaDB store)
  - pip install ragas datasets
  - export OPENAI_API_KEY="sk-..."
============================================================
"""

import os
import sys

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
# STEP 1: Generate answers from your RAG pipeline
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating answers from your RAG pipeline...")
print("=" * 60)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

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
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True,
)


# ──────────────────────────────────────────────────────────
# STEP 2: Define evaluation questions with ground truth
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preparing evaluation dataset...")
print("=" * 60)

# These are questions where we KNOW the correct answer from the paper
eval_questions = [
    "What is multi-head attention?",
    "How many encoder layers are in the base model?",
    "What optimizer was used for training?",
    "What is the dimensionality of the model?",
]

ground_truths = [
    "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. It runs h parallel attention functions.",
    "The base Transformer model uses N=6 identical encoder layers.",
    "The Adam optimizer was used with specific learning rate scheduling.",
    "The base model uses d_model = 512 dimensions.",
]

# Generate RAG answers and collect retrieved contexts
print(f"\nRunning {len(eval_questions)} questions through the RAG pipeline...\n")

answers = []
contexts = []

for i, question in enumerate(eval_questions):
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    context_texts = [doc.page_content for doc in result["source_documents"]]

    answers.append(answer)
    contexts.append(context_texts)

    print(f"  Q{i+1}: {question}")
    print(f"  A:  {answer[:100]}...")
    print(f"  Retrieved {len(context_texts)} chunks\n")


# ──────────────────────────────────────────────────────────
# STEP 3: Evaluate with RAGAS
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Evaluating with RAGAS metrics...")
print("=" * 60)

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    # Build the evaluation dataset in RAGAS format
    eval_data = {
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    dataset = Dataset.from_dict(eval_data)

    print("\nRunning RAGAS evaluation (this may take 1-2 minutes)...\n")

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    # Display results
    print("=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)

    for metric, score in results.items():
        if isinstance(score, float):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"\n  {metric:25s}  {bar}  {score:.3f}")

    print(f"\n{'─' * 60}")
    print("\n  Score interpretation:")
    print("    0.9 - 1.0  →  Excellent")
    print("    0.7 - 0.9  →  Good (typical for well-built RAG)")
    print("    0.5 - 0.7  →  Needs improvement")
    print("    < 0.5      →  Significant issues")

    # Per-question breakdown
    df = results.to_pandas()
    print(f"\n{'─' * 60}")
    print("\n  Per-question breakdown:\n")
    for i, row in df.iterrows():
        print(f"  Q{i+1}: {row['question'][:50]}...")
        print(f"      Faithfulness: {row.get('faithfulness', 'N/A'):.3f}  |  "
              f"Relevancy: {row.get('answer_relevancy', 'N/A'):.3f}  |  "
              f"Precision: {row.get('context_precision', 'N/A'):.3f}")
        print()

except ImportError:
    print("\n  RAGAS is not installed. Install it with:")
    print("    pip install ragas datasets")
    print("\n  Here's what the output would look like:\n")
    print("  faithfulness              ████████████████████  0.950")
    print("  answer_relevancy          █████████████████░░░  0.870")
    print("  context_precision         ██████████████████░░  0.910")
    print("\n  These scores tell you:")
    print("    - Faithfulness: 95% of claims are supported by retrieved context")
    print("    - Answer Relevancy: 87% of answers directly address the question")
    print("    - Context Precision: 91% of retrieved chunks were actually useful")


print("\n" + "=" * 60)
print("  WHAT TO DO WITH THESE SCORES")
print("=" * 60)
print("""
  Low Faithfulness?
    → Your LLM is adding information not in the context
    → Fix: Stricter prompt, lower temperature, better chunking

  Low Answer Relevancy?
    → Answers are off-topic or incomplete
    → Fix: Better retrieval (more chunks, re-ranking)

  Low Context Precision?
    → Retrieved chunks are irrelevant noise
    → Fix: Smaller chunks, hybrid search, metadata filtering

  The #1 improvement for most RAG systems:
    Add re-ranking (see 02_multi_query_rag.py)
""")
print("=" * 60)
