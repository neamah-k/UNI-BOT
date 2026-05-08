"""
rag_query.py
------------
Takes a user question, retrieves relevant chunks from ChromaDB,
and sends them to Gemini (free) to generate a grounded answer with citations.

Usage:
    python rag_query.py

Requirements:
    pip install google-generativeai chromadb sentence-transformers python-dotenv
"""

import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# --- Load API key from .env file ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found. Make sure your .env file exists and has the key.")

CHROMA_DIR = "chroma_db"
TOP_K      = 5

# --- Load embedding model ---
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Connect to ChromaDB ---
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("university_docs")

# --- Configure Gemini ---
groq_client = Groq(api_key=API_KEY)


def retrieve(question: str) -> list[dict]:
    """Embed the question and fetch top K matching chunks."""
    embedding = embed_model.encode([question]).tolist()
    results   = collection.query(query_embeddings=embedding, n_results=TOP_K)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text"    : doc,
            "source"  : meta["source"],
            "section" : meta["section"],
            "category": meta["category"],
        })
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build the prompt with retrieved context."""
    context_blocks = ""
    for i, chunk in enumerate(chunks, 1):
        context_blocks += (
            f"[Source {i}: {chunk['source']} — {chunk['section']}]\n"
            f"{chunk['text']}\n\n"
        )

    return f"""You are a helpful university helpdesk assistant for FAST-NUCES university.
Answer the student's question using ONLY the information provided in the sources below.
If the answer is not in the sources, say: "I don't have that information. Please contact the admin office directly."
Always end your answer by citing which source(s) you used, like: [Source: filename — section].

--- SOURCES ---
{context_blocks}
--- END OF SOURCES ---

Student question: {question}
Note: If the question is about merit, grades, or admission calculation, look carefully in all sources provided.
Answer:"""


def ask(question: str) -> str:
    chunks  = retrieve(question)
    prompt  = build_prompt(question, chunks)
    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",   # free model
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# --- Interactive loop ---
if __name__ == "__main__":
    print("\n UniBot — University Helpdesk (powered by Gemini)")
    print(" Type your question or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print("\nUniBot:", end=" ", flush=True)
        answer = ask(question)
        print(answer)
        print()
