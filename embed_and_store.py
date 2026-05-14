"""
Reads all_chunks.json, generates embeddings using a free HuggingFace model,
and stores everything in ChromaDB for fast retrieval.

Usage:
    python embed_and_store.py

Requirements:
    pip install chromadb sentence-transformers
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

CHUNKS_FILE = Path("chunks/all_chunks.json")
CHROMA_DIR  = Path("chroma_db")

# --- Load chunks ---
print("Loading chunks...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"  {len(chunks)} chunks loaded.")

# --- Load embedding model (downloads once, cached after) ---
print("\nLoading embedding model (first run downloads ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Model ready.")

# --- Generate embeddings ---
print("\nGenerating embeddings...")
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)
print(f"  Done. Shape: {embeddings.shape}")

# --- Set up ChromaDB ---
print("\nSetting up ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Delete existing collection if re-running to avoid duplicates
existing = [c.name for c in client.list_collections()]
if "university_docs" in existing:
    client.delete_collection("university_docs")
    print("  Existing collection cleared.")

collection = client.create_collection(
    name="university_docs",
    metadata={"hnsw:space": "cosine"}   # cosine similarity for text
)

# --- Store in ChromaDB ---
print("Storing chunks in ChromaDB...")
collection.add(
    ids        = [chunk["id"] for chunk in chunks],
    embeddings = embeddings.tolist(),
    documents  = [chunk["text"] for chunk in chunks],
    metadatas  = [
        {
            "source"  : chunk["source"],
            "category": chunk["category"],
            "section" : chunk["section"],
            "keywords": ", ".join(chunk["keywords"])
        }
        for chunk in chunks
    ]
)

print(f"\n All done! {len(chunks)} chunks stored in '{CHROMA_DIR}/'")
print("You can now build the retrieval + LLM layer.")

# --- Quick test query ---
print("\n--- Quick Test ---")
test_query = "What are the admission requirements?"
test_embedding = model.encode([test_query]).tolist()

results = collection.query(
    query_embeddings = test_embedding,
    n_results        = 3
)

print(f"Query: '{test_query}'\n")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"Result {i+1}: [{meta['category']}] {meta['section']}")
    print(f"  Source : {meta['source']}")
    print(f"  Preview: {doc[:120]}...")
    print()
