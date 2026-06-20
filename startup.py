"""
Rebuilds the ChromaDB vector database from cleaned_text/ files if it
doesn't already exist. Run this once before starting the Streamlit app.

This is needed for deployment platforms (like HuggingFace Spaces) where
the chroma_db/ folder is not committed to the repo (since it contains
binary files) and must be rebuilt fresh on each deploy.
"""

import os
import subprocess
from pathlib import Path

CHROMA_DIR = Path("chroma_db")

def needs_rebuild() -> bool:
    """Check if ChromaDB needs to be built from scratch."""
    if not CHROMA_DIR.exists():
        return True
    sqlite_file = CHROMA_DIR / "chroma.sqlite3"
    if not sqlite_file.exists():
        return True
    return False


def rebuild():
    print("=" * 50)
    print("ChromaDB not found — rebuilding from cleaned_text/...")
    print("=" * 50)

    # Step 1: Run the chunker
    print("\n[1/2] Running chunker.py...")
    result = subprocess.run(["python", "chunker.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR in chunker.py:")
        print(result.stderr)
        raise SystemExit(1)

    # Step 2: Run the embedder
    print("\n[2/2] Running embed_and_store.py...")
    result = subprocess.run(["python", "embed_and_store.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR in embed_and_store.py:")
        print(result.stderr)
        raise SystemExit(1)

    print("\n" + "=" * 50)
    print("ChromaDB rebuild complete!")
    print("=" * 50)


if __name__ == "__main__":
    if needs_rebuild():
        rebuild()
    else:
        print("ChromaDB already exists — skipping rebuild.")