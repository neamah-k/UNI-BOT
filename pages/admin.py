import os
import re
import json
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from docx import Document
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
CHROMA_DIR = "chroma_db"

st.set_page_config(page_title="Admin — UniBot", page_icon="⚙️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: #0d1117 !important; }
.main .block-container { padding-top: 1.5rem !important; max-width: 700px !important; }
h1 { font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; font-weight: 800 !important;
     background: linear-gradient(90deg, #58a6ff, #a5d6ff); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent; background-clip: text; }
h3 { color: #c9d1d9 !important; font-size: 1rem !important; font-weight: 600 !important; margin-top: 1.5rem !important; }
p, li { color: #8b949e !important; font-size: 0.88rem !important; }
hr { border-color: #21262d !important; }
[data-testid="stFileUploader"] {
    background: #161b22 !important; border: 1px dashed #30363d !important;
    border-radius: 12px !important; padding: 1rem !important;
}
[data-testid="stSelectbox"] > div { background: #161b22 !important; border-color: #30363d !important; }
.stButton > button {
    background: #238636 !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-size: 0.88rem !important;
    padding: 0.5rem 1.2rem !important; font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover { background: #2ea043 !important; }
.stTextInput input { background: #161b22 !important; border-color: #30363d !important; color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load resources ──
@st.cache_resource
def load_resources():
    em  = SentenceTransformer("all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection("university_docs")
    return em, col

embed_model, collection = load_resources()


# ── Helpers ──
def extract_text_from_docx(file) -> str:
    doc   = Document(file)
    lines = []
    for para in doc.paragraphs:
        text  = para.text.strip()
        style = para.style.name.lower()
        if not text:
            if lines and lines[-1] != "": lines.append("")
            continue
        if "heading" in style:
            if lines and lines[-1] != "": lines.append("")
            lines.append(text); lines.append("")
        elif "list" in style:
            lines.append(f"- {text}" if not text.startswith(("-","•","*")) else text)
        else:
            lines.append(text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8")

def extract_keywords(text: str) -> list[str]:
    stopwords = {"this","that","with","from","have","will","must","into","their","they","been","were","shall"}
    words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", text.lower()) if w not in stopwords]
    return list(dict.fromkeys(words))[:8]

def split_sections(text: str) -> list[str]:
    pattern  = r"\n(?=[A-Z][A-Za-z\s&()/-]{3,}:?\n)"
    sections = re.split(pattern, text)
    return [s.strip() for s in sections if s.strip()]

def chunk_text(text: str, source_name: str, category: str) -> list[dict]:
    sections = split_sections(text)
    chunks   = []
    for i, section in enumerate(sections, 1):
        lines = section.split("\n")
        title = lines[0].replace(":", "").strip()
        body  = "\n".join(lines[1:]).strip()
        if not body:
            continue
        chunks.append({
            "id"      : f"{Path(source_name).stem}_{i:03}",
            "source"  : source_name,
            "category": category,
            "section" : title,
            "keywords": extract_keywords(body),
            "text"    : body
        })
    return chunks

def add_to_chromadb(chunks: list[dict]) -> int:
    texts      = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts).tolist()
    collection.add(
        ids        = [c["id"] for c in chunks],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [{"source": c["source"], "category": c["category"],
                       "section": c["section"], "keywords": ", ".join(c["keywords"])}
                      for c in chunks]
    )
    return len(chunks)


# ── UI ──
st.title("⚙️ Admin Panel")
st.caption("Upload new documents to expand UniBot's knowledge base")
st.divider()

# Stats
try:
    total = collection.count()
    st.markdown(f"**📦 Current knowledge base:** `{total}` chunks indexed")
except:
    pass

st.divider()

st.markdown("### 📤 Upload New Document")
st.markdown("Supported formats: `.docx`, `.txt`")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["docx", "txt"],
    label_visibility="collapsed"
)

CATEGORIES = ["Admissions", "Fees", "Scholarships", "Schedules", "Programs", "Rules", "Facilities", "General"]

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Category", CATEGORIES)
with col2:
    custom_name = st.text_input("Display name (optional)", placeholder="e.g. Fee Structure 2026")

if uploaded_file and st.button("⚡ Process & Add to Knowledge Base", use_container_width=True):
    with st.spinner("Processing document..."):
        try:
            # Extract text
            if uploaded_file.name.endswith(".docx"):
                text = extract_text_from_docx(uploaded_file)
            else:
                text = extract_text_from_txt(uploaded_file)

            if not text.strip():
                st.error("Could not extract text from this file. Please check the file and try again.")
                st.stop()

            # Use custom name or original filename
            source_name = (custom_name.strip() or Path(uploaded_file.name).stem) + ".txt"

            # Chunk
            chunks = chunk_text(text, source_name, category)
            if not chunks:
                st.warning("No sections found in the document. The file may be too short or unstructured.")
                st.stop()

            # Embed & store
            added = add_to_chromadb(chunks)

            st.success(f"✅ Done! Added **{added} chunks** from *{source_name}* to the knowledge base.")

            # Show preview
            with st.expander("📋 Preview chunks added"):
                for c in chunks[:5]:
                    st.markdown(f"**Section:** {c['section']}")
                    st.caption(c["text"][:200] + "...")
                    st.divider()
                if len(chunks) > 5:
                    st.caption(f"...and {len(chunks) - 5} more chunks")

        except Exception as e:
            st.error(f"Error processing file: {e}")

st.divider()
st.markdown("### 🗂️ Knowledge Base Contents")

if st.button("🔍 Show all indexed sources"):
    try:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        sources  = {}
        for m in all_meta:
            src = m.get("source", "Unknown")
            cat = m.get("category", "General")
            sources[src] = sources.get(src, {"count": 0, "category": cat})
            sources[src]["count"] += 1

        for src, info in sorted(sources.items()):
            st.markdown(f"- **{src}** — {info['category']} — `{info['count']} chunks`")
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
st.divider()
st.markdown("### ❓ Unanswered Questions")
st.caption("Questions where UniBot said it didn't have information — use these to identify document gaps.")

import pandas as pd
log_path = "question_log.csv"

if os.path.isfile(log_path):
    df = pd.read_csv(log_path)
    unanswered = df[df["answered"] == "NO"]

    if unanswered.empty:
        st.success("No unanswered questions yet!")
    else:
        st.warning(f"**{len(unanswered)}** unanswered question(s) found")
        st.dataframe(
            unanswered[["timestamp", "question"]],
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.markdown("### 📊 All Questions")
    st.dataframe(df[["timestamp", "question", "answered"]], use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download full log", csv_data, "question_log.csv", "text/csv")
else:
    st.info("No questions logged yet. Start chatting to generate logs.")