"""
app.py
------
Streamlit chat interface for UniBot — FAST-NUCES University Helpdesk.

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit chromadb sentence-transformers groq python-dotenv
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# --- Config ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR   = "chroma_db"
TOP_K        = 5

# --- Page setup ---
st.set_page_config(
    page_title = "UniBot — FAST-NUCES Helpdesk",
    page_icon  = "🎓",
    layout     = "centered"
)

# --- Custom CSS (styling only — no logic changes) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Page background */
.stApp {
    background: #0d1117 !important;
}
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 750px !important;
}

/* Title */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #58a6ff, #a5d6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px !important;
}
[data-testid="stCaptionContainer"] p {
    color: #6e7681 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
}
hr {
    border-color: #21262d !important;
    margin: 0.75rem 0 1.25rem !important;
}

/* Chat message containers */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.15rem 0 !important;
    gap: 0.75rem !important;
}

/* User bubble */
[data-testid="stChatMessageAvatarUser"] ~ div .stMarkdown {
    background: linear-gradient(135deg, #0d2847, #0f3460) !important;
    border: 1px solid #1f4a7a !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.8rem 1rem !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    color: #cdd9e5 !important;
    box-shadow: 0 2px 12px rgba(13,40,71,0.5) !important;
}

/* Assistant bubble */
[data-testid="stChatMessageAvatarAssistant"] ~ div .stMarkdown {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 0.85rem 1.05rem !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    color: #c9d1d9 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
}
[data-testid="stChatMessageAvatarAssistant"] ~ div .stMarkdown p  { color: #c9d1d9 !important; }
[data-testid="stChatMessageAvatarAssistant"] ~ div .stMarkdown strong { color: #58a6ff !important; }
[data-testid="stChatMessageAvatarAssistant"] ~ div .stMarkdown code {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.82rem !important;
    color: #79c0ff !important;
}

/* Avatars */
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #0f3460, #58a6ff) !important;
    border-radius: 50% !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.3) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #161b22, #21262d) !important;
    border: 1px solid #30363d !important;
    border-radius: 50% !important;
}

/* Source expander */
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    margin-top: 0.4rem !important;
}
[data-testid="stExpander"] details summary {
    color: #58a6ff !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stExpander"] details summary:hover { color: #a5d6ff !important; }
[data-testid="stExpander"] p  { color: #6e7681 !important; font-size: 0.79rem !important; }
[data-testid="stExpander"] strong { color: #58a6ff !important; font-weight: 600 !important; }

/* Chat input */
[data-testid="stChatInput"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    color: #c9d1d9 !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    caret-color: #58a6ff !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #484f58 !important; }
[data-testid="stChatInputSubmitButton"] button {
    background: #238636 !important;
    border-radius: 8px !important;
    border: none !important;
    transition: background 0.2s !important;
}
[data-testid="stChatInputSubmitButton"] button:hover { background: #2ea043 !important; }

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #58a6ff !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #30363d; }

/* FAQ pill buttons */
div[data-testid="stButton"] button[kind="secondary"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 20px !important;
    color: #8b949e !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 0.7rem !important;
    transition: all 0.18s ease !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #1c2128 !important;
    border-color: #58a6ff !important;
    color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.12) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# --- Load models (cached so they only load once) ---
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma      = chromadb.PersistentClient(path=CHROMA_DIR)
    collection  = chroma.get_collection("university_docs")
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embed_model, collection, groq_client

embed_model, collection, groq_client = load_resources()

# --- RAG functions ---
def retrieve(question: str) -> list[dict]:
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
Note: If the question is about merit, grades, or admission calculation, look carefully in all sources provided.

--- SOURCES ---
{context_blocks}
--- END OF SOURCES ---

Student question: {question}

Answer:"""


def ask(question: str) -> tuple[str, list[dict]]:
    chunks   = retrieve(question)
    prompt   = build_prompt(question, chunks)
    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, chunks


# --- UI ---
st.title("🎓 UniBot")
st.caption("AI Helpdesk for FAST-NUCES University — Powered by RAG")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages    = []
    st.session_state.sources     = {}
    st.session_state.faq_trigger = None

# FAQ questions
FAQ_QUESTIONS = [
    ("🎓", "What is the application fee?"),
    ("📊", "How is merit calculated?"),
    ("🏆", "What scholarships are available?"),
    ("📅", "What are the admission dates?"),
    ("📚", "What programs are offered?"),
    ("💰", "What is the fee structure?"),
]

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and i in st.session_state.sources:
            with st.expander("📄 Sources used"):
                for chunk in st.session_state.sources[i]:
                    st.markdown(f"**{chunk['source']}** — *{chunk['section']}*")
                    st.caption(chunk["text"][:200] + "...")
                    st.divider()

# FAQ buttons — only show when chat is empty
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; margin: 2.5rem 0 1rem;">
        <p style="color:#484f58; font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:1rem;">
            Frequently Asked Questions
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: first 3
    cols1 = st.columns(3)
    for idx, (icon, q) in enumerate(FAQ_QUESTIONS[:3]):
        with cols1[idx]:
            if st.button(f"{icon} {q}", key=f"faq_{idx}", use_container_width=True):
                st.session_state.faq_trigger = q
                st.rerun()

    # Row 2: last 3
    cols2 = st.columns(3)
    for idx, (icon, q) in enumerate(FAQ_QUESTIONS[3:]):
        with cols2[idx]:
            if st.button(f"{icon} {q}", key=f"faq_{idx+3}", use_container_width=True):
                st.session_state.faq_trigger = q
                st.rerun()

# Resolve FAQ trigger as if user typed it
question = st.session_state.pop("faq_trigger", None) if st.session_state.get("faq_trigger") else None

# Chat input (overrides faq trigger if both somehow set)
if typed := st.chat_input("Ask about fees, admissions, scholarships, rules..."):
    question = typed

if question:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get and show bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching university documents..."):
            answer, chunks = ask(question)

        st.markdown(answer)

        # Store sources linked to this message index
        msg_index = len(st.session_state.messages)
        st.session_state.sources[msg_index] = chunks
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("📄 Sources used"):
            for chunk in chunks:
                st.markdown(f"**{chunk['source']}** — *{chunk['section']}*")
                st.caption(chunk["text"][:200] + "...")
                st.divider()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About UniBot")
    st.markdown("""
    UniBot answers questions using official FAST-NUCES documents including:
    - 📋 Admission Information
    - 💰 Fee Structure
    - 🏆 Scholarships
    - 📅 Academic Schedule
    - 📚 Programs Offered
    - 🏫 Rules & Regulations
    - 🚌 Transport & Hostel
    """)
    st.divider()
    st.caption("Answers are based only on official university documents. For urgent matters, contact the admin office directly.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages    = []
        st.session_state.sources     = {}
        st.session_state.faq_trigger = None
        st.rerun()
