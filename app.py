"""
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import csv
from datetime import datetime

import subprocess
import sys
from pathlib import Path

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

import subprocess
import sys
from pathlib import Path

if not Path("chroma_db/chroma.sqlite3").exists():
    with st.spinner("First-time setup: building knowledge base... (this takes 2-3 minutes)"):
        subprocess.run([sys.executable, "chunker.py"], check=True)
        subprocess.run([sys.executable, "embed_and_store.py"], check=True)

# --- CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }

.stApp { background: #0d1117 !important; }
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 750px !important;
}

h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #b388ff, #ADEFD1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px !important;
}
[data-testid="stCaptionContainer"] p {
    color: #7a6a9a !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
}
hr { border-color: #2d1f4a !important; margin: 0.75rem 0 1.25rem !important; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.15rem 0 !important;
    gap: 0.75rem !important;
}

[data-testid="stChatMessageAvatarUser"] ~ div .stMarkdown {
    background: linear-gradient(135deg, #2a1045, #3d1f6e) !important;
    border: 1px solid #5a2d9a !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.8rem 1rem !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    color: #d8c8f0 !important;
    box-shadow: 0 2px 12px rgba(90,45,154,0.4) !important;
}

[data-testid="stChatMessageAvatarAssistant"] ~ div .stMarkdown {
    background: #1e1030 !important;
    border: 1px solid #3a2060 !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 0.85rem 1.05rem !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    color: #e8e0f8 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4) !important;
}
[data-testid="stChatMessageAvatarAssistant"] ~ div .stMarkdown p  { color: #e8e0f8 !important; }
[data-testid="stChatMessageAvatarAssistant"] ~ div .stMarkdown strong { color: #b388ff !important; }
[data-testid="stChatMessageAvatarAssistant"] ~ div .stMarkdown code {
    background: #140820 !important;
    border: 1px solid #3a2060 !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.82rem !important;
    color: #ce93d8 !important;
}

/* Avatars */
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #b388ff, #7c4dff) !important;
    border-radius: 50% !important;
    box-shadow: 0 0 0 2px rgba(179,136,255,0.4) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #2a1045, #3d1f6e) !important;
    border: 1px solid #5a2d9a !important;
    border-radius: 50% !important;
}

/* Source expander */
[data-testid="stExpander"] {
    background: #1e1030 !important;
    border: 1px solid #3a2060 !important;
    border-radius: 10px !important;
    margin-top: 0.4rem !important;
}
[data-testid="stExpander"] details summary {
    color: #b388ff !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stExpander"] details summary:hover { color: #ce93d8 !important; }
[data-testid="stExpander"] p  { color: #7a6a9a !important; font-size: 0.79rem !important; }
[data-testid="stExpander"] strong { color: #b388ff !important; font-weight: 600 !important; }

/* Chat input */
[data-testid="stChatInput"] {
    background: #1e1030 !important;
    border: 1px solid #3a2060 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #b388ff !important;
    box-shadow: 0 0 0 3px rgba(179,136,255,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e8e0f8 !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    caret-color: #b388ff !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4a3a6a !important; }
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(90deg, #b388ff, #7c4dff) !important;
    border-radius: 8px !important;
    border: none !important;
    transition: opacity 0.2s !important;
}
[data-testid="stChatInputSubmitButton"] button:hover { opacity: 0.85 !important; }

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #b388ff !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #3a2060; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #5a2d9a; }

/* FAQ / follow-up buttons */
div[data-testid="stButton"] button[kind="secondary"] {
    background: #1e1030 !important;
    border: 1px solid #3a2060 !important;
    border-radius: 20px !important;
    color: #7a6a9a !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 0.7rem !important;
    transition: all 0.18s ease !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #2a1545 !important;
    border-color: #b388ff !important;
    color: #b388ff !important;
    box-shadow: 0 0 0 2px rgba(179,136,255,0.12) !important;
    transform: translateY(-1px) !important;
}

/* Message text formatting */
[data-testid="stChatMessage"] p { color: #e8e0f8 !important; font-size: 0.92rem !important; line-height: 1.75 !important; margin-bottom: 0.5rem !important; }
[data-testid="stChatMessage"] strong { color: #b388ff !important; font-weight: 600 !important; }
[data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol { color: #e8e0f8 !important; padding-left: 1.4rem !important; margin: 0.4rem 0 0.6rem !important; }
[data-testid="stChatMessage"] li { font-size: 0.9rem !important; line-height: 1.7 !important; margin-bottom: 0.25rem !important; color: #e8e0f8 !important; }
[data-testid="stChatMessage"] h3 { color: #b388ff !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; margin: 0.8rem 0 0.4rem !important; }

/* Category tags */
.cat-tag {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 100px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.cat-Fees         { background: #2a1a0a; color: #ce93d8; border: 1px solid #8b4a10; }
.cat-Admissions   { background: #1a0a30; color: #b388ff; border: 1px solid #5a2d9a; }
.cat-Rules        { background: #2a0a0a; color: #ef9a9a; border: 1px solid #8b2020; }
.cat-Scholarships { background: #200a28; color: #e040fb; border: 1px solid #6a1a7a; }
.cat-Schedules    { background: #0a1020; color: #64b5f6; border: 1px solid #1a4a8b; }
.cat-Programs     { background: #201a0a; color: #ffeb3b; border: 1px solid #8b7a10; }
.cat-Facilities   { background: #0a2010; color: #69f0ae; border: 1px solid #1a6a3a; }
.cat-Contact      { background: #0a1a20; color: #39d0d8; border: 1px solid #0e7490; }
.cat-General      { background: #1a1020; color: #9a8ab0; border: 1px solid #3a2060; }
.cat-Faculty      { background: #1a1020; color: #9a8ab0; border: 1px solid #3a2060; }
</style>

""", unsafe_allow_html=True)


# --- Load models ---
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
    embedding = embed_model.encode([question]).tolist()  # text to vector
    # vector database -> chromaDB
    results   = collection.query(query_embeddings=embedding, n_results=TOP_K)  # search through vector database 
    # it looks for vectors that are "geometrically close" to the question's vector
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]): # data unpacking
        chunks.append({  
            "text"    : doc,
            "source"  : meta["source"],
            "section" : meta["section"],
            "category": meta["category"],
        })
    return chunks
def is_valid_question(question: str) -> bool:
    """Reject empty, too short, or obviously nonsense inputs."""
    q = question.strip()
    if len(q) < 3:
        return False
    if q in ["?", "??", "!", "...", "lol", "ok", "hi", "hello", "hey"]:
        return False
    return True

HISTORY_TURNS = 4  # number of past exchanges to pass as memory

def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Build messages array with conversation history for memory.""" 
    ctx = ""
    for i, chunk in enumerate(chunks, 1):
        ctx += f"[Source {i}: {chunk['source']} — {chunk['section']}]\n{chunk['text']}\n\n"

    system = f"""You are a strict university helpdesk assistant for FAST-NUCES university.
Your ONLY job is to answer questions related to FAST-NUCES university topics such as admissions, fees, scholarships, programs, schedules, rules, facilities, and faculty.

If the student's question is unrelated to the university (e.g. general knowledge, coding help, jokes, current events, personal advice), respond with:
"I can only answer questions about FAST-NUCES university. Please ask me about admissions, fees, programs, or other university matters."

If the question is university-related but the answer is not in the sources below, say:
"I don't have that information. Please contact the admin office directly."

Otherwise, answer using ONLY the information provided in the sources below.
Format your response clearly using markdown:
- **Bold** for important values like fees, deadlines, percentages, and key terms
- Bullet points for lists of items or steps
- Short paragraphs for explanations
Always end your answer by citing which source(s) you used, like: [Source: filename — section].
Note: If the question is about merit, grades, or admission calculation, look carefully in all sources provided.

--- SOURCES ---
{ctx}--- END OF SOURCES ---"""

    messages = [{"role": "system", "content": system}]

    # Add last N turns of history for memory
    history = st.session_state.messages[-(HISTORY_TURNS * 2):]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})
    return messages


def ask(question: str) -> tuple[str, list[dict]]:
    chunks = retrieve(question)
    messages = build_messages(question, chunks)

    # Stream response token by token
    stream = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = messages,
        stream   = True          #  enables streaming
    )
    return stream, chunks

def get_followups(question: str, answer: str) -> list[str]:
    prompt = f"""A student asked: "{question}"
The answer given was: "{answer}"

Generate exactly 3 short follow-up questions a student might ask next based on this conversation.
Return ONLY a JSON array of 3 strings, no explanation, no markdown, no extra text.
Example: ["Question 1?", "Question 2?", "Question 3?"]"""

    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [{"role": "user", "content": prompt}]
    )
    import json
    try:
        text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return []

def log_question(question: str, answer: str):
    log_file    = "question_log.csv"
    unanswered  = (
        "i don't have that information" in answer.lower() or
        "i can only answer questions about fast-nuces" in answer.lower() or
        "I can only answer questions about FAST-NUCES university. Please ask me about admissions, fees, programs, or other university matters." 
    )
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answered", "answer_preview", "feedback"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            "NO" if unanswered else "YES",
            answer[:150].replace("\n", " "),
            ""
        ])

def detect_category(chunks: list[dict]) -> str:
    from collections import Counter
    cats = [c["category"] for c in chunks]
    return Counter(cats).most_common(1)[0][0] if cats else "General"

def log_feedback(question: str, feedback: str):
    log_file = "question_log.csv"
    if not os.path.isfile(log_file):
        return
    rows = []
    with open(log_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for i in range(len(rows) - 1, 0, -1):
        if len(rows[i]) >= 2 and rows[i][1] == question:
            # append feedback at end if column missing
            while len(rows[i]) < 5:
                rows[i].append("")
            rows[i][4] = feedback
            break
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)    

# --- UI ---
st.title("🎓 UniBot")
st.caption("AI Helpdesk for FAST-NUCES University — Powered by RAG")
st.divider()

# Initialize chat history
if "messages"    not in st.session_state: st.session_state.messages    = []
if "sources"     not in st.session_state: st.session_state.sources     = {}
if "followups"   not in st.session_state: st.session_state.followups   = {}
if "categories"  not in st.session_state: st.session_state.categories  = {}
if "feedback"    not in st.session_state: st.session_state.feedback    = {}
if "faq_trigger" not in st.session_state: st.session_state.faq_trigger = None

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
            cat = st.session_state.categories.get(i, "General")
            st.markdown(f'<div class="cat-tag cat-{cat}">📁 {cat}</div>', unsafe_allow_html=True)

            fb = st.session_state.feedback.get(i)
            if fb:
                st.caption(f"You rated this: {fb}")
            else:
                q_text = st.session_state.messages[i-1]["content"] if i > 0 else ""
                fb_col1, fb_col2, _ = st.columns([1, 1, 8])
                with fb_col1:
                    if st.button("👍", key=f"up_{i}"):
                        st.session_state.feedback[i] = "👍 Helpful"
                        log_feedback(q_text, "helpful")
                        st.rerun()
                with fb_col2:
                    if st.button("👎", key=f"dn_{i}"):
                        st.session_state.feedback[i] = "👎 Not helpful"
                        log_feedback(q_text, "not_helpful")
                        st.rerun()

            with st.expander("📄 Sources used"):
                for chunk in st.session_state.sources[i]:
                    st.markdown(f"**{chunk['source']}** — *{chunk['section']}*")
                    st.caption(chunk["text"][:200] + "...")
                    st.divider()

            # Show follow-ups only on the last assistant message
            if i == len(st.session_state.messages) - 1:
                followups = st.session_state.followups.get(i, [])
                if followups:
                    st.markdown("<p style='color:#484f58; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.8rem;'>You might also ask</p>", unsafe_allow_html=True)
                    fu_cols = st.columns(len(followups))
                    for fi, fq in enumerate(followups):
                        with fu_cols[fi]:
                            if st.button(fq, key=f"fu_{i}_{fi}", use_container_width=True):
                                st.session_state.faq_trigger = fq
                                st.rerun()

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
    if not is_valid_question(question):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown("Please type a valid question about FAST-NUCES university.")
        st.session_state.messages.append({"role": "assistant", "content": "Please type a valid question about FAST-NUCES university."})
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get and show bot response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                stream, chunks = ask(question)

            # Stream tokens live — no st.markdown needed
            answer = st.write_stream(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices[0].delta.content
            )

            category  = detect_category(chunks)
            followups = get_followups(question, answer)
            log_question(question, answer)

            msg_index = len(st.session_state.messages)
            st.session_state.sources[msg_index]    = chunks
            st.session_state.followups[msg_index]  = followups
            st.session_state.categories[msg_index] = category
            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.markdown(f'<div class="cat-tag cat-{category}">📁 {category}</div>', unsafe_allow_html=True)

            fb_col1, fb_col2, _ = st.columns([1, 1, 8])
            with fb_col1:
                if st.button("👍", key="up_new"):
                    st.session_state.feedback[msg_index] = "👍 Helpful"
                    log_feedback(question, "helpful")
                    st.rerun()
            with fb_col2:
                if st.button("👎", key="dn_new"):
                    st.session_state.feedback[msg_index] = "👎 Not helpful"
                    log_feedback(question, "not_helpful")
                    st.rerun()

            with st.expander("📄 Sources used"):
                for chunk in chunks:
                    st.markdown(f"**{chunk['source']}** — *{chunk['section']}*")
                    st.caption(chunk["text"][:200] + "...")
                    st.divider()
            if followups:
                st.markdown("<p style='color:#484f58; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.8rem;'>You might also ask</p>", unsafe_allow_html=True)
                fu_cols = st.columns(len(followups))
                for fi, fq in enumerate(followups):
                    with fu_cols[fi]:
                        if st.button(fq, key=f"fu_new_{fi}", use_container_width=True):
                            st.session_state.faq_trigger = fq
                            st.rerun()

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
    - 📞 Contact info
    """)
    st.divider()
    st.caption("Answers are based only on official university documents. For urgent matters, contact the admin office directly.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages    = []
        st.session_state.sources     = {}
        st.session_state.followups   = {}
        st.session_state.categories  = {} 
        st.session_state.feedback    = {} 
        st.session_state.faq_trigger = None
        st.rerun()
