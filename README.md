# UniBot — AI Helpdesk for FAST-NUCES

An AI-powered university helpdesk chatbot using Retrieval-Augmented Generation (RAG). UniBot answers student questions about fees, admissions, scholarships, academic rules, programs, transport, hostel facilities, and faculty — sourced exclusively from official university documents, with every answer cited to its source.

## 🔗 Live Demo

[https://uni-bot-pevehupkhvenvm5x4bxsl7.streamlit.app/](https://uni-bot-pevehupkhvenvm5x4bxsl7.streamlit.app/)

## Features

- **Grounded answers** — responses are generated only from retrieved document chunks, with source and section citations on every answer
- **Conversation memory** — understands follow-up questions in context of the previous exchange
- **Streaming responses** — answers appear token-by-token in real time
- **Topic categorization** — each answer is tagged with a colored category badge (Fees, Admissions, Scholarships, etc.)
- **AI-generated follow-up questions** — suggests 3 relevant next questions after each answer
- **Feedback collection** — thumbs up/down on every answer, logged for review
- **Admin panel** — upload new `.docx`/`.txt` documents and they're automatically chunked, embedded, and added to the knowledge base
- **Question logging** — every query is logged with category, answered status, and feedback for gap analysis

## Tech Stack

| Component | Technology |
|---|---|
| Web Interface | Streamlit |
| LLM Inference | Groq API — LLaMA 3.1 8B |
| Vector Database | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 (SentenceTransformers) |
| Document Parsing | python-docx |
| Data Logging | pandas, csv |

## Architecture

```
University Documents → Chunker → Embeddings → ChromaDB (Knowledge Base)

Student Question → Embedding → ChromaDB Search → Top 5 Chunks
                                                        ↓
                                    Groq API (LLaMA 3.1) + Conversation History
                                                        ↓
                                          Streamed, Cited Answer
```

## Setup (Run Locally)

**1. Clone the repository**
```bash
git clone https://github.com/neamah-k/UNI-BOT.git
cd UNI-BOT
```

**2. Create a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Get a free key at [console.groq.com](https://console.groq.com), then create a `.env` file in the project root:
```
GROQ_API_KEY=your-key-here
```

**5. Build the knowledge base**
```bash
python chunker.py
python embed_and_store.py
```

**6. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The admin panel is available at `http://localhost:8501/admin`.

## Deployment

This app is deployed on **Streamlit Community Cloud**. To deploy your own copy:

1. Push your fork to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app** → select your repo, branch `main`, main file `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
5. A `runtime.txt` file (pinned to `python-3.11`) is included to avoid compatibility issues with newer Python versions
6. On first load, the app automatically builds the ChromaDB knowledge base from `cleaned_text/` (takes 2-3 minutes); subsequent loads are instant

## Project Structure

```
├── app.py                  # Main Streamlit chat interface
├── pages/
│   └── admin.py             # Admin panel for document uploads
├── chunker.py               # Splits cleaned text into chunks
├── docx_to_txt.py            # Converts .docx sources to .txt
├── embed_and_store.py        # Generates embeddings, builds ChromaDB
├── cleaned_text/             # Source documents (university info)
├── runtime.txt                # Pins Python version for deployment
└── requirements.txt
```

## Adding New Documents

Use the admin panel (`/admin`) to upload `.docx` or `.txt` files directly — they're automatically chunked, embedded, and added to the knowledge base without restarting the app.

## Notes

This project was built as an AI course project demonstrating a complete RAG pipeline: document ingestion, semantic chunking, vector embedding, retrieval, and LLM-based generation with source grounding.

## License
This project is open source and available under the [MIT License](LICENSE).