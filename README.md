---
title: UniBot
emoji: 🎓
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
startup_duration_timeout: 30m
---

# UniBot — AI Helpdesk for FAST-NUCES

An AI-powered university helpdesk chatbot using Retrieval-Augmented Generation (RAG).

## 🔗 Live Demo
https://huggingface.co/spaces/neamahk/UNIBOT

## Features
- Answers questions about fees, admissions, scholarships, rules, programs, and faculty
- Powered by LLaMA 3.1 via Groq API
- Semantic search using ChromaDB + SentenceTransformers
- Streaming responses, conversation memory, follow-up suggestions
- Admin panel for uploading new documents

## Tech Stack
Streamlit, ChromaDB, Groq API, SentenceTransformers, Python