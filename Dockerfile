FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Build ChromaDB at image build time (not at runtime)
# This runs chunker.py then embed_and_store.py once, baking the
# vector database into the image so app startup is instant.
RUN python chunker.py && python embed_and_store.py

# HuggingFace Spaces expects the app on port 7860
EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
