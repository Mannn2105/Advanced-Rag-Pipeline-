FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_TF=0 \
    TRANSFORMERS_NO_TF=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.4.1 \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "if ! python -c 'import json,pathlib,config,sys; p=pathlib.Path(config.QDRANT_DIR)/\"embedding_config.json\"; expected={\"model\":config.EMBED_MODEL,\"dimension\":config.EMBED_DIMENSION}; sys.exit(0 if p.exists() and json.loads(p.read_text())==expected else 1)'; then python ingest.py; fi; uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
