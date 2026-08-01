FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/model-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3-pip \
      stockfish git curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# arcmark package (adjust path if yours is elsewhere in the repo)
COPY arcmark/ /arcmark/
RUN pip install --no-cache-dir -e /arcmark

# Application source
COPY bam/            ./bam/
COPY server.py       .
COPY session.py      .
COPY chess_engine.py .
COPY lm_backend_shared.py .
COPY static/         ./static/

# Model weights are downloaded at runtime into a mounted network volume
VOLUME ["/model-cache"]
EXPOSE 8000

ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
    STOCKFISH_PATH="/usr/games/stockfish" \
    SKILL_LEVEL="5" \
    MAX_SESSIONS="4"

CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "120"]
