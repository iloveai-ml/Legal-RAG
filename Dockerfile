# syntax=docker/dockerfile:1

# Nyaya runs comfortably inside a 512 MB Render instance. Two choices keep it
# there: the embedding model is ONNX rather than PyTorch, and the search index
# ships as prebuilt files that are memory mapped instead of loaded into a
# vector database process.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the 90 MB embedding model into the image. Downloading it at boot instead
# would put a cold start on the critical path of somebody's first question, and
# would fail outright if the Hugging Face CDN were unreachable.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')" \
 && find /opt/models -name '*.onnx' -size +1k | head -1

COPY app ./app
COPY data/index ./data/index

# Fail the build rather than deploy an image with no corpus in it
RUN python -c "\
import json,pathlib;\
m=json.loads(pathlib.Path('data/index/manifest.json').read_text());\
assert m['n_chunks']>0, 'index manifest reports no chunks';\
print('index baked in:', m['n_chunks'], 'chunks')"

RUN useradd --create-home --uid 10001 nyaya \
 && chown -R nyaya:nyaya /app /opt/models
USER nyaya

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# One worker on purpose. A second would double the resident index and the
# onnxruntime session for no throughput gain on a shared vCPU.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65"]
