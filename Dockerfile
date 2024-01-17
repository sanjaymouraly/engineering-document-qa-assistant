# Engineering RAG Assistant — container image.
#
# One image serves both entry points:
#   docker run … python -m engrag web   # Streamlit UI on 8501
#   docker run … python -m engrag api   # FastAPI on 8502
#
# Everything runs offline inside the image: ingestion, FTS5/hybrid search,
# extractive answering, citation and engineering-unit verification.  An LLM
# provider (Ollama / OpenAI / Anthropic / OpenRouter / custom) is optional and
# configured at runtime through the UI or ENGRAG_* environment variables.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENGRAG_DATA_DIR=/data

WORKDIR /app

# Install the package first (only the source tree is needed at build time).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . \
 && useradd --create-home --uid 10001 engrag \
 && mkdir -p /data \
 && chown -R engrag:engrag /app /data

USER engrag

VOLUME ["/data"]
EXPOSE 8501 8502

# Default to the human-facing UI; override with `python -m engrag api`.
CMD ["python", "-m", "engrag", "web"]
