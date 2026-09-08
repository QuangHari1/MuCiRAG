FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/workspace/MuCiRAG/.venv/bin:$PATH"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY MuCiRAG/pyproject.toml MuCiRAG/uv.lock /workspace/MuCiRAG/
WORKDIR /workspace/MuCiRAG
RUN uv sync --frozen --no-dev
WORKDIR /workspace

# Ship the implementation and prepared runtime artifacts. Source documents,
# intermediate metadata and experiment outputs are excluded by .dockerignore.
COPY MuCiRAG /workspace/MuCiRAG
COPY dataset /workspace/dataset
COPY README.md AGENTS.md mucirag.py /workspace/
COPY docs /workspace/docs

# Keep a writable destination for a fresh benchmark (or a mounted named volume)
# without shipping a prior run in the image.
RUN mkdir -p /workspace/MuCiRAG/results/teleqna /workspace/MuCiRAG/results/mlflow

# Do not bake API keys into the image. The corpus and stored vectors are fully
# local; a live benchmark still needs the configured query-embedding and LLM
# provider credentials at `docker run` time.
ENTRYPOINT ["python"]
CMD ["mucirag.py", "--help"]
