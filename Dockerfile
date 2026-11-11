FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    DOCLING_NUM_THREADS=2 \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[docling,gcp]" \
    && docling-tools models download layout tableformer --output-dir /opt/docling-models

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd --create-home worker && chown -R worker:worker /opt/docling-models
USER worker
CMD ["python", "-m", "resolveops.jobs.pdf_job"]
