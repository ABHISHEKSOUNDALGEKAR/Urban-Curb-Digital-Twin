# ---------------------------------------------------------------------------
# Reproducible environment for the Urban Curb Digital Twin.
#
#   docker build -t curb-twin .
#   docker run --rm -v "$PWD/results:/app/results" curb-twin \
#       python -m src.experiments.runner --all --seeds 30
#
# The image includes SUMO (via the eclipse-sumo wheel), so the microsimulation
# backend and its tests run inside the container without any host setup.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# libxml2/proj are runtime dependencies of the SUMO binaries; the rest keeps the
# layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 libproj25 libgdal32 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir eclipse-sumo traci sumolib

COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY README.md Makefile ./
RUN mkdir -p results data

ENV PYTHONPATH=/app

# Fail the build if the model itself is broken.
RUN python -m pytest tests -q -m "not slow" \
    && python scripts/generate_data.py

CMD ["python", "-m", "src.experiments.runner", "--scenario", "baseline", "--seeds", "10"]
