FROM ubuntu:latest
LABEL authors="javed"

ENTRYPOINT ["top", "-b"]

# ─────────────────────────────────────────────────────────────────
# Dockerfile
# Builds the Padel Analytics API as a Docker container.
#
# Build:
#   docker build -t padel-analytics .
#
# Run locally:
#   docker run -p 8000:8000 padel-analytics
#
# Push to Docker Hub:
#   docker tag padel-analytics your-dockerhub-username/padel-analytics
#   docker push your-dockerhub-username/padel-analytics
# ─────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first (Docker layer cache)
COPY requirements.txt .
COPY deploy/requirements_deploy.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements_deploy.txt

# Copy source code
COPY src/       ./src/
COPY deploy/    ./deploy/
COPY main.py    .
COPY setup.py   .

# Install project in editable mode so src/ is on the path
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p data/raw data/outputs data/uploads models/pretrained

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Start the API server
CMD ["uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]