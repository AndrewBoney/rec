# syntax=docker/dockerfile:1
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY rec/ ./rec/
COPY config/ ./config/

# Create directories for runtime data
RUN mkdir -p /data/chroma /models/retrieval /models/ranking

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /data /models
USER appuser

# Environment defaults (override with -e or docker-compose)
ENV PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run command (mount models/config as volumes)
CMD ["python", "-m", "rec.deploy_api", \
     "--config", "/app/config/movielens/movielens_1m_large.yaml", \
     "--retrieval-bundle", "/models/retrieval", \
     "--ranking-bundle", "/models/ranking", \
     "--chroma-path", "/data/chroma"]
