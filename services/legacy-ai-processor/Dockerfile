# AI Processor Dockerfile - Threading and Talon cleaning focused
FROM python:3.11-slim

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create directories
RUN mkdir -p /app/logs /app/models

# Install uv for fast Python package management
RUN pip install uv

# Copy requirements and install Python dependencies with uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Note: Qwen model will be downloaded automatically on first run by modern_llm.py

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose port
EXPOSE 8080

# Start the service (no LLM model download for now)
CMD ["python", "main_queue.py"]