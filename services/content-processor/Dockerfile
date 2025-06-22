# Content Processor Dockerfile - Unstructured.io integration
FROM python:3.11-slim

# Install system dependencies for Unstructured
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    pandoc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create directories
RUN mkdir -p /app/logs /app/models

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download MiniLM embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8082/health')" || exit 1

# Expose port
EXPOSE 8082

# Start the content processor service
CMD ["python", "main.py"]