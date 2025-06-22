# Email Scorer Dockerfile - Lightweight scoring service
FROM python:3.11-slim

# Install system dependencies including llama.cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install llama.cpp for Qwen-0.5B inference  
RUN git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp \
    && cd /tmp/llama.cpp \
    && make \
    && cp main /usr/local/bin/llama-cli \
    && rm -rf /tmp/llama.cpp

# Set working directory
WORKDIR /app

# Create directories
RUN mkdir -p /app/logs /app/models

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create script to download Qwen model if not present (shared with ai-processor)
RUN echo '#!/bin/bash\n\
if [ ! -f "/app/models/qwen-0.5b-q4_0.gguf" ]; then\n\
    echo "Downloading Qwen-0.5B model..."\n\
    curl -L "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf" \\\n\
         -o "/app/models/qwen-0.5b-q4_0.gguf"\n\
    echo "Model downloaded successfully"\n\
fi' > /app/download_model.sh && chmod +x /app/download_model.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8081/health')" || exit 1

# Expose port
EXPOSE 8081

# Start script that downloads model and starts the service
CMD ["/bin/bash", "-c", "/app/download_model.sh && python main.py"]