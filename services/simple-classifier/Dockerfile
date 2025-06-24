FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    fastapi \
    uvicorn \
    python-multipart \
    requests

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8083/health')" || exit 1

# Expose port
EXPOSE 8083

# Start the service
CMD ["python", "main.py"]