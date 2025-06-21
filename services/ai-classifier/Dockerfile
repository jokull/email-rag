FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install llm CLI
RUN pip install llm

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 classifier
RUN chown -R classifier:classifier /app
USER classifier

# Set up llm models directory
RUN mkdir -p /home/classifier/.cache/llm

CMD ["python", "main.py"]