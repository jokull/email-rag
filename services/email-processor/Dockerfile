# Multi-stage build for email processor with Rust bindings
FROM rust:1.75-slim as rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    python3-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv in builder
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy Rust project files
COPY Cargo.toml pyproject.toml ./
COPY src/ ./src/

# Install maturin and build the Python wheel
RUN uv pip install --system --break-system-packages maturin
RUN maturin build --release --out dist/

# Python runtime stage
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd -r emailprocessor && useradd -r -g emailprocessor emailprocessor

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy the built Rust wheel from the builder stage
COPY --from=rust-builder /app/dist/*.whl /tmp/
RUN uv pip install --system /tmp/*.whl

# Copy Python application code
COPY *.py ./

# Set ownership to non-root user
RUN chown -R emailprocessor:emailprocessor /app

# Switch to non-root user
USER emailprocessor

# Expose port for FastAPI
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Default command runs the FastAPI server
CMD ["python", "main.py"]