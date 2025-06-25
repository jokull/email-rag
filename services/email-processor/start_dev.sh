#!/bin/bash
set -e

echo "Starting Email Processor in development mode..."

# Check if we have the required environment variables
if [ -z "$DATABASE_URL" ]; then
    export DATABASE_URL="postgresql://email_user:email_pass@localhost:5433/email_rag"
    echo "Using default DATABASE_URL: $DATABASE_URL"
fi

# Install Python dependencies if requirements.txt is newer than last install
if [ ! -f .venv/pyvenv.cfg ] || [ requirements.txt -nt .venv/pyvenv.cfg ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Try to build Rust bindings (this will fail initially but show the user what to do)
echo "Attempting to build Rust bindings..."
if ! maturin develop; then
    echo ""
    echo "⚠️  Rust bindings build failed. This is expected on first run."
    echo "   The service will use Python email parsing as fallback."
    echo "   To enable Rust bindings, ensure Rust is installed:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo ""
fi

echo "Starting FastAPI server on http://localhost:8080"
python main.py