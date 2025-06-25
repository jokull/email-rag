#!/bin/bash

# Email Classification Worker Runner
# This script runs the host-based email classification queue worker

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="$SCRIPT_DIR/email_classifier_worker.py"
LOG_FILE="$SCRIPT_DIR/email_classifier_worker.log"

# Environment variables (can be overridden)
export DATABASE_URL="${DATABASE_URL:-postgresql://email_user:email_pass@localhost:5432/email_rag}"
export LLM_MODEL_NAME="${LLM_MODEL_NAME:-gpt-4o-mini}"
export MY_EMAIL="${MY_EMAIL:-jokull@solberg.is}"
export BATCH_SIZE="${BATCH_SIZE:-10}"
export SLEEP_INTERVAL="${SLEEP_INTERVAL:-30}"
export MAILBOX_FILTER="${MAILBOX_FILTER:-INBOX}"

echo "🚀 Starting Email Classification Worker"
echo "📁 Working directory: $SCRIPT_DIR"
echo "📧 Processing: $MAILBOX_FILTER mailbox"
echo "🤖 Model: $LLM_MODEL_NAME"
echo "📦 Batch size: $BATCH_SIZE"
echo "⏰ Sleep interval: ${SLEEP_INTERVAL}s"
echo "📋 Log file: $LOG_FILE"
echo ""

# Check dependencies
echo "🔍 Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found"
    exit 1
fi

if ! command -v llm &> /dev/null; then
    echo "❌ llm CLI not found - install with: brew install llm"
    exit 1
fi

# Check if we can import required modules
if ! python3 -c "import llm, sqlalchemy, psycopg2" 2>/dev/null; then
    echo "❌ Required Python modules not found"
    echo "💡 Run: uv add llm sqlalchemy psycopg2-binary email-reply-parser"
    exit 1
fi

echo "✅ Dependencies OK"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Run the worker
echo "🔄 Starting classification worker..."
echo "Press Ctrl+C to stop"
echo ""

# Run with unbuffered output so logs appear immediately
exec python3 -u "$WORKER_SCRIPT"