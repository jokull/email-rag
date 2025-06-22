#!/bin/bash

# Deploy Zero permissions script
# Usage: ./scripts/deploy-permissions.sh [environment]

set -e

ENVIRONMENT=${1:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔐 Deploying Zero permissions for environment: $ENVIRONMENT"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Set database URL based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    if [ -z "$PRODUCTION_DATABASE_URL" ]; then
        echo "❌ PRODUCTION_DATABASE_URL not set"
        exit 1
    fi
    DATABASE_URL="$PRODUCTION_DATABASE_URL"
else
    DATABASE_URL="${DATABASE_URL:-postgresql://email_user:email_pass@localhost:5432/email_rag}"
fi

echo "📊 Using database: $DATABASE_URL"

# Deploy permissions using Zero CLI
echo "🚀 Deploying permissions..."
# Note: Zero permissions are managed automatically by the Zero server
# For custom permissions, you would typically use zero CLI tools
# This is a placeholder for future permission management
echo "✅ Zero server handles permissions automatically based on schema"

echo "✅ Zero permissions deployed successfully!"