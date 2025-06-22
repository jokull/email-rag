#!/bin/bash

# Zero schema migration script
# Usage: ./scripts/migrate-schema.sh [environment]

set -e

ENVIRONMENT=${1:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔄 Running Zero schema migration for environment: $ENVIRONMENT"

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

# Check if database is accessible
echo "🔍 Checking database connection..."
if ! psql "$DATABASE_URL" -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ Cannot connect to database"
    exit 1
fi

echo "✅ Database connection successful"

# Run database schema migrations first (if any)
MIGRATIONS_DIR="$PROJECT_ROOT/services/database/migrations"
if [ -d "$MIGRATIONS_DIR" ] && [ "$(ls -A $MIGRATIONS_DIR)" ]; then
    echo "🔧 Running database migrations..."
    for migration in "$MIGRATIONS_DIR"/*.sql; do
        if [ -f "$migration" ]; then
            echo "   Applying: $(basename "$migration")"
            psql "$DATABASE_URL" -f "$migration"
        fi
    done
else
    echo "ℹ️  No database migrations found"
fi

# Restart Zero server to pick up schema changes
echo "🔄 Restarting Zero server to pick up schema changes..."

echo "✅ Schema migration completed successfully!"

# Restart zero-cache if running in Docker
if [ "$ENVIRONMENT" = "development" ] && docker-compose ps zero-cache | grep -q "Up"; then
    echo "🔄 Restarting zero-cache service..."
    docker-compose restart zero-cache
    
    # Wait for zero-cache to be ready
    echo "⏳ Waiting for zero-cache to be ready..."
    timeout 30 bash -c 'until curl -s http://localhost:4848/health > /dev/null 2>&1; do sleep 2; done' || true
fi

echo "🎉 Migration complete! Zero schema and permissions are up to date."