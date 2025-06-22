#!/bin/bash

# Reset development environment
# Cleans up all data and starts fresh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🗑️  Resetting development environment..."
echo "⚠️  This will delete ALL data in the development database!"

read -p "Are you sure you want to continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Reset cancelled"
    exit 1
fi

# Stop all services
echo "🛑 Stopping all services..."
cd "$PROJECT_ROOT"
docker-compose down

# Remove volumes (this deletes all data)
echo "🗂️  Removing volumes..."
docker-compose down -v

# Remove zero cache data
echo "🧹 Cleaning up Zero cache files..."
rm -rf services/zero-cache/node_modules/.cache/zero-cache
rm -f services/zero-cache/*.db*

# Clean up any temporary files
echo "🧽 Cleaning temporary files..."
rm -f scripts/seed-data.sql

# Rebuild and restart
echo "🔄 Rebuilding services..."
docker-compose build --no-cache

echo "🚀 Starting fresh environment..."
./scripts/dev-setup.sh

echo "✨ Development environment reset complete!"
echo "🌐 Access the UI at http://localhost:3001"