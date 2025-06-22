#!/bin/bash

# Development setup script
# Sets up the complete development environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Setting up email-rag development environment"

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

if ! command -v bun &> /dev/null; then
    echo "❌ Bun is required but not installed"
    echo "   Install from: https://bun.sh"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create .env if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "📝 Creating .env file from template..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "⚠️  Please edit .env file with your IMAP credentials before continuing"
    read -p "Press Enter to continue after editing .env file..."
fi

# Load environment variables
export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)

echo "📦 Installing dependencies..."

# Install UI dependencies
cd "$PROJECT_ROOT/web"
bun install

# Return to project root
cd "$PROJECT_ROOT"

echo "🐳 Starting Docker services..."
docker-compose up -d postgres

# Wait for postgres to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
timeout 60 bash -c 'until docker-compose exec postgres pg_isready -U email_user -d email_rag; do sleep 2; done'

echo "🔐 Deploying Zero permissions..."
./scripts/deploy-permissions.sh development

echo "🌱 Seeding database with sample data..."
./scripts/seed-database.sh development

echo "🎉 Development environment setup complete!"
echo ""
echo "🚀 To start the complete system:"
echo "   docker-compose up -d"
echo ""
echo "🔧 Individual service commands:"
echo "   UI Development:     cd web && bun run dev"
echo "   Zero Server:        docker-compose up zero-cache"
echo "   View Logs:          docker-compose logs -f [service-name]"
echo ""
echo "🌐 Access URLs:"
echo "   UI:                 http://localhost:3001"
echo "   Zero Server:        http://localhost:4848"
echo "   PostgreSQL:         localhost:5432"