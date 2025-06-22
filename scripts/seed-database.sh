#!/bin/bash

# Seed database with sample data
# Usage: ./scripts/seed-database.sh [environment]

set -e

ENVIRONMENT=${1:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🌱 Seeding database for environment: $ENVIRONMENT"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Set database URL based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "❌ Cannot seed production database"
    exit 1
else
    DATABASE_URL="${DATABASE_URL:-postgresql://email_user:email_pass@localhost:5432/email_rag}"
fi

echo "📊 Using database: $DATABASE_URL"

# Create seed data SQL
SEED_FILE="$PROJECT_ROOT/scripts/seed-data.sql"

cat > "$SEED_FILE" << 'EOF'
-- Seed data for email-rag development

-- Insert sample user preferences
INSERT INTO user_preferences (id, preference_key, preference_value, created_at, updated_at) VALUES
  ('pref_001', 'daily_processing_budget', '{"classification_tokens": 100000, "embedding_tokens": 50000}', NOW(), NOW()),
  ('pref_002', 'processing_enabled', 'true', NOW(), NOW()),
  ('pref_003', 'ui_theme', '"dark"', NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

-- Insert sample sender rules
INSERT INTO sender_rules (id, rule_type, pattern, is_active, created_at, updated_at) VALUES
  ('rule_001', 'blacklist', '@noreply.com', true, NOW(), NOW()),
  ('rule_002', 'blacklist', '@notifications.', true, NOW(), NOW()),
  ('rule_003', 'whitelist', '@important-client.com', true, NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

-- Insert sample thread (this would normally come from IMAP sync)
INSERT INTO threads (id, subject_normalized, participants, message_count, last_message_date, created_at, updated_at) VALUES
  ('thread_001', 'Welcome to the project', '["alice@example.com", "bob@example.com"]', 3, NOW() - INTERVAL '1 hour', NOW() - INTERVAL '2 days', NOW())
ON CONFLICT (id) DO NOTHING;

-- Insert sample email
INSERT INTO emails (id, thread_id, message_id, from_email, from_name, to_email, subject, body_text, date_sent, created_at, updated_at) VALUES
  ('email_001', 'thread_001', '<msg001@example.com>', 'alice@example.com', 'Alice Smith', 'bob@example.com', 'Welcome to the project', 'Hi Bob, welcome to our new email RAG project! This is a sample conversation to demonstrate the system.', NOW() - INTERVAL '2 days', NOW() - INTERVAL '2 days', NOW())
ON CONFLICT (id) DO NOTHING;

-- Insert sample classification
INSERT INTO classifications (id, thread_id, classification, human_score, personal_score, relevance_score, should_process, model_used, tokens_used, created_at, updated_at) VALUES
  ('class_001', 'thread_001', 'human', 0.95, 0.85, 0.90, true, 'qwen2.5:3b', 125, NOW() - INTERVAL '1 day', NOW() - INTERVAL '1 day')
ON CONFLICT (id) DO NOTHING;

-- Insert sample processing queue entry
INSERT INTO processing_queue (id, thread_id, queue_type, status, priority, retry_count, created_at, updated_at) VALUES
  ('queue_001', 'thread_001', 'embedding', 'pending', 1, 0, NOW() - INTERVAL '1 hour', NOW() - INTERVAL '1 hour')
ON CONFLICT (id) DO NOTHING;

-- Insert sample processing stats
INSERT INTO processing_stats (id, date, stat_type, stat_value, created_at, updated_at) VALUES
  ('stat_001', CURRENT_DATE::text, 'classification_tokens', 5000, NOW(), NOW()),
  ('stat_002', CURRENT_DATE::text, 'embedding_tokens', 2500, NOW(), NOW()),
  ('stat_003', CURRENT_DATE::text, 'threads_processed', 12, NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

COMMIT;
EOF

# Execute seed data
echo "🔧 Executing seed data..."
psql "$DATABASE_URL" -f "$SEED_FILE"

# Clean up
rm -f "$SEED_FILE"

echo "✅ Database seeded successfully!"
echo "🔍 You can now view sample data in the UI at http://localhost:3001"