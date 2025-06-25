-- Reset threading tables and prepare for comprehensive threading implementation
-- Migration: 006_reset_threading_tables.sql

BEGIN;

-- Drop and recreate conversations table with enhanced schema
DROP TABLE IF EXISTS conversations CASCADE;

CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE, -- Unique thread identifier
    genesis_message_id TEXT NOT NULL, -- Message-ID of the first message in thread
    subject_normalized TEXT NOT NULL, -- Normalized subject (Re:, Fwd: removed)
    
    -- Conversation metadata
    participants JSONB DEFAULT '[]'::jsonb, -- Denormalized list of participants
    message_count INTEGER DEFAULT 0,
    first_message_date TIMESTAMP,
    last_message_date TIMESTAMP,
    
    -- AI-generated summary (updated on each new message)
    summary TEXT, -- One-liner summary of the conversation
    key_topics JSONB DEFAULT '[]'::jsonb, -- Array of key topics/entities
    
    -- Processing status
    summary_generated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Reset thread_id in messages table to reprocess threading
UPDATE messages SET thread_id = NULL WHERE thread_id IS NOT NULL;

-- Indexes for conversations table
CREATE INDEX idx_conversations_thread_id ON conversations(thread_id);
CREATE INDEX idx_conversations_genesis_message_id ON conversations(genesis_message_id);
CREATE INDEX idx_conversations_last_message_date ON conversations(last_message_date);
CREATE INDEX idx_conversations_participants ON conversations USING GIN(participants);
CREATE INDEX idx_conversations_key_topics ON conversations USING GIN(key_topics);
CREATE INDEX idx_conversations_message_count ON conversations(message_count);

-- Index for threading lookup on messages (if not exists)
CREATE INDEX IF NOT EXISTS idx_messages_threading_lookup ON messages(message_id, in_reply_to, email_references);

COMMIT;