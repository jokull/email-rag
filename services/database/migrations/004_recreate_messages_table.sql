-- Migration 004: Recreate messages table with fixed column types and names
-- Remove timezone info and fix date_received -> processed_at

BEGIN;

-- Drop existing messages table (this will lose current data)
DROP TABLE IF EXISTS messages CASCADE;

-- Recreate messages table with proper column types
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    imap_message_id INTEGER NOT NULL REFERENCES imap_messages(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    thread_id TEXT,
    subject TEXT,
    from_email TEXT NOT NULL,
    to_emails JSONB DEFAULT '[]'::jsonb,
    cc_emails JSONB DEFAULT '[]'::jsonb,
    reply_to TEXT,
    
    -- Raw threading headers for debugging/reprocessing
    email_references TEXT,  -- References header (renamed from 'references' to avoid SQL keyword)
    in_reply_to TEXT,  -- In-Reply-To header
    thread_topic TEXT,  -- Thread-Topic header if present
    date_sent TIMESTAMP NOT NULL,
    processed_at TIMESTAMP DEFAULT NOW(),
    
    -- Cleaned content from mail-parser + email_reply_parser
    body_text TEXT,
    body_html TEXT,
    
    -- Participant information (normalized)
    participants JSONB DEFAULT '[]'::jsonb,
    
    -- Email classification (from future AI step)
    category TEXT,
    
    -- Granular processing pipeline stages
    parsed_at TIMESTAMP,
    cleaned_at TIMESTAMP,
    classified_at TIMESTAMP,
    embedded_at TIMESTAMP,
    processing_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_messages_imap_message_id ON messages(imap_message_id);
CREATE INDEX idx_messages_message_id ON messages(message_id);
CREATE INDEX idx_messages_thread_id ON messages(thread_id);
CREATE INDEX idx_messages_from_email ON messages(from_email);
CREATE INDEX idx_messages_date_sent ON messages(date_sent);
CREATE INDEX idx_messages_processing_status ON messages(processing_status);

-- Also recreate contacts and conversations tables for consistency
DROP TABLE IF EXISTS contacts CASCADE;
CREATE TABLE contacts (
    email TEXT PRIMARY KEY,
    name TEXT,
    seen_names JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS conversations CASCADE;
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    genesis_message_id TEXT,
    subject_normalized TEXT,
    
    -- Conversation metadata
    participants JSONB DEFAULT '[]'::jsonb,
    message_count INTEGER DEFAULT 0,
    first_message_date TIMESTAMP,
    last_message_date TIMESTAMP,
    
    -- AI-generated summary (from future step)
    summary TEXT,
    
    -- Processing status
    summary_generated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for conversations
CREATE INDEX idx_conversations_thread_id ON conversations(thread_id);
CREATE INDEX idx_conversations_genesis_message_id ON conversations(genesis_message_id);

COMMIT;