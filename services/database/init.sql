-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
-- Enable uuid extension for Zero-compatible IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Create go-imap-sql compatible schema
-- go-imap-sql expects specific table structures for IMAP operation
-- IMAP Users table for go-imap-sql
CREATE TABLE imap_users (
    username VARCHAR(255) PRIMARY KEY,
    password_hash VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- IMAP Mailboxes table for go-imap-sql
CREATE TABLE imap_mailboxes (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL REFERENCES imap_users(username) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    uidvalidity BIGINT NOT NULL,
    uidnext BIGINT NOT NULL DEFAULT 1,
    special_use TEXT [],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(username, name)
);
-- IMAP Messages table for go-imap-sql (maps to our emails table)
CREATE TABLE imap_messages (
    id SERIAL PRIMARY KEY,
    mailbox_id INTEGER NOT NULL REFERENCES imap_mailboxes(id) ON DELETE CASCADE,
    uid BIGINT NOT NULL,
    flags TEXT [] DEFAULT ARRAY []::TEXT [],
    internal_date TIMESTAMP WITH TIME ZONE NOT NULL,
    size BIGINT NOT NULL,
    body_structure JSONB,
    envelope JSONB,
    raw_message BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(mailbox_id, uid)
);
-- Create indexes for IMAP tables
CREATE INDEX idx_imap_mailboxes_username ON imap_mailboxes(username);
CREATE INDEX idx_imap_messages_mailbox_id ON imap_messages(mailbox_id);
CREATE INDEX idx_imap_messages_uid ON imap_messages(uid);
CREATE INDEX idx_imap_messages_flags ON imap_messages USING GIN(flags);
-- Insert default IMAP user for email sync
INSERT INTO imap_users (username, password_hash)
VALUES ('email_sync_user', 'not_used_for_sync') ON CONFLICT (username) DO NOTHING;

-- New pipeline tables for email processing

-- Contacts table - normalized contact management
CREATE TABLE contacts (
    email TEXT PRIMARY KEY, -- Normalized email address
    name TEXT, -- Optional current display name
    seen_names JSONB DEFAULT '[]'::jsonb, -- Array of historical names seen
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Messages table - cleaned and parsed emails from imap_messages
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    imap_message_id INTEGER NOT NULL REFERENCES imap_messages(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL, -- Email Message-ID header
    thread_id TEXT, -- Thread identifier from headers
    subject TEXT,
    from_email TEXT NOT NULL,
    to_emails JSONB DEFAULT '[]'::jsonb, -- Array of email addresses
    cc_emails JSONB DEFAULT '[]'::jsonb,
    reply_to TEXT,
    
    -- Raw threading headers for debugging/reprocessing
    references TEXT, -- References header
    in_reply_to TEXT, -- In-Reply-To header
    thread_topic TEXT, -- Thread-Topic header if present
    date_sent TIMESTAMP WITH TIME ZONE NOT NULL,
    date_received TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Cleaned content from mail-parser + email_reply_parser
    body_text TEXT, -- Cleaned plaintext body
    body_html TEXT, -- Original HTML (may be null)
    
    -- Participant information (normalized)
    participants JSONB DEFAULT '[]'::jsonb, -- Array of {email, name} objects
    
    -- Email classification (from Qwen)
    category TEXT, -- personal/promotion/automated
    
    -- Granular processing pipeline stages
    parsed_at TIMESTAMP WITH TIME ZONE, -- mail-parser completed
    cleaned_at TIMESTAMP WITH TIME ZONE, -- email_reply_parser completed
    classified_at TIMESTAMP WITH TIME ZONE, -- Qwen categorization completed
    embedded_at TIMESTAMP WITH TIME ZONE, -- Unstructured + embeddings completed
    processing_status TEXT DEFAULT 'pending', -- pending/processing/completed/failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique message processing
    UNIQUE(imap_message_id)
);

-- Conversations table - threaded email conversations
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL, -- From threading logic
    genesis_message_id TEXT, -- The first message that started the thread
    subject_normalized TEXT, -- Normalized subject line
    
    -- Conversation metadata
    participants JSONB DEFAULT '[]'::jsonb, -- Denormalized participant list
    message_count INTEGER DEFAULT 0,
    first_message_date TIMESTAMP WITH TIME ZONE,
    last_message_date TIMESTAMP WITH TIME ZONE,
    
    -- AI-generated summary (from Qwen)
    summary TEXT, -- One-liner summary of the conversation
    
    -- Processing status
    summary_generated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique threads
    UNIQUE(thread_id)
);

-- Embeddings table - vector embeddings for messages
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    
    -- Embedding metadata
    embedding_model TEXT NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    text_chunk TEXT NOT NULL, -- The text that was embedded
    chunk_index INTEGER DEFAULT 0, -- For multi-chunk messages
    
    -- Vector embedding
    embedding vector(384), -- all-MiniLM-L6-v2 produces 384-dimensional vectors
    
    -- Processing metadata
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique embeddings per message chunk
    UNIQUE(message_id, chunk_index)
);

-- Create indexes for performance

-- Contacts table indexes
CREATE INDEX idx_contacts_name ON contacts(name);
CREATE INDEX idx_contacts_seen_names ON contacts USING GIN(seen_names);
CREATE INDEX idx_contacts_updated_at ON contacts(updated_at);

-- Messages table indexes
CREATE INDEX idx_messages_imap_message_id ON messages(imap_message_id);
CREATE INDEX idx_messages_message_id ON messages(message_id);
CREATE INDEX idx_messages_thread_id ON messages(thread_id);
CREATE INDEX idx_messages_from_email ON messages(from_email);
CREATE INDEX idx_messages_date_sent ON messages(date_sent);
CREATE INDEX idx_messages_category ON messages(category);
CREATE INDEX idx_messages_participants ON messages USING GIN(participants);
-- Threading header indexes
CREATE INDEX idx_messages_references ON messages(references);
CREATE INDEX idx_messages_in_reply_to ON messages(in_reply_to);
-- Processing stage indexes for queue management
CREATE INDEX idx_messages_parsed_at ON messages(parsed_at);
CREATE INDEX idx_messages_cleaned_at ON messages(cleaned_at);
CREATE INDEX idx_messages_classified_at ON messages(classified_at);
CREATE INDEX idx_messages_embedded_at ON messages(embedded_at);
CREATE INDEX idx_messages_processing_status ON messages(processing_status);
-- Composite index for pipeline queue queries
CREATE INDEX idx_messages_pipeline_status ON messages(processing_status, created_at);

-- Conversations table indexes
CREATE INDEX idx_conversations_thread_id ON conversations(thread_id);
CREATE INDEX idx_conversations_genesis_message_id ON conversations(genesis_message_id);
CREATE INDEX idx_conversations_last_message_date ON conversations(last_message_date);
CREATE INDEX idx_conversations_participants ON conversations USING GIN(participants);
CREATE INDEX idx_conversations_message_count ON conversations(message_count);

-- Embeddings table indexes
CREATE INDEX idx_embeddings_message_id ON embeddings(message_id);
CREATE INDEX idx_embeddings_model ON embeddings(embedding_model);
-- Vector similarity search index
CREATE INDEX idx_embeddings_vector ON embeddings USING hnsw (embedding vector_cosine_ops);
