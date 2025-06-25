-- Add new pipeline tables for email processing service
-- This migration adds the cleaned up schema for the new email processor

-- Messages table - cleaned and parsed emails from imap_messages
CREATE TABLE IF NOT EXISTS messages (
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
    
    -- Email classification (from future AI step)
    category TEXT, -- personal/promotion/automated
    
    -- Granular processing pipeline stages
    parsed_at TIMESTAMP WITH TIME ZONE, -- mail-parser completed
    cleaned_at TIMESTAMP WITH TIME ZONE, -- email_reply_parser completed
    classified_at TIMESTAMP WITH TIME ZONE, -- AI categorization completed
    embedded_at TIMESTAMP WITH TIME ZONE, -- Unstructured + embeddings completed
    processing_status TEXT DEFAULT 'pending', -- pending/processing/completed/failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique message processing
    UNIQUE(imap_message_id)
);

-- Create indexes for messages table
CREATE INDEX IF NOT EXISTS idx_messages_imap_message_id ON messages(imap_message_id);
CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_from_email ON messages(from_email);
CREATE INDEX IF NOT EXISTS idx_messages_date_sent ON messages(date_sent);
CREATE INDEX IF NOT EXISTS idx_messages_category ON messages(category);
CREATE INDEX IF NOT EXISTS idx_messages_participants ON messages USING GIN(participants);
-- Threading header indexes
CREATE INDEX IF NOT EXISTS idx_messages_references ON messages(references);
CREATE INDEX IF NOT EXISTS idx_messages_in_reply_to ON messages(in_reply_to);
-- Processing stage indexes for queue management
CREATE INDEX IF NOT EXISTS idx_messages_parsed_at ON messages(parsed_at);
CREATE INDEX IF NOT EXISTS idx_messages_cleaned_at ON messages(cleaned_at);
CREATE INDEX IF NOT EXISTS idx_messages_classified_at ON messages(classified_at);
CREATE INDEX IF NOT EXISTS idx_messages_embedded_at ON messages(embedded_at);
CREATE INDEX IF NOT EXISTS idx_messages_processing_status ON messages(processing_status);
-- Composite index for pipeline queue queries
CREATE INDEX IF NOT EXISTS idx_messages_pipeline_status ON messages(processing_status, created_at);