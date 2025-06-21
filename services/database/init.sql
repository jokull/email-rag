-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable uuid extension for Zero-compatible IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create emails table (Zero-compatible with string primary keys)
CREATE TABLE emails (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    message_id VARCHAR(255) UNIQUE NOT NULL,
    thread_id VARCHAR(255),
    imap_uid BIGINT,
    mailbox_name VARCHAR(255) DEFAULT 'INBOX',
    from_email VARCHAR(255) NOT NULL,
    from_name VARCHAR(255),
    to_emails TEXT[] NOT NULL,
    cc_emails TEXT[],
    bcc_emails TEXT[],
    subject TEXT,
    body_text TEXT,
    body_html TEXT,
    date_sent TIMESTAMP WITH TIME ZONE NOT NULL,
    date_received TIMESTAMP WITH TIME ZONE NOT NULL,
    raw_headers JSONB,
    attachments JSONB DEFAULT '[]'::jsonb,
    flags TEXT[] DEFAULT ARRAY[]::TEXT[],
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create threads table (Zero-compatible)
CREATE TABLE threads (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    subject_normalized VARCHAR(500) NOT NULL,
    participants TEXT[] NOT NULL,
    first_message_date TIMESTAMP WITH TIME ZONE NOT NULL,
    last_message_date TIMESTAMP WITH TIME ZONE NOT NULL,
    message_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create contacts table (Zero-compatible)
CREATE TABLE contacts (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    frequency_score INTEGER DEFAULT 1,
    relationship_strength FLOAT DEFAULT 0.0,
    first_contact TIMESTAMP WITH TIME ZONE,
    last_contact TIMESTAMP WITH TIME ZONE,
    total_messages INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create classifications table (Zero-compatible)
CREATE TABLE classifications (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    thread_id VARCHAR(255) NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    classification VARCHAR(50) NOT NULL CHECK (classification IN ('human', 'promotional', 'transactional', 'automated')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    model_used VARCHAR(100) NOT NULL,
    reasoning TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(thread_id)
);

-- Create embeddings table with pgvector (Zero-compatible)
CREATE TABLE embeddings (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    thread_id VARCHAR(255) NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(1024), -- Qwen3-Embedding-0.6B produces 1024-dimensional vectors
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(thread_id, chunk_index)
);

-- Add foreign key constraint for thread_id in emails
ALTER TABLE emails ADD CONSTRAINT fk_emails_thread_id 
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE SET NULL;

-- Create indexes for performance
CREATE INDEX idx_emails_message_id ON emails(message_id);
CREATE INDEX idx_emails_thread_id ON emails(thread_id);
CREATE INDEX idx_emails_from_email ON emails(from_email);
CREATE INDEX idx_emails_date_sent ON emails(date_sent);
CREATE INDEX idx_emails_date_received ON emails(date_received);
CREATE INDEX idx_emails_imap_uid ON emails(imap_uid);
CREATE INDEX idx_emails_mailbox ON emails(mailbox_name);

CREATE INDEX idx_threads_subject_normalized ON threads(subject_normalized);
CREATE INDEX idx_threads_participants ON threads USING GIN(participants);
CREATE INDEX idx_threads_last_message_date ON threads(last_message_date);

CREATE INDEX idx_contacts_email ON contacts(email);
CREATE INDEX idx_contacts_relationship_strength ON contacts(relationship_strength);
CREATE INDEX idx_contacts_last_contact ON contacts(last_contact);

CREATE INDEX idx_classifications_thread_id ON classifications(thread_id);
CREATE INDEX idx_classifications_classification ON classifications(classification);
CREATE INDEX idx_classifications_confidence ON classifications(confidence);

CREATE INDEX idx_embeddings_thread_id ON embeddings(thread_id);
CREATE INDEX idx_embeddings_embedding ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_emails_updated_at BEFORE UPDATE ON emails
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_threads_updated_at BEFORE UPDATE ON threads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_contacts_updated_at BEFORE UPDATE ON contacts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for thread normalization
CREATE OR REPLACE FUNCTION normalize_subject(subject TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Remove Re:, Fwd:, etc. and normalize spacing
    RETURN TRIM(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                LOWER(COALESCE(subject, '')),
                '^(re:|fwd:|fw:|forward:|reply:)\s*',
                '',
                'gi'
            ),
            '\s+',
            ' ',
            'g'
        )
    );
END;
$$ LANGUAGE plpgsql;

-- Create notification function for real-time sync (Zero will handle this)
CREATE OR REPLACE FUNCTION notify_email_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM pg_notify('email_changes', json_build_object(
            'operation', 'INSERT',
            'table', TG_TABLE_NAME,
            'id', NEW.id,
            'data', row_to_json(NEW)
        )::text);
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        PERFORM pg_notify('email_changes', json_build_object(
            'operation', 'UPDATE',
            'table', TG_TABLE_NAME,
            'id', NEW.id,
            'data', row_to_json(NEW)
        )::text);
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('email_changes', json_build_object(
            'operation', 'DELETE',
            'table', TG_TABLE_NAME,
            'id', OLD.id
        )::text);
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for notifications
CREATE TRIGGER notify_emails_change
    AFTER INSERT OR UPDATE OR DELETE ON emails
    FOR EACH ROW EXECUTE FUNCTION notify_email_change();

CREATE TRIGGER notify_threads_change
    AFTER INSERT OR UPDATE OR DELETE ON threads
    FOR EACH ROW EXECUTE FUNCTION notify_email_change();

CREATE TRIGGER notify_classifications_change
    AFTER INSERT OR UPDATE OR DELETE ON classifications
    FOR EACH ROW EXECUTE FUNCTION notify_email_change();

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
    special_use TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(username, name)
);

-- IMAP Messages table for go-imap-sql (maps to our emails table)
CREATE TABLE imap_messages (
    id SERIAL PRIMARY KEY,
    mailbox_id INTEGER NOT NULL REFERENCES imap_mailboxes(id) ON DELETE CASCADE,
    uid BIGINT NOT NULL,
    flags TEXT[] DEFAULT ARRAY[]::TEXT[],
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

-- Function to sync from imap_messages to emails table
CREATE OR REPLACE FUNCTION sync_imap_to_emails()
RETURNS TRIGGER AS $$
DECLARE
    parsed_envelope JSONB;
    email_id VARCHAR(255);
    thread_id VARCHAR(255);
    extracted_text TEXT;
BEGIN
    -- Parse envelope data
    parsed_envelope := NEW.envelope;
    
    -- Generate email ID
    email_id := uuid_generate_v4()::text;
    
    -- Extract message-id from envelope for deduplication
    IF parsed_envelope->>'message_id' IS NOT NULL THEN
        -- Check if email already exists
        IF EXISTS (SELECT 1 FROM emails WHERE message_id = parsed_envelope->>'message_id') THEN
            RETURN NEW;
        END IF;
    END IF;
    
    -- Basic text extraction from raw message (simplified)
    -- In a real implementation, you'd parse MIME structure properly
    extracted_text := '';
    IF NEW.raw_message IS NOT NULL THEN
        -- Simple text extraction - convert bytes to text and extract readable parts
        extracted_text := convert_from(NEW.raw_message, 'UTF8');
        -- Remove email headers and extract body (very basic)
        IF position(E'\r\n\r\n' in extracted_text) > 0 THEN
            extracted_text := substring(extracted_text from position(E'\r\n\r\n' in extracted_text) + 4);
        END IF;
        -- Limit length to prevent issues
        extracted_text := substring(extracted_text from 1 for 50000);
    END IF;
    
    -- Insert into emails table
    INSERT INTO emails (
        id,
        message_id,
        imap_uid,
        mailbox_name,
        from_email,
        from_name,
        to_emails,
        cc_emails,
        subject,
        body_text,
        date_sent,
        date_received,
        raw_headers,
        flags,
        created_at
    ) VALUES (
        email_id,
        COALESCE(parsed_envelope->>'message_id', 'missing-' || NEW.id::text),
        NEW.uid,
        (SELECT name FROM imap_mailboxes WHERE id = NEW.mailbox_id),
        COALESCE(parsed_envelope->'from'->0->>'email', 'unknown@unknown.com'),
        parsed_envelope->'from'->0->>'name',
        COALESCE(ARRAY(SELECT jsonb_array_elements_text(parsed_envelope->'to')), ARRAY[]::TEXT[]),
        COALESCE(ARRAY(SELECT jsonb_array_elements_text(parsed_envelope->'cc')), ARRAY[]::TEXT[]),
        parsed_envelope->>'subject',
        extracted_text,
        COALESCE((parsed_envelope->>'date')::timestamp with time zone, NEW.internal_date),
        NEW.internal_date,
        parsed_envelope,
        NEW.flags,
        NOW()
    );
    
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN
        -- Log error and continue
        RAISE WARNING 'Error syncing message %: %', NEW.id, SQLERRM;
        RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to sync IMAP messages to emails
CREATE TRIGGER sync_imap_messages_to_emails
    AFTER INSERT ON imap_messages
    FOR EACH ROW EXECUTE FUNCTION sync_imap_to_emails();

-- Function to find or create thread for an email
CREATE OR REPLACE FUNCTION find_or_create_thread(
    email_subject TEXT,
    from_email TEXT,
    to_emails TEXT[],
    cc_emails TEXT[],
    date_sent TIMESTAMP WITH TIME ZONE
) RETURNS VARCHAR(255) AS $$
DECLARE
    normalized_subject TEXT;
    all_participants TEXT[];
    existing_thread_id VARCHAR(255);
    new_thread_id VARCHAR(255);
BEGIN
    -- Normalize subject
    normalized_subject := normalize_subject(email_subject);
    
    -- Combine all participants
    all_participants := ARRAY[from_email] || COALESCE(to_emails, ARRAY[]::TEXT[]) || COALESCE(cc_emails, ARRAY[]::TEXT[]);
    all_participants := array_remove(all_participants, NULL);
    all_participants := (SELECT ARRAY(SELECT DISTINCT unnest(all_participants) ORDER BY 1));
    
    -- Look for existing thread
    SELECT id INTO existing_thread_id
    FROM threads 
    WHERE subject_normalized = normalized_subject
    AND participants @> all_participants 
    AND participants <@ all_participants
    ORDER BY last_message_date DESC 
    LIMIT 1;
    
    IF existing_thread_id IS NOT NULL THEN
        -- Update existing thread
        UPDATE threads 
        SET last_message_date = GREATEST(last_message_date, date_sent),
            message_count = message_count + 1,
            updated_at = NOW()
        WHERE id = existing_thread_id;
        
        RETURN existing_thread_id;
    ELSE
        -- Create new thread
        new_thread_id := uuid_generate_v4()::text;
        
        INSERT INTO threads (
            id, subject_normalized, participants, 
            first_message_date, last_message_date, message_count
        ) VALUES (
            new_thread_id, normalized_subject, all_participants,
            date_sent, date_sent, 1
        );
        
        RETURN new_thread_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Update the sync function to create threads
CREATE OR REPLACE FUNCTION sync_imap_to_emails()
RETURNS TRIGGER AS $$
DECLARE
    parsed_envelope JSONB;
    email_id VARCHAR(255);
    thread_id VARCHAR(255);
    extracted_text TEXT;
    from_email TEXT;
    to_emails TEXT[];
    cc_emails TEXT[];
    subject TEXT;
    date_sent TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Parse envelope data
    parsed_envelope := NEW.envelope;
    
    -- Generate email ID
    email_id := uuid_generate_v4()::text;
    
    -- Extract message-id from envelope for deduplication
    IF parsed_envelope->>'message_id' IS NOT NULL THEN
        -- Check if email already exists
        IF EXISTS (SELECT 1 FROM emails WHERE message_id = parsed_envelope->>'message_id') THEN
            RETURN NEW;
        END IF;
    END IF;
    
    -- Extract email data
    from_email := COALESCE(parsed_envelope->'from'->0->>'email', 'unknown@unknown.com');
    to_emails := COALESCE(ARRAY(SELECT jsonb_array_elements_text(parsed_envelope->'to')), ARRAY[]::TEXT[]);
    cc_emails := COALESCE(ARRAY(SELECT jsonb_array_elements_text(parsed_envelope->'cc')), ARRAY[]::TEXT[]);
    subject := parsed_envelope->>'subject';
    date_sent := COALESCE((parsed_envelope->>'date')::timestamp with time zone, NEW.internal_date);
    
    -- Find or create thread
    thread_id := find_or_create_thread(subject, from_email, to_emails, cc_emails, date_sent);
    
    -- Basic text extraction from raw message (simplified)
    extracted_text := '';
    IF NEW.raw_message IS NOT NULL THEN
        -- Simple text extraction - convert bytes to text and extract readable parts
        extracted_text := convert_from(NEW.raw_message, 'UTF8');
        -- Remove email headers and extract body (very basic)
        IF position(E'\r\n\r\n' in extracted_text) > 0 THEN
            extracted_text := substring(extracted_text from position(E'\r\n\r\n' in extracted_text) + 4);
        END IF;
        -- Limit length to prevent issues
        extracted_text := substring(extracted_text from 1 for 50000);
    END IF;
    
    -- Insert into emails table
    INSERT INTO emails (
        id,
        message_id,
        thread_id,
        imap_uid,
        mailbox_name,
        from_email,
        from_name,
        to_emails,
        cc_emails,
        subject,
        body_text,
        date_sent,
        date_received,
        raw_headers,
        flags,
        created_at
    ) VALUES (
        email_id,
        COALESCE(parsed_envelope->>'message_id', 'missing-' || NEW.id::text),
        thread_id,
        NEW.uid,
        (SELECT name FROM imap_mailboxes WHERE id = NEW.mailbox_id),
        from_email,
        parsed_envelope->'from'->0->>'name',
        to_emails,
        cc_emails,
        subject,
        extracted_text,
        date_sent,
        NEW.internal_date,
        parsed_envelope,
        NEW.flags,
        NOW()
    );
    
    -- Update contact information
    PERFORM upsert_contact(from_email, parsed_envelope->'from'->0->>'name', date_sent);
    PERFORM upsert_contact(unnest(to_emails), NULL, date_sent);
    PERFORM upsert_contact(unnest(cc_emails), NULL, date_sent);
    
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN
        -- Log error and continue
        RAISE WARNING 'Error syncing message %: %', NEW.id, SQLERRM;
        RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to upsert contact information
CREATE OR REPLACE FUNCTION upsert_contact(
    contact_email TEXT,
    contact_name TEXT,
    message_date TIMESTAMP WITH TIME ZONE
) RETURNS VOID AS $$
BEGIN
    IF contact_email IS NULL OR contact_email = '' THEN
        RETURN;
    END IF;
    
    INSERT INTO contacts (id, email, name, frequency_score, first_contact, last_contact, total_messages)
    VALUES (uuid_generate_v4()::text, contact_email, contact_name, 1, message_date, message_date, 1)
    ON CONFLICT (email) DO UPDATE SET
        name = COALESCE(EXCLUDED.name, contacts.name, contact_name),
        frequency_score = contacts.frequency_score + 1,
        last_contact = GREATEST(contacts.last_contact, EXCLUDED.last_contact),
        total_messages = contacts.total_messages + 1,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Create sender rules table for whitelist/blacklist
CREATE TABLE sender_rules (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    email_pattern VARCHAR(255) NOT NULL, -- Can be exact email or pattern like @company.com
    rule_type VARCHAR(20) NOT NULL CHECK (rule_type IN ('whitelist', 'blacklist')),
    priority INTEGER DEFAULT 0, -- Higher priority = more important
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255) DEFAULT 'user',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(email_pattern, rule_type)
);

-- Create user preferences table
CREATE TABLE user_preferences (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    preference_key VARCHAR(100) NOT NULL,
    preference_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(preference_key)
);

-- Create processing queue table for RAG pipeline
CREATE TABLE processing_queue (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    thread_id VARCHAR(255) NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    queue_type VARCHAR(50) NOT NULL CHECK (queue_type IN ('classification', 'embedding', 'reprocessing')),
    priority INTEGER DEFAULT 0, -- Higher = more important (new messages = higher priority)
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(thread_id, queue_type)
);

-- Create processing stats table for budgeting and progress
CREATE TABLE processing_stats (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    stat_type VARCHAR(50) NOT NULL, -- 'classification_tokens', 'embedding_tokens', 'processing_time_ms'
    stat_value BIGINT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(date, stat_type)
);

-- Create thread actions table for user overrides
CREATE TABLE thread_actions (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    thread_id VARCHAR(255) NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    action_type VARCHAR(50) NOT NULL CHECK (action_type IN ('force_process', 'skip_processing', 'mark_important', 'mark_spam')),
    created_by VARCHAR(255) DEFAULT 'user',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(thread_id, action_type)
);

-- Update classifications table to include more detailed scoring
ALTER TABLE classifications ADD COLUMN relevance_score FLOAT DEFAULT 0.0 CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0);
ALTER TABLE classifications ADD COLUMN human_score FLOAT DEFAULT 0.0 CHECK (human_score >= 0.0 AND human_score <= 1.0);
ALTER TABLE classifications ADD COLUMN personal_score FLOAT DEFAULT 0.0 CHECK (personal_score >= 0.0 AND personal_score <= 1.0);
ALTER TABLE classifications ADD COLUMN should_process BOOLEAN DEFAULT FALSE;

-- Create indexes for new tables
CREATE INDEX idx_sender_rules_email_pattern ON sender_rules(email_pattern);
CREATE INDEX idx_sender_rules_rule_type ON sender_rules(rule_type);
CREATE INDEX idx_sender_rules_priority ON sender_rules(priority DESC);

CREATE INDEX idx_processing_queue_status ON processing_queue(status);
CREATE INDEX idx_processing_queue_priority ON processing_queue(priority DESC, created_at DESC);
CREATE INDEX idx_processing_queue_thread_id ON processing_queue(thread_id);
CREATE INDEX idx_processing_queue_type ON processing_queue(queue_type);

CREATE INDEX idx_processing_stats_date ON processing_stats(date);
CREATE INDEX idx_processing_stats_type ON processing_stats(stat_type);

CREATE INDEX idx_thread_actions_thread_id ON thread_actions(thread_id);
CREATE INDEX idx_thread_actions_type ON thread_actions(action_type);

-- Function to check sender rules
CREATE OR REPLACE FUNCTION check_sender_rules(sender_email TEXT)
RETURNS TABLE(rule_type VARCHAR(20), priority INTEGER) AS $$
BEGIN
    RETURN QUERY
    SELECT sr.rule_type, sr.priority
    FROM sender_rules sr
    WHERE sr.is_active = TRUE
    AND (
        sr.email_pattern = sender_email 
        OR (sr.email_pattern LIKE '@%' AND sender_email LIKE '%' || sr.email_pattern)
        OR sender_email ILIKE sr.email_pattern
    )
    ORDER BY sr.priority DESC, sr.created_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to add item to processing queue
CREATE OR REPLACE FUNCTION add_to_processing_queue(
    p_thread_id VARCHAR(255),
    p_queue_type VARCHAR(50),
    p_priority INTEGER DEFAULT NULL
) RETURNS VOID AS $$
DECLARE
    calculated_priority INTEGER;
BEGIN
    -- Calculate priority if not provided (newer threads = higher priority)
    IF p_priority IS NULL THEN
        SELECT EXTRACT(EPOCH FROM (NOW() - t.first_message_date))::INTEGER / 3600 -- Hours ago, inverted
        INTO calculated_priority
        FROM threads t WHERE t.id = p_thread_id;
        
        calculated_priority := GREATEST(0, 10000 - COALESCE(calculated_priority, 10000));
    ELSE
        calculated_priority := p_priority;
    END IF;

    INSERT INTO processing_queue (thread_id, queue_type, priority)
    VALUES (p_thread_id, p_queue_type, calculated_priority)
    ON CONFLICT (thread_id, queue_type) DO UPDATE SET
        priority = GREATEST(processing_queue.priority, calculated_priority),
        status = CASE 
            WHEN processing_queue.status = 'failed' THEN 'pending'
            ELSE processing_queue.status
        END,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to update processing stats
CREATE OR REPLACE FUNCTION update_processing_stats(
    p_stat_type VARCHAR(50),
    p_stat_value BIGINT,
    p_metadata JSONB DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO processing_stats (date, stat_type, stat_value, metadata)
    VALUES (CURRENT_DATE, p_stat_type, p_stat_value, p_metadata)
    ON CONFLICT (date, stat_type) DO UPDATE SET
        stat_value = processing_stats.stat_value + p_stat_value,
        metadata = COALESCE(p_metadata, processing_stats.metadata),
        created_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically queue new threads for classification
CREATE OR REPLACE FUNCTION auto_queue_thread_classification()
RETURNS TRIGGER AS $$
BEGIN
    -- Add to classification queue
    PERFORM add_to_processing_queue(NEW.id, 'classification');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER auto_queue_new_threads
    AFTER INSERT ON threads
    FOR EACH ROW EXECUTE FUNCTION auto_queue_thread_classification();

-- Insert default user preferences
INSERT INTO user_preferences (preference_key, preference_value, description) VALUES
('daily_processing_budget', '{"classification_tokens": 100000, "embedding_tokens": 50000, "max_processing_time_hours": 2}', 'Daily limits for AI processing'),
('classification_thresholds', '{"min_human_score": 0.7, "min_personal_score": 0.6, "min_relevance_score": 0.5}', 'Minimum scores for processing threads'),
('processing_priorities', '{"new_message_boost": 1000, "user_forced_boost": 5000, "whitelist_boost": 500}', 'Priority boosts for different scenarios'),
('ui_settings', '{"show_progress_bar": true, "auto_refresh_interval": 30, "items_per_page": 50}', 'UI behavior settings')
ON CONFLICT (preference_key) DO NOTHING;

-- Insert default IMAP user for email sync
INSERT INTO imap_users (username, password_hash) 
VALUES ('email_sync_user', 'not_used_for_sync') 
ON CONFLICT (username) DO NOTHING;