-- Fix Schema Inconsistencies for Queue-Based Pipeline
-- This migration fixes all the schema mismatches found in the codebase

-- 1. Fix classifications table to match expected schema
-- Drop and recreate with correct structure
DROP TABLE IF EXISTS classifications CASCADE;

CREATE TABLE classifications (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    email_id VARCHAR(255) NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    thread_id VARCHAR(255) REFERENCES threads(id) ON DELETE CASCADE, -- Keep both for compatibility
    classification VARCHAR(50) NOT NULL CHECK (classification IN ('human', 'promotional', 'transactional', 'automated')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Multi-dimensional classification fields
    sentiment VARCHAR(20) DEFAULT 'neutral' CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    sentiment_score FLOAT DEFAULT 0.0 CHECK (sentiment_score >= 0.0 AND sentiment_score <= 1.0),
    formality VARCHAR(20) DEFAULT 'neutral' CHECK (formality IN ('formal', 'informal', 'casual', 'neutral')),
    formality_score FLOAT DEFAULT 0.0 CHECK (formality_score >= 0.0 AND formality_score <= 1.0),
    personalization VARCHAR(30) DEFAULT 'generic' CHECK (personalization IN ('highly_personal', 'somewhat_personal', 'generic')),
    personalization_score FLOAT DEFAULT 0.0 CHECK (personalization_score >= 0.0 AND personalization_score <= 1.0),
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('urgent', 'normal', 'low')),
    priority_score FLOAT DEFAULT 0.0 CHECK (priority_score >= 0.0 AND priority_score <= 1.0),
    
    -- Processing decision fields
    should_process BOOLEAN DEFAULT FALSE,
    processing_priority INTEGER DEFAULT 0,
    
    -- Enhanced scoring fields
    human_score FLOAT DEFAULT 0.0 CHECK (human_score >= 0.0 AND human_score <= 1.0),
    personal_score FLOAT DEFAULT 0.0 CHECK (personal_score >= 0.0 AND personal_score <= 1.0),
    relevance_score FLOAT DEFAULT 0.0 CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
    importance_score FLOAT DEFAULT 0.0 CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    commercial_score FLOAT DEFAULT 0.0 CHECK (commercial_score >= 0.0 AND commercial_score <= 1.0),
    
    -- Metadata
    model_used VARCHAR(100) NOT NULL DEFAULT 'qwen-0.5b-v1',
    scorer_version VARCHAR(50) DEFAULT 'qwen-0.5b-v1',
    reasoning TEXT,
    content_length INTEGER DEFAULT 0,
    processing_time_ms FLOAT DEFAULT 0.0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(email_id)
);

-- 2. Fix cleaned_emails table to match expected schema  
DROP TABLE IF EXISTS cleaned_emails CASCADE;

CREATE TABLE cleaned_emails (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    email_id VARCHAR(255) NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    clean_content TEXT NOT NULL,
    signature_removed TEXT DEFAULT '',
    quotes_removed TEXT DEFAULT '',
    
    -- Add missing columns that code expects
    chunk_index INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    
    -- Original columns
    original_length INTEGER NOT NULL,
    cleaned_length INTEGER NOT NULL,
    cleaning_confidence FLOAT NOT NULL CHECK (cleaning_confidence >= 0.0 AND cleaning_confidence <= 1.0),
    cleaning_method VARCHAR(100) NOT NULL,
    content_type VARCHAR(20) DEFAULT 'text' CHECK (content_type IN ('text', 'html')),
    reduction_ratio FLOAT DEFAULT 0.0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(email_id)
);

-- 3. Standardize embeddings table (use 384 dimensions for sentence-transformers/all-MiniLM-L6-v2)
DROP TABLE IF EXISTS embeddings CASCADE;

CREATE TABLE embeddings (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    email_id VARCHAR(255) NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    thread_id VARCHAR(255) REFERENCES threads(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    element_type VARCHAR(50) DEFAULT 'text',
    embedding vector(384), -- Standard sentence-transformers dimension
    chunk_metadata JSONB DEFAULT '{}'::jsonb,
    quality_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(email_id, chunk_index)
);

-- 4. Remove problematic triggers that cause pg_notify issues
DROP TRIGGER IF EXISTS notify_emails_change ON emails;
DROP TRIGGER IF EXISTS notify_threads_change ON threads;
DROP TRIGGER IF EXISTS notify_classifications_change ON classifications;
DROP TRIGGER IF EXISTS notify_conversations_change ON conversations;
DROP TRIGGER IF EXISTS notify_conversation_turns_change ON conversation_turns;

-- Drop the problematic notification function
DROP FUNCTION IF EXISTS notify_email_change();

-- 5. Remove complex auto-queue triggers (pgqueuer will handle this)
DROP TRIGGER IF EXISTS auto_queue_new_threads ON threads;
DROP TRIGGER IF EXISTS queue_content_processing_after_classification ON classifications;
DROP TRIGGER IF EXISTS auto_queue_conversations_for_rag ON conversations;

-- Drop related functions
DROP FUNCTION IF EXISTS auto_queue_thread_classification();
DROP FUNCTION IF EXISTS auto_queue_after_classification();
DROP FUNCTION IF EXISTS auto_queue_conversation_for_rag();

-- 6. Simplify processing_queue table for pgqueuer compatibility
DROP TABLE IF EXISTS processing_queue;

-- pgqueuer will create its own job table, but we can keep a simple status table
CREATE TABLE processing_status (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    email_id VARCHAR(255) NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    thread_id VARCHAR(255) REFERENCES threads(id) ON DELETE CASCADE,
    stage VARCHAR(50) NOT NULL CHECK (stage IN ('pending', 'classified', 'content_processed', 'embeddings_generated', 'completed', 'failed')),
    last_processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(email_id)
);

-- 7. Create proper indexes for the fixed tables
CREATE INDEX idx_classifications_email_id ON classifications(email_id);
CREATE INDEX idx_classifications_thread_id ON classifications(thread_id);  
CREATE INDEX idx_classifications_classification ON classifications(classification);
CREATE INDEX idx_classifications_confidence ON classifications(confidence);
CREATE INDEX idx_classifications_should_process ON classifications(should_process);
CREATE INDEX idx_classifications_processing_priority ON classifications(processing_priority);

CREATE INDEX idx_cleaned_emails_email_id ON cleaned_emails(email_id);
CREATE INDEX idx_cleaned_emails_confidence ON cleaned_emails(cleaning_confidence);
CREATE INDEX idx_cleaned_emails_method ON cleaned_emails(cleaning_method);

CREATE INDEX idx_embeddings_email_id ON embeddings(email_id);
CREATE INDEX idx_embeddings_thread_id ON embeddings(thread_id);
CREATE INDEX idx_embeddings_chunk_index ON embeddings(chunk_index);
CREATE INDEX idx_embeddings_embedding ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_embeddings_quality ON embeddings(quality_score);

CREATE INDEX idx_processing_status_email_id ON processing_status(email_id);
CREATE INDEX idx_processing_status_thread_id ON processing_status(thread_id);
CREATE INDEX idx_processing_status_stage ON processing_status(stage);
CREATE INDEX idx_processing_status_last_processed ON processing_status(last_processed_at);

-- 8. Create updated_at triggers for new tables
CREATE TRIGGER update_classifications_updated_at BEFORE UPDATE ON classifications
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cleaned_emails_updated_at BEFORE UPDATE ON cleaned_emails
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_processing_status_updated_at BEFORE UPDATE ON processing_status
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 9. Helper function to update processing status
CREATE OR REPLACE FUNCTION update_processing_status(
    p_email_id VARCHAR(255),
    p_stage VARCHAR(50),
    p_error TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO processing_status (email_id, thread_id, stage, last_error, error_count)
    SELECT p_email_id, e.thread_id, p_stage, p_error, CASE WHEN p_error IS NULL THEN 0 ELSE 1 END
    FROM emails e WHERE e.id = p_email_id
    ON CONFLICT (email_id) DO UPDATE SET
        stage = EXCLUDED.stage,
        last_processed_at = NOW(),
        last_error = EXCLUDED.last_error,
        error_count = CASE 
            WHEN EXCLUDED.last_error IS NULL THEN 0
            ELSE processing_status.error_count + 1
        END,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- 10. Function to get unprocessed emails for queue initialization
CREATE OR REPLACE FUNCTION get_unprocessed_emails(limit_count INTEGER DEFAULT 100)
RETURNS TABLE(
    email_id VARCHAR(255),
    thread_id VARCHAR(255),
    from_email VARCHAR(255),
    subject TEXT,
    date_received TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.thread_id,
        e.from_email,
        e.subject,
        e.date_received
    FROM emails e
    LEFT JOIN processing_status ps ON e.id = ps.email_id
    WHERE ps.email_id IS NULL 
       OR ps.stage IN ('pending', 'failed')
    ORDER BY e.date_received DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Clean up any remaining problematic data
DELETE FROM processing_stats WHERE stat_type LIKE '%notification%';

-- Insert initial processing status for existing emails without classifications
INSERT INTO processing_status (email_id, thread_id, stage)
SELECT e.id, e.thread_id, 'pending'
FROM emails e
LEFT JOIN processing_status ps ON e.id = ps.email_id
WHERE ps.email_id IS NULL;

COMMIT;