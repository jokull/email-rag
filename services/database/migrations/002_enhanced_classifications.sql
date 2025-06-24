-- Enhanced Classification Migration
-- Adds multi-dimensional analysis fields to classifications table

-- Add email_id column and make it the main reference instead of thread_id
ALTER TABLE classifications ADD COLUMN email_id VARCHAR(255) REFERENCES emails(id) ON DELETE CASCADE;
ALTER TABLE classifications DROP CONSTRAINT classifications_thread_id_key;
ALTER TABLE classifications ADD CONSTRAINT classifications_email_id_unique UNIQUE(email_id);

-- Add multi-dimensional classification fields
ALTER TABLE classifications ADD COLUMN sentiment VARCHAR(20) DEFAULT 'neutral' CHECK (sentiment IN ('positive', 'neutral', 'negative'));
ALTER TABLE classifications ADD COLUMN sentiment_score FLOAT DEFAULT 0.0 CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0);
ALTER TABLE classifications ADD COLUMN formality VARCHAR(20) DEFAULT 'neutral' CHECK (formality IN ('formal', 'informal', 'casual', 'neutral'));
ALTER TABLE classifications ADD COLUMN formality_score FLOAT DEFAULT 0.0 CHECK (formality_score >= 0.0 AND formality_score <= 1.0);
ALTER TABLE classifications ADD COLUMN personalization VARCHAR(30) DEFAULT 'generic' CHECK (personalization IN ('highly_personal', 'somewhat_personal', 'generic'));
ALTER TABLE classifications ADD COLUMN personalization_score FLOAT DEFAULT 0.0 CHECK (personalization_score >= 0.0 AND personalization_score <= 1.0);
ALTER TABLE classifications ADD COLUMN priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('urgent', 'normal', 'low'));
ALTER TABLE classifications ADD COLUMN priority_score FLOAT DEFAULT 0.0 CHECK (priority_score >= 0.0 AND priority_score <= 1.0);

-- Add contact history analysis fields
ALTER TABLE classifications ADD COLUMN sender_frequency_score FLOAT DEFAULT 0.0 CHECK (sender_frequency_score >= 0.0 AND sender_frequency_score <= 1.0);
ALTER TABLE classifications ADD COLUMN response_likelihood FLOAT DEFAULT 0.0 CHECK (response_likelihood >= 0.0 AND response_likelihood <= 1.0);
ALTER TABLE classifications ADD COLUMN relationship_strength FLOAT DEFAULT 0.0 CHECK (relationship_strength >= 0.0 AND relationship_strength <= 1.0);

-- Create index on new email_id column
CREATE INDEX idx_classifications_email_id ON classifications(email_id);

-- Create indexes for new analysis fields for efficient querying
CREATE INDEX idx_classifications_sentiment ON classifications(sentiment);
CREATE INDEX idx_classifications_priority ON classifications(priority);
CREATE INDEX idx_classifications_personalization ON classifications(personalization);
CREATE INDEX idx_classifications_priority_score ON classifications(priority_score);
CREATE INDEX idx_classifications_relationship_strength ON classifications(relationship_strength);

-- Update existing data to populate email_id from thread_id (if any exists)
UPDATE classifications 
SET email_id = (
    SELECT e.id 
    FROM emails e 
    WHERE e.thread_id = classifications.thread_id 
    ORDER BY e.date_sent DESC 
    LIMIT 1
)
WHERE email_id IS NULL;

-- Clean up orphaned classifications that couldn't be mapped
DELETE FROM classifications WHERE email_id IS NULL;