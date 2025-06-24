-- Add thread-level summary fields to threads table
-- Migration: Add thread summary support

ALTER TABLE threads ADD COLUMN IF NOT EXISTS summary_oneliner TEXT;
ALTER TABLE threads ADD COLUMN IF NOT EXISTS summary_embedding vector(384);
ALTER TABLE threads ADD COLUMN IF NOT EXISTS last_summary_update TIMESTAMP;
ALTER TABLE threads ADD COLUMN IF NOT EXISTS summary_version INTEGER DEFAULT 1;
ALTER TABLE threads ADD COLUMN IF NOT EXISTS key_entities TEXT[];
ALTER TABLE threads ADD COLUMN IF NOT EXISTS thread_mood VARCHAR(50);
ALTER TABLE threads ADD COLUMN IF NOT EXISTS action_items TEXT[];

-- Index for similarity search on summary embeddings
CREATE INDEX IF NOT EXISTS idx_threads_summary_embedding ON threads USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);

-- Index for fast summary lookups
CREATE INDEX IF NOT EXISTS idx_threads_summary_update ON threads (last_summary_update DESC);

COMMENT ON COLUMN threads.summary_oneliner IS 'Short one-line summary for notifications (10-15 words)';
COMMENT ON COLUMN threads.summary_embedding IS 'Vector embedding of the summary for similarity search';
COMMENT ON COLUMN threads.key_entities IS 'Important entities mentioned in thread (people, topics, dates)';
COMMENT ON COLUMN threads.thread_mood IS 'Overall thread mood: planning, urgent, social, work, problem_solving';
COMMENT ON COLUMN threads.action_items IS 'Extracted action items from the conversation';