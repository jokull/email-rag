-- Scrub LLM classification columns to start fresh with new LLM integration
-- Migration: 007_scrub_llm_classification.sql

BEGIN;

-- Reset classification data in messages table
UPDATE messages SET 
    category = NULL,
    confidence = NULL,
    classified_at = NULL
WHERE classified_at IS NOT NULL;

-- Reset threading data to force re-threading with new LLM summaries
UPDATE messages SET 
    thread_id = NULL
WHERE thread_id IS NOT NULL;

-- Clear all conversations to regenerate with new LLM summaries
DELETE FROM conversations;

-- Reset auto-increment for conversations table
ALTER SEQUENCE conversations_id_seq RESTART WITH 1;

-- Log the cleanup
DO $$
DECLARE
    message_count INTEGER;
    conversation_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO message_count FROM messages WHERE category IS NULL;
    SELECT COUNT(*) INTO conversation_count FROM conversations;
    
    RAISE NOTICE 'Scrubbed % messages, % conversations remaining', message_count, conversation_count;
END $$;

COMMIT;