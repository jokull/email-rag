-- Add confidence column to messages table for classification confidence scores
-- Migration: 005_add_confidence_column.sql

BEGIN;

-- Add confidence column to store classification confidence scores
ALTER TABLE messages 
ADD COLUMN confidence REAL;

-- Add index on confidence for filtering by confidence thresholds
CREATE INDEX idx_messages_confidence ON messages(confidence);

-- Add composite index for category + confidence queries
CREATE INDEX idx_messages_category_confidence ON messages(category, confidence);

COMMIT;