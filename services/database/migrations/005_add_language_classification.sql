-- Migration: Add language classification columns to messages table
-- Date: 2025-06-26
-- Description: Add language detection with confidence scoring

-- Add language classification columns
ALTER TABLE messages 
ADD COLUMN language TEXT,
ADD COLUMN language_confidence FLOAT;

-- Add index for language filtering
CREATE INDEX IF NOT EXISTS idx_messages_language ON messages(language);

-- Add index for language confidence filtering  
CREATE INDEX IF NOT EXISTS idx_messages_language_confidence ON messages(language_confidence);

-- Add comment explaining the language column format
COMMENT ON COLUMN messages.language IS 'ISO 639-1 language code (en, es, fr, de, etc.) detected from email content';
COMMENT ON COLUMN messages.language_confidence IS 'Language detection confidence score from 0.0 to 1.0';