-- Migration: Add embeddings table for RAG indexing
-- Description: Adds message_chunks table for storing text chunks and vector embeddings

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create message_chunks table for RAG embeddings
CREATE TABLE IF NOT EXISTS message_chunks (
    id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    
    -- Unstructured.io metadata
    element_type TEXT,
    chunk_metadata JSONB DEFAULT '{}',
    
    -- Vector embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding vector(384),
    
    -- Processing metadata
    processed_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Ensure unique chunk ordering per message
    UNIQUE(message_id, chunk_index)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_message_chunks_message_id ON message_chunks(message_id);
CREATE INDEX IF NOT EXISTS idx_message_chunks_element_type ON message_chunks(element_type);
CREATE INDEX IF NOT EXISTS idx_message_chunks_processed_at ON message_chunks(processed_at);

-- Create vector similarity search index (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS idx_message_chunks_embedding_hnsw 
ON message_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create IVFFlat index as alternative for exact searches (can coexist with HNSW)
-- CREATE INDEX IF NOT EXISTS idx_message_chunks_embedding_ivfflat 
-- ON message_chunks USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- Add helpful comments
COMMENT ON TABLE message_chunks IS 'Text chunks and embeddings for RAG search from personal emails';
COMMENT ON COLUMN message_chunks.embedding IS '384-dimensional vector from sentence-transformers/all-MiniLM-L6-v2';
COMMENT ON COLUMN message_chunks.element_type IS 'Unstructured.io element type (Title, NarrativeText, etc.)';
COMMENT ON COLUMN message_chunks.metadata IS 'Additional metadata from Unstructured.io processing';
COMMENT ON INDEX idx_message_chunks_embedding_hnsw IS 'HNSW index for fast approximate vector similarity search';

-- Verify the setup
SELECT 'message_chunks table created successfully' as status;