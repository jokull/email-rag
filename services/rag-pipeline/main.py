import os
import json
import time
import subprocess
import psycopg2
import tiktoken
import numpy as np
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '50'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '5'))
        self.sleep_interval = int(os.getenv('SLEEP_INTERVAL', '120'))
        
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Initialize tokenizer for text chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.setup_model()
    
    def setup_model(self):
        """Install and setup the embedding model if not already available"""
        try:
            # Install sentence-transformers plugin for embeddings
            subprocess.run(['llm', 'install', 'llm-sentence-transformers'], check=True)
            print("Installed llm-sentence-transformers plugin")
            
            # Register the Qwen embedding model
            if 'Qwen3-Embedding' in self.embedding_model:
                subprocess.run(['llm', 'sentence-transformers', 'register', 'Qwen/Qwen3-Embedding-0.6B'], 
                             check=True)
                print(f"Registered Qwen3 embedding model")
            
            # List available models
            result = subprocess.run(['llm', 'models', 'list'], 
                                  capture_output=True, text=True, check=True)
            print(f"Using embedding model: {self.embedding_model}")
            print("Available models:", result.stdout)
                
        except subprocess.CalledProcessError as e:
            print(f"Error setting up embedding model: {e}")
            raise

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    def get_pending_embedding_queue(self, limit: int = None) -> List[Dict]:
        """Get threads from processing queue that need embedding generation"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                SELECT 
                    t.id::text,
                    t.subject_normalized,
                    t.participants,
                    t.message_count,
                    t.first_message_date,
                    t.last_message_date,
                    pq.priority,
                    pq.attempts,
                    c.should_process,
                    c.human_score,
                    c.personal_score,
                    c.relevance_score,
                    string_agg(
                        COALESCE(e.body_text, e.body_html, e.subject, ''),
                        E'\n\n---MESSAGE---\n\n'
                        ORDER BY e.date_sent
                    ) as full_thread_text,
                    -- Check for user actions
                    (SELECT COUNT(*) FROM thread_actions ta 
                     WHERE ta.thread_id = t.id AND ta.action_type = 'force_process'
                    ) > 0 as user_forced,
                    (SELECT COUNT(*) FROM thread_actions ta 
                     WHERE ta.thread_id = t.id AND ta.action_type = 'skip_processing'
                    ) > 0 as user_skipped
                FROM processing_queue pq
                JOIN threads t ON pq.thread_id = t.id
                JOIN classifications c ON t.id = c.thread_id
                LEFT JOIN emails e ON t.id = e.thread_id
                WHERE pq.queue_type = 'embedding'
                AND pq.status = 'pending'
                AND pq.attempts < pq.max_attempts
                AND (c.should_process = TRUE OR EXISTS (
                    SELECT 1 FROM thread_actions ta 
                    WHERE ta.thread_id = t.id AND ta.action_type = 'force_process'
                ))
                AND NOT EXISTS (
                    SELECT 1 FROM thread_actions ta 
                    WHERE ta.thread_id = t.id AND ta.action_type = 'skip_processing'
                )
                GROUP BY t.id, t.subject_normalized, t.participants, t.message_count, 
                         t.first_message_date, t.last_message_date, pq.priority, pq.attempts,
                         c.should_process, c.human_score, c.personal_score, c.relevance_score
                HAVING string_agg(
                    COALESCE(e.body_text, e.body_html, e.subject, ''),
                    E'\n\n---MESSAGE---\n\n'
                    ORDER BY e.date_sent
                ) IS NOT NULL
                ORDER BY 
                    CASE WHEN EXISTS (
                        SELECT 1 FROM thread_actions ta 
                        WHERE ta.thread_id = t.id AND ta.action_type = 'force_process'
                    ) THEN 1 ELSE 0 END DESC,  -- User-forced items first
                    pq.priority DESC,  -- Then by priority (newer first)
                    t.last_message_date DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cur.execute(query)
                return cur.fetchall()

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean and prepare text
        text = text.strip()
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Clean up the chunk
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            if end >= len(tokens):
                break
                
            start = end - self.chunk_overlap
        
        return chunks

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using LLM CLI with Qwen3"""
        try:
            # Use llm embed command with sentence-transformers model
            result = subprocess.run([
                'llm', 'embed', 
                '-m', self.embedding_model,
                '-c', text
            ], capture_output=True, text=True, check=True)
            
            # Parse the embedding from the output
            embedding_str = result.stdout.strip()
            
            # The output should be JSON array of floats
            try:
                embedding = json.loads(embedding_str)
                if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                    # Qwen3-Embedding-0.6B produces 1024-dimensional vectors
                    if len(embedding) == 1024:
                        return embedding
                    else:
                        print(f"Unexpected embedding dimension: {len(embedding)}, expected 1024")
                        return embedding  # Still return it, might work
                else:
                    print(f"Invalid embedding format: {type(embedding)}")
                    return None
            except json.JSONDecodeError:
                print(f"Failed to parse embedding JSON: {embedding_str}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error generating embedding: {e}")
            print(f"stderr: {e.stderr}")
            return None

    def save_embeddings(self, thread_id: str, chunks: List[str], embeddings: List[List[float]]):
        """Save chunk embeddings to database"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Delete existing embeddings for this thread
                cur.execute("DELETE FROM embeddings WHERE thread_id = %s", (thread_id,))
                
                # Insert new embeddings
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    cur.execute("""
                        INSERT INTO embeddings (thread_id, chunk_text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s)
                    """, (thread_id, chunk, i, embedding))

    def update_processing_queue_status(self, thread_id: str, status: str, error_msg: str = None):
        """Update processing queue status for embedding"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                if status == 'processing':
                    cur.execute("""
                        UPDATE processing_queue 
                        SET status = %s, 
                            processing_started_at = NOW(),
                            attempts = attempts + 1,
                            updated_at = NOW()
                        WHERE thread_id = %s AND queue_type = 'embedding'
                    """, (status, thread_id))
                elif status == 'completed':
                    cur.execute("""
                        UPDATE processing_queue 
                        SET status = %s, 
                            processing_completed_at = NOW(),
                            updated_at = NOW()
                        WHERE thread_id = %s AND queue_type = 'embedding'
                    """, (status, thread_id))
                elif status == 'failed':
                    cur.execute("""
                        UPDATE processing_queue 
                        SET status = %s, 
                            error_message = %s,
                            processing_completed_at = NOW(),
                            updated_at = NOW()
                        WHERE thread_id = %s AND queue_type = 'embedding'
                    """, (status, error_msg, thread_id))

    def check_embedding_budget(self) -> bool:
        """Check if we're within daily embedding budget"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Get today's stats
                    cur.execute("""
                        SELECT stat_value FROM processing_stats 
                        WHERE date = CURRENT_DATE AND stat_type = 'embedding_tokens'
                    """)
                    result = cur.fetchone()
                    used_tokens = result['stat_value'] if result else 0
                    
                    # Get budget from preferences
                    cur.execute("""
                        SELECT preference_value FROM user_preferences 
                        WHERE preference_key = 'daily_processing_budget'
                    """)
                    result = cur.fetchone()
                    if result:
                        budget = result['preference_value'].get('embedding_tokens', 50000)
                    else:
                        budget = 50000
                    
                    return used_tokens < budget
        except:
            return True  # Default to allowing processing

    def process_thread(self, thread: Dict):
        """Process a single thread - chunk and generate embeddings"""
        try:
            print(f"Processing thread {thread['id']}: {thread['subject_normalized']}")
            print(f"  Scores - Human: {thread.get('human_score', 0):.2f}, " +
                  f"Personal: {thread.get('personal_score', 0):.2f}, " +
                  f"Relevance: {thread.get('relevance_score', 0):.2f}")
            
            # Mark as processing
            self.update_processing_queue_status(thread['id'], 'processing')
            
            # Extract and clean the full thread text
            full_text = thread['full_thread_text']
            if not full_text or len(full_text.strip()) < 10:
                print(f"  -> Skipping thread {thread['id']} (insufficient content)")
                self.update_processing_queue_status(thread['id'], 'failed', 'Insufficient content')
                return
            
            # Chunk the text
            chunks = self.chunk_text(full_text)
            if not chunks:
                print(f"  -> No chunks generated for thread {thread['id']}")
                self.update_processing_queue_status(thread['id'], 'failed', 'No chunks generated')
                return
            
            print(f"  -> Generated {len(chunks)} chunks")
            
            # Generate embeddings for each chunk
            embeddings = []
            total_tokens = 0
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:  # Skip very short chunks
                    continue
                
                # Check budget before processing each chunk
                if not self.check_embedding_budget():
                    print(f"  -> Embedding budget exceeded, stopping at chunk {i+1}")
                    break
                    
                embedding = self.generate_embedding(chunk)
                if embedding:
                    embeddings.append(embedding)
                    total_tokens += len(chunk) // 4  # Rough token estimate
                    print(f"  -> Generated embedding for chunk {i+1}/{len(chunks)}")
                else:
                    print(f"  -> Failed to generate embedding for chunk {i+1}")
                
                # Small delay to avoid overwhelming the embedding service
                time.sleep(0.5)
            
            if embeddings:
                # Only save chunks that have embeddings
                valid_chunks = [chunk for chunk, emb in zip(chunks, embeddings) if emb is not None]
                valid_embeddings = [emb for emb in embeddings if emb is not None]
                
                self.save_embeddings(thread['id'], valid_chunks, valid_embeddings)
                print(f"  -> Saved {len(valid_embeddings)} embeddings for thread {thread['id']}")
                
                # Update processing stats
                with self.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT update_processing_stats('embedding_tokens', %s)", 
                                  (total_tokens,))
                
                # Mark as completed
                self.update_processing_queue_status(thread['id'], 'completed')
            else:
                print(f"  -> No valid embeddings generated for thread {thread['id']}")
                self.update_processing_queue_status(thread['id'], 'failed', 'No valid embeddings generated')
                
        except Exception as e:
            print(f"Error processing thread {thread['id']}: {e}")
            self.update_processing_queue_status(thread['id'], 'failed', str(e))

    def process_batch(self):
        """Process a batch of threads from embedding queue"""
        if not self.check_embedding_budget():
            print("Daily embedding budget exceeded, skipping RAG processing")
            return
            
        threads = self.get_pending_embedding_queue(self.batch_size)
        
        if not threads:
            print("No threads in embedding queue")
            return
        
        print(f"Processing {len(threads)} threads for RAG pipeline...")
        print(f"Queue priorities: {[t.get('priority', 0) for t in threads]}")
        
        for thread in threads:
            if not self.check_embedding_budget():
                print("Budget exceeded during batch processing, stopping")
                break
            self.process_thread(thread)

    def get_embedding_stats(self):
        """Get statistics about embeddings"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT thread_id) as threads_with_embeddings,
                        COUNT(*) as total_chunks,
                        AVG(array_length(embedding, 1)) as avg_embedding_dimension
                    FROM embeddings
                """)
                
                stats = cur.fetchone()
                if stats:
                    print(f"Embedding Stats:")
                    print(f"  - Threads with embeddings: {stats['threads_with_embeddings']}")
                    print(f"  - Total chunks: {stats['total_chunks']}")
                    print(f"  - Avg embedding dimension: {stats['avg_embedding_dimension']}")

    def run_continuous(self):
        """Run the RAG pipeline continuously"""
        print(f"Starting RAG pipeline with embedding model: {self.embedding_model}")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        print(f"Processing {self.batch_size} threads every {self.sleep_interval} seconds")
        
        while True:
            try:
                self.process_batch()
                self.get_embedding_stats()
                print(f"Sleeping for {self.sleep_interval} seconds...")
                time.sleep(self.sleep_interval)
                
            except KeyboardInterrupt:
                print("Shutting down RAG pipeline...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(self.sleep_interval)

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.run_continuous()