# Email RAG System - Claude Instructions

## Project Overview

This is a self-hosted email RAG system that syncs IMAP emails to PostgreSQL, processes them with
ML-based threading and classification, and provides a modern React UI for conversation management.

Pipeline:

1. **IMAP Sync**: `go-imap-sql` dumps emails to Postgres (pgvector) - main table is `imap_messages`
2. **Email Processing** (host-based):
   1. Python email library extracts plaintext version in UTF-8
   2. Extracts participants with normalized email addresses, date, message-id, thread-id, from,
      to, cc, reply-to, thread-topic, references etc.
   3. Python library `email_reply_parser` removes threaded responses and signatures from plaintext bodies
   4. **Language detection** with langdetect library - identifies language and confidence (en, fr, es, etc.)
   5. Result is saved in the `messages` table with participants in JSONB column with proper indexes
   6. Attachments, unprocessable raw bodies, HTML are all ignored and discarded
3. **Classification** (host-based):
   1. Host-based `classify.py` CLI with elegant interface and multiple run modes
   2. First-line defense patterns catch 65%+ emails instantly (noreply@, notifications@, %invoice%, etc.)
   3. Correspondence intelligence uses IMAP Sent mailbox for better classification
   4. LLM classification (via `llm` Python library) for remaining emails: personal/promotional/automated
   5. Results written to `category`, `confidence`, `classified_at` columns in `messages` table
4. **RAG Indexing** (host-based):
   1. Host-based `email-processor/worker.py rag-indexing` processes personal emails for semantic search
   2. Uses Unstructured library directly for intelligent text chunking (no Docker)
   3. Generates 384D embeddings with sentence-transformers/all-MiniLM-L6-v2
   4. Stores chunks and embeddings in `message_chunks` table with pgvector HNSW index
   5. Batch processing with comprehensive error handling and statistics
5. **Threading & Summarization** (working):
   1. SQL lookups on headers establish if this is _in reply to existing email_ (threading)
   2. Thread creation based on "genesis message id" (`conversations` table)
   3. LLM summarization of conversation threads using Qwen 3 model (max 120 chars)
   4. Two-phase threading algorithm prevents constraint violations
   5. Beautiful CLI output with conversation progress tracking
6. **UI** (future):
   1. Modern React interface for conversation management
   2. Vector similarity search for personal email RAG queries

## Current state

- **Working**: IMAP sync, PostgreSQL with pgvector, host-based email processing
- **Working**: Language detection with confidence scoring (50+ languages supported)
- **Working**: Host-based email classification with elegant CLI interface
- **Working**: First-line defense + LLM classification pipeline
- **Working**: RAG indexing for personal email embeddings (470 personal emails ready)
- **Working**: Direct Unstructured library integration for text processing
- **Working**: Email threading and conversation summarization with Qwen 3 LLM
- **Future**: UI with semantic search, advanced conversation intelligence features

## Conversation Intelligence Opportunities

The threading and summarization foundation enables advanced conversation intelligence features:

### 🧠 Semantic Conversation Features
- **Conversation Embeddings**: Generate 384D embeddings for entire conversation summaries using same model as RAG indexing
- **Similar Conversation Discovery**: Find conversations with similar topics/outcomes using cosine similarity
- **Conversation Clustering**: Group conversations by topic automatically using embedding clusters
- **Thread Continuation Detection**: Identify when old conversations are revived or continued

### 📊 Conversation Quality & Analytics  
- **Thread Quality Scoring**: Use LLM to score conversation quality (1-10) based on resolution, clarity, outcomes
- **Key Decision Extraction**: Identify and extract key decisions/outcomes from conversation summaries
- **Participant Sentiment Analysis**: Track sentiment changes throughout conversation threads
- **Topic Evolution Tracking**: How conversation topics shift over time within threads

### 🔗 Smart Threading Enhancements
- **Smart Thread Merging**: LLM-powered detection of conversations that should be merged despite different subjects
- **Thread Split Detection**: Identify when single thread splits into multiple conversation topics
- **Cross-Thread Relationship Mapping**: Find related conversations that reference each other
- **Genesis Message Classification**: Better categorization of conversation starters (question, announcement, decision, etc.)

### 🎯 Advanced Summarization
- **Multi-Level Summaries**: Generate both brief (120 char) and detailed (500 char) summaries per conversation
- **Outcome-Focused Summarization**: Emphasize decisions made, action items assigned, problems solved
- **Executive Summary Generation**: Weekly/monthly rollups of important conversation outcomes
- **Topic Tagging**: Automatic extraction and tagging of conversation topics (project names, initiatives, etc.)

### 🔍 Future RAG Integration
- **Conversation Context RAG**: "Find all conversations where we discussed X topic"
- **Decision History RAG**: "What decisions have we made about Y in the past?"
- **Participant Expertise Discovery**: Identify who has been involved in specific topic conversations
- **Cross-Conversation Knowledge Graph**: Build knowledge graph connecting people, topics, decisions across all conversations

### Implementation Notes
- All features build on existing `conversations` table and Qwen 3 LLM integration
- Conversation embeddings would use same sentence-transformers model as message RAG
- Quality scoring and topic extraction can leverage existing LLM infrastructure
- Advanced features provide foundation for sophisticated conversation management UI

## Database Access & Migrations

### PostgreSQL Connection
- **Container user**: `email_user` (password: `email_pass`)
- **Database**: `email_rag`
- **Host port**: `5433` (maps to container port 5432)
- **Connection string**: `postgresql://email_user:email_pass@localhost:5433/email_rag`

### psql Access
```bash
# Connect to database via Docker
docker compose exec postgres psql -U email_user -d email_rag

# Run SQL files
docker compose exec postgres psql -U email_user -d email_rag -c "SQL_COMMAND_HERE"

# Check table structure
docker compose exec postgres psql -U email_user -d email_rag -c "\d+ table_name"

# View column information
docker compose exec postgres psql -U email_user -d email_rag -c "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'messages';"
```

### Migration Rules
- **Never recreate imap_\* tables** - mail sync will take too long and this part will remain as-is
- **Safe to recreate**: `messages`, `contacts`, `conversations`, `message_chunks` tables
- **Always use `IF NOT EXISTS`** when adding columns/indexes to avoid errors
- **Migration files location**: `services/database/migrations/`
- **Naming convention**: `001_description.sql`, `002_description.sql`, etc.

### Recent Migrations
- `004_add_embeddings_table.sql` - Added message_chunks table for RAG embeddings
- `005_add_language_classification.sql` - Added language detection columns

### Example Migration Commands
```sql
-- Add new columns safely
ALTER TABLE messages 
ADD COLUMN IF NOT EXISTS new_column TEXT,
ADD COLUMN IF NOT EXISTS another_column FLOAT;

-- Add indexes safely  
CREATE INDEX IF NOT EXISTS idx_messages_new_column ON messages(new_column);

-- Check if migration worked
SELECT column_name, data_type FROM information_schema.columns 
WHERE table_name = 'messages' AND column_name LIKE '%new%';
```

## Package Management

- **ALWAYS use uv for python packages** - NEVER use pip directly
- Modern Python tooling with uv for fast, reliable dependency management
- Use `uv add <package>` to add dependencies
- Use `uv pip install -r requirements.txt` for bulk installs
- Use `uv run <command>` to run scripts in virtual environment

### Key Dependencies Setup
- **pgvector**: Required for PostgreSQL vector operations
  - Install: `uv pip install pgvector>=0.2.0`
  - Requires PostgreSQL with vector extension: `CREATE EXTENSION IF NOT EXISTS vector`
  - Must be in same environment as SQLAlchemy models that use `Vector()` columns
- **LLM library**: For local Qwen 3 model access
  - Install: `uv pip install llm`
  - **MLX Plugin**: Required for Qwen 3 model access: `uv pip install llm-mlx`
  - Available models: `mlx-community/Qwen3-8B-4bit` (local reasoning model)
  - Note: LLM models are stored globally but plugins must be installed per environment
- **Email processing**: langdetect, email-reply-parser, unstructured for text chunking

## Architecture

**Containers** (via docker-compose):
- `postgres` - Database with pgvector extension
- `imap-sync` - IMAP email ingestion

**Host** (Python with uv):
- `email-processor/` - All email processing (parsing, cleaning, RAG indexing)
- `classify.py` - Email classification with elegant CLI interface  
- `worker.py` - Queue-based email processing and RAG indexing worker
- `threader.py` - Threading and summarization worker with Qwen 3 LLM

## Deployment Tips

### Docker Services
- Only `postgres` and `imap-sync` run in containers
- Use `docker compose up -d postgres imap-sync` to start essential services

### Host-based Workers (run with uv)
- **Email Processing**: `cd email-processor && uv run python worker.py email-processing`
- **RAG Indexing**: `cd email-processor && uv run python worker.py rag-indexing`
- **Threading & Summarization**: `cd email-processor && uv run python threader.py`
- **Interactive Mode**: Add `--interactive` or `-i` for live commands (stats, view, help, etc.)
- **Classification**: `uv run python classify.py worker` for continuous classification
- **Classification Batch**: `uv run python classify.py batch --limit N` for one-off processing

### Threading Worker Options
- **Basic run**: `uv run python threader.py`
- **Interactive mode**: `uv run python threader.py --interactive`
- **Limited run**: `uv run python threader.py --max 10` (exit after processing 10 conversations)
- **Environment**: Requires DATABASE_URL and local Qwen 3 model via LLM library

### Beautiful Output & Interactive CLI
- All workers display rich emoji-based progress output with timing and statistics
- **Language detection** shown inline with processing results (e.g., `en:0.95`, `fr:1.00`)
- Interactive mode supports real-time commands: `stats`, `view <id>`, `pause`, `resume`, `help`, `quit`
- Language statistics in worker stats: distribution, detection rate, confidence averages
- Example output:
  ```
  19:21:45 │ 📦 Processing batch of 3 emails:
  19:21:45 │ 📧 #   52 web.check-in@iport.aero   → ✅ parsed   (250 chars en:1.00)  985ms
  19:21:45 │ 📧 #   50 marie@example.fr          → ✅ parsed   (180 chars fr:1.00)  120ms
  19:21:45 │ ✅ Batch complete: 3/3 parsed in 2.1s (1.4/s, 430 chars)
  ```

### Setup
- Install dependencies: `cd email-processor && uv pip install -r requirements.txt`
- All Python workers run on host for better performance and GPU access
- Direct library usage (no containers) for ML workloads
