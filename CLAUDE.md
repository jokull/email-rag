# Email RAG System - Claude Instructions

## Project Overview
This is a self-hosted email RAG system that syncs IMAP emails to PostgreSQL, processes them with ML-based threading and classification, and provides a modern React UI for conversation management.

## Architecture
- **IMAP Sync**: go-imap-sql syncs emails to PostgreSQL with FSStore for raw email storage
- **Database**: PostgreSQL with pgvector for vector embeddings
- **Email Scorer**: Lightweight Qwen-0.5B service for rapid email triage and multi-dimensional scoring
- **Content Processor**: Queue-based service using Unstructured.io for HTML email processing and embeddings
- **Threading**: Python service with JWZ + BERT-based ML threading (90%+ accuracy)
- **UI**: React with TanStack Router, Zero for real-time sync, @tanstack/react-virtual
- **Icons**: justd-icons (IntentUI icons)
- **Runtime**: Bun (not Node.js)

## 🧠 LLM Agent Configuration: Qwen-0.5B on Mac mini M2 (16GB RAM)

### Model Setup
- **Model**: Qwen-0.5B (GGUF quantized Q4_0)
- **Disk Size**: ~395 MB
- **RAM at Runtime**: ~0.5–1.5 GB (incl. context buffers)
- **Machine**: Apple Mac mini M2, 16GB unified memory
- **Acceleration**: Metal GPU / Apple Neural Engine (via llama.cpp)

### Ideal Use Cases
- ✅ Multi-dimensional email scoring: importance, sentiment, commercial, human detection
- ✅ Rapid email triage and intelligent queue processing
- ✅ HTML email content extraction with Unstructured.io
- ✅ Semantic email chunking with structure preservation
- ✅ Element-level embeddings for enhanced RAG retrieval
- ✅ Queue-based processing with priority scoring

### New Pipeline Architecture
1. **IMAP → PostgreSQL**: Raw email ingestion
2. **Email Scorer**: Qwen-0.5B rapid scoring (sentiment, importance, commercial, human)
3. **Queue Management**: Priority-based content processing queue
4. **Content Processor**: Unstructured.io + embeddings for qualifying emails
5. **pgvector**: Enhanced vector storage with element-level granularity

### Resource Allocation (Mac mini M2 16GB)
- **Email Scorer**: ~1GB RAM (rapid triage)
- **Content Processor**: ~2GB RAM (Unstructured + embeddings)
- **Legacy AI Processor**: ~1GB RAM (reduced scope)
- **Total**: ~4GB RAM for AI services (25% of system memory)

## Key Commands
- Start services: `docker-compose up -d`
- Format/lint/typecheck: `cd web && bun format lint:fix typecheck`
- Build UI: `cd web && bun run build`

## Database Schema
Core tables: `emails`, `threads`, `conversations`, `conversation_turns`, `cleaned_emails`, `embeddings`, `classifications`

## Development Notes
- Keep things simple for self-hosting - avoid production complexity
- Zero cache server runs as `rocicorp/zero:{version}` Docker container
- Thread processor uses Talon library for production-grade email cleaning
- UI based on zbugs patterns with virtualization and real-time filtering
- Classification scores: Human/Personal/Relevance for RAG pipeline filtering

## Package Management
- Use uv for python packages