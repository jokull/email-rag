# Email Cleaning Service Implementation Plan

## Overview

Create a Python service that processes raw emails from imap_messages table, extracts clean content and metadata using Rust mail_parse
crate bindings, and saves to messages table.

## Architecture

- FastAPI service (uvicorn) for debugging/status endpoints
- Simple worker loop (separate process) polling imap_messages table
- Custom Python bindings for Rust mail_parse crate
- SQLAlchemy for database operations
- No AI/classification - pure cleaning only

## Implementation Steps

### Phase 1: Rust Bindings Setup

1. Create services/email-processor/ directory
2. Set up Rust project with mail_parse crate dependency
3. Create Python bindings using PyO3 to expose:
   - Header extraction (message-id, subject, from, to, cc, reply-to, references, in-reply-to, thread-topic, date)
   - Clean UTF-8 plaintext body extraction
   - Participant normalization
4. Build wheel and integrate with Python requirements

### Phase 2: Python Service Structure

1. FastAPI app (main.py):
   - Health check endpoint
   - Processing status endpoint
   - Queue stats endpoint
2. Worker process (worker.py):
   - Poll imap_messages for unprocessed records
   - Use Rust bindings to parse raw email bytes
   - Apply email_reply_parser to remove quotes/signatures
   - Save results to messages table
   - Update processing timestamps
3. Database models using SQLAlchemy
4. Docker setup with multi-stage build

### Phase 3: Processing Pipeline

1. Input: Raw bytes from imap_messages.raw_message
2. Parse: Rust bindings extract headers + plaintext
3. Clean: email_reply_parser removes quotes/signatures
4. Save: Insert into messages table with metadata
5. Update: Mark processing stages (parsed_at, cleaned_at)

### Phase 4: Service Integration

1. Add service to docker-compose.yml
2. Environment configuration
3. Database connection setup
4. Error handling and logging
5. Health checks and monitoring

## Technical Stack

- Rust: mail_parse crate + PyO3 for bindings
- Python: FastAPI, SQLAlchemy, email_reply_parser
- Database: PostgreSQL with existing schema
- Docker: Multi-stage build (Rust compile → Python runtime)

## Success Criteria

- Raw email bytes → clean structured data in messages table
- FastAPI endpoints for monitoring
- Simple polling worker (no queue complexity)
- Foundation ready for future AI classification step
