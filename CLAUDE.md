# Email RAG System - Claude Instructions

## Project Overview

This is a self-hosted email RAG system that syncs IMAP emails to PostgreSQL, processes them with
ML-based threading and classification, and provides a modern React UI for conversation management.

Pipeline:

1. IMAP Sync: `go-imap-sql` dumps emails to Postgres (pgvector) - main table is `imap_messages`
2. Cleaning:
   1. Python bindings with Rust library `mail-parser`
      1. Extracts plaintext version in UTF-8
      2. Extracts participants with normalized email addresses, date, message-id, thread-id, from,
         to, cc, reply-to, thread-topic, references etc.
   2. Python library `email_reply_parser` removes threaded responses and signatures from plaintext
      bodies
   3. Result is saved in the `clean_messages` and participants are saved in JSONB column with proper
      indexes Notes. Attachments, unprocessable raw bodies, HTML are all ignored and discarded.
3. Classification:
   1. Qwen-0.5B model pulls `clean_messages` and is prompted "Is this email
      human/promotional/transactional? Respond only with one of those words" and the `category`
      column is updated in `clean_messages`.
   2. SQL lookups on headers establish if this is _in reply to to existing email_ (threading), and a
      thread is created based on the "genesis message id" (`conversations` table)
   3. The thread table also has a denormalized `participants` JSONB that is kept up to date
   4. Another Qwen prompt summarizes in a one-liner the thread and saves in the `conversations`
      table as `summary`
4. Chunking & Embedding
   1. When clean messages are labeled "human" a service pulls messages and uses Unstructured service
      (docker-to-docker comms)
   2. Unstructured returns a list of vector embeddings per message and saves with known pgvector
      patterns
5. UI
   1. To be decided

## Current state

- Mail sync works great. An outdated ai-processor container service gives examples of how to get
  Qwen running.
- What's working well is imap-sync, postgres, the Unstructured container (and ai-processor insofar
  as I'm able to use qwen which I want to use)
-

## Migrations

- Never recreate imap\_ tables - mail sync will take too long and this part will remain as-is
- Recreate other tables like

## Package Management

- Use uv for python packages
- Modern Python tooling with uv for fast, reliable dependency management

## Deployment Tips

- Use container targeting when running the restart script
