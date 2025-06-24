# Email RAG System

A self-hosted, AI-powered email RAG system that transforms your email conversations into a
searchable, intelligent knowledge base.

Pipeline:

1. IMAP Sync: `go-imap-sql` dumps emails to Postgres - main table is `imap_messages`
2. Cleaning:
   1. Python bindings with Rust library `mail-parser`
      1. Extracts plaintext version in UTF-8
      2. Extracts participants with normalized email addresses, date, message-id, thread-id, from,
         to, cc, reply-to, thread-topic, references etc.
   2. Python library `email_reply_parser` removes threaded responses and signatures from plaintext
      bodies
   3. Result is saved in the `clean_messages` and participants are saved in JSONB column with proper
      indexes Notes: Attachments, unprocessable raw bodies, HTML are ignored
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

Essentially - Unstructured is receiving the plaintext email and:

```py
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.text import partition_text

def process(clean_message_text):
  elements = partition_text(clean_message_text)
  embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
  )
  for element in chunk_elements(elements):
    # Get the element's "text" field.
    text = element["text"]
    # Generate the embeddings for that "text" field.
    query_result = embeddings.embed_query(text)
    # Add the embeddings to that element as an "embeddings" field.
    yield query_result
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Mac mini M2 (16GB RAM) or equivalent
- **Software**: Docker, Docker Compose
- **Email**: IMAP-enabled email account

### Setup

1. **Clone and Configure**

   ```bash
   git clone https://github.com/your-repo/email-rag.git
   cd email-rag
   cp .env.example .env
   ```

2. **Configure Email Access**

   ```bash
   # Edit .env with your IMAP credentials
   nano .env

   # Required settings:
   IMAP_HOST=imap.gmail.com
   IMAP_USER=your-email@gmail.com
   IMAP_PASS=your-app-password  # Generate app password for Gmail
   ```

3. **Start Services**

   ```bash
   # Start all services
   docker-compose up -d
   ```

4. **Verify Services**

   ```bash
   # Check service health
   curl http://localhost:8080/health  # AI Processor
   curl http://localhost:8082/health  # Content Processor

   # Check processing status
   curl http://localhost:8080/process/status
   ```

## 📄 License

MIT License - see LICENSE file for details
