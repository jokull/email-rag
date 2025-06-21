# Email RAG System

A dockerized AI-powered email analysis system that intelligently processes, classifies, and enables semantic search through your email conversations.

## Features

- **IMAP Email Synchronization**: Automatically syncs emails from any IMAP server
- **AI-Powered Classification**: Uses local LLM models to filter human conversations from promotional/automated emails
- **Semantic Search**: RAG pipeline with vector embeddings for natural language email search
- **Real-time Updates**: Rocicorp Zero provides instant sync across all components
- **Thread Intelligence**: Automatically groups emails into conversations with participant tracking
- **Contact Insights**: Tracks relationship strength and communication frequency
- **Read-only Interface**: Safe, search-focused email exploration without modification risks

## Architecture

### Core Services

1. **PostgreSQL Database** - Central data store with pgvector extension for embeddings
2. **Go IMAP Server** - Go service using go-imap-sql for efficient email synchronization
3. **AI Classifier** - Python service using simonw/llm with Qwen models for content classification
4. **RAG Pipeline** - Python service for text chunking and embedding generation using Qwen3-Embedding
5. **Zero Server** - Real-time data synchronization server
6. **Web UI** - React application with TanStack Router and Justd components

### Data Flow

```
External IMAP → Go IMAP Server → go-imap-sql → PostgreSQL → Event Stream
                                                    ↓
                                              [ Zero UI / RAG Pipeline ]
                                                    ↓
                                          PostgreSQL (vectors, classifications)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Email account with IMAP access (Gmail, Outlook, etc.)
- 8GB+ RAM recommended for local LLM models

### Setup

1. **Clone and Configure**
   ```bash
   git clone <repository-url>
   cd email-rag
   cp .env.example .env
   ```

2. **Edit Environment Variables**
   ```bash
   # IMAP Configuration
   IMAP_HOST=imap.gmail.com
   IMAP_PORT=993
   IMAP_USER=your-email@gmail.com
   IMAP_PASS=your-app-password  # Use app password for Gmail
   IMAP_TLS=true

   # AI Models (will be auto-downloaded)
   LLM_MODEL=llama3.2:3b
   EMBEDDING_MODEL=nomic-embed-text
   ```

3. **Start the System**
   ```bash
   docker-compose up -d
   ```

4. **Access the Interface**
   - Web UI: http://localhost:3001
   - Zero Server: http://localhost:3000

### Initial Sync

The system will begin syncing your emails immediately. Processing stages:

1. **Email Sync** (5-30 minutes depending on inbox size)
2. **AI Classification** (10-60 minutes for classification)
3. **RAG Processing** (20-90 minutes for embeddings)

Monitor progress in docker logs:
```bash
docker-compose logs -f imap-server
docker-compose logs -f ai-classifier
docker-compose logs -f rag-pipeline
```

## Configuration

### Email Provider Setup

#### Gmail
1. Enable 2-factor authentication
2. Generate an App Password
3. Use `imap.gmail.com:993` with TLS

#### Outlook/Hotmail
1. Enable IMAP in account settings
2. Use `outlook.office365.com:993`

#### Custom IMAP
Update `.env` with your provider's IMAP settings

### AI Model Configuration

The system uses local LLM models via Simon Willison's `llm` CLI with Qwen models:

- **Classification Model**: `qwen2.5:3b` (lightweight, fast, multilingual)
- **Embedding Model**: `sentence-transformers/Qwen/Qwen3-Embedding-0.6B` (1024-dimensional vectors, Apache 2.0 licensed)

Models are automatically downloaded on first run. Ensure sufficient disk space (~4GB+).

#### Why Qwen Models?
- **Qwen3 Embeddings**: Highest scoring open-weight models on MTEB leaderboard
- **Apache 2.0 Licensed**: Commercial-friendly licensing
- **Efficient**: 0.6B parameter embedding model with excellent performance
- **Multilingual**: Strong support for multiple languages

### Performance Tuning

Adjust processing parameters in `.env`:

```bash
# Sync frequency (milliseconds)
SYNC_INTERVAL=300000  # 5 minutes

# AI processing batch sizes
BATCH_SIZE=10
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## Usage

### Browse Conversations
- Navigate to the home page to see threaded conversations
- Only human-classified conversations are shown
- Click threads to view full email exchanges

### Semantic Search
- Use the search page for natural language queries
- Examples:
  - "emails about project deadlines"
  - "vacation requests from last month"
  - "budget discussions with finance team"

### Understanding Classifications

The AI classifier categorizes emails as:
- **Human**: Real person-to-person communication
- **Promotional**: Marketing, newsletters, sales
- **Transactional**: Receipts, confirmations, alerts
- **Automated**: System notifications, reports

Only "Human" conversations appear in the UI and search results.

## API Integration

The Zero server exposes real-time data APIs:

```javascript
import { Zero } from '@rocicorp/zero'

const zero = new Zero({
  server: 'http://localhost:3000',
  schema: { /* See services/api/src/schema.ts */ }
})

// Query threads
const threads = zero.query.threads
  .related('classification', q => q.where('classification', 'human'))
  .orderBy('lastMessageDate', 'desc')
  .useQuery()
```

## Development

### Service Structure

```
services/
├── database/       # PostgreSQL schema and init scripts
├── imap-server/    # Go IMAP server with go-imap-sql
├── ai-classifier/  # Python AI classification service with Qwen models
├── rag-pipeline/   # Python RAG processing service with Qwen3 embeddings
├── api/           # Zero server for real-time sync
└── ui/            # React web interface
```

### Running Individual Services

```bash
# Start only database
docker-compose up postgres

# Start IMAP server in development
cd services/imap-server
go mod tidy
go run .

# Start UI in development
cd services/ui
npm install
npm run dev
```

### Database Access

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U email_user -d email_rag

# View sync status
SELECT COUNT(*) FROM emails;
SELECT classification, COUNT(*) FROM classifications GROUP BY classification;
```

## Troubleshooting

### Common Issues

1. **IMAP Connection Failed**
   - Verify credentials and app passwords
   - Check firewall settings
   - Ensure IMAP is enabled in email provider

2. **AI Models Not Loading**
   - Check available disk space (>4GB needed)
   - Monitor classifier logs: `docker-compose logs ai-classifier`
   - Models download automatically on first run

3. **Slow Processing**
   - Reduce batch sizes in `.env`
   - Monitor system resources
   - Consider using lighter models for testing

4. **No Search Results**
   - Verify embeddings are generated: `SELECT COUNT(*) FROM embeddings;`
   - Check if threads are classified as 'human'
   - Allow more time for RAG processing

### Performance Monitoring

```bash
# Check service health
docker-compose ps

# Monitor resource usage
docker stats

# View processing progress
docker-compose logs -f rag-pipeline
```

## Security Notes

- Email credentials are stored in environment variables only
- No email content is sent to external services
- All AI processing happens locally
- Database access is containerized and isolated
- UI is read-only - no email modification capabilities

## Implementation Notes

This system integrates [go-imap-sql](https://github.com/foxcpp/go-imap-sql) for efficient IMAP-to-PostgreSQL email storage, combined with Simon Willison's LLM CLI tools for local AI processing. The architecture follows the principle: **External IMAP → go-imap-sql → PostgreSQL → Event Stream → [Zero UI / RAG Pipeline]**

### Key Technical Decisions:
- **go-imap-sql**: Proven SQL-based IMAP storage backend
- **Qwen3 Embeddings**: State-of-the-art open-weight embedding models  
- **Rocicorp Zero**: Eliminates complex state management for real-time UI
- **PostgreSQL + pgvector**: Unified storage for structured data and vector embeddings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details