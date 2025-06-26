# Email Processor Service

Email cleaning and processing service for the Email RAG system. This service processes raw emails from the `imap_messages` table, extracts metadata and clean content, and saves structured data to the `messages` table.

## Architecture

- **FastAPI Service**: Provides REST API for monitoring and manual processing
- **Background Worker**: Polls for unprocessed emails and processes them
- **Rust Bindings**: High-performance email parsing using `mail-parser` crate
- **Email Cleaning**: Removes quoted replies and signatures using `email_reply_parser`

## Pipeline

1. **Input**: Raw email bytes from `imap_messages.raw_message` 
2. **Parse**: Extract headers and plaintext using Rust `mail-parser`
3. **Clean**: Remove quotes/signatures with `email_reply_parser`
4. **Normalize**: Update contacts table with participant information
5. **Save**: Insert structured data into `messages` table

## Development

### Prerequisites

- Python 3.11+
- Rust (for building native bindings)
- PostgreSQL running with email_rag database

### Quick Start

```bash
# Start in development mode
./start_dev.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
maturin develop  # Build Rust bindings
python main.py   # Start FastAPI server
```

### Manual Worker

```bash
# Run worker separately
python worker.py
```

## API Endpoints

- `GET /` - Service info
- `GET /health` - Health check with database test
- `GET /status` - Processing statistics
- `GET /queue` - Queue status information
- `POST /process` - Manually trigger processing
- `GET /messages/{id}` - Get processed message details

## Configuration

Environment variables:

```bash
DATABASE_URL=postgresql://user:pass@host:port/db
HOST=0.0.0.0
PORT=8080
LOG_LEVEL=info
POLL_INTERVAL=30      # Worker polling interval (seconds)
BATCH_SIZE=50         # Messages processed per batch
SQL_DEBUG=false       # Enable SQL query logging
```

## Docker

### Build and Run

```bash
# Build image
docker build -t email-processor .

# Run FastAPI service
docker run -p 8080:8080 \
  -e DATABASE_URL=postgresql://... \
  email-processor

# Run worker
docker run \
  -e DATABASE_URL=postgresql://... \
  email-processor python worker.py
```

### Docker Compose

The service is integrated into the main `docker-compose.yml`:

```bash
# Start all services including email processor
docker-compose up -d

# View logs
docker-compose logs email-processor
docker-compose logs email-processor-worker
```

## Monitoring

### Health Checks

```bash
# Service health
curl http://localhost:8080/health

# Processing status
curl http://localhost:8080/status

# Queue information
curl http://localhost:8080/queue
```

### Processing Stats

The `/status` endpoint provides:
- Total IMAP messages
- Total processed messages
- Pending message count
- Processing status breakdown
- Processing rate

## Troubleshooting

### Rust Bindings

If Rust bindings fail to build:
1. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Run: `maturin develop`
3. The service falls back to Python email parsing if bindings unavailable

### Database Connection

- Ensure PostgreSQL is running on correct port
- Verify DATABASE_URL format
- Check network connectivity in Docker

### Processing Issues

- Check worker logs: `docker-compose logs email-processor-worker`
- Monitor API status: `curl http://localhost:8080/status`
- Check database for failed processing: Look for `processing_status='failed'`

## Future Enhancements

- [ ] Complete Rust bindings integration
- [ ] Add threading detection logic
- [ ] Implement classification service integration
- [ ] Add metrics and monitoring
- [ ] Performance optimizations