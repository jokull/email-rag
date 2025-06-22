# Email RAG System - Quick Start Guide

Get your self-hosted email RAG system running in minutes on Mac mini M2.

## Prerequisites

- Mac mini M2 (16GB RAM recommended)
- Docker & Docker Compose
- [uv](https://docs.astral.sh/uv/) for Python package management
- Email account with IMAP access

## 🚀 Quick Setup (5 minutes)

### 1. Clone and Setup Environment

```bash
git clone https://github.com/your-repo/email-rag.git
cd email-rag

# Interactive environment setup
cd scripts
uv sync
uv run python setup_env.py
```

The setup script will:

- ✅ Prompt for your email credentials
- ✅ Auto-detect IMAP settings for Gmail, Outlook, etc.
- ✅ Generate secure secrets with openssl
- ✅ Create a complete `.env` file

### 2. Start Services

```bash
cd ..
docker-compose up -d
```

### 3. Monitor Progress

```bash
cd scripts
pipeline-monitor --continuous
```

### 4. Access Your System

- **Web UI**: http://localhost:3001
- **Email Scorer API**: http://localhost:8081/health
- **Content Processor API**: http://localhost:8082/health

## 📊 What Happens Next

The system will automatically:

1. **IMAP Sync** (immediate): Download emails from your account
2. **AI Classification** (5-30 min): Multi-dimensional scoring with Qwen-0.5B
3. **Content Processing** (10-60 min): Clean emails → structured elements → embeddings
4. **UI Updates** (real-time): Live conversation browser with markdown display

## 📈 Monitor Your Pipeline

Watch the processing in real-time:

```bash
# One-time stats
pipeline-monitor --once

# Continuous monitoring
pipeline-monitor --continuous --interval 30
```

Example output:

```
📨 IMAP SYNC STATUS:
   Raw IMAP messages:            15,234
   Synced to emails table:       15,180 (99.6%)
   IMAP sync rate (last hour):   45 emails
   Sync lag:                     2.3 minutes

📧 EMAIL PROCESSING OVERVIEW:
   Total emails in database:     15,180
   Emails processed:             8,750 (57.6%)
   Processing rate:              12.5 emails/minute

🧠 AI CLASSIFICATION RESULTS:
   Threads classified:           3,420
   Avg human score:              0.742
   Avg importance score:         0.321
```

## 🔧 Common Email Providers

The setup script auto-detects settings for:

| Provider        | IMAP Server           | Port | Notes            |
| --------------- | --------------------- | ---- | ---------------- |
| Gmail           | imap.gmail.com        | 993  | Use App Password |
| Outlook/Hotmail | outlook.office365.com | 993  | Use App Password |
| Yahoo           | imap.mail.yahoo.com   | 993  | Regular password |
| iCloud          | imap.mail.me.com      | 993  | Use App Password |

## ⚡ Performance (Mac mini M2 16GB)

Expected processing speeds:

- **Email Scoring**: 5-10 emails/second (1GB RAM)
- **Content Processing**: 1-3 emails/second (2GB RAM)
- **Total Throughput**: 500-1000 emails/hour
- **Memory Usage**: ~4GB (25% of system RAM)

## 🎯 Usage Examples

### Browse Clean Conversations

- Navigate to `/conversations` for threaded email display
- Filter by human/promotional/transactional
- Search cleaned email content
- Discover similar conversations

### AI-Powered Insights

- **Sentiment Analysis**: Identify negative feedback quickly
- **Importance Scoring**: Focus on high-priority communications
- **Commercial Filtering**: Hide marketing emails
- **Similar Discovery**: Find related discussions across time

## 🛠️ Troubleshooting

### Email Sync Issues

```bash
# Check IMAP sync service
docker-compose logs imap-sync

# Test IMAP connection
telnet your-imap-server 993
```

### Processing Stuck

```bash
# Check service health
curl http://localhost:8081/health
curl http://localhost:8082/health

# Monitor queue status
pipeline-monitor --once
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Reduce batch sizes in .env
SCORING_BATCH_SIZE=5
PROCESSING_BATCH_SIZE=3
```

### Reset Everything

```bash
docker-compose down -v  # ⚠️ Removes all data
docker-compose up -d
```

## 📚 Next Steps

- **Large Accounts**: The initial sync may take hours for 10k+ emails
- **Customization**: Edit `.env` for advanced configuration
- **Monitoring**: Set up continuous monitoring for production use
- **Backup**: Consider backing up the PostgreSQL database

For detailed configuration options, see the full [README.md](./README.md).
