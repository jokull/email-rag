# Pipeline Monitoring Scripts

## Overview

The pipeline monitor script provides comprehensive statistics for the email RAG processing pipeline, tracking everything from IMAP sync to AI processing.

## Setup

1. **Quick Environment Setup** (recommended):
   ```bash
   cd scripts
   uv sync
   setup-env
   ```
   
   This interactive script will:
   - Prompt for your IMAP email settings
   - Auto-detect common providers (Gmail, Outlook, etc.)
   - Generate secure secrets with openssl
   - Create a complete .env file

2. **Manual Setup**:
   ```bash
   cd scripts
   uv sync
   cp ../.env.example ../.env
   # Edit .env with your settings
   ```

3. **Configure monitoring** (optional):
   ```bash
   export DATABASE_URL="postgresql://email_user:email_pass@localhost:5432/email_rag"
   export EMAIL_SCORER_URL="http://localhost:8081"
   export CONTENT_PROCESSOR_URL="http://localhost:8082"
   ```

## Usage

### One-time Statistics
```bash
# Using installed script
pipeline-monitor --once

# Or directly with uv
uv run python pipeline_monitor.py --once
```

### Continuous Monitoring
```bash
# Monitor every 60 seconds (default)
pipeline-monitor --continuous

# Monitor every 30 seconds  
pipeline-monitor --continuous --interval 30

# Or with uv
uv run python pipeline_monitor.py --continuous --interval 30
```

## Statistics Tracked

### 📨 IMAP Sync Status
- Raw IMAP messages vs synced emails
- Sync rate (emails/hour)
- Sync lag time

### 📧 Email Processing
- Total emails processed
- Processing rate
- Overall completion percentage

### 🧠 AI Classification
- Threads classified
- Average scores (human, importance, commercial, sentiment)

### 🔧 Content Processing
- Elements extracted via Unstructured.io
- Chunks created and embeddings generated
- Processing quality scores

### ⏳ Queue Status
- Pending items by queue type
- Failed processing attempts

### 🏥 Service Health
- Email scorer service status
- Content processor service status
- Database connectivity

### 💻 Resource Usage
- Memory and CPU usage from services

## Output

The script outputs both:
1. **Console display** - Formatted, human-readable statistics
2. **JSON files** - Machine-readable data for analysis

Example output:
```
================================================================================
📊 EMAIL RAG PIPELINE STATISTICS - 2024-06-22T14:30:00.000Z
================================================================================

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
   Avg commercial score:         0.156
   Avg sentiment score:          0.634
```

## Use Cases

- **Development**: Monitor pipeline performance during testing
- **Deployment**: Track processing progress on large email accounts
- **Debugging**: Identify bottlenecks and service issues
- **Optimization**: Measure impact of configuration changes
- **Reporting**: Generate regular status reports

## Tips

- Use continuous monitoring during initial IMAP sync of large accounts
- Monitor resource usage to optimize Mac mini M2 performance
- Check queue status to ensure processing isn't stuck
- Track quality scores to validate AI processing effectiveness