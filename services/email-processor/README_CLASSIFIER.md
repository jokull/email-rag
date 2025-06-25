# Email Classification Worker

Host-based email classification service using enhanced LLM classifier with first-line defense and correspondence intelligence.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv add llm sqlalchemy psycopg2-binary email-reply-parser
   ```

2. **Set environment variables (optional):**
   ```bash
   export DATABASE_URL="postgresql://email_user:email_pass@localhost:5432/email_rag"
   export LLM_MODEL_NAME="gpt-4o-mini"
   export MY_EMAIL="jokull@solberg.is"
   ```

3. **Run the classifier:**
   ```bash
   ./run_classifier.sh
   ```

## Manual Operation

**Start classification worker:**
```bash
python email_classifier_worker.py
```

**Check health:**
```bash
python check_classifier_health.py
```

**Test classifier directly:**
```bash
python -c "
from llm_classifier import LLMEmailClassifier, initialize_llm_classifier
initialize_llm_classifier()
classifier = LLMEmailClassifier()
classifier.initialize()
result = classifier.classify_email('test@example.com', 'Hello', 'Hi there!')
print(f'Result: {result.category} ({result.confidence})')
"
```

## Configuration

Environment variables:
- `DATABASE_URL` - PostgreSQL connection (default: localhost)
- `LLM_MODEL_NAME` - Model to use (default: gpt-4o-mini)
- `MY_EMAIL` - Your email for correspondence analysis (default: jokull@solberg.is)
- `BATCH_SIZE` - Emails per batch (default: 10)
- `SLEEP_INTERVAL` - Seconds between batches (default: 30)
- `MAILBOX_FILTER` - Which mailbox to process (default: INBOX)

## Performance

- **First-line defense**: 65%+ emails classified instantly (0ms)
- **LLM classification**: ~1.2s average for remaining emails
- **Patterns caught**: noreply@, notifications@, outbound@, hello@, %invoice%, %noreply%
- **Correspondence intelligence**: Uses sent email history for better classification

## Architecture

```
Host Machine:
├── email_classifier_worker.py (queue worker)
├── llm_classifier.py (enhanced classifier)
├── run_classifier.sh (runner script)
├── check_classifier_health.py (health check)
└── Database (PostgreSQL container)
```

## Logs

- Worker logs: `email_classifier_worker.log`
- Health check: Run `./check_classifier_health.py`
- Real-time: Watch console output when running manually

## Stopping

Press `Ctrl+C` to stop the worker gracefully.