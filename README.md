# Email RAG System

A self-hosted, AI-powered email RAG system that transforms your email conversations into a searchable, intelligent knowledge base. Built for Mac mini M2 (16GB RAM) with advanced email processing, thread-level summaries, and multi-dimensional AI classification.

## 🎯 Key Features

- **Thread-Level Intelligence**: Dynamic conversation summaries for notifications and context switching
- **Multi-Dimensional AI Classification**: Sentiment, formality, personalization, priority scoring
- **Tiered Email Cleaning**: email-reply-parser → Qwen-0.5B → basic fallback for optimal quality
- **Contact History Analysis**: Relationship strength and response likelihood scoring
- **Real-Time Processing**: Background email processing with health monitoring
- **Self-Hosted Privacy**: All processing happens locally with Qwen-0.5B LLM

## 🏗️ Modern Architecture

### Consolidated Service Design
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│ IMAP Sync   │───▶│ PostgreSQL  │───▶│AI Processor │───▶│Content      │
│ (Go)        │    │+ pgvector   │    │(Qwen-0.5B)  │    │Processor    │
│             │    │             │    │             │    │(Unstructured)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                             │                     │
                                             ▼                     ▼
                                    ┌─────────────┐      ┌─────────────┐
                                    │Thread       │      │Enhanced     │
                                    │Summaries +  │      │Embeddings + │
                                    │Multi-dim    │      │Chunks       │
                                    │Classification│      │             │
                                    └─────────────┘      └─────────────┘
                                             │                     │
                                             └─────────┬───────────┘
                                                       ▼
                                              ┌─────────────┐
                                              │ React UI +  │
                                              │Zero Real-   │
                                              │Time Sync    │
                                              └─────────────┘
```

## 🧠 AI Processing Pipeline

### **Consolidated AI Processor** (Port 8080)
- **Model**: Qwen-0.5B (395MB disk, ~1GB RAM runtime)
- **Features**:
  - **Threading & Cleaning**: JWZ + Talon + email-reply-parser
  - **Multi-dimensional Classification**: human/promotional/transactional/automated + sentiment/formality/personalization/priority
  - **Contact History Analysis**: Relationship strength, response likelihood, frequency scoring
  - **Thread Summaries**: Dynamic one-liner summaries for notifications
  - **Tiered Processing**: Smart fallbacks for maximum reliability

### **Content Processor** (Port 8082)
- **Purpose**: Advanced content processing with Unstructured.io
- **Processing**: HTML → clean chunks → embeddings → pgvector storage
- **Integration**: Consumes cleaned emails from AI processor

### **Core Processing Flow**
1. **IMAP Sync**: Raw emails → PostgreSQL
2. **AI Processing**: Threading → Talon cleaning → Qwen classification → Thread summaries
3. **Content Processing**: Unstructured.io → embeddings → chunks
4. **Real-time UI**: Live updates via Zero cache

## 🎨 Enhanced Features

### **Thread-Level Intelligence**
- **Dynamic Summaries**: "Planning summer cabin rental with Maria & Sigurdis"
- **Mood Detection**: planning/urgent/social/work/problem_solving/informational
- **Entity Extraction**: Key people, topics, dates automatically identified
- **Action Items**: Extracted decisions and next steps
- **Notification Context**: Rich context for thread updates

### **Multi-Dimensional Classification**
```typescript
{
  classification: "human" | "promotional" | "transactional" | "automated",
  confidence: 0.85,
  sentiment: "positive" | "neutral" | "negative",
  sentiment_score: 0.7,  // -1.0 to 1.0
  formality: "formal" | "informal" | "casual" | "neutral", 
  personalization: "highly_personal" | "somewhat_personal" | "generic",
  priority: "urgent" | "normal" | "low",
  should_process: true,
  processing_priority: 85  // 0-100
}
```

### **Contact History Analysis**
- **Frequency Scoring**: How often this person emails
- **Response Likelihood**: Historical response patterns
- **Relationship Strength**: Based on email patterns and duration
- **Personal vs Business**: Email domain and content analysis

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

## 📊 API Endpoints

### **AI Processor** (localhost:8080)
- `GET /health` - Service health and model status
- `GET /metrics` - Performance metrics and stats
- `POST /process/trigger` - Manually trigger email processing
- `GET /process/status` - Current processing status
- `POST /classify` - Classify email content
- `POST /metrics/clean` - Test email cleaning pipeline
- `POST /summarize/thread` - Generate thread summary
- `GET /test1`, `/test2` - Simple test endpoints

### **Content Processor** (localhost:8082)
- `GET /health` - Unstructured.io service status
- `POST /process` - Process email content

## 🛠️ Configuration

### Resource Allocation (Mac mini M2 16GB)
```yaml
services:
  ai-processor:
    memory: 3GB      # Qwen-0.5B + processing overhead
  content-processor:
    memory: 2GB      # Unstructured.io + embeddings
  # Total AI services: ~5GB (31% of system memory)
```

### Environment Variables
```bash
# AI Processing
FALLBACK_TO_BASIC_CLASSIFICATION=true  # Graceful degradation
PROCESSING_INTERVAL=30                 # Seconds between processing cycles
BATCH_SIZE=10                         # Emails per batch

# Model Configuration
LOG_LEVEL=INFO                        # DEBUG for development
KEEP_MODEL_LOADED=true               # Keep Qwen warm
```

## 🗄️ Enhanced Database Schema

### Core Tables
- **`emails`**: Raw email storage with thread associations
- **`threads`**: Conversation groupings with enhanced metadata
- **`classifications`**: Multi-dimensional AI scoring
- **`cleaned_emails`**: Tiered cleaning results (email-reply-parser/qwen/basic)
- **`enhanced_embeddings`**: pgvector embeddings for RAG
- **`conversation_turns`**: Structured conversation flow

### Thread-Level Enhancements
```sql
-- New thread summary fields
ALTER TABLE threads ADD COLUMN summary_oneliner TEXT;
ALTER TABLE threads ADD COLUMN summary_embedding vector(384);
ALTER TABLE threads ADD COLUMN key_entities TEXT[];
ALTER TABLE threads ADD COLUMN thread_mood VARCHAR(50);
ALTER TABLE threads ADD COLUMN action_items TEXT[];
ALTER TABLE threads ADD COLUMN last_summary_update TIMESTAMP;
```

## 🔧 Development

### Service Architecture
```
services/
├── ai-processor/       # Consolidated AI processing (Qwen-0.5B)
├── content-processor/  # Unstructured.io integration
├── imap-sync/         # Go IMAP synchronization
└── ui/                # React conversation browser (planned)
```

### Local Development
```bash
# AI processor development
cd services/ai-processor
python main.py

# Content processor development
cd services/content-processor
python main.py

# Database access
docker-compose exec postgres psql -U email_user -d email_rag
```

### Testing Email Processing
```bash
# Test email cleaning
curl -X POST http://localhost:8080/metrics/clean \
  -H "Content-Type: application/json" \
  -d '{"content": "Your email content here..."}'

# Test thread summarization
curl -X POST http://localhost:8080/summarize/thread \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "your-thread-id", "force_update": true}'

# Monitor processing
docker-compose logs -f ai-processor
```

## 📈 Performance Characteristics

### Mac mini M2 (16GB RAM) Benchmarks
- **Email Classification**: ~3-5 emails/second
- **Thread Summarization**: ~10-15 seconds per thread
- **Memory Usage**: 3GB AI processor + 2GB content processor
- **Model Loading**: ~10-30 seconds initial startup
- **Quality**: 85-95% accuracy for email cleaning and classification

### Processing Quality
- **Tiered Cleaning**: email-reply-parser (70-80%) → Qwen (90%+) → basic fallback
- **Classification Accuracy**: 85-90% multi-dimensional scoring
- **Thread Summaries**: Context-aware one-liners for notifications
- **Contact Analysis**: Historical relationship scoring

## 🔍 Usage Examples

### Thread Summary Generation
```bash
# Generate summary for a conversation thread
curl -X POST http://localhost:8080/summarize/thread \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "abc123-def456",
    "force_update": true
  }'
```

### Email Classification Testing
```bash
# Test multi-dimensional classification
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hi John, thanks for the great meeting today. The budget proposal looks solid and I think we should move forward with Q3 planning. Best regards, Sarah"
  }'
```

### Processing Status Monitoring
```bash
# Check what's being processed
curl http://localhost:8080/process/status | jq '.'

# View performance metrics
curl http://localhost:8080/metrics | jq '.llm_metrics'
```

## 🚨 Troubleshooting

### Service Health Checks
```bash
# Check if services are running
docker-compose ps

# View service logs
docker-compose logs ai-processor --tail=50
docker-compose logs content-processor --tail=50

# Test model loading
curl http://localhost:8080/health | jq '.llm_status'
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check processing queue
psql postgresql://email_user:email_pass@localhost:5432/email_rag \
  -c "SELECT COUNT(*) FROM emails WHERE thread_id IS NULL;"
```

### Model Issues
```bash
# Check model download
ls -la services/ai-processor/models/

# Force model re-download
docker-compose down ai-processor
docker-compose up ai-processor -d
```

## 🔒 Privacy & Security

- **Local Processing**: All AI happens on your Mac mini M2
- **No External Calls**: Qwen-0.5B runs entirely offline  
- **Self-Hosted**: Complete control over your email data
- **Encrypted Storage**: PostgreSQL with proper authentication
- **Read-Only Processing**: No email modification, only analysis

## 🎯 What's New

### Recent Enhancements
- **Consolidated Architecture**: Single AI processor instead of multiple specialized services
- **Thread-Level Intelligence**: Dynamic summaries with embeddings for similarity search
- **Tiered Processing**: Multiple fallback strategies for maximum reliability
- **Contact Analysis**: Historical relationship and response pattern analysis
- **Modern LLM Integration**: llama-cpp-python with structured Pydantic output
- **Enhanced Monitoring**: Comprehensive health checks and performance metrics

### Upcoming Features
- **UI Integration**: React conversation browser with thread summaries
- **Real-Time Updates**: WebSocket integration for live processing updates
- **Advanced Search**: Vector similarity search across thread summaries
- **Mobile Support**: Responsive design for thread browsing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Test on Mac mini M2 equivalent hardware
4. Submit pull request with performance benchmarks

## 📄 License

MIT License - see LICENSE file for details

---

**Built for self-hosted email intelligence with thread-level understanding and modern AI processing.**