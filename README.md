# Email RAG System

A self-hosted, AI-powered email RAG system that transforms your email conversations into a
searchable, intelligent knowledge base.

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
