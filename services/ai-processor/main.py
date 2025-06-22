"""
Consolidated AI Processor for Email RAG System
Handles all LLM tasks in a single service optimized for Qwen-0.5B on Mac mini M2

This service consolidates:
- Email classification (human vs promotional/transactional/automated)
- Thread detection and conversation building  
- Content cleaning and chunking
- Embedding generation for RAG pipeline
- All ML-based processing tasks

Optimized for:
- Mac mini M2 16GB RAM
- Qwen-0.5B (620M params, Q4_0 GGUF quantized)
- Low memory footprint (~1-2GB total)
- Fast inference with Metal acceleration
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime, timedelta

# Import our consolidated modules
from config import get_config, LLMConfig, ProcessorConfig
from llm_interface import QwenInterface, create_qwen_interface
from email_processor import EmailProcessor, ProcessingResult
from embedding_service import EmbeddingService
from health_monitor import HealthMonitor

# Global instances
llm_interface: Optional[QwenInterface] = None
email_processor: Optional[EmailProcessor] = None
embedding_service: Optional[EmbeddingService] = None
health_monitor: Optional[HealthMonitor] = None
database_engine = None
SessionLocal = None

# Configuration
llm_config: LLMConfig
processor_config: ProcessorConfig

async def setup_services():
    """Initialize all AI processing services"""
    global llm_interface, email_processor, embedding_service, health_monitor
    global database_engine, SessionLocal, llm_config, processor_config
    
    logging.info("🚀 Starting AI Processor - Qwen-0.5B on Mac mini M2")
    
    # Load configuration
    llm_config, processor_config = get_config()
    
    # Setup database connection
    database_engine = create_engine(processor_config.database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=database_engine)
    
    # Initialize LLM interface (this loads and warms the model)
    logging.info("🧠 Initializing Qwen-0.5B model...")
    llm_interface = create_qwen_interface(llm_config)
    
    # Wait for model to warm up
    model_ready = False
    max_retries = 30
    for i in range(max_retries):
        if await llm_interface.health_check():
            model_ready = True
            break
        logging.info(f"⏳ Waiting for model to load... ({i+1}/{max_retries})")
        await asyncio.sleep(2)
    
    if not model_ready:
        raise RuntimeError("Failed to load Qwen-0.5B model")
    
    logging.info("✅ Qwen-0.5B model loaded and ready")
    
    # Initialize other services
    embedding_service = EmbeddingService(processor_config)
    email_processor = EmailProcessor(llm_interface, embedding_service, processor_config)
    health_monitor = HealthMonitor(llm_interface, email_processor, embedding_service)
    
    logging.info("🎯 All AI services initialized successfully")

async def shutdown_services():
    """Clean shutdown of all services"""
    logging.info("🛑 Shutting down AI Processor...")
    
    # Stop background processing
    if email_processor:
        email_processor.stop_processing()
    
    # Close database connections
    if database_engine:
        database_engine.dispose()
    
    logging.info("✅ AI Processor shutdown complete")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    await setup_services()
    yield
    await shutdown_services()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Email RAG AI Processor",
    description="Consolidated AI processing service for email RAG system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    if not health_monitor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    health_status = await health_monitor.get_health_status()
    
    if not health_status["healthy"]:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    if not health_monitor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return await health_monitor.get_detailed_metrics()

# Manual processing trigger (useful for debugging)
@app.post("/process/trigger")
async def trigger_processing(background_tasks: BackgroundTasks):
    """Manually trigger email processing cycle"""
    if not email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    background_tasks.add_task(email_processor.process_pending_emails)
    return {"status": "processing_triggered", "timestamp": datetime.utcnow()}

# Get processing status
@app.get("/process/status")
async def get_processing_status():
    """Get current processing status and queue information"""
    if not email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    return await email_processor.get_processing_status()

# Classification endpoint (for external API access)
@app.post("/classify")
async def classify_content(content: Dict[str, Any]):
    """Classify email content via API"""
    if not llm_interface:
        raise HTTPException(status_code=503, detail="LLM interface not ready")
    
    email_content = content.get("content", "")
    if not email_content:
        raise HTTPException(status_code=400, detail="Content required")
    
    try:
        result = await llm_interface.classify_email(email_content)
        return {
            "classification": result.classification,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "tokens_used": result.tokens_used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Embedding endpoint
@app.post("/embed")
async def generate_embedding(content: Dict[str, Any]):
    """Generate embedding for text content"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not ready")
    
    text = content.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text required")
    
    try:
        embedding = await embedding_service.generate_embedding(text)
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "model": processor_config.embedding_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

async def main_processing_loop():
    """Main background processing loop"""
    logging.info("🔄 Starting main processing loop...")
    
    while True:
        try:
            if email_processor and health_monitor:
                # Check if we should process
                health_status = await health_monitor.get_health_status()
                
                if health_status["healthy"]:
                    # Process pending emails
                    await email_processor.process_pending_emails()
                else:
                    logging.warning("⚠️ Skipping processing due to unhealthy status")
                
                # Sleep for the configured interval
                await asyncio.sleep(processor_config.processing_interval)
            else:
                # Services not ready yet
                await asyncio.sleep(5)
                
        except Exception as e:
            logging.error(f"❌ Error in main processing loop: {e}")
            await asyncio.sleep(30)  # Wait longer on error

def setup_logging():
    """Configure logging for the service"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("/app/logs/ai-processor.log") if os.path.exists("/app/logs") else logging.NullHandler()
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logging.info(f"📡 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start background processing task
    async def run_app():
        # Start the main processing loop in the background
        processing_task = asyncio.create_task(main_processing_loop())
        
        # Start the FastAPI server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8080,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        try:
            await server.serve()
        finally:
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
    
    # Run the application
    asyncio.run(run_app())