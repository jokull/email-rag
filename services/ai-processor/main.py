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
from modern_llm import ModernQwenInterface
from email_processor import EmailProcessor, ProcessingResult
from embedding_service import EmbeddingService
from health_monitor import HealthMonitor

# Global instances
llm_interface: Optional[ModernQwenInterface] = None
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
    
    logging.info("🚀 Starting Email Threading & Cleaning Service (with Qwen)")
    
    # Load configuration
    llm_config, processor_config = get_config()
    
    # Setup database connection
    database_engine = create_engine(processor_config.database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=database_engine)
    
    # Try to initialize LLM interface
    llm_interface = None
    fallback_mode = os.getenv("FALLBACK_TO_BASIC_CLASSIFICATION", "false").lower() == "true"
    disable_llm = os.getenv("DISABLE_LLM_INIT", "false").lower() == "true"
    
    if disable_llm:
        logging.info("🔧 LLM initialization disabled for debugging")
        llm_interface = None
    else:
        try:
            logging.info("🧠 Initializing Modern Qwen-0.5B interface...")
            llm_interface = ModernQwenInterface(llm_config, processor_config)
            
            # Initialize the model
            model_ready = await llm_interface.initialize()
            
            if model_ready:
                logging.info("✅ Modern Qwen-0.5B interface loaded and ready")
            else:
                raise RuntimeError("Model failed to initialize")
                
        except Exception as e:
            if fallback_mode:
                logging.warning(f"⚠️ Qwen model failed to load: {e}")
                logging.info("📧 Falling back to basic classification (threading + Talon cleaning only)")
                llm_interface = None
            else:
                raise RuntimeError(f"Failed to load Qwen-0.5B model: {e}")
    
    # Initialize other services
    try:
        embedding_service = EmbeddingService(processor_config)
    except Exception as e:
        logging.warning(f"⚠️ Embedding service failed to initialize: {e}")
        embedding_service = None
        
    email_processor = EmailProcessor(llm_interface, embedding_service, processor_config, SessionLocal)
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

# Email cleaning test endpoint (using existing metrics route pattern)
@app.post("/metrics/clean")
async def test_email_cleaning(content: Dict[str, Any]):
    """Test email cleaning functionality"""
    if not email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    raw_content = content.get("content", "")
    if not raw_content:
        raise HTTPException(status_code=400, detail="Content required")
    
    try:
        # Test our tiered cleaning approach
        cleaned = await email_processor._clean_with_tiered_approach(raw_content)
        method_used = getattr(email_processor, '_last_cleaning_method', 'unknown')
        
        return {
            "success": True,
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned),
            "cleaned_content": cleaned,
            "cleaning_method": method_used,
            "reduction_ratio": round((len(raw_content) - len(cleaned)) / len(raw_content), 2) if len(raw_content) > 0 else 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original_length": len(raw_content)
        }

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
        # Create a simple processing request for classification
        from classification_models import EmailProcessingRequest
        request = EmailProcessingRequest(
            email_id="api_request",
            sender="unknown@example.com",
            content=email_content
        )
        
        result = await llm_interface.classify_email(request)
        return {
            "classification": result.classification,
            "confidence": result.confidence,
            "sentiment": result.sentiment,
            "sentiment_score": result.sentiment_score,
            "formality": result.formality,
            "personalization": result.personalization,
            "priority": result.priority,
            "should_process": result.should_process,
            "processing_time_ms": result.processing_time_ms,
            "reasoning": result.reasoning
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

# Email cleaning endpoint (for testing our tiered approach)
@app.post("/clean")
async def clean_email_content(content: Dict[str, Any]):
    """Clean email content using our tiered approach"""
    if not email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    raw_content = content.get("content", "")
    if not raw_content:
        raise HTTPException(status_code=400, detail="Content required")
    
    try:
        cleaned = await email_processor._clean_with_tiered_approach(raw_content)
        method_used = getattr(email_processor, '_last_cleaning_method', 'unknown')
        
        return {
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned),
            "cleaned_content": cleaned,
            "cleaning_method": method_used,
            "reduction_ratio": round((len(raw_content) - len(cleaned)) / len(raw_content), 2) if len(raw_content) > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email cleaning failed: {str(e)}")

# Thread summary endpoint
@app.post("/summarize/thread")
async def generate_thread_summary(request: Dict[str, Any]):
    """Generate or update thread summary"""
    if not email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    thread_id = request.get("thread_id")
    force_update = request.get("force_update", False)
    
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    
    try:
        summary = await email_processor.generate_thread_summary(thread_id, force_update)
        
        if summary:
            return {
                "success": True,
                "summary": summary.dict(),
                "message": f"Generated summary: {summary.summary_oneliner}"
            }
        else:
            raise HTTPException(status_code=404, detail="Thread not found or summary generation failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

# Test endpoint for incremental development
@app.get("/test1")
async def test_endpoint():
    """Simple test endpoint"""
    return {"status": "auto-reload working!", "timestamp": "2025-06-23", "reload": True}

@app.get("/test2")
async def test_endpoint2():
    """Another test endpoint"""
    return {"test": "working"}

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
    
    # Check if we should use uvicorn directly for reload
    use_reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    
    # Always use uvicorn directly for now (disable background processing during startup issues)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=use_reload,
        log_level="info"
    )