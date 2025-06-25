"""
Queue-Based AI Processor for Email RAG System
Uses pgqueuer for controlled pipeline processing instead of triggers
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime, timedelta

# Import our queue-based modules
from config import get_config, LLMConfig, ProcessorConfig
from modern_llm import ModernQwenInterface
from queue_manager import EmailQueueManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global queue manager
queue_manager: Optional[EmailQueueManager] = None

async def setup_queue_service(app: FastAPI):
    """Initialize queue-based processing service"""
    global queue_manager
    
    try:
        logger.info("🚀 SETUP: Starting Queue-Based Email Processor")
        
        # Load configuration
        logger.info("🔧 SETUP: Loading configuration...")
        llm_config, processor_config = get_config()
        app.state.llm_config = llm_config
        app.state.processor_config = processor_config
        logger.info(f"✅ SETUP: Configuration loaded - batch_size: {processor_config.batch_size}")
        
        # Initialize LLM interface
        llm_interface = None
        disable_llm = os.getenv("DISABLE_LLM_INIT", "false").lower() == "true"
        
        if not disable_llm:
            try:
                logger.info("🧠 SETUP: Initializing Qwen-0.5B interface...")
                llm_interface = ModernQwenInterface(llm_config, processor_config)
                
                model_ready = await llm_interface.initialize()
                if model_ready:
                    logger.info("✅ SETUP: Qwen-0.5B interface ready")
                else:
                    logger.warning("⚠️ SETUP: LLM failed to initialize, using fallback")
                    llm_interface = None
                    
            except Exception as e:
                logger.warning(f"⚠️ SETUP: LLM initialization failed: {e}")
                llm_interface = None
        else:
            logger.info("🔧 SETUP: LLM disabled for debugging")
        
        # Initialize queue manager
        logger.info("📥 SETUP: Initializing queue manager...")
        queue_manager = EmailQueueManager(processor_config, llm_interface)
        
        queue_ready = await queue_manager.initialize()
        if not queue_ready:
            raise RuntimeError("Failed to initialize queue manager")
        
        app.state.queue_manager = queue_manager
        app.state.llm_interface = llm_interface
        
        logger.info("✅ SETUP: Queue-based processor initialized successfully")
        
        # Start queue workers in background
        logger.info("🏃 SETUP: Starting queue workers...")
        asyncio.create_task(queue_manager.start_workers())
        
    except Exception as e:
        logger.error(f"❌ SETUP: Failed to initialize queue service: {e}")
        raise

async def shutdown_queue_service(app: FastAPI):
    """Shutdown queue service"""
    global queue_manager
    
    logger.info("🛑 Shutting down queue processor...")
    
    try:
        if queue_manager:
            await queue_manager.shutdown()
        
        logger.info("✅ Queue processor shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    logger.info("🚀 LIFESPAN: Starting queue-based processor")
    
    try:
        # Startup
        await setup_queue_service(app)
        yield
        
        # Shutdown
        await shutdown_queue_service(app)
        
    except Exception as e:
        logger.error(f"❌ LIFESPAN: Error: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Email RAG Queue Processor",
    description="Queue-based email processing pipeline using pgqueuer",
    version="2.0.0",
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

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    if not hasattr(request.app.state, 'queue_manager') or not request.app.state.queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not ready")
    
    stats = await request.app.state.queue_manager.get_queue_stats()
    
    return {
        "status": "healthy",
        "service": "queue-based-email-processor",
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "llm_available": request.app.state.llm_interface is not None
    }

@app.get("/stats")
async def get_processing_stats(request: Request):
    """Get detailed processing statistics"""
    if not hasattr(request.app.state, 'queue_manager'):
        raise HTTPException(status_code=503, detail="Queue manager not ready")
    
    return await request.app.state.queue_manager.get_queue_stats()

@app.post("/process/enqueue")
async def enqueue_emails_for_processing(request: Request, background_tasks: BackgroundTasks):
    """Enqueue pending emails for processing"""
    if not hasattr(request.app.state, 'queue_manager'):
        raise HTTPException(status_code=503, detail="Queue manager not ready")
    
    queue_manager = request.app.state.queue_manager
    
    try:
        # Get unprocessed emails from database
        unprocessed_emails = await queue_manager.connection.fetch("""
            SELECT email_id, thread_id FROM get_unprocessed_emails(50)
        """)
        
        # Enqueue emails for classification
        enqueued_count = 0
        for row in unprocessed_emails:
            await queue_manager.enqueue_email_for_classification(
                row['email_id'], 
                row['thread_id']
            )
            enqueued_count += 1
        
        return {
            "message": f"Enqueued {enqueued_count} emails for processing",
            "enqueued_count": enqueued_count
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to enqueue emails: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/trigger")
async def trigger_processing_legacy(request: Request, background_tasks: BackgroundTasks):
    """Legacy endpoint - redirects to queue-based processing"""
    return await enqueue_emails_for_processing(request, background_tasks)

@app.get("/queue/status")
async def get_queue_status(request: Request):
    """Get current queue status and depth"""
    if not hasattr(request.app.state, 'queue_manager'):
        raise HTTPException(status_code=503, detail="Queue manager not ready")
    
    queue_manager = request.app.state.queue_manager
    
    try:
        # Get queue stats from pgqueuer
        queue_stats = await queue_manager.connection.fetch("""
            SELECT 
                entrypoint,
                status,
                COUNT(*) as count,
                MIN(created_at) as oldest_job,
                MAX(created_at) as newest_job
            FROM pgqueuer.job 
            GROUP BY entrypoint, status
            ORDER BY entrypoint, status
        """)
        
        # Get processing status from our table
        processing_stats = await queue_manager.connection.fetch("""
            SELECT 
                stage,
                COUNT(*) as count
            FROM processing_status
            GROUP BY stage
            ORDER BY stage
        """)
        
        return {
            "queue_stats": [dict(row) for row in queue_stats],
            "processing_stats": [dict(row) for row in processing_stats],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/clear-queue")
async def clear_queue(request: Request):
    """Clear all pending jobs from queue (admin only)"""
    if not hasattr(request.app.state, 'queue_manager'):
        raise HTTPException(status_code=503, detail="Queue manager not ready")
    
    queue_manager = request.app.state.queue_manager
    
    try:
        # Clear pgqueuer jobs
        result = await queue_manager.connection.execute("""
            DELETE FROM pgqueuer.job WHERE status = 'queued'
        """)
        
        logger.info(f"🧹 Cleared queue - removed jobs")
        
        return {
            "message": "Queue cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/requeue-failed")
async def requeue_failed_jobs(request: Request):
    """Requeue failed processing jobs"""
    if not hasattr(request.app.state, 'queue_manager'):
        raise HTTPException(status_code=503, detail="Queue manager not ready")
    
    queue_manager = request.app.state.queue_manager
    
    try:
        # Get failed processing statuses
        failed_emails = await queue_manager.connection.fetch("""
            SELECT email_id, thread_id, stage, error_count
            FROM processing_status 
            WHERE stage = 'failed' AND error_count < 3
        """)
        
        requeued_count = 0
        for row in failed_emails:
            await queue_manager.enqueue_email_for_classification(
                row['email_id'], 
                row['thread_id'],
                priority=50  # Lower priority for retries
            )
            requeued_count += 1
        
        return {
            "message": f"Requeued {requeued_count} failed jobs",
            "requeued_count": requeued_count
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to requeue failed jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Handle graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
    asyncio.create_task(shutdown_queue_service(app))
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main_queue:app",
        host="0.0.0.0",
        port=8080,
        reload=False,  # Disable reload for queue workers
        log_level="info"
    )