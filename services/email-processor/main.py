from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging
import os
from datetime import datetime

from database import get_db
from processor import EmailProcessor
from models import Message, ImapMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Email Processor API",
    description="Email cleaning and processing service for Email RAG system",
    version="1.0.0"
)

# Initialize processor
processor = EmailProcessor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "email-processor",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        from sqlalchemy import text
        result = db.execute(text("SELECT 1")).fetchone()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/status")
async def processing_status():
    """Get processing status and statistics"""
    try:
        stats = processor.get_processing_stats()
        return {
            "status": "ok",
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/process")
async def trigger_processing():
    """Manually trigger email processing"""
    try:
        processed_count = processor.process_unprocessed_emails(limit=50)
        
        return {
            "status": "completed",
            "processed_count": processed_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Manual processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/queue")
async def queue_status(db: Session = Depends(get_db)):
    """Get queue status information"""
    try:
        # Count unprocessed emails
        total_imap = db.query(ImapMessage).count()
        processed_messages = db.query(Message).count()
        pending = total_imap - processed_messages
        
        # Get recent processing activity
        recent_processed = db.query(Message).filter(
            Message.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
        
        return {
            "queue_status": {
                "total_emails": total_imap,
                "processed": processed_messages,
                "pending": pending,
                "processed_today": recent_processed
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")

@app.get("/messages/{message_id}")
async def get_message(message_id: int, db: Session = Depends(get_db)):
    """Get details of a specific processed message"""
    try:
        message = db.query(Message).filter(Message.id == message_id).first()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        return {
            "message": {
                "id": message.id,
                "imap_message_id": message.imap_message_id,
                "message_id": message.message_id,
                "subject": message.subject,
                "from_email": message.from_email,
                "participants": message.participants,
                "processing_status": message.processing_status,
                "parsed_at": message.parsed_at.isoformat() if message.parsed_at else None,
                "cleaned_at": message.cleaned_at.isoformat() if message.cleaned_at else None,
                "created_at": message.created_at.isoformat(),
                "body_preview": message.body_text[:200] + "..." if message.body_text and len(message.body_text) > 200 else message.body_text
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get message {message_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get message: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting Email Processor API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )