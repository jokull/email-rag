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
from llm_classifier import get_llm_classifier, initialize_llm_classifier

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

@app.get("/classification/status")
async def classification_status():
    """Get classification status and statistics"""
    try:
        stats = processor.get_classification_stats()
        classifier = get_llm_classifier()
        classifier_stats = classifier.get_stats() if classifier else {}
        
        return {
            "status": "ok",
            "classification_statistics": stats,
            "classifier_performance": classifier_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get classification status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get classification status: {str(e)}")

@app.post("/classification/process")
async def trigger_classification():
    """Manually trigger email classification"""
    try:
        # Initialize LLM classifier if not ready
        classifier = get_llm_classifier()
        if not classifier.ready:
            logger.info("Initializing LLM classifier...")
            if not initialize_llm_classifier():
                raise HTTPException(status_code=503, detail="Failed to initialize LLM classifier")
        
        classified_count = processor.process_unclassified_emails(limit=50)
        
        return {
            "status": "completed",
            "classified_count": classified_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/classification/categories")
async def get_category_breakdown(db: Session = Depends(get_db)):
    """Get breakdown of emails by category"""
    try:
        from sqlalchemy import func
        
        # Get category counts
        category_stats = dict(
            db.query(Message.category, func.count(Message.id))
            .filter(Message.category.isnot(None))
            .group_by(Message.category)
            .all()
        )
        
        # Get recent classifications (last 24 hours)
        recent_classifications = db.query(Message).filter(
            Message.classified_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
            Message.category.isnot(None)
        ).count()
        
        return {
            "category_breakdown": category_stats,
            "recent_classifications": recent_classifications,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get category breakdown: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get category breakdown: {str(e)}")

@app.get("/classification/personal")
async def get_personal_emails(limit: int = 20, db: Session = Depends(get_db)):
    """Get recent personal emails (for testing embedding pipeline)"""
    try:
        personal_messages = db.query(Message).filter(
            Message.category == 'personal',
            Message.classified_at.isnot(None)
        ).order_by(Message.date_sent.desc()).limit(limit).all()
        
        results = []
        for msg in personal_messages:
            results.append({
                "id": msg.id,
                "subject": msg.subject,
                "from_email": msg.from_email,
                "date_sent": msg.date_sent.isoformat() if msg.date_sent else None,
                "classified_at": msg.classified_at.isoformat() if msg.classified_at else None,
                "category": msg.category,
                "confidence": msg.confidence,
                "body_preview": msg.body_text[:150] + "..." if msg.body_text and len(msg.body_text) > 150 else msg.body_text
            })
        
        return {
            "personal_emails": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get personal emails: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get personal emails: {str(e)}")

@app.get("/threading/status")
async def threading_status():
    """Get email threading status and statistics"""
    try:
        stats = processor.get_threading_stats()
        
        return {
            "status": "ok",
            "threading_statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get threading status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get threading status: {str(e)}")

@app.post("/threading/process")
async def trigger_threading():
    """Manually trigger email threading for personal emails"""
    try:
        threaded_count = processor.process_unthreaded_personal_emails(limit=50)
        
        return {
            "status": "completed",
            "threaded_count": threaded_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Manual threading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Threading failed: {str(e)}")

@app.post("/threading/regenerate-summaries")
async def regenerate_summaries():
    """Regenerate AI summaries for existing conversations"""
    try:
        regenerated_count = processor.threading_service.regenerate_conversation_summaries(limit=50)
        
        return {
            "status": "completed",
            "regenerated_count": regenerated_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Summary regeneration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summary regeneration failed: {str(e)}")

@app.get("/conversations")
async def get_conversations(limit: int = 20, db: Session = Depends(get_db)):
    """Get recent conversations with their summaries"""
    try:
        from models import Conversation
        
        conversations = db.query(Conversation).order_by(
            Conversation.last_message_date.desc()
        ).limit(limit).all()
        
        results = []
        for conv in conversations:
            results.append({
                "id": conv.id,
                "thread_id": conv.thread_id,
                "subject_normalized": conv.subject_normalized,
                "summary": conv.summary,
                "key_topics": conv.key_topics,
                "message_count": conv.message_count,
                "participants": conv.participants,
                "first_message_date": conv.first_message_date.isoformat() if conv.first_message_date else None,
                "last_message_date": conv.last_message_date.isoformat() if conv.last_message_date else None,
                "summary_generated_at": conv.summary_generated_at.isoformat() if conv.summary_generated_at else None
            })
        
        return {
            "conversations": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

@app.get("/conversations/{thread_id}/messages")
async def get_conversation_messages(thread_id: str, db: Session = Depends(get_db)):
    """Get all messages in a conversation thread"""
    try:
        messages = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.date_sent).all()
        
        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        results = []
        for msg in messages:
            results.append({
                "id": msg.id,
                "message_id": msg.message_id,
                "subject": msg.subject,
                "from_email": msg.from_email,
                "to_emails": msg.to_emails,
                "date_sent": msg.date_sent.isoformat() if msg.date_sent else None,
                "body_text": msg.body_text,
                "participants": msg.participants
            })
        
        return {
            "thread_id": thread_id,
            "messages": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation messages: {str(e)}")

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