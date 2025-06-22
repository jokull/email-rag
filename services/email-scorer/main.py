"""
Email Scorer Service
Lightweight Qwen-0.5B service for rapid email triage and multi-dimensional scoring
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import uvicorn
import psutil

from scorer import EmailScorer, ScoringResult
from config import ScorerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email Scorer Service",
    description="Rapid email triage and multi-dimensional scoring using Qwen-0.5B",
    version="1.0.0"
)

# Global state
scorer: Optional[EmailScorer] = None
config: Optional[ScorerConfig] = None
db_engine = None
SessionLocal = None

@dataclass
class HealthStatus:
    healthy: bool
    scorer_loaded: bool
    database_connected: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    scores_processed: int
    average_processing_time_ms: float
    uptime_seconds: float
    errors_last_hour: int = 0

class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.total_scores = 0
        self.total_processing_time = 0.0
        self.recent_errors = []
    
    def record_scoring(self, processing_time: float):
        self.total_scores += 1
        self.total_processing_time += processing_time
    
    def record_error(self, error: Exception):
        self.recent_errors.append({
            'timestamp': time.time(),
            'error': str(error)
        })
        # Keep only last hour of errors
        cutoff = time.time() - 3600
        self.recent_errors = [e for e in self.recent_errors if e['timestamp'] > cutoff]
    
    def get_status(self) -> HealthStatus:
        # Memory and CPU usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent()
        
        # Calculate averages
        avg_time = (self.total_processing_time / self.total_scores * 1000) if self.total_scores > 0 else 0
        uptime = time.time() - self.start_time
        
        return HealthStatus(
            healthy=scorer is not None and db_engine is not None,
            scorer_loaded=scorer is not None and scorer.model_loaded,
            database_connected=db_engine is not None,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            scores_processed=self.total_scores,
            average_processing_time_ms=avg_time,
            uptime_seconds=uptime,
            errors_last_hour=len(self.recent_errors)
        )

health_monitor = HealthMonitor()

async def setup_database():
    """Initialize database connection"""
    global db_engine, SessionLocal
    
    database_url = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@postgres:5432/email_rag")
    
    try:
        db_engine = create_engine(database_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=db_engine)
        
        # Test connection
        with db_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        logger.info("✅ Database connection established")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        health_monitor.record_error(e)
        return False

async def setup_scorer():
    """Initialize email scorer"""
    global scorer, config
    
    try:
        config = ScorerConfig()
        scorer = EmailScorer(config)
        
        # Warm up the model
        await scorer.initialize()
        
        logger.info("✅ Email scorer initialized and warmed up")
        return True
        
    except Exception as e:
        logger.error(f"❌ Email scorer initialization failed: {e}")
        health_monitor.record_error(e)
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Email Scorer Service...")
    
    # Setup database connection
    db_success = await setup_database()
    if not db_success:
        logger.error("Failed to setup database connection")
        
    # Setup scorer
    scorer_success = await setup_scorer()
    if not scorer_success:
        logger.error("Failed to setup email scorer")
    
    if db_success and scorer_success:
        logger.info("✅ Email Scorer Service ready")
        # Start background processing
        asyncio.create_task(process_classification_queue())
    else:
        logger.error("❌ Service startup incomplete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = health_monitor.get_status()
    
    return {
        "status": "healthy" if status.healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "details": {
            "scorer_loaded": status.scorer_loaded,
            "database_connected": status.database_connected,
            "memory_usage_mb": round(status.memory_usage_mb, 1),
            "cpu_usage_percent": round(status.cpu_usage_percent, 1),
            "scores_processed": status.scores_processed,
            "average_processing_time_ms": round(status.average_processing_time_ms, 1),
            "uptime_seconds": round(status.uptime_seconds, 1),
            "errors_last_hour": status.errors_last_hour
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Detailed metrics endpoint"""
    status = health_monitor.get_status()
    
    # Get queue stats from database
    queue_stats = {}
    if db_engine:
        try:
            with SessionLocal() as session:
                result = session.execute(text("""
                    SELECT 
                        status,
                        COUNT(*) as count,
                        AVG(actual_processing_time) as avg_time
                    FROM processing_queue 
                    WHERE queue_type = 'classification'
                    GROUP BY status
                """)).fetchall()
                
                queue_stats = {row.status: {
                    'count': row.count,
                    'avg_time_ms': row.avg_time or 0
                } for row in result}
                
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
    
    return {
        "service": {
            "uptime_seconds": status.uptime_seconds,
            "scores_processed": status.scores_processed,
            "average_processing_time_ms": status.average_processing_time_ms,
            "errors_last_hour": status.errors_last_hour
        },
        "resources": {
            "memory_usage_mb": status.memory_usage_mb,
            "cpu_usage_percent": status.cpu_usage_percent,
            "model_loaded": status.scorer_loaded
        },
        "queue": queue_stats,
        "model": {
            "name": config.model_name if config else "unknown",
            "version": config.model_version if config else "unknown"
        }
    }

@app.post("/score")
async def score_email(email_data: Dict[str, Any]):
    """Score a single email (for manual testing)"""
    if not scorer:
        raise HTTPException(status_code=503, detail="Scorer not initialized")
    
    try:
        start_time = time.time()
        
        result = await scorer.score_email(
            subject=email_data.get('subject', ''),
            content=email_data.get('content', ''),
            sender=email_data.get('sender', ''),
            html_content=email_data.get('html_content')
        )
        
        processing_time = time.time() - start_time
        health_monitor.record_scoring(processing_time)
        
        return {
            "scores": result.to_dict(),
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
    except Exception as e:
        health_monitor.record_error(e)
        raise HTTPException(status_code=500, detail=str(e))

async def process_classification_queue():
    """Background task to process classification queue"""
    logger.info("🔄 Starting background classification processing")
    
    while True:
        try:
            if not scorer or not db_engine:
                await asyncio.sleep(5)
                continue
            
            # Get pending items from queue
            with SessionLocal() as session:
                pending_items = session.execute(text("""
                    SELECT pq.id, pq.thread_id, pq.priority, e.id as email_id,
                           e.subject, e.body_text, e.body_html, e.from_email
                    FROM processing_queue pq
                    JOIN threads t ON pq.thread_id = t.id
                    JOIN emails e ON e.thread_id = t.id
                    WHERE pq.queue_type = 'classification' 
                    AND pq.status = 'pending'
                    AND pq.attempts < pq.max_attempts
                    ORDER BY pq.priority DESC, pq.created_at ASC
                    LIMIT 10
                """)).fetchall()
                
                if not pending_items:
                    await asyncio.sleep(2)
                    continue
                
                logger.info(f"📧 Processing {len(pending_items)} emails for classification")
                
                for item in pending_items:
                    try:
                        # Mark as processing
                        session.execute(text("""
                            UPDATE processing_queue 
                            SET status = 'processing',
                                processing_started_at = NOW(),
                                attempts = attempts + 1
                            WHERE id = :queue_id
                        """), {"queue_id": item.id})
                        session.commit()
                        
                        # Score the email
                        start_time = time.time()
                        result = await scorer.score_email(
                            subject=item.subject or '',
                            content=item.body_text or '',
                            sender=item.from_email or '',
                            html_content=item.body_html
                        )
                        processing_time = time.time() - start_time
                        
                        # Save classification results
                        session.execute(text("""
                            INSERT INTO classifications (
                                id, thread_id, classification, confidence, model_used,
                                human_score, personal_score, relevance_score,
                                sentiment_score, importance_score, commercial_score,
                                processing_priority, scorer_version, should_process
                            ) VALUES (
                                uuid_generate_v4()::text, :thread_id, :classification, :confidence, :model_used,
                                :human_score, :personal_score, :relevance_score,
                                :sentiment_score, :importance_score, :commercial_score,
                                :processing_priority, :scorer_version, :should_process
                            ) ON CONFLICT (thread_id) DO UPDATE SET
                                classification = EXCLUDED.classification,
                                confidence = EXCLUDED.confidence,
                                human_score = EXCLUDED.human_score,
                                personal_score = EXCLUDED.personal_score,
                                relevance_score = EXCLUDED.relevance_score,
                                sentiment_score = EXCLUDED.sentiment_score,
                                importance_score = EXCLUDED.importance_score,
                                commercial_score = EXCLUDED.commercial_score,
                                processing_priority = EXCLUDED.processing_priority,
                                scorer_version = EXCLUDED.scorer_version,
                                should_process = EXCLUDED.should_process,
                                created_at = NOW()
                        """), {
                            "thread_id": item.thread_id,
                            "classification": result.classification,
                            "confidence": result.confidence,
                            "model_used": config.model_name,
                            "human_score": result.human_score,
                            "personal_score": result.personal_score,
                            "relevance_score": result.relevance_score,
                            "sentiment_score": result.sentiment_score,
                            "importance_score": result.importance_score,
                            "commercial_score": result.commercial_score,
                            "processing_priority": result.processing_priority,
                            "scorer_version": config.model_version,
                            "should_process": result.should_process
                        })
                        
                        # Mark queue item as completed
                        session.execute(text("""
                            UPDATE processing_queue 
                            SET status = 'completed',
                                processing_completed_at = NOW(),
                                actual_processing_time = :processing_time
                            WHERE id = :queue_id
                        """), {
                            "queue_id": item.id,
                            "processing_time": int(processing_time * 1000)
                        })
                        
                        session.commit()
                        health_monitor.record_scoring(processing_time)
                        
                        logger.info(f"✅ Scored email {item.email_id}: {result.classification} (confidence: {result.confidence:.2f})")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process email {item.email_id}: {e}")
                        health_monitor.record_error(e)
                        
                        # Mark as failed
                        session.execute(text("""
                            UPDATE processing_queue 
                            SET status = CASE 
                                    WHEN attempts >= max_attempts THEN 'failed'
                                    ELSE 'pending'
                                END,
                                error_message = :error_message,
                                processing_completed_at = NOW()
                            WHERE id = :queue_id
                        """), {
                            "queue_id": item.id,
                            "error_message": str(e)[:500]
                        })
                        session.commit()
        
        except Exception as e:
            logger.error(f"❌ Queue processing error: {e}")
            health_monitor.record_error(e)
            await asyncio.sleep(5)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
        log_level="info"
    )