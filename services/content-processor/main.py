"""
Content Processor Service
Queue-based email content processing using Unstructured.io and embeddings
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

from processor import ContentProcessor, ProcessingResult
from config import ProcessorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Content Processor Service", 
    description="Advanced email content processing using Unstructured.io",
    version="1.0.0"
)

# Global state
processor: Optional[ContentProcessor] = None
config: Optional[ProcessorConfig] = None
db_engine = None
SessionLocal = None

@dataclass
class HealthStatus:
    healthy: bool
    processor_ready: bool
    database_connected: bool
    unstructured_available: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    emails_processed: int
    average_processing_time_ms: float
    queue_size: int
    uptime_seconds: float
    errors_last_hour: int = 0

class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.total_processed = 0
        self.total_processing_time = 0.0
        self.recent_errors = []
    
    def record_processing(self, processing_time: float):
        self.total_processed += 1
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
        
        # Get queue size
        queue_size = 0
        if db_engine:
            try:
                with SessionLocal() as session:
                    result = session.execute(text("""
                        SELECT COUNT(*) FROM processing_queue 
                        WHERE queue_type = 'content_processing' AND status = 'pending'
                    """)).fetchone()
                    queue_size = result[0] if result else 0
            except:
                pass
        
        # Calculate averages
        avg_time = (self.total_processing_time / self.total_processed * 1000) if self.total_processed > 0 else 0
        uptime = time.time() - self.start_time
        
        return HealthStatus(
            healthy=processor is not None and db_engine is not None,
            processor_ready=processor is not None and processor.ready,
            database_connected=db_engine is not None,
            unstructured_available=processor is not None and processor.unstructured_available,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            emails_processed=self.total_processed,
            average_processing_time_ms=avg_time,
            queue_size=queue_size,
            uptime_seconds=uptime,
            errors_last_hour=len(self.recent_errors)
        )

health_monitor = HealthMonitor()

async def store_processing_results(session: Session, email_id: str, result: ProcessingResult, processing_time: float):
    """Store processed elements, embeddings, and metadata to database"""
    try:
        import uuid
        
        # Store processing metadata
        metadata_id = str(uuid.uuid4())
        session.execute(text("""
            INSERT INTO processing_metadata (
                id, email_id, processing_stage, processor_version, 
                processing_time_ms, quality_metrics, resource_usage
            ) VALUES (
                :id, :email_id, :stage, :version, :time_ms, :quality, :resources
            )
        """), {
            "id": metadata_id,
            "email_id": email_id,
            "stage": "content_processing",
            "version": "unstructured+sentence-transformers-v1",
            "time_ms": int(processing_time * 1000),
            "quality": json.dumps({"quality_score": result.quality_score}),
            "resources": json.dumps(result.metadata or {})
        })
        
        # Get processed data from processor (we need to modify processor to return this)
        if hasattr(processor, 'last_processing_data'):
            elements = processor.last_processing_data.get('elements', [])
            chunks = processor.last_processing_data.get('chunks', [])
            
            # Store email elements with markdown content
            for element in elements:
                element_id = str(uuid.uuid4())
                session.execute(text("""
                    INSERT INTO email_elements (
                        id, email_id, element_id, element_type, content, 
                        markdown_content, element_metadata, sequence_number,
                        extraction_confidence, processing_method, is_cleaned
                    ) VALUES (
                        :id, :email_id, :element_id, :element_type, :content,
                        :markdown_content, :metadata, :sequence_number,
                        :confidence, :method, :is_cleaned
                    )
                """), {
                    "id": element_id,
                    "email_id": email_id,
                    "element_id": getattr(element, 'element_id', f'elem_{element.sequence_number}'),
                    "element_type": element.element_type,
                    "content": element.content,
                    "markdown_content": getattr(element, 'markdown_content', None),
                    "metadata": json.dumps(element.metadata),
                    "sequence_number": element.sequence_number,
                    "confidence": getattr(element, 'extraction_confidence', 1.0),
                    "method": "unstructured",
                    "is_cleaned": True
                })
            
            # Store enhanced embeddings
            for chunk in chunks:
                if chunk.embedding is not None:
                    embedding_id = str(uuid.uuid4())
                    # Convert numpy array to list for postgres
                    embedding_list = chunk.embedding.tolist() if hasattr(chunk.embedding, 'tolist') else list(chunk.embedding)
                    
                    session.execute(text("""
                        INSERT INTO enhanced_embeddings (
                            id, email_id, chunk_text, chunk_index, element_type,
                            embedding, chunk_metadata, chunking_method, quality_score
                        ) VALUES (
                            :id, :email_id, :chunk_text, :chunk_index, :element_type,
                            :embedding, :metadata, :method, :quality_score
                        )
                    """), {
                        "id": embedding_id,
                        "email_id": email_id,
                        "chunk_text": chunk.text,
                        "chunk_index": getattr(chunk, 'chunk_index', 0),
                        "element_type": chunk.element_type,
                        "embedding": embedding_list,
                        "metadata": json.dumps(chunk.metadata or {}),
                        "method": "unstructured_semantic",
                        "quality_score": chunk.quality_score
                    })
        
        session.commit()
        logger.info(f"💾 Stored processing results for email {email_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to store processing results for email {email_id}: {e}")
        session.rollback()
        raise

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

async def setup_processor():
    """Initialize content processor"""
    global processor, config
    
    try:
        config = ProcessorConfig()
        processor = ContentProcessor(config)
        
        # Initialize processor
        await processor.initialize()
        
        logger.info("✅ Content processor initialized")
        return True
        
    except Exception as e:
        logger.error(f"❌ Content processor initialization failed: {e}")
        health_monitor.record_error(e)
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Content Processor Service...")
    
    # Setup database connection
    db_success = await setup_database()
    if not db_success:
        logger.error("Failed to setup database connection")
        
    # Setup processor
    processor_success = await setup_processor()
    if not processor_success:
        logger.error("Failed to setup content processor")
    
    if db_success and processor_success:
        logger.info("✅ Content Processor Service ready")
        # Start background processing
        asyncio.create_task(process_content_queue())
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
            "processor_ready": status.processor_ready,
            "database_connected": status.database_connected,
            "unstructured_available": status.unstructured_available,
            "memory_usage_mb": round(status.memory_usage_mb, 1),
            "cpu_usage_percent": round(status.cpu_usage_percent, 1),
            "emails_processed": status.emails_processed,
            "average_processing_time_ms": round(status.average_processing_time_ms, 1),
            "queue_size": status.queue_size,
            "uptime_seconds": round(status.uptime_seconds, 1),
            "errors_last_hour": status.errors_last_hour
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Detailed metrics endpoint"""
    status = health_monitor.get_status()
    
    # Get detailed queue stats
    queue_stats = {}
    processing_stats = {}
    if db_engine:
        try:
            with SessionLocal() as session:
                # Queue status breakdown
                queue_result = session.execute(text("""
                    SELECT 
                        status,
                        content_type,
                        COUNT(*) as count,
                        AVG(actual_processing_time) as avg_time,
                        AVG(quality_score) as avg_quality
                    FROM processing_queue 
                    WHERE queue_type = 'content_processing'
                    GROUP BY status, content_type
                """)).fetchall()
                
                queue_stats = {}
                for row in queue_result:
                    key = f"{row.status}_{row.content_type}"
                    queue_stats[key] = {
                        'count': row.count,
                        'avg_time_ms': row.avg_time or 0,
                        'avg_quality': row.avg_quality or 0
                    }
                
                # Processing statistics
                stats_result = session.execute(text("""
                    SELECT 
                        processing_stage,
                        AVG(processing_time_ms) as avg_time,
                        COUNT(*) as count
                    FROM processing_metadata 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY processing_stage
                """)).fetchall()
                
                processing_stats = {row.processing_stage: {
                    'avg_time_ms': row.avg_time or 0,
                    'count': row.count
                } for row in stats_result}
                
        except Exception as e:
            logger.error(f"Failed to get detailed stats: {e}")
    
    return {
        "service": {
            "uptime_seconds": status.uptime_seconds,
            "emails_processed": status.emails_processed,
            "average_processing_time_ms": status.average_processing_time_ms,
            "errors_last_hour": status.errors_last_hour
        },
        "resources": {
            "memory_usage_mb": status.memory_usage_mb,
            "cpu_usage_percent": status.cpu_usage_percent,
            "processor_ready": status.processor_ready,
            "unstructured_available": status.unstructured_available
        },
        "queue": {
            "pending_count": status.queue_size,
            "breakdown": queue_stats
        },
        "processing": processing_stats,
        "processor": processor.get_stats() if processor else {}
    }

@app.post("/process")
async def process_email(email_data: Dict[str, Any]):
    """Process a single email (for manual testing)"""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        start_time = time.time()
        
        result = await processor.process_email(
            email_id=email_data.get('email_id', 'test'),
            subject=email_data.get('subject', ''),
            content=email_data.get('content', ''),
            html_content=email_data.get('html_content')
        )
        
        processing_time = time.time() - start_time
        health_monitor.record_processing(processing_time)
        
        return {
            "result": result.to_dict(),
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
    except Exception as e:
        health_monitor.record_error(e)
        raise HTTPException(status_code=500, detail=str(e))

async def process_content_queue():
    """Background task to process content processing queue"""
    logger.info("🔄 Starting background content processing")
    
    while True:
        try:
            if not processor or not db_engine:
                await asyncio.sleep(5)
                continue
            
            # Get pending items from queue
            with SessionLocal() as session:
                pending_items = session.execute(text("""
                    SELECT pq.id, pq.thread_id, pq.priority, pq.content_type,
                           e.id as email_id, e.subject, e.body_text, e.body_html, e.from_email
                    FROM processing_queue pq
                    JOIN threads t ON pq.thread_id = t.id
                    JOIN emails e ON e.thread_id = t.id
                    WHERE pq.queue_type = 'content_processing' 
                    AND pq.status = 'pending'
                    AND pq.attempts < pq.max_attempts
                    ORDER BY pq.priority DESC, pq.created_at ASC
                    LIMIT 5  -- Process fewer at a time due to complexity
                """)).fetchall()
                
                if not pending_items:
                    await asyncio.sleep(10)  # Longer sleep for content processing
                    continue
                
                logger.info(f"📧 Processing {len(pending_items)} emails for content extraction")
                
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
                        
                        # Process the email content
                        start_time = time.time()
                        result = await processor.process_email(
                            email_id=item.email_id,
                            subject=item.subject or '',
                            content=item.body_text or '',
                            html_content=item.body_html if item.content_type == 'html' else None
                        )
                        processing_time = time.time() - start_time
                        
                        # Store processed data to database
                        if result.success:
                            await store_processing_results(session, item.email_id, result, processing_time)
                        
                        # Update queue with results
                        session.execute(text("""
                            UPDATE processing_queue 
                            SET status = 'completed',
                                processing_completed_at = NOW(),
                                actual_processing_time = :processing_time,
                                quality_score = :quality_score,
                                elements_extracted = :elements_extracted,
                                chunks_created = :chunks_created,
                                embeddings_created = :embeddings_created
                            WHERE id = :queue_id
                        """), {
                            "queue_id": item.id,
                            "processing_time": int(processing_time * 1000),
                            "quality_score": result.quality_score,
                            "elements_extracted": result.elements_extracted,
                            "chunks_created": result.chunks_created,
                            "embeddings_created": result.embeddings_created
                        })
                        
                        session.commit()
                        health_monitor.record_processing(processing_time)
                        
                        logger.info(f"✅ Processed email {item.email_id}: {result.elements_extracted} elements, {result.chunks_created} chunks, {result.embeddings_created} embeddings")
                        
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
            await asyncio.sleep(10)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8082,
        reload=False,
        log_level="info"
    )