"""
Queue-based Email Processing Pipeline using pgqueuer
Replaces trigger-based processing with controlled queue workers
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

import asyncpg
from pgqueuer import PgQueuer
from pgqueuer.db import AsyncpgDriver
from pgqueuer.models import Job

from config import ProcessorConfig
from modern_llm import ModernQwenInterface
from classification_models import EmailProcessingRequest, ContactHistory
from email_processor import EmailProcessor, ProcessingResult

logger = logging.getLogger(__name__)

@dataclass
class QueueJobData:
    """Data structure for queue jobs"""
    email_id: str
    thread_id: str
    stage: str  # 'classify', 'process_content', 'generate_embeddings'
    payload: Dict[str, Any]
    priority: int = 0
    retry_count: int = 0

class EmailQueueManager:
    """
    Queue-based email processing pipeline using pgqueuer
    
    Pipeline stages:
    1. email_classify - Qwen-0.5B rapid scoring (sentiment, importance, commercial)
    2. content_process - Unstructured.io processing for qualifying emails  
    3. generate_embeddings - Vector embeddings for processed content
    """
    
    def __init__(self, config: ProcessorConfig, llm_interface: ModernQwenInterface = None):
        self.config = config
        self.llm_interface = llm_interface
        self.connection: Optional[asyncpg.Connection] = None
        self.pgq: Optional[PgQueuer] = None
        self.is_running = False
        
        # Processing stats
        self.stats = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "total_processing_time": 0.0,
            "last_processed": None,
            "stage_stats": {
                "email_classify": {"processed": 0, "failed": 0, "avg_time_ms": 0},
                "content_process": {"processed": 0, "failed": 0, "avg_time_ms": 0}, 
                "generate_embeddings": {"processed": 0, "failed": 0, "avg_time_ms": 0}
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize pgqueuer connection and workers"""
        try:
            logger.info("🔄 Initializing Email Queue Manager...")
            
            # Connect to PostgreSQL
            self.connection = await asyncpg.connect(self.config.database_url)
            logger.info("✅ Connected to PostgreSQL")
            
            # Initialize pgqueuer
            driver = AsyncpgDriver(self.connection)
            self.pgq = PgQueuer(driver)
            
            # Register job handlers
            self._register_job_handlers()
            
            logger.info("✅ Queue manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize queue manager: {e}")
            return False
    
    def _register_job_handlers(self):
        """Register job handlers for each pipeline stage"""
        
        @self.pgq.entrypoint("email_classify")
        async def classify_email_job(job: Job) -> None:
            """Handle email classification jobs"""
            await self._handle_classification_job(job)
        
        @self.pgq.entrypoint("content_process")
        async def process_content_job(job: Job) -> None:
            """Handle content processing jobs"""
            await self._handle_content_processing_job(job)
        
        @self.pgq.entrypoint("generate_embeddings")
        async def embeddings_job(Job) -> None:
            """Handle embedding generation jobs"""
            await self._handle_embeddings_job(job)
    
    async def _handle_classification_job(self, job: Job) -> None:
        """Process email classification job"""
        start_time = time.time()
        
        try:
            # Parse job data
            job_data = json.loads(job.payload.decode('utf-8'))
            email_id = job_data['email_id']
            thread_id = job_data['thread_id']
            
            logger.info(f"🧠 Processing classification job for email {email_id}")
            
            # Get email data from database
            email_data = await self._get_email_data(email_id)
            if not email_data:
                raise ValueError(f"Email {email_id} not found")
            
            # Create processing request
            request = EmailProcessingRequest(
                email_id=email_id,
                sender=email_data['from_email'],
                subject=email_data.get('subject', ''),
                content=email_data.get('body_text', ''),
                contact_history=await self._get_contact_history(email_data['from_email'])
            )
            
            # Classify with LLM
            if self.llm_interface and self.llm_interface.ready:
                classification = await self.llm_interface.classify_email(request)
                
                # Save classification to database
                await self._save_classification(thread_id, email_id, classification)
                
                # Queue next stage if email qualifies
                if classification.should_process:
                    await self.enqueue_job("content_process", {
                        "email_id": email_id,
                        "thread_id": thread_id,
                        "classification": classification.dict()
                    }, priority=classification.processing_priority)
                
                # Update stats
                processing_time = (time.time() - start_time) * 1000
                self._update_stage_stats("email_classify", processing_time, success=True)
                
                logger.info(f"✅ Classified email {email_id} as {classification.classification} (confidence: {classification.confidence:.2f})")
            else:
                raise RuntimeError("LLM interface not available")
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stage_stats("email_classify", processing_time, success=False)
            logger.error(f"❌ Classification job failed for email {job_data.get('email_id', 'unknown')}: {e}")
            raise
    
    async def _handle_content_processing_job(self, job: Job) -> None:
        """Process content using Unstructured.io and prepare for embeddings"""
        start_time = time.time()
        
        try:
            job_data = json.loads(job.payload.decode('utf-8'))
            email_id = job_data['email_id']
            thread_id = job_data['thread_id']
            
            logger.info(f"📄 Processing content for email {email_id}")
            
            # Get email content
            email_data = await self._get_email_data(email_id)
            if not email_data:
                raise ValueError(f"Email {email_id} not found")
            
            # Process with Unstructured.io (simplified for now)
            processed_content = await self._process_with_unstructured(email_data)
            
            # Save processed content
            await self._save_processed_content(email_id, processed_content)
            
            # Queue for embeddings
            await self.enqueue_job("generate_embeddings", {
                "email_id": email_id,
                "thread_id": thread_id,
                "processed_content": processed_content
            }, priority=job_data.get('priority', 100))
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stage_stats("content_process", processing_time, success=True)
            
            logger.info(f"✅ Processed content for email {email_id}")
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stage_stats("content_process", processing_time, success=False)
            logger.error(f"❌ Content processing failed for email {job_data.get('email_id', 'unknown')}: {e}")
            raise
    
    async def _handle_embeddings_job(self, job: Job) -> None:
        """Generate embeddings for processed content"""
        start_time = time.time()
        
        try:
            job_data = json.loads(job.payload.decode('utf-8'))
            email_id = job_data['email_id']
            thread_id = job_data['thread_id']
            
            logger.info(f"🔢 Generating embeddings for email {email_id}")
            
            # Generate embeddings (simplified for now)
            embeddings_created = await self._generate_embeddings(email_id, job_data['processed_content'])
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stage_stats("generate_embeddings", processing_time, success=True)
            
            logger.info(f"✅ Generated {embeddings_created} embeddings for email {email_id}")
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stage_stats("generate_embeddings", processing_time, success=False)
            logger.error(f"❌ Embeddings generation failed for email {job_data.get('email_id', 'unknown')}: {e}")
            raise
    
    async def enqueue_job(self, job_type: str, payload: Dict[str, Any], priority: int = 0) -> None:
        """Enqueue a job for processing"""
        try:
            if not self.pgq:
                raise RuntimeError("Queue manager not initialized")
            
            # Encode payload
            payload_bytes = json.dumps(payload).encode('utf-8')
            
            # Enqueue job using pgqueuer
            from pgqueuer.queries import Queries
            driver = AsyncpgDriver(self.connection)
            queries = Queries(driver)
            
            await queries.enqueue(
                [job_type],
                [payload_bytes],
                [priority]
            )
            
            logger.debug(f"📥 Enqueued {job_type} job with priority {priority}")
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue {job_type} job: {e}")
            raise
    
    async def start_workers(self) -> None:
        """Start processing workers"""
        if not self.pgq:
            raise RuntimeError("Queue manager not initialized")
        
        logger.info("🚀 Starting queue workers...")
        self.is_running = True
        
        try:
            # Start pgqueuer (this blocks until shutdown)
            await self.pgq.run()
        except Exception as e:
            logger.error(f"❌ Queue workers failed: {e}")
            raise
        finally:
            self.is_running = False
    
    async def stop_workers(self) -> None:
        """Stop processing workers"""
        logger.info("🛑 Stopping queue workers...")
        self.is_running = False
        
        if self.pgq:
            # pgqueuer handles graceful shutdown
            pass
    
    async def enqueue_email_for_classification(self, email_id: str, thread_id: str, priority: int = None) -> None:
        """Enqueue an email for classification (entry point to pipeline)"""
        if priority is None:
            # Calculate priority based on email age and importance signals
            priority = await self._calculate_email_priority(email_id)
        
        await self.enqueue_job("email_classify", {
            "email_id": email_id,
            "thread_id": thread_id,
            "enqueued_at": time.time()
        }, priority=priority)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue processing statistics"""
        if not self.connection:
            return {"error": "Not connected"}
        
        try:
            # Get queue depth for each job type
            queue_stats = await self.connection.fetch("""
                SELECT 
                    entrypoint,
                    COUNT(*) as pending_jobs,
                    AVG(priority) as avg_priority
                FROM pgqueuer.job 
                WHERE status = 'queued'
                GROUP BY entrypoint
            """)
            
            return {
                "processing_stats": self.stats,
                "queue_depth": {row['entrypoint']: {
                    "pending": row['pending_jobs'],
                    "avg_priority": float(row['avg_priority']) if row['avg_priority'] else 0
                } for row in queue_stats},
                "is_running": self.is_running
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get queue stats: {e}")
            return {"error": str(e)}
    
    # Helper methods
    
    async def _get_email_data(self, email_id: str) -> Optional[Dict[str, Any]]:
        """Get email data from database"""
        try:
            result = await self.connection.fetchrow("""
                SELECT id, from_email, subject, body_text, body_html, date_sent, thread_id
                FROM emails WHERE id = $1
            """, email_id)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"❌ Failed to get email data for {email_id}: {e}")
            return None
    
    async def _get_contact_history(self, email: str) -> ContactHistory:
        """Get contact history for email address"""
        try:
            result = await self.connection.fetchrow("""
                SELECT frequency_score, relationship_strength, total_messages
                FROM contacts WHERE email = $1
            """, email)
            
            if result:
                return ContactHistory(
                    sender_frequency_score=float(result['frequency_score']) / 100.0,
                    response_likelihood=float(result['relationship_strength']),
                    relationship_strength=float(result['relationship_strength']),
                    total_emails=result['total_messages'],
                    recent_emails=min(result['total_messages'], 10)
                )
            else:
                return ContactHistory(
                    sender_frequency_score=0.1,
                    response_likelihood=0.3,
                    relationship_strength=0.1,
                    total_emails=1,
                    recent_emails=1
                )
        except Exception as e:
            logger.error(f"❌ Failed to get contact history for {email}: {e}")
            return ContactHistory()
    
    async def _save_classification(self, thread_id: str, email_id: str, classification) -> None:
        """Save classification results to database"""
        try:
            await self.connection.execute("""
                INSERT INTO classifications (
                    id, email_id, classification, confidence, sentiment, sentiment_score,
                    formality, formality_score, personalization, personalization_score,
                    priority, priority_score, should_process, processing_priority,
                    model_used, reasoning
                ) VALUES (
                    gen_random_uuid()::text, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                ) ON CONFLICT (email_id) DO UPDATE SET
                    classification = EXCLUDED.classification,
                    confidence = EXCLUDED.confidence,
                    sentiment = EXCLUDED.sentiment,
                    sentiment_score = EXCLUDED.sentiment_score,
                    formality = EXCLUDED.formality,
                    formality_score = EXCLUDED.formality_score,
                    personalization = EXCLUDED.personalization,
                    personalization_score = EXCLUDED.personalization_score,
                    priority = EXCLUDED.priority,
                    priority_score = EXCLUDED.priority_score,
                    should_process = EXCLUDED.should_process,
                    processing_priority = EXCLUDED.processing_priority,
                    updated_at = NOW()
            """, 
                email_id, classification.classification, classification.confidence,
                classification.sentiment, classification.sentiment_score,
                classification.formality, classification.formality_score,
                classification.personalization, classification.personalization_score,
                classification.priority, classification.priority_score,
                classification.should_process, classification.processing_priority,
                "qwen-0.5b", classification.reasoning
            )
        except Exception as e:
            logger.error(f"❌ Failed to save classification for email {email_id}: {e}")
            raise
    
    async def _calculate_email_priority(self, email_id: str) -> int:
        """Calculate priority for email processing"""
        try:
            result = await self.connection.fetchrow("""
                SELECT 
                    EXTRACT(EPOCH FROM (NOW() - date_received))::INTEGER / 3600 as hours_old,
                    CASE 
                        WHEN from_email LIKE '%noreply%' OR from_email LIKE '%no-reply%' THEN -100
                        ELSE 0 
                    END as sender_penalty
                FROM emails WHERE id = $1
            """, email_id)
            
            if result:
                age_priority = max(0, 1000 - result['hours_old'])
                return age_priority + result['sender_penalty']
            else:
                return 500  # Default priority
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate priority for email {email_id}: {e}")
            return 500
    
    async def _process_with_unstructured(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process email content with Unstructured.io (placeholder)"""
        # TODO: Implement Unstructured.io processing
        return {
            "cleaned_content": email_data.get('body_text', ''),
            "elements": [],
            "chunks": [email_data.get('body_text', '')],
            "processing_method": "placeholder"
        }
    
    async def _save_processed_content(self, email_id: str, processed_content: Dict[str, Any]) -> None:
        """Save processed content to database"""
        try:
            await self.connection.execute("""
                INSERT INTO cleaned_emails (
                    id, email_id, clean_content, original_length, cleaned_length,
                    cleaning_confidence, cleaning_method
                ) VALUES (
                    gen_random_uuid()::text, $1, $2, $3, $4, $5, $6
                ) ON CONFLICT (email_id) DO UPDATE SET
                    clean_content = EXCLUDED.clean_content,
                    cleaned_length = EXCLUDED.cleaned_length,
                    cleaning_confidence = EXCLUDED.cleaning_confidence,
                    updated_at = NOW()
            """,
                email_id,
                processed_content['cleaned_content'],
                len(processed_content.get('original_content', '')),
                len(processed_content['cleaned_content']),
                0.9,  # High confidence for now
                processed_content['processing_method']
            )
        except Exception as e:
            logger.error(f"❌ Failed to save processed content for email {email_id}: {e}")
            raise
    
    async def _generate_embeddings(self, email_id: str, processed_content: Dict[str, Any]) -> int:
        """Generate embeddings for processed content (placeholder)"""
        # TODO: Implement embedding generation
        return len(processed_content.get('chunks', []))
    
    def _update_stage_stats(self, stage: str, processing_time_ms: float, success: bool) -> None:
        """Update processing statistics for a stage"""
        stage_stats = self.stats["stage_stats"][stage]
        
        if success:
            stage_stats["processed"] += 1
            # Update running average
            current_avg = stage_stats["avg_time_ms"]
            total_processed = stage_stats["processed"]
            stage_stats["avg_time_ms"] = ((current_avg * (total_processed - 1)) + processing_time_ms) / total_processed
        else:
            stage_stats["failed"] += 1
        
        self.stats["last_processed"] = time.time()
        
    async def shutdown(self) -> None:
        """Graceful shutdown"""
        await self.stop_workers()
        
        if self.connection:
            await self.connection.close()
            logger.info("✅ Database connection closed")