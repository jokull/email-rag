"""
Consolidated Email Processor
Handles threading, classification, cleaning, and chunking
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session

from llm_interface import QwenInterface, ClassificationResult
from embedding_service import EmbeddingService
from config import ProcessorConfig

@dataclass
class ProcessingResult:
    """Result of processing an email"""
    email_id: str
    success: bool
    classification: Optional[ClassificationResult] = None
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None
    chunks_created: int = 0
    embeddings_created: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None

class EmailProcessor:
    """
    Consolidated email processor that handles all AI tasks:
    - Email classification (human/promotional/transactional/automated)
    - Thread detection using JWZ + ML similarity
    - Content cleaning and chunking 
    - Conversation building
    - Embedding generation for RAG
    """
    
    def __init__(
        self, 
        llm_interface: QwenInterface,
        embedding_service: EmbeddingService,
        config: ProcessorConfig
    ):
        self.llm = llm_interface
        self.embeddings = embedding_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processing_stats = {
            "total_processed": 0,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "threads_created": 0,
            "conversations_created": 0,
            "embeddings_generated": 0,
            "total_processing_time": 0.0,
            "last_processed": None
        }
        self.is_processing = False
        
    async def process_pending_emails(self) -> List[ProcessingResult]:
        """Process all pending emails in the database"""
        if self.is_processing:
            self.logger.info("⏳ Processing already in progress, skipping...")
            return []
        
        self.is_processing = True
        start_time = time.time()
        results = []
        
        try:
            self.logger.info("🔄 Starting email processing cycle...")
            
            # Get unprocessed emails from database
            from main import SessionLocal
            with SessionLocal() as session:
                unprocessed_emails = self._get_unprocessed_emails(session)
                
                if not unprocessed_emails:
                    self.logger.info("✅ No emails to process")
                    return []
                
                self.logger.info(f"📧 Processing {len(unprocessed_emails)} emails...")
                
                # Process in batches for memory efficiency
                batch_size = self.config.batch_size
                for i in range(0, len(unprocessed_emails), batch_size):
                    batch = unprocessed_emails[i:i + batch_size]
                    batch_results = await self._process_email_batch(session, batch)
                    results.extend(batch_results)
                    
                    # Small delay between batches to prevent overload
                    if i + batch_size < len(unprocessed_emails):
                        await asyncio.sleep(1)
                
                # Update statistics
                processing_time = time.time() - start_time
                self.processing_stats["total_processing_time"] += processing_time
                self.processing_stats["last_processed"] = datetime.utcnow()
                
                successful = sum(1 for r in results if r.success)
                self.logger.info(
                    f"✅ Processed {len(results)} emails in {processing_time:.2f}s "
                    f"({successful} successful, {len(results) - successful} failed)"
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error in email processing: {e}")
            
        finally:
            self.is_processing = False
            
        return results
    
    def _get_unprocessed_emails(self, session: Session) -> List[Dict[str, Any]]:
        """Get emails that haven't been processed yet"""
        query = text("""
            SELECT 
                e.id,
                e.message_id,
                e.subject,
                e.sender,
                e.recipients,
                e.content,
                e.received_date,
                e.in_reply_to,
                e.references
            FROM emails e
            LEFT JOIN classifications c ON e.id = c.email_id
            WHERE c.id IS NULL  -- Not yet classified
            AND e.content IS NOT NULL
            AND LENGTH(e.content) >= :min_length
            ORDER BY e.received_date DESC
            LIMIT :limit
        """)
        
        result = session.execute(query, {
            "min_length": self.config.min_email_length,
            "limit": self.config.batch_size * 2  # Get more than one batch
        })
        
        return [dict(row._mapping) for row in result]
    
    async def _process_email_batch(
        self, 
        session: Session, 
        emails: List[Dict[str, Any]]
    ) -> List[ProcessingResult]:
        """Process a batch of emails"""
        results = []
        
        # Step 1: Classify emails in batch
        self.logger.info(f"🧠 Classifying {len(emails)} emails...")
        classifications = await self._batch_classify_emails(emails)
        
        # Step 2: Process each email individually
        for email, classification in zip(emails, classifications):
            result = await self._process_single_email(session, email, classification)
            results.append(result)
            
            # Update global stats
            self.processing_stats["total_processed"] += 1
            if result.success:
                self.processing_stats["successful_classifications"] += 1
            else:
                self.processing_stats["failed_classifications"] += 1
        
        return results
    
    async def _batch_classify_emails(
        self, 
        emails: List[Dict[str, Any]]
    ) -> List[ClassificationResult]:
        """Classify multiple emails efficiently"""
        contents = []
        
        for email in emails:
            content = email.get("content", "")
            
            # Pre-process content for classification
            if email.get("subject"):
                content = f"Subject: {email['subject']}\n\n{content}"
            
            # Truncate if too long
            if len(content) > self.config.max_email_length:
                content = content[:self.config.max_email_length] + "..."
            
            contents.append(content)
        
        try:
            return await self.llm.batch_classify(contents)
        except Exception as e:
            self.logger.error(f"❌ Batch classification failed: {e}")
            # Return default classifications
            return [
                ClassificationResult(
                    classification="automated",
                    confidence=0.5,
                    processing_time=0.0,
                    tokens_used=0
                )
                for _ in emails
            ]
    
    async def _process_single_email(
        self,
        session: Session,
        email: Dict[str, Any],
        classification: ClassificationResult
    ) -> ProcessingResult:
        """Process a single email through the complete pipeline"""
        email_id = email["id"]
        start_time = time.time()
        
        try:
            # 1. Save classification to database
            await self._save_classification(session, email_id, classification)
            
            # 2. Thread detection and conversation building
            thread_id, conversation_id = await self._handle_threading(session, email)
            
            # 3. Content cleaning and chunking (only for human conversations)
            chunks_created = 0
            embeddings_created = 0
            
            if classification.classification == "human" and classification.confidence >= self.config.human_classification_threshold:
                chunks_created = await self._clean_and_chunk_content(session, email_id, email["content"])
                
                # 4. Generate embeddings for RAG pipeline
                if self.config.enable_embedding_generation:
                    embeddings_created = await self._generate_embeddings(session, email_id)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                email_id=email_id,
                success=True,
                classification=classification,
                thread_id=thread_id,
                conversation_id=conversation_id,
                chunks_created=chunks_created,
                embeddings_created=embeddings_created,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process email {email_id}: {e}")
            return ProcessingResult(
                email_id=email_id,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _save_classification(
        self, 
        session: Session, 
        email_id: str, 
        classification: ClassificationResult
    ):
        """Save classification results to database"""
        # Calculate additional scores based on classification
        human_score = 0.9 if classification.classification == "human" else 0.1
        personal_score = 0.7 if classification.classification == "human" and classification.confidence > 0.8 else 0.2
        relevance_score = classification.confidence
        should_process = (
            classification.classification == "human" and 
            classification.confidence >= self.config.human_classification_threshold
        )
        
        query = text("""
            INSERT INTO classifications (
                email_id, classification, confidence, human_score, 
                personal_score, relevance_score, should_process, created_at
            ) VALUES (
                :email_id, :classification, :confidence, :human_score,
                :personal_score, :relevance_score, :should_process, :created_at
            )
            ON CONFLICT (email_id) DO UPDATE SET
                classification = EXCLUDED.classification,
                confidence = EXCLUDED.confidence,
                human_score = EXCLUDED.human_score,
                personal_score = EXCLUDED.personal_score,
                relevance_score = EXCLUDED.relevance_score,
                should_process = EXCLUDED.should_process,
                updated_at = :created_at
        """)
        
        session.execute(query, {
            "email_id": email_id,
            "classification": classification.classification,
            "confidence": classification.confidence,
            "human_score": human_score,
            "personal_score": personal_score,
            "relevance_score": relevance_score,
            "should_process": should_process,
            "created_at": datetime.utcnow()
        })
        session.commit()
    
    async def _handle_threading(
        self, 
        session: Session, 
        email: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Handle thread detection and conversation building"""
        # Simple threading based on message-id, in-reply-to, and references
        # This is a simplified version - the original thread-processor had more sophisticated ML
        
        thread_id = None
        conversation_id = None
        
        try:
            # Check if thread already exists
            if email.get("in_reply_to") or email.get("references"):
                # Look for existing thread
                query = text("""
                    SELECT t.id FROM threads t
                    JOIN emails e ON e.thread_id = t.id
                    WHERE e.message_id = :in_reply_to
                    OR e.message_id = ANY(string_to_array(:references, ' '))
                    LIMIT 1
                """)
                
                result = session.execute(query, {
                    "in_reply_to": email.get("in_reply_to"),
                    "references": email.get("references", "")
                }).fetchone()
                
                if result:
                    thread_id = result[0]
            
            # Create new thread if none found
            if not thread_id:
                thread_id = f"thread_{email['id']}"
                query = text("""
                    INSERT INTO threads (id, subject, participants, created_at)
                    VALUES (:id, :subject, :participants, :created_at)
                    ON CONFLICT (id) DO NOTHING
                """)
                
                participants = [email.get("sender", "")]
                if email.get("recipients"):
                    participants.extend(email["recipients"].split(","))
                
                session.execute(query, {
                    "id": thread_id,
                    "subject": email.get("subject", ""),
                    "participants": participants,
                    "created_at": datetime.utcnow()
                })
            
            # Update email with thread_id
            query = text("UPDATE emails SET thread_id = :thread_id WHERE id = :email_id")
            session.execute(query, {"thread_id": thread_id, "email_id": email["id"]})
            
            session.commit()
            self.processing_stats["threads_created"] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Threading failed for email {email['id']}: {e}")
            session.rollback()
        
        return thread_id, conversation_id
    
    async def _clean_and_chunk_content(
        self, 
        session: Session, 
        email_id: str, 
        content: str
    ) -> int:
        """Clean and chunk email content"""
        try:
            # Use the LLM interface's chunking (which uses heuristics for efficiency)
            chunking_result = await self.llm.chunk_content(content)
            
            # Save cleaned chunks
            chunks_created = 0
            for i, chunk in enumerate(chunking_result.chunks):
                query = text("""
                    INSERT INTO cleaned_emails (
                        id, email_id, chunk_index, clean_content, 
                        word_count, char_count, created_at
                    ) VALUES (
                        :id, :email_id, :chunk_index, :clean_content,
                        :word_count, :char_count, :created_at
                    )
                """)
                
                chunk_id = f"{email_id}_chunk_{i}"
                session.execute(query, {
                    "id": chunk_id,
                    "email_id": email_id,
                    "chunk_index": i,
                    "clean_content": chunk.strip(),
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk),
                    "created_at": datetime.utcnow()
                })
                chunks_created += 1
            
            session.commit()
            return chunks_created
            
        except Exception as e:
            self.logger.error(f"❌ Content cleaning failed for email {email_id}: {e}")
            session.rollback()
            return 0
    
    async def _generate_embeddings(self, session: Session, email_id: str) -> int:
        """Generate embeddings for email chunks"""
        try:
            # Get cleaned chunks for this email
            query = text("""
                SELECT id, clean_content FROM cleaned_emails 
                WHERE email_id = :email_id
            """)
            
            chunks = session.execute(query, {"email_id": email_id}).fetchall()
            embeddings_created = 0
            
            for chunk_id, content in chunks:
                # Generate embedding
                embedding = await self.embeddings.generate_embedding(content)
                
                # Save to database
                query = text("""
                    INSERT INTO embeddings (
                        id, email_id, chunk_id, embedding, model_name, 
                        dimension, created_at
                    ) VALUES (
                        :id, :email_id, :chunk_id, :embedding, :model_name,
                        :dimension, :created_at
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        updated_at = :created_at
                """)
                
                session.execute(query, {
                    "id": f"emb_{chunk_id}",
                    "email_id": email_id,
                    "chunk_id": chunk_id,
                    "embedding": embedding.tolist(),
                    "model_name": self.config.embedding_model,
                    "dimension": len(embedding),
                    "created_at": datetime.utcnow()
                })
                embeddings_created += 1
            
            session.commit()
            self.processing_stats["embeddings_generated"] += embeddings_created
            return embeddings_created
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed for email {email_id}: {e}")
            session.rollback()
            return 0
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            "is_processing": self.is_processing,
            "stats": self.processing_stats.copy(),
            "llm_metrics": self.llm.get_performance_metrics() if self.llm else {},
            "config": {
                "batch_size": self.config.batch_size,
                "processing_interval": self.config.processing_interval,
                "human_threshold": self.config.human_classification_threshold
            }
        }
    
    def stop_processing(self):
        """Stop background processing"""
        self.logger.info("🛑 Stopping email processor...")
        self.is_processing = False