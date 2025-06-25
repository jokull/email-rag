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

from modern_llm import ModernQwenInterface
from classification_models import EmailClassification, EmailProcessingRequest, ContactHistory, ThreadSummary, ThreadSummaryRequest
from embedding_service import EmbeddingService
from config import ProcessorConfig
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

@dataclass
class ProcessingResult:
    """Result of processing an email"""
    email_id: str
    success: bool
    classification: Optional[EmailClassification] = None
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
        llm_interface: ModernQwenInterface,
        embedding_service: EmbeddingService,
        config: ProcessorConfig,
        session_maker: sessionmaker = None
    ):
        self.llm = llm_interface
        self.embeddings = embedding_service
        self.config = config
        self.session_maker = session_maker
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
            
            # Check if we have a session maker
            if not self.session_maker:
                self.logger.warning("⚠️ No database session available, skipping email processing")
                return []
            
            # Get unprocessed emails from database
            with self.session_maker() as session:
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
                e.from_email as sender,
                e.to_emails as recipients,
                COALESCE(e.body_text, e.body_html) as content,
                e.date_received as received_date,
                e.raw_headers->>'in-reply-to' as in_reply_to,
                e.raw_headers->>'references' as references
            FROM emails e
            LEFT JOIN classifications c ON e.id = c.email_id
            WHERE c.id IS NULL  -- Not yet classified
            AND COALESCE(e.body_text, e.body_html) IS NOT NULL
            AND LENGTH(COALESCE(e.body_text, e.body_html)) >= :min_length
            ORDER BY e.date_received DESC
            LIMIT :limit
        """)
        
        result = session.execute(query, {
            "min_length": self.config.min_email_length,
            "limit": min(10, self.config.batch_size * 2)  # Start with just 10 emails max
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
            try:
                result = await self._process_single_email(session, email, classification)
                results.append(result)
                
                # Commit if successful
                if result.success:
                    session.commit()
                    self.processing_stats["successful_classifications"] += 1
                else:
                    session.rollback()
                    self.processing_stats["failed_classifications"] += 1
                    
            except Exception as e:
                session.rollback()
                self.logger.error(f"❌ Email processing failed: {e}")
                result = ProcessingResult(
                    email_id=email.get("id", "unknown"),
                    success=False,
                    error=str(e),
                    processing_time=0
                )
                results.append(result)
                self.processing_stats["failed_classifications"] += 1
            
            # Update global stats
            self.processing_stats["total_processed"] += 1
        
        return results
    
    async def _batch_classify_emails(
        self, 
        emails: List[Dict[str, Any]]
    ) -> List[EmailClassification]:
        """Classify multiple emails efficiently"""
        if self.llm is None:
            # Fallback to basic heuristic classification
            return [self._basic_classify_email(email) for email in emails]
        
        # Convert emails to EmailProcessingRequest objects
        requests = []
        
        for email in emails:
            content = email.get("content", "")
            
            # Pre-process content for classification
            if email.get("subject"):
                content = f"Subject: {email['subject']}\n\n{content}"
            
            # Truncate if too long
            if len(content) > self.config.max_email_length:
                content = content[:self.config.max_email_length] + "..."
            
            # Analyze contact history for this email
            contact_history = self._get_contact_history(email.get("sender", ""))
            
            # Create processing request
            request = EmailProcessingRequest(
                email_id=str(email.get("id", "")),
                sender=email.get("sender", ""),
                subject=email.get("subject"),
                content=content,
                contact_history=contact_history
            )
            requests.append(request)
        
        try:
            return await self.llm.classify_batch(requests)
        except Exception as e:
            self.logger.error(f"❌ Batch classification failed: {e}")
            # Fallback to basic classification
            return [self._basic_classify_email(email) for email in emails]
    
    def _basic_classify_email(self, email: Dict[str, Any]) -> EmailClassification:
        """Enhanced heuristic email classification with multi-dimensional analysis"""
        content = email.get("content", "").lower()
        subject = email.get("subject", "").lower()
        sender = email.get("sender", "").lower()
        
        # Get contact history analysis
        contact_analysis = self._analyze_contact_history(sender)
        
        # Basic classification
        classification = "human"
        confidence = 0.6
        
        if any(word in content or word in subject for word in [
            "unsubscribe", "promotional", "offer", "deal", "sale", "discount"
        ]):
            classification = "promotional"
            confidence = 0.8
        elif any(word in content or word in subject for word in [
            "receipt", "invoice", "payment", "transaction", "order", "shipping"
        ]):
            classification = "transactional"
            confidence = 0.8
        elif any(word in sender for word in [
            "noreply", "no-reply", "donotreply", "automated", "bot"
        ]):
            classification = "automated"
            confidence = 0.9
        
        # Sentiment analysis
        sentiment, sentiment_score = self._analyze_sentiment(content, subject)
        
        # Formality analysis
        formality, formality_score = self._analyze_formality(content, subject)
        
        # Personalization analysis
        personalization, personalization_score = self._analyze_personalization(content, subject, sender)
        
        # Priority analysis
        priority, priority_score = self._analyze_priority(content, subject, contact_analysis)
        
        return EmailClassification(
            classification=classification,
            confidence=confidence,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            formality=formality,
            formality_score=formality_score,
            personalization=personalization,
            personalization_score=personalization_score,
            priority=priority,
            priority_score=priority_score,
            should_process=(classification == "human" and confidence > 0.6),
            processing_priority=80 if priority == "urgent" else 50 if classification == "human" else 20,
            reasoning=f"Heuristic classification: {classification} based on content patterns",
            content_length=len(email.get("content", "")),
            processing_time_ms=1.0
        )
    
    def _get_contact_history(self, sender: str) -> ContactHistory:
        """Get contact history as ContactHistory object for the new interface"""
        contact_analysis = self._analyze_contact_history(sender)
        
        return ContactHistory(
            sender_frequency_score=contact_analysis["frequency_score"],
            response_likelihood=contact_analysis["response_likelihood"],
            relationship_strength=contact_analysis["relationship_strength"],
            total_emails=int(contact_analysis.get("total_emails", 0)),
            recent_emails=int(contact_analysis.get("recent_emails", 0))
        )
    
    def _analyze_contact_history(self, sender: str) -> Dict[str, float]:
        """Analyze contact history for this sender"""
        try:
            if not self.session_maker:
                return {
                    "frequency_score": 0.0,
                    "response_likelihood": 0.0,
                    "relationship_strength": 0.0,
                    "total_emails": 0,
                    "recent_emails": 0
                }
                
            with self.session_maker() as session:
                # Get sender statistics
                stats = session.execute(text("""
                    SELECT 
                        COUNT(*) as total_emails,
                        COUNT(CASE WHEN DATE(e.date_received) >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as recent_emails,
                        MAX(e.date_received) as last_email,
                        AVG(CASE WHEN LENGTH(e.body_text) > 0 THEN LENGTH(e.body_text) ELSE NULL END) as avg_length,
                        COUNT(CASE WHEN e.from_email LIKE '%@gmail.com' OR e.from_email LIKE '%@outlook.com' THEN 1 END) as personal_email_count
                    FROM emails e 
                    WHERE LOWER(e.from_email) = LOWER(:sender)
                """), {"sender": sender}).fetchone()
                
                if not stats or stats.total_emails == 0:
                    return {
                        "frequency_score": 0.0,
                        "response_likelihood": 0.0,
                        "relationship_strength": 0.0,
                        "total_emails": 0,
                        "recent_emails": 0
                    }
                
                # Calculate frequency score (0-1 based on recent activity)
                frequency_score = min(1.0, float(stats.recent_emails) / 10.0)
                
                # Estimate response likelihood based on email patterns
                personal_ratio = float(stats.personal_email_count) / float(stats.total_emails)
                avg_length = float(stats.avg_length or 0)
                response_likelihood = min(1.0, (personal_ratio * 0.5) + (min(avg_length, 1000) / 2000))
                
                # Relationship strength based on frequency and duration
                relationship_strength = min(1.0, (float(stats.total_emails) / 50.0) * (frequency_score + 0.5))
                
                return {
                    "frequency_score": frequency_score,
                    "response_likelihood": response_likelihood,
                    "relationship_strength": relationship_strength,
                    "total_emails": stats.total_emails,
                    "recent_emails": stats.recent_emails
                }
                
        except Exception as e:
            self.logger.warning(f"Contact history analysis failed: {e}")
            return {
                "frequency_score": 0.0,
                "response_likelihood": 0.0,
                "relationship_strength": 0.0,
                "total_emails": 0,
                "recent_emails": 0
            }
    
    def _analyze_sentiment(self, content: str, subject: str) -> Tuple[str, float]:
        """Basic sentiment analysis"""
        positive_words = ["thank", "great", "awesome", "excellent", "love", "happy", "congratulations"]
        negative_words = ["sorry", "problem", "issue", "error", "failed", "urgent", "deadline"]
        
        text = f"{subject} {content}"
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "positive", min(1.0, positive_count / 5.0)
        elif negative_count > positive_count:
            return "negative", min(1.0, negative_count / 5.0)  # Keep positive for DB constraint
        else:
            return "neutral", 0.5
    
    def _analyze_formality(self, content: str, subject: str) -> Tuple[str, float]:
        """Basic formality analysis"""
        formal_words = ["please", "kindly", "regards", "sincerely", "respectfully"]
        informal_words = ["hey", "hi", "thanks", "btw", "lol", "yeah"]
        
        text = f"{subject} {content}"
        formal_count = sum(1 for word in formal_words if word in text)
        informal_count = sum(1 for word in informal_words if word in text)
        
        if formal_count > informal_count:
            return "formal", min(1.0, formal_count / 3.0)
        elif informal_count > formal_count:
            return "informal", min(1.0, informal_count / 3.0)
        else:
            return "neutral", 0.5
    
    def _analyze_personalization(self, content: str, subject: str, sender: str) -> Tuple[str, float]:
        """Basic personalization analysis"""
        personal_indicators = ["you", "your", "personally", "specifically"]
        generic_indicators = ["dear customer", "dear sir/madam", "to whom it may concern"]
        
        text = f"{subject} {content}"
        personal_count = sum(1 for word in personal_indicators if word in text)
        generic_count = sum(1 for phrase in generic_indicators if phrase in text)
        
        # Check if sender appears to be personal email
        is_personal_domain = any(domain in sender for domain in ["@gmail.com", "@outlook.com", "@icloud.com"])
        
        if personal_count > 0 and generic_count == 0:
            score = min(1.0, (personal_count / 3.0) + (0.3 if is_personal_domain else 0))
            if score > 0.7:
                return "highly_personal", score
            else:
                return "somewhat_personal", score
        elif generic_count > 0:
            return "generic", max(0.0, 1.0 - generic_count / 2.0)
        else:
            return "somewhat_personal" if is_personal_domain else "generic", 0.3 if is_personal_domain else 0.1
    
    def _analyze_priority(self, content: str, subject: str, contact_analysis: Dict[str, float]) -> Tuple[str, float]:
        """Basic priority analysis"""
        urgent_words = ["urgent", "asap", "immediately", "deadline", "emergency", "critical"]
        low_priority_words = ["fyi", "no rush", "when you can", "newsletter"]
        
        text = f"{subject} {content}"
        urgent_count = sum(1 for word in urgent_words if word in text)
        low_count = sum(1 for word in low_priority_words if word in text)
        
        # Factor in relationship strength
        relationship_boost = contact_analysis.get("relationship_strength", 0.0) * 0.3
        
        if urgent_count > 0:
            return "urgent", min(1.0, urgent_count / 2.0 + relationship_boost)
        elif low_count > 0:
            return "low", max(0.0, 0.2 - low_count / 3.0)
        else:
            return "normal", 0.5 + relationship_boost
    
    async def _process_single_email(
        self,
        session: Session,
        email: Dict[str, Any],
        classification: EmailClassification
    ) -> ProcessingResult:
        """Process a single email through the complete pipeline"""
        email_id = email["id"]
        start_time = time.time()
        
        try:
            # 1. Thread detection and conversation building (do this first to get thread_id)
            thread_id, conversation_id = await self._handle_threading(session, email)
            
            # 2. Save classification to database with proper thread_id
            await self._save_classification(session, email_id, classification, email, thread_id)
            
            # 3. Content cleaning and chunking (only for human conversations)
            chunks_created = 0
            embeddings_created = 0
            
            if classification.classification == "human" and classification.confidence >= self.config.human_classification_threshold:
                chunks_created = await self._clean_and_chunk_content(session, email_id, email["content"])
                
                # 4. Generate embeddings for RAG pipeline (skip if no embedding service)
                if self.config.enable_embedding_generation and self.embeddings is not None:
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
        classification: EmailClassification,
        email: dict,
        thread_id: str
    ):
        """Save enhanced classification results to database"""
        # Calculate additional scores based on classification
        human_score = 0.9 if classification.classification == "human" else 0.1
        personal_score = classification.personalization_score
        relevance_score = classification.confidence
        should_process = (
            classification.classification == "human" and 
            classification.confidence >= self.config.human_classification_threshold
        )
        
        # Get contact history from sender
        sender = email.get("from_email", "unknown@example.com")
        contact_analysis = self._analyze_contact_history(sender)
        
        query = text("""
            INSERT INTO classifications (
                email_id, thread_id, classification, confidence, human_score, 
                personal_score, relevance_score, should_process, model_used,
                sentiment, sentiment_score, formality, formality_score,
                personalization, personalization_score, priority, priority_score,
                sender_frequency_score, response_likelihood, relationship_strength,
                created_at
            ) VALUES (
                :email_id, :thread_id, :classification, :confidence, :human_score,
                :personal_score, :relevance_score, :should_process, :model_used,
                :sentiment, :sentiment_score, :formality, :formality_score,
                :personalization, :personalization_score, :priority, :priority_score,
                :sender_frequency_score, :response_likelihood, :relationship_strength,
                :created_at
            )
            ON CONFLICT (email_id) DO UPDATE SET
                thread_id = EXCLUDED.thread_id,
                classification = EXCLUDED.classification,
                confidence = EXCLUDED.confidence,
                human_score = EXCLUDED.human_score,
                personal_score = EXCLUDED.personal_score,
                relevance_score = EXCLUDED.relevance_score,
                should_process = EXCLUDED.should_process,
                model_used = EXCLUDED.model_used,
                sentiment = EXCLUDED.sentiment,
                sentiment_score = EXCLUDED.sentiment_score,
                formality = EXCLUDED.formality,
                formality_score = EXCLUDED.formality_score,
                personalization = EXCLUDED.personalization,
                personalization_score = EXCLUDED.personalization_score,
                priority = EXCLUDED.priority,
                priority_score = EXCLUDED.priority_score,
                sender_frequency_score = EXCLUDED.sender_frequency_score,
                response_likelihood = EXCLUDED.response_likelihood,
                relationship_strength = EXCLUDED.relationship_strength
        """)
        
        session.execute(query, {
            "email_id": email_id,
            "thread_id": thread_id,  # Use the thread_id from threading step
            "classification": classification.classification,
            "confidence": classification.confidence,
            "human_score": human_score,
            "personal_score": personal_score,
            "relevance_score": relevance_score,
            "should_process": should_process,
            "model_used": "qwen-0.5b-fallback" if not self.llm else "qwen-0.5b",
            "sentiment": classification.sentiment,
            "sentiment_score": classification.sentiment_score,
            "formality": classification.formality,
            "formality_score": classification.formality_score,
            "personalization": classification.personalization,
            "personalization_score": classification.personalization_score,
            "priority": classification.priority,
            "priority_score": classification.priority_score,
            "sender_frequency_score": contact_analysis["frequency_score"],
            "response_likelihood": contact_analysis["response_likelihood"],
            "relationship_strength": contact_analysis["relationship_strength"],
            "created_at": datetime.utcnow()
        })
        # Don't commit here - let the calling method handle the transaction
    
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
            # Check if thread already exists by looking in raw_headers JSONB
            in_reply_to = None
            references = None
            
            if email.get("raw_headers"):
                # Parse raw_headers JSONB for threading information
                headers = email["raw_headers"]
                if isinstance(headers, str):
                    import json
                    headers = json.loads(headers)
                
                in_reply_to = headers.get("in_reply_to")
                references = headers.get("references")
            
            if in_reply_to or references:
                # Look for existing thread
                query = text("""
                    SELECT t.id FROM threads t
                    JOIN emails e ON e.thread_id = t.id
                    WHERE e.message_id = :in_reply_to
                    OR (:references IS NOT NULL AND e.message_id = ANY(string_to_array(:references, ' ')))
                    LIMIT 1
                """)
                
                result = session.execute(query, {
                    "in_reply_to": in_reply_to,
                    "references": references or ""
                }).fetchone()
                
                if result:
                    thread_id = result[0]
            
            # Create new thread if none found
            if not thread_id:
                thread_id = f"thread_{email['id']}"
                
                # Clean participants data first
                participants = []
                if email.get("from_email"):
                    participants.append(email["from_email"])
                
                # Handle recipients - they are PostgreSQL arrays
                if email.get("recipients"):
                    recipients_array = email["recipients"]
                    if isinstance(recipients_array, list):
                        # Handle PostgreSQL array format
                        for recipient in recipients_array:
                            if isinstance(recipient, str):
                                # If it's a JSON string, try to parse it
                                if recipient.startswith('{') and recipient.endswith('}'):
                                    try:
                                        import json
                                        parsed_recipient = json.loads(recipient)
                                        if isinstance(parsed_recipient, dict) and "email" in parsed_recipient:
                                            participants.append(parsed_recipient["email"])
                                        else:
                                            participants.append(recipient)
                                    except:
                                        participants.append(recipient)
                                else:
                                    participants.append(recipient)
                            elif isinstance(recipient, dict) and "email" in recipient:
                                participants.append(recipient["email"])
                    else:
                        # Fallback - treat as comma-separated string
                        recipients_str = str(recipients_array)
                        participants.extend([r.strip() for r in recipients_str.split(",") if r.strip()])
                
                # Clean and normalize subject
                subject = email.get("subject", "No Subject")
                subject_normalized = subject.lower().strip()
                # Truncate if too long for database field (500 char limit)
                if len(subject_normalized) > 500:
                    subject_normalized = subject_normalized[:497] + "..."
                
                # Keep participants as Python list for SQLAlchemy to handle
                
                query = text("""
                    INSERT INTO threads (
                        id, subject_normalized, participants, 
                        first_message_date, last_message_date, created_at
                    )
                    VALUES (
                        :id, :subject_normalized, :participants, 
                        :first_message_date, :last_message_date, :created_at
                    )
                    ON CONFLICT (id) DO UPDATE SET 
                        subject_normalized = EXCLUDED.subject_normalized,
                        participants = EXCLUDED.participants,
                        last_message_date = EXCLUDED.last_message_date
                """)
                
                # Get email date
                email_date = email.get("date_received", datetime.utcnow())
                
                thread_params = {
                    "id": thread_id,
                    "subject_normalized": subject_normalized,
                    "participants": participants,
                    "first_message_date": email_date,
                    "last_message_date": email_date,
                    "created_at": datetime.utcnow()
                }
                
                # Debug: Log thread creation parameters with detailed lengths
                participants_str = str(participants)
                self.logger.info(f"📊 Creating thread: {thread_id[:20]}...")
                self.logger.info(f"   📏 subject_len={len(subject_normalized)} participants_count={len(participants)}")
                self.logger.info(f"   📏 participants_str_len={len(participants_str)} participants_preview={participants_str[:100]}...")
                self.logger.info(f"   📏 subject_preview={subject_normalized[:100]}...")
                
                try:
                    session.execute(query, thread_params)
                except Exception as db_error:
                    self.logger.error(f"💥 Thread creation failed with params:")
                    self.logger.error(f"   thread_id: {thread_id} (len={len(thread_id)})")
                    self.logger.error(f"   subject_normalized: {subject_normalized[:200]}... (len={len(subject_normalized)})")
                    self.logger.error(f"   participants: {participants} (len={len(str(participants))})")
                    self.logger.error(f"   email_date: {email_date}")
                    raise db_error
            
            # Update email with thread_id
            query = text("UPDATE emails SET thread_id = :thread_id WHERE id = :email_id")
            session.execute(query, {"thread_id": thread_id, "email_id": email["id"]})
            
            # Don't commit here - let the calling method handle the transaction
            self.processing_stats["threads_created"] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Threading failed for email {email['id']}: {e}")
            # Don't rollback here - let the calling method handle the transaction
            raise e  # Re-raise to stop processing this email
        
        return thread_id, conversation_id
    
    async def _clean_and_chunk_content(
        self, 
        session: Session, 
        email_id: str, 
        content: str
    ) -> int:
        """Clean email content using tiered approach (email-reply-parser + Qwen) and simple chunking"""
        cleaning_method = "unknown"
        
        try:
            # Step 1: Use tiered cleaning approach
            cleaned_content = await self._clean_with_tiered_approach(content)
            
            # Determine which method was used based on logs or content analysis
            if hasattr(self, '_last_cleaning_method'):
                cleaning_method = self._last_cleaning_method
            else:
                cleaning_method = "tiered_approach"
            
            # Step 2: Simple chunking (no LLM needed for basic chunking)
            chunks = self._simple_chunk_content(cleaned_content)
            
            # Step 3: Save cleaned chunks
            chunks_created = 0
            for i, chunk in enumerate(chunks):
                query = text("""
                    INSERT INTO cleaned_emails (
                        id, email_id, clean_content, original_length,
                        cleaned_length, cleaning_confidence, cleaning_method, created_at
                    ) VALUES (
                        :id, :email_id, :clean_content, :original_length,
                        :cleaned_length, :cleaning_confidence, :cleaning_method, :created_at
                    )
                """)
                
                chunk_id = f"{email_id}_chunk_{i}"
                session.execute(query, {
                    "id": chunk_id,
                    "email_id": email_id,
                    "clean_content": chunk.strip(),
                    "original_length": len(content),
                    "cleaned_length": len(chunk.strip()),
                    "cleaning_confidence": 0.8,  # Default confidence
                    "cleaning_method": cleaning_method,
                    "created_at": datetime.utcnow()
                })
                chunks_created += 1
            
            session.commit()
            return chunks_created
            
        except Exception as e:
            self.logger.error(f"❌ Content cleaning failed for email {email_id}: {e}")
            session.rollback()
            return 0
    
    async def _clean_with_tiered_approach(self, content: str) -> str:
        """Tiered email cleaning: email-reply-parser → Qwen-0.5B for complex cases"""
        
        # Tier 1: Try email-reply-parser (fast, 70-80% accuracy)
        try:
            from email_reply_parser import EmailReplyParser
            parsed = EmailReplyParser().read(content)
            cleaned = parsed.reply
            
            # Quality check: if cleaned content is reasonable, use it
            if self._is_good_quality_cleaning(content, cleaned):
                self.logger.debug("✅ Used email-reply-parser for cleaning")
                self._last_cleaning_method = "email-reply-parser"
                return cleaned.strip()
                
        except ImportError:
            self.logger.warning("⚠️ email-reply-parser not available")
        except Exception as e:
            self.logger.warning(f"⚠️ email-reply-parser failed: {e}")
        
        # Tier 2: Use Qwen-0.5B for complex email cleaning
        if self.llm and self.llm.ready:
            try:
                cleaned = await self._clean_with_qwen(content)
                if cleaned and len(cleaned.strip()) > 20:  # Sanity check
                    self.logger.debug("🧠 Used Qwen for complex email cleaning")
                    self._last_cleaning_method = "qwen-0.5b"
                    return cleaned.strip()
            except Exception as e:
                self.logger.warning(f"⚠️ Qwen cleaning failed: {e}")
        
        # Tier 3: Fallback to basic cleaning
        self.logger.debug("🔄 Using basic cleaning fallback")
        self._last_cleaning_method = "basic-fallback"
        return self._basic_email_cleaning(content)
    
    async def _clean_with_qwen(self, content: str) -> str:
        """Use Qwen-0.5B for complex email cleaning with optimized prompt"""
        
        # Truncate very long emails to fit context window
        if len(content) > 4000:
            content = content[:4000] + "..."
        
        prompt = f"""You are an expert email processor. Extract only the NEW email content, removing all quoted text, signatures, and email artifacts.

RULES:
1. Remove lines starting with ">" (quoted replies)
2. Remove "From:", "To:", "Subject:" blocks from forwards
3. Remove email signatures (patterns like "Best regards", "Sent from iPhone", "--")
4. Remove automatic disclaimers and footers
5. Keep only the author's new message content
6. Preserve formatting and line breaks in the main content

EMAIL:
{content}

NEW CONTENT ONLY:"""

        try:
            response = self.llm.model(
                prompt,
                max_tokens=1024,
                temperature=0.1,  # Low temperature for consistent cleaning
                stop=["EMAIL:", "RULES:", "\n\n---"],
                echo=False
            )
            
            cleaned = response['choices'][0]['text'].strip()
            
            # Basic validation
            if len(cleaned) > 0 and not cleaned.startswith("I can't") and not cleaned.startswith("I cannot"):
                return cleaned
            else:
                raise ValueError("Invalid Qwen response")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Qwen cleaning failed: {e}")
            raise
    
    def _is_good_quality_cleaning(self, original: str, cleaned: str) -> bool:
        """Check if email cleaning produced reasonable results"""
        if not cleaned or len(cleaned.strip()) < 10:
            return False
        
        # Cleaned content should be meaningful portion of original
        if len(cleaned) < len(original) * 0.1:  # Less than 10% might be too aggressive
            return False
        
        # Should not be mostly punctuation
        alpha_chars = sum(1 for c in cleaned if c.isalpha())
        if alpha_chars < len(cleaned) * 0.3:  # Less than 30% letters
            return False
            
        # Common signs of failed parsing
        failed_indicators = [
            "failed to parse",
            "error processing",
            "unable to extract",
            "see original message"
        ]
        
        if any(indicator in cleaned.lower() for indicator in failed_indicators):
            return False
            
        return True
    
    def _basic_email_cleaning(self, content: str) -> str:
        """Basic email cleaning fallback"""
        import re
        
        # Remove common signature patterns
        content = re.sub(r'\n--\s*\n.*', '', content, flags=re.DOTALL)
        content = re.sub(r'\nSent from my.*', '', content)
        content = re.sub(r'\n>.*', '', content, flags=re.MULTILINE)
        
        # Clean up whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content
    
    def _simple_chunk_content(self, content: str) -> List[str]:
        """Simple content chunking without LLM"""
        words = content.split()
        chunk_size = self.config.chunk_size_tokens if hasattr(self.config, 'chunk_size_tokens') else 600
        
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining words as final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [content]
    
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
            "llm_metrics": self.llm.get_stats() if self.llm else {},
            "config": {
                "batch_size": self.config.batch_size,
                "processing_interval": self.config.processing_interval,
                "human_threshold": self.config.human_classification_threshold
            }
        }
    
    async def generate_thread_summary(self, thread_id: str, force_update: bool = False) -> Optional[ThreadSummary]:
        """Generate or update thread summary using Qwen"""
        if not self.session_maker or not self.llm or not self.llm.ready:
            self.logger.warning("⚠️ Cannot generate thread summary: missing dependencies")
            return None
        
        try:
            with self.session_maker() as session:
                # Get thread emails (last 20 for context)
                emails_query = text("""
                    SELECT e.content, e.subject, e.from_email, e.date_sent, ce.clean_content
                    FROM emails e
                    LEFT JOIN cleaned_emails ce ON e.id = ce.email_id
                    WHERE e.thread_id = :thread_id
                    ORDER BY e.date_sent ASC
                    LIMIT 20
                """)
                emails = session.execute(emails_query, {"thread_id": thread_id}).fetchall()
                
                if not emails:
                    self.logger.warning(f"No emails found for thread {thread_id}")
                    return None
                
                # Check if summary needs updating
                thread_query = text("""
                    SELECT summary_oneliner, summary_version, message_count, last_summary_update
                    FROM threads WHERE id = :thread_id
                """)
                thread_info = session.execute(thread_query, {"thread_id": thread_id}).fetchone()
                
                if not force_update and thread_info and thread_info.summary_oneliner:
                    # Check if summary is recent enough
                    if thread_info.message_count == len(emails):
                        self.logger.debug(f"Thread {thread_id} summary is up to date")
                        return ThreadSummary(
                            thread_id=thread_id,
                            summary_oneliner=thread_info.summary_oneliner,
                            confidence=0.8,  # Existing summary
                            last_email_count=thread_info.message_count
                        )
                
                # Generate new summary
                self.logger.info(f"🧠 Generating thread summary for {thread_id} ({len(emails)} emails)")
                
                # Prepare email content for summarization
                email_contents = []
                for email in emails:
                    content = email.clean_content or email.content
                    if content:
                        # Keep only first 500 chars per email for context
                        truncated = content[:500] + "..." if len(content) > 500 else content
                        email_contents.append(f"From: {email.from_email}\n{truncated}")
                
                # Create summary request
                request = ThreadSummaryRequest(
                    thread_id=thread_id,
                    emails=email_contents,
                    existing_summary=thread_info.summary_oneliner if thread_info else None
                )
                
                # Generate summary with Qwen
                summary = await self._generate_summary_with_qwen(request)
                
                if summary:
                    # Save to database
                    await self._save_thread_summary(session, summary)
                    session.commit()
                    
                    self.logger.info(f"✅ Generated thread summary: {summary.summary_oneliner}")
                    return summary
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate thread summary for {thread_id}: {e}")
            return None
    
    async def _generate_summary_with_qwen(self, request: ThreadSummaryRequest) -> Optional[ThreadSummary]:
        """Use Qwen to generate thread summary with structured output"""
        
        # Build context from emails
        emails_text = "\n\n".join(request.emails)
        existing_context = f"\nPrevious summary: {request.existing_summary}" if request.existing_summary else ""
        
        prompt = f"""Analyze this email thread and provide a concise summary suitable for notifications.

{existing_context}

THREAD EMAILS:
{emails_text}

Create a JSON response with:
{{
    "summary_oneliner": "Short descriptive summary (10-15 words max)",
    "key_entities": ["person1", "topic1", "date1"],
    "thread_mood": "planning|urgent|social|work|problem_solving|informational",
    "action_items": ["action1", "action2"],
    "confidence": 0.8
}}

Focus on the main topic, key participants, and any decisions or actions needed."""

        try:
            response = self.llm.model(
                prompt,
                max_tokens=256,
                temperature=0.2,
                stop=["THREAD EMAILS:", "}}"],
                echo=False
            )
            
            response_text = response['choices'][0]['text'].strip()
            
            # Parse JSON response
            import json
            data = self._parse_summary_response(response_text)
            
            return ThreadSummary(
                thread_id=request.thread_id,
                summary_oneliner=data.get("summary_oneliner", "Email conversation"),
                key_entities=data.get("key_entities", []),
                thread_mood=data.get("thread_mood", "informational"),
                action_items=data.get("action_items", []),
                confidence=float(data.get("confidence", 0.7)),
                last_email_count=len(request.emails)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Qwen summary generation failed: {e}")
            return None
    
    def _parse_summary_response(self, response_text: str) -> dict:
        """Parse JSON summary response from Qwen"""
        import json
        
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                return json.loads(json_text)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse summary JSON: {e}")
            # Fallback to basic parsing
            return {
                "summary_oneliner": "Email conversation",
                "key_entities": [],
                "thread_mood": "informational", 
                "action_items": [],
                "confidence": 0.5
            }
    
    async def _save_thread_summary(self, session: Session, summary: ThreadSummary):
        """Save thread summary to database"""
        
        # Generate embedding for the summary if embedding service is available
        summary_embedding = None
        if self.embeddings:
            try:
                embedding = await self.embeddings.generate_embedding(summary.summary_oneliner)
                summary_embedding = embedding.tolist()
            except Exception as e:
                self.logger.warning(f"Failed to generate summary embedding: {e}")
        
        # Update threads table
        update_query = text("""
            UPDATE threads SET 
                summary_oneliner = :summary,
                summary_embedding = :embedding,
                key_entities = :entities,
                thread_mood = :mood,
                action_items = :actions,
                last_summary_update = NOW(),
                summary_version = COALESCE(summary_version, 0) + 1
            WHERE id = :thread_id
        """)
        
        session.execute(update_query, {
            "thread_id": summary.thread_id,
            "summary": summary.summary_oneliner,
            "embedding": summary_embedding,
            "entities": summary.key_entities,
            "mood": summary.thread_mood,
            "actions": summary.action_items
        })

    def stop_processing(self):
        """Stop background processing"""
        self.logger.info("🛑 Stopping email processor...")
        self.is_processing = False