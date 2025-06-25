"""
Email Threading Service
Implements RFC 5322 compliant email threading with AI-powered conversation summarization
"""

import hashlib
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from sqlalchemy.orm import Session
from sqlalchemy import func

from models import Message, Conversation
from database import get_db_session
from llm_summarizer import get_llm_summarizer, EmailForSummary, ThreadSummary

logger = logging.getLogger(__name__)


class EmailThreadingService:
    """
    Service for threading personal emails and generating AI conversation summaries
    """
    
    def __init__(self):
        self.llm_summarizer = get_llm_summarizer()  # For conversation summarization
        
        # Initialize LLM summarizer
        logger.info("🤖 Initializing LLM summarizer for threading service...")
        if not self.llm_summarizer.initialize():
            logger.error("❌ Failed to initialize LLM summarizer")
        else:
            logger.info("✅ LLM summarizer initialized successfully")
    
    def normalize_subject(self, subject: str) -> str:
        """
        Normalize email subject by removing Re:, RE:, Fwd:, etc.
        """
        if not subject:
            return "No Subject"
        
        # Remove common reply/forward prefixes (case insensitive)
        normalized = re.sub(r'^(re|RE|fwd|FWD|fw|FW):\s*', '', subject.strip())
        normalized = re.sub(r'^\[.*?\]\s*', '', normalized)  # Remove [list] prefixes
        normalized = normalized.strip()
        
        return normalized if normalized else "No Subject"
    
    def generate_thread_id(self, genesis_message_id: str, normalized_subject: str) -> str:
        """
        Generate a unique thread ID based on genesis message and subject
        """
        # Create deterministic thread ID from genesis message + normalized subject
        thread_data = f"{genesis_message_id}::{normalized_subject}"
        thread_hash = hashlib.sha256(thread_data.encode()).hexdigest()[:16]
        return f"thread_{thread_hash}"
    
    def find_genesis_message(self, message: Message, session: Session) -> Optional[Message]:
        """
        Find the genesis (first) message in a thread by following reply chains
        """
        if not message.in_reply_to:
            # This is already a genesis message
            return message
        
        # Follow the reply chain backwards
        current_reply_to = message.in_reply_to
        visited = set()  # Prevent infinite loops
        
        while current_reply_to and current_reply_to not in visited:
            visited.add(current_reply_to)
            
            # Look for the message this one replies to
            parent = session.query(Message).filter(
                Message.message_id == current_reply_to,
                Message.category == 'personal'  # Only personal emails
            ).first()
            
            if parent:
                if not parent.in_reply_to:
                    # Found the genesis message
                    return parent
                current_reply_to = parent.in_reply_to
            else:
                # Parent not found, current message becomes genesis
                break
        
        # If we can't find the actual genesis, treat current message as genesis
        return message
    
    def get_thread_messages(self, thread_id: str, session: Session) -> List[Message]:
        """
        Get all messages in a thread, ordered by date (most recent first for processing)
        Only returns personal messages since we only create conversations for personal emails
        """
        return session.query(Message).filter(
            Message.thread_id == thread_id,
            Message.category == 'personal'
        ).order_by(Message.date_sent.desc()).all()  # Most recent first
    
    def update_conversation_metadata(self, conversation: Conversation, messages: List[Message], session: Session):
        """
        Update conversation metadata based on current messages
        """
        if not messages:
            return
        
        # Update basic metadata
        conversation.message_count = len(messages)
        conversation.first_message_date = min(msg.date_sent for msg in messages if msg.date_sent)
        conversation.last_message_date = max(msg.date_sent for msg in messages if msg.date_sent)
        
        # Collect unique participants
        participants = set()
        for message in messages:
            # Add sender
            if message.from_email:
                participants.add(message.from_email)
            
            # Add recipients
            for email in (message.to_emails or []):
                participants.add(email)
            for email in (message.cc_emails or []):
                participants.add(email)
        
        # Store as list of participant objects with metadata
        participant_list = []
        for email in participants:
            # Get name from any message's participants
            name = None
            for message in messages:
                for participant in (message.participants or []):
                    if participant.get('email') == email:
                        name = participant.get('name')
                        break
                if name:
                    break
            
            participant_list.append({
                'email': email,
                'name': name
            })
        
        conversation.participants = participant_list
        conversation.updated_at = datetime.utcnow()
    
    def generate_conversation_summary(self, conversation: Conversation, messages: List[Message]) -> Tuple[str, List[str]]:
        """
        Generate AI-powered conversation summary using Qwen and extract key topics
        """
        if not messages:
            return "Empty conversation", []
        
        # Convert messages to EmailForSummary format for LLM
        emails_for_summary = []
        for msg in messages:
            emails_for_summary.append(EmailForSummary(
                from_email=msg.from_email or "unknown@example.com",
                date_sent=msg.date_sent or datetime.utcnow(),
                subject=msg.subject or "",
                body_text=msg.body_text or ""
            ))
        
        # Use LLM to generate structured summary
        thread_summary = self.llm_summarizer.summarize_conversation(
            conversation.thread_id,
            emails_for_summary
        )
        
        if thread_summary:
            # Extract summary and topics from LLM response
            summary = thread_summary.summary_oneliner
            key_topics = thread_summary.key_entities
            
            logger.info(f"Generated LLM summary for {conversation.thread_id}: {summary}")
            return summary, key_topics
        else:
            # No fallback - fail if LLM doesn't work
            raise RuntimeError(f"LLM summarization returned None for {conversation.thread_id}")
    
    def _extract_key_topics(self, conversation_text: str) -> List[str]:
        """
        Extract key topics from conversation (simple keyword extraction for now)
        """
        # Simple keyword extraction (can be enhanced with NER later)
        text_lower = conversation_text.lower()
        topics = []
        
        # Look for common business/personal topics
        topic_patterns = {
            'meeting': ['meeting', 'call', 'zoom', 'conference'],
            'project': ['project', 'work', 'task', 'deadline'],
            'planning': ['plan', 'schedule', 'calendar', 'date'],
            'problem': ['problem', 'issue', 'bug', 'error', 'help'],
            'payment': ['payment', 'invoice', 'money', 'cost', 'price'],
            'travel': ['travel', 'trip', 'flight', 'hotel', 'vacation'],
            'family': ['family', 'kids', 'children', 'parent'],
            'technical': ['code', 'software', 'server', 'website', 'app']
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics[:5]  # Limit to 5 topics
    
    def process_message_threading(self, message: Message, session: Session) -> Optional[str]:
        """
        Process threading for a single personal message
        Only create conversations for threads with at least one personal message
        Returns the thread_id if successful
        """
        if message.category != 'personal':
            logger.debug(f"Skipping threading for non-personal message {message.id}")
            return None
        
        try:
            # Normalize subject
            normalized_subject = self.normalize_subject(message.subject)
            
            # Find genesis message
            genesis = self.find_genesis_message(message, session)
            
            # Generate thread ID
            thread_id = self.generate_thread_id(genesis.message_id, normalized_subject)
            
            # Update message with thread ID
            message.thread_id = thread_id
            
            # Find or create conversation (we know this thread has at least one personal message)
            conversation = session.query(Conversation).filter(
                Conversation.thread_id == thread_id
            ).first()
            
            if not conversation:
                # Create new conversation only for threads with personal messages
                conversation = Conversation(
                    thread_id=thread_id,
                    genesis_message_id=genesis.message_id,
                    subject_normalized=normalized_subject
                )
                session.add(conversation)
                session.flush()  # Get ID
            
            # Get all messages in this thread
            thread_messages = self.get_thread_messages(thread_id, session)
            
            # Update conversation metadata
            self.update_conversation_metadata(conversation, thread_messages, session)
            
            # Generate/update conversation summary
            summary, key_topics = self.generate_conversation_summary(conversation, thread_messages)
            conversation.summary = summary
            conversation.key_topics = key_topics
            conversation.summary_generated_at = datetime.utcnow()
            
            session.commit()
            
            logger.info(f"Threaded message {message.id} into conversation {thread_id} "
                       f"({len(thread_messages)} messages)")
            
            return thread_id
            
        except Exception as e:
            logger.error(f"Failed to process threading for message {message.id}: {e}")
            session.rollback()
            raise
    
    def process_unthreaded_personal_messages(self, limit: int = 100) -> int:
        """
        Process threading for personal messages that haven't been threaded yet
        Process from most recent to oldest for better performance and relevance
        """
        threaded_count = 0
        
        with get_db_session() as session:
            # Find personal messages without thread_id, ordered by most recent first
            unthreaded = session.query(Message).filter(
                Message.category == 'personal',
                Message.thread_id.is_(None),
                Message.classified_at.isnot(None)  # Only classified messages
            ).order_by(Message.date_sent.desc()).limit(limit).all()  # Most recent first
            
            logger.info(f"Found {len(unthreaded)} unthreaded personal messages to process")
            
            for message in unthreaded:
                try:
                    thread_id = self.process_message_threading(message, session)
                    if thread_id:
                        threaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to thread message {message.id}: {e}")
                    raise
        
        return threaded_count
    
    def get_threading_stats(self) -> Dict[str, int]:
        """
        Get threading statistics
        """
        with get_db_session() as session:
            total_personal = session.query(func.count(Message.id)).filter(
                Message.category == 'personal'
            ).scalar()
            
            threaded_messages = session.query(func.count(Message.id)).filter(
                Message.category == 'personal',
                Message.thread_id.isnot(None)
            ).scalar()
            
            total_conversations = session.query(func.count(Conversation.id)).scalar()
            
            summarized_conversations = session.query(func.count(Conversation.id)).filter(
                Conversation.summary.isnot(None)
            ).scalar()
            
            return {
                'total_personal_messages': total_personal,
                'threaded_messages': threaded_messages,
                'pending_threading': total_personal - threaded_messages,
                'total_conversations': total_conversations,
                'summarized_conversations': summarized_conversations,
                'pending_summarization': total_conversations - summarized_conversations
            }
    
    def regenerate_conversation_summaries(self, limit: int = 50) -> int:
        """
        Regenerate AI summaries for existing conversations (useful after upgrading summarization)
        """
        regenerated_count = 0
        
        with get_db_session() as session:
            # Find conversations that need better summaries, prioritize recent ones
            conversations = session.query(Conversation).filter(
                Conversation.message_count > 0
            ).order_by(Conversation.last_message_date.desc()).limit(limit).all()  # Most recent first
            
            logger.info(f"Regenerating summaries for {len(conversations)} conversations")
            
            for conversation in conversations:
                try:
                    # Get messages for this conversation
                    messages = self.get_thread_messages(conversation.thread_id, session)
                    
                    if messages:
                        # Generate new summary
                        summary, key_topics = self.generate_conversation_summary(conversation, messages)
                        
                        # Update conversation
                        conversation.summary = summary
                        conversation.key_topics = key_topics
                        conversation.summary_generated_at = datetime.utcnow()
                        
                        session.commit()
                        regenerated_count += 1
                        
                        logger.info(f"Regenerated summary for conversation {conversation.thread_id}: {summary}")
                        
                except Exception as e:
                    logger.error(f"Failed to regenerate summary for conversation {conversation.thread_id}: {e}")
                    session.rollback()
                    raise
        
        return regenerated_count