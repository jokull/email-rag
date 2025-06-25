"""
Qwen-based conversation summarization service
Based on legacy ai-processor patterns with modern improvements
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import llama-cpp-python for Qwen
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

class ThreadSummary(BaseModel):
    """Thread summary model based on legacy patterns"""
    thread_id: str
    summary_oneliner: str = Field(..., description="10-15 words max for notifications")
    key_entities: List[str] = Field(default_factory=list, description="People, topics, dates")
    thread_mood: Literal["planning", "urgent", "social", "work", "problem_solving", "informational"] = "informational"
    action_items: List[str] = Field(default_factory=list, description="Up to 5 action items")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    last_email_count: int = 0

@dataclass
class EmailForSummary:
    """Email data for summarization"""
    from_email: str
    date_sent: datetime
    subject: str
    body_text: str
    
class QwenConversationSummarizer:
    """
    Qwen-based conversation summarizer following legacy patterns
    """
    
    def __init__(self, model_path: Optional[str] = None):
        import os
        self.model_path = model_path or os.getenv("QWEN_MODEL_PATH", "/app/models/qwen2.5-0.5b-instruct-q4_0.gguf")
        self.model = None
        self.ready = False
        
        # Performance tracking
        self.total_summarizations = 0
        self.total_time_ms = 0.0
        
        # Qwen parameters from legacy
        self.max_tokens = 512
        self.temperature = 0.1
        self.context_window = 2048
        
        logger.info(f"🤖 Initializing Qwen summarizer with model: {self.model_path}")
    
    def initialize(self) -> bool:
        """Initialize the Qwen model"""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("❌ llama-cpp-python not available. Install with: pip install llama-cpp-python")
            return False
        
        if not Path(self.model_path).exists():
            logger.warning(f"⚠️ Qwen model not found at: {self.model_path}")
            logger.info(f"💡 To enable Qwen summarization, download the model:")
            logger.info(f"   wget -O {self.model_path} https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf")
            # Still return True but mark as ready=False for fallback mode
            self.ready = False
            return True
        
        try:
            logger.info(f"📥 Loading Qwen model: {self.model_path}")
            start_time = time.time()
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_window,  # Context window
                n_threads=4,  # CPU threads
                verbose=False,
                use_mmap=True,  # Memory mapping for efficiency
                use_mlock=False,
                n_gpu_layers=0  # Use CPU only (more reliable in containers)
            )
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Qwen model loaded successfully in {load_time:.0f}ms")
            
            self.ready = True
            
            # Test the model
            self._test_model()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen model: {e}")
            return False
    
    def _test_model(self):
        """Test the model with a simple summarization"""
        try:
            test_emails = [
                EmailForSummary(
                    from_email="test@example.com",
                    date_sent=datetime.now(),
                    subject="Test meeting",
                    body_text="Let's schedule a meeting for next week to discuss the project."
                )
            ]
            
            start_time = time.time()
            result = self.summarize_conversation("test_thread", test_emails)
            test_time = (time.time() - start_time) * 1000
            
            if result:
                logger.info(f"🧪 Model test successful: '{result.summary_oneliner}' in {test_time:.0f}ms")
            else:
                logger.warning("⚠️ Model test returned no result")
                
        except Exception as e:
            logger.warning(f"⚠️ Model test failed: {e}")
    
    def summarize_conversation(self, thread_id: str, emails: List[EmailForSummary], 
                             existing_summary: Optional[ThreadSummary] = None) -> Optional[ThreadSummary]:
        """
        Generate conversation summary using Qwen
        Following legacy patterns for prompt structure and output validation
        """
        if not self.ready:
            logger.debug("Qwen model not ready, using enhanced fallback")
            return self._generate_enhanced_fallback_summary(thread_id, emails)
        
        if not emails:
            return None
        
        start_time = time.time()
        
        try:
            # Prepare context from emails (max 20, following legacy pattern)
            emails_for_context = emails[-20:] if len(emails) > 20 else emails
            
            # Build existing context if available
            existing_context = ""
            if existing_summary:
                existing_context = f"Previous summary: {existing_summary.summary_oneliner}\n"
                existing_context += f"Previous entities: {', '.join(existing_summary.key_entities)}\n"
                existing_context += f"Previous mood: {existing_summary.thread_mood}\n"
            
            # Prepare emails text
            emails_text = self._prepare_emails_for_prompt(emails_for_context)
            
            # Create prompt following legacy pattern
            prompt = self._create_summarization_prompt(existing_context, emails_text)
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                stop=["\\n\\n", "###"]
            )
            
            if not response or 'choices' not in response or not response['choices']:
                logger.warning("Empty response from Qwen")
                return None
            
            # Parse response
            response_text = response['choices'][0]['text'].strip()
            summary = self._parse_qwen_response(thread_id, response_text, len(emails))
            
            if summary:
                processing_time = (time.time() - start_time) * 1000
                self.total_summarizations += 1
                self.total_time_ms += processing_time
                
                logger.info(f"✅ Generated summary for {thread_id}: '{summary.summary_oneliner}' in {processing_time:.1f}ms")
                return summary
            else:
                logger.warning(f"Failed to parse Qwen response for {thread_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Qwen summarization failed for {thread_id}: {e}")
            return None
    
    def _prepare_emails_for_prompt(self, emails: List[EmailForSummary]) -> str:
        """Prepare emails text for prompt, following legacy patterns"""
        email_texts = []
        
        for i, email in enumerate(emails):
            # Get sender name (first part of email)
            sender = email.from_email.split('@')[0] if email.from_email else "Unknown"
            
            # Format date
            date_str = email.date_sent.strftime("%Y-%m-%d %H:%M") if email.date_sent else "Unknown"
            
            # Truncate body to 500 chars (legacy pattern)
            body = (email.body_text or "")[:500]
            if len(email.body_text or "") > 500:
                body += "..."
            
            email_text = f"Email {i+1} ({date_str}) from {sender}:\nSubject: {email.subject}\n{body}\n"
            email_texts.append(email_text)
        
        return "\\n".join(email_texts)
    
    def _create_summarization_prompt(self, existing_context: str, emails_text: str) -> str:
        """Create summarization prompt following legacy patterns"""
        
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

Focus on the main topic, key participants, and any decisions or actions needed. Keep the summary under 15 words."""
        
        return prompt
    
    def _parse_qwen_response(self, thread_id: str, response_text: str, email_count: int) -> Optional[ThreadSummary]:
        """Parse Qwen JSON response into ThreadSummary"""
        try:
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning(f"No JSON found in Qwen response: {response_text[:100]}")
                return None
            
            json_text = response_text[json_start:json_end]
            parsed = json.loads(json_text)
            
            # Validate and create ThreadSummary
            summary = ThreadSummary(
                thread_id=thread_id,
                summary_oneliner=parsed.get('summary_oneliner', 'Email conversation'),
                key_entities=parsed.get('key_entities', [])[:10],  # Limit entities
                thread_mood=parsed.get('thread_mood', 'informational'),
                action_items=parsed.get('action_items', [])[:5],  # Limit actions
                confidence=float(parsed.get('confidence', 0.8)),
                last_email_count=email_count
            )
            
            # Validate summary length
            if len(summary.summary_oneliner) > 100:
                summary.summary_oneliner = summary.summary_oneliner[:97] + "..."
            
            return summary
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse Qwen response: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Try to extract basic summary from response
            return self._extract_fallback_summary(thread_id, response_text, email_count)
    
    def _extract_fallback_summary(self, thread_id: str, response_text: str, email_count: int) -> Optional[ThreadSummary]:
        """Extract fallback summary from malformed response"""
        try:
            # Simple text extraction fallback
            lines = response_text.split('\\n')
            summary_line = None
            
            for line in lines:
                if 'summary' in line.lower() and len(line.strip()) > 10:
                    summary_line = line.strip()
                    break
            
            if not summary_line:
                summary_line = "Email conversation"
            
            # Clean up the summary
            summary_line = summary_line.replace('summary_oneliner', '').replace(':', '').replace('"', '').strip()
            
            return ThreadSummary(
                thread_id=thread_id,
                summary_oneliner=summary_line[:80],  # Truncate
                key_entities=[],
                thread_mood="informational",
                action_items=[],
                confidence=0.5,  # Lower confidence for fallback
                last_email_count=email_count
            )
            
        except Exception as e:
            logger.error(f"Fallback summary extraction failed: {e}")
            return None
    
    def _generate_enhanced_fallback_summary(self, thread_id: str, emails: List[EmailForSummary]) -> ThreadSummary:
        """
        Generate enhanced fallback summary when Qwen model is not available
        Uses improved heuristics based on email content analysis
        """
        if not emails:
            return ThreadSummary(
                thread_id=thread_id,
                summary_oneliner="Empty conversation",
                confidence=0.1,
                last_email_count=0
            )
        
        # Analyze email content for patterns
        all_text = " ".join([email.body_text + " " + email.subject for email in emails]).lower()
        
        # Participant analysis
        participants = set()
        for email in emails:
            if email.from_email:
                name = email.from_email.split('@')[0]
                participants.add(name)
        
        participant_list = list(participants)[:3]  # Limit to 3 main participants
        
        # Pattern-based mood and summary detection
        mood = "informational"
        summary_template = "Conversation"
        key_entities = participant_list.copy()
        action_items = []
        
        # Enhanced pattern matching
        if any(word in all_text for word in ['meeting', 'meet', 'zoom', 'call', 'conference']):
            mood = "planning"
            summary_template = "Meeting discussion"
            if 'schedule' in all_text or 'when' in all_text:
                action_items.append("Schedule meeting")
        
        elif any(word in all_text for word in ['urgent', 'asap', 'emergency', 'important', 'critical']):
            mood = "urgent"
            summary_template = "Urgent matter"
        
        elif any(word in all_text for word in ['problem', 'issue', 'bug', 'error', 'help', 'fix']):
            mood = "problem_solving"
            summary_template = "Problem resolution"
            action_items.append("Resolve issue")
        
        elif any(word in all_text for word in ['plan', 'schedule', 'organize', 'arrange']):
            mood = "planning"
            summary_template = "Planning discussion"
        
        elif any(word in all_text for word in ['thanks', 'thank you', 'appreciate']):
            mood = "social"
            summary_template = "Appreciation exchange"
        
        elif any(word in all_text for word in ['work', 'project', 'task', 'deadline', 'business']):
            mood = "work"
            summary_template = "Work discussion"
        
        # Extract subject for context
        main_subject = emails[0].subject if emails else "Unknown topic"
        if main_subject.lower().startswith(('re:', 'fwd:', 'fw:')):
            # Clean up subject
            main_subject = main_subject[3:].strip() if len(main_subject) > 3 else main_subject
        
        # Generate summary based on analysis
        if len(emails) == 1:
            summary = f"{summary_template} about {main_subject}"
        else:
            summary = f"{summary_template} with {len(participant_list)} people about {main_subject}"
        
        # Truncate summary to keep it concise
        if len(summary) > 80:
            summary = summary[:77] + "..."
        
        # Extract additional entities (dates, topics)
        import re
        date_matches = re.findall(r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}/\d{1,2})\b', all_text)
        key_entities.extend(date_matches[:2])  # Add up to 2 dates
        
        return ThreadSummary(
            thread_id=thread_id,
            summary_oneliner=summary,
            key_entities=key_entities[:5],  # Limit to 5 entities
            thread_mood=mood,
            action_items=action_items[:3],  # Limit to 3 actions
            confidence=0.6,  # Reasonable confidence for enhanced fallback
            last_email_count=len(emails)
        )
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        avg_time = self.total_time_ms / self.total_summarizations if self.total_summarizations > 0 else 0
        
        return {
            "ready": self.ready,
            "total_summarizations": self.total_summarizations,
            "total_time_ms": self.total_time_ms,
            "average_time_ms": avg_time,
            "summarizations_per_second": 1000 / avg_time if avg_time > 0 else 0,
            "model_loaded": self.model is not None
        }
    
    def health_check(self) -> bool:
        """Check if summarizer is ready and working"""
        if not self.ready or not self.model:
            return False
        
        try:
            # Quick test
            test_emails = [EmailForSummary(
                from_email="test@example.com",
                date_sent=datetime.now(),
                subject="Health check",
                body_text="Test message"
            )]
            result = self.summarize_conversation("health_check", test_emails)
            return result is not None
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

# Global summarizer instance
_summarizer_instance = None

def get_qwen_summarizer() -> QwenConversationSummarizer:
    """Get singleton summarizer instance"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = QwenConversationSummarizer()
    return _summarizer_instance

def initialize_qwen_summarizer() -> bool:
    """Initialize the global summarizer"""
    summarizer = get_qwen_summarizer()
    return summarizer.initialize()