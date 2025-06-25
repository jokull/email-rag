"""
LLM-powered conversation summarization service
Using Ollama directly for reliable model access
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import Ollama library
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

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

class OllamaConversationSummarizer:
    """
    LLM-powered conversation summarizer using Ollama directly
    Simpler and more reliable than the LLM library approach
    """
    
    def __init__(self, model_name: Optional[str] = None):
        import os
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "qwen2.5:0.5b-instruct")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = None
        self.ready = False
        
        # Performance tracking
        self.total_summarizations = 0
        self.total_time_ms = 0.0
        
        logger.info(f"🤖 Initializing Ollama summarizer with model: {self.model_name}")
    
    def initialize(self) -> bool:
        """Initialize the Ollama client"""
        if not OLLAMA_AVAILABLE:
            logger.error("❌ Ollama library not available. Install with: pip install ollama")
            return False
        
        try:
            logger.info(f"📥 Connecting to Ollama: {self.ollama_base_url}")
            logger.info(f"🤖 Using model: {self.model_name}")
            start_time = time.time()
            
            # Create Ollama client
            self.client = ollama.Client(host=self.ollama_base_url)
            
            # Test connection by listing models
            models = self.client.list()
            available_models = [model.model for model in models.models]
            logger.info(f"📋 Available models: {available_models}")
            
            # Check if our model is available
            if self.model_name not in available_models:
                logger.error(f"❌ Model {self.model_name} not found in available models")
                return False
            
            connect_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Ollama client connected successfully in {connect_time:.0f}ms")
            
            self.ready = True
            
            # Test the model
            self._test_model()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            logger.info("💡 Make sure Ollama is running and the model is installed. Try:")
            logger.info(f"   curl {self.ollama_base_url}/api/tags")
            logger.info(f"   ollama pull {self.model_name}")
            
            self.ready = False
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
        Generate conversation summary using Ollama
        Direct API access for reliability
        """
        if not self.ready:
            raise RuntimeError(f"Ollama client not ready. Cannot generate summary without Ollama.")
        
        if not emails:
            return None
        
        start_time = time.time()
        
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
        
        # Generate response using Ollama directly
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 512
            }
        )
        response_text = response.message.content.strip()
        
        # Parse response
        summary = self._parse_llm_response(thread_id, response_text, len(emails))
        
        if summary:
            processing_time = (time.time() - start_time) * 1000
            self.total_summarizations += 1
            self.total_time_ms += processing_time
            
            logger.info(f"✅ Generated Ollama summary for {thread_id}: '{summary.summary_oneliner}' in {processing_time:.1f}ms")
            return summary
        else:
            raise RuntimeError(f"Failed to parse Ollama response for {thread_id}")
                
    
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
        
        return "\n".join(email_texts)
    
    def _create_summarization_prompt(self, existing_context: str, emails_text: str) -> str:
        """Create simple summarization prompt for mobile notifications"""
        
        prompt = f"""What is this email conversation about? Respond with just the topic in 2-5 words, like "Thailand vacation planning" or "Invoice payment reminder" or "Meeting schedule change". No prefixes like "Email about" or "Conversation about".

{emails_text}

Topic:"""
        
        return prompt
    
    def _parse_llm_response(self, thread_id: str, response_text: str, email_count: int) -> Optional[ThreadSummary]:
        """Parse simple text response into ThreadSummary"""
        try:
            # Clean up the response text
            summary_text = response_text.strip()
            
            # Limit to 200 characters as requested
            if len(summary_text) > 200:
                summary_text = summary_text[:197] + "..."
            
            # Create simplified ThreadSummary with just the summary
            summary = ThreadSummary(
                thread_id=thread_id,
                summary_oneliner=summary_text,
                key_entities=[],  # Simplified - no entities
                thread_mood="informational",  # Default
                action_items=[],  # Simplified - no action items
                confidence=0.8,  # Default confidence
                last_email_count=email_count
            )
            
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # No fallback - fail if we can't parse
            raise ValueError(f"Failed to parse LLM response: {e}")
    
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        avg_time = self.total_time_ms / self.total_summarizations if self.total_summarizations > 0 else 0
        
        return {
            "ready": self.ready,
            "total_summarizations": self.total_summarizations,
            "total_time_ms": self.total_time_ms,
            "average_time_ms": avg_time,
            "summarizations_per_second": 1000 / avg_time if avg_time > 0 else 0,
            "client_connected": self.client is not None
        }
    
    def health_check(self) -> bool:
        """Check if summarizer is ready and working"""
        if not self.ready or not self.client:
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

def get_llm_summarizer() -> OllamaConversationSummarizer:
    """Get singleton summarizer instance"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = OllamaConversationSummarizer()
    return _summarizer_instance

def initialize_llm_summarizer() -> bool:
    """Initialize the global summarizer"""
    summarizer = get_llm_summarizer()
    return summarizer.initialize()