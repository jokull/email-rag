"""
LLM-based email classifier using the same Ollama infrastructure as conversation summarization
Much more reliable than the broken SetFit model
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

@dataclass
class LLMClassificationResult:
    """Result of LLM email classification"""
    category: str  # personal/promotional/automated
    confidence: float  # 0.0 to 1.0
    processing_time_ms: float
    reasoning: str  # LLM's explanation


class LLMEmailClassifier:
    """
    Email classifier using Simon Willison's LLM library with correspondence intelligence
    """
    
    def __init__(self, model_name: Optional[str] = None):
        import os
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.my_email = os.getenv("MY_EMAIL", "jokull@solberg.is")  # Your email for correspondence analysis
        self.model = None
        self.ready = False
        
        # Performance tracking
        self.total_classifications = 0
        self.total_time_ms = 0.0
        
        logger.info(f"🤖 Initializing LLM email classifier with model: {self.model_name}")
    
    def initialize(self) -> bool:
        """Initialize the LLM client"""
        if not LLM_AVAILABLE:
            logger.error("❌ LLM library not available. Install with: uv add llm")
            return False
        
        try:
            # Initialize the model using llm library
            self.model = llm.get_model(self.model_name)
            
            logger.info(f"✅ LLM email classifier ready with model: {self.model_name}")
            self.ready = True
            
            # Test the classifier
            self._test_classifier()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            return False
    
    def _test_classifier(self):
        """Test the classifier with simple examples"""
        try:
            test_result = self.classify_email(
                from_email="test@gmail.com",
                subject="Test message",
                body="Hi honey, can you pick up milk? Love you"
            )
            if test_result and test_result.category == "personal":
                logger.info(f"🧪 Classifier test successful: {test_result.category}")
            else:
                logger.warning(f"⚠️ Classifier test unexpected result: {test_result}")
        except Exception as e:
            logger.warning(f"⚠️ Classifier test failed: {e}")
    
    def classify_email(self, from_email: str, subject: str = "", body: str = "") -> Optional[LLMClassificationResult]:
        """
        Classify email using LLM with correspondence intelligence
        """
        if not self.ready:
            raise RuntimeError(f"LLM classifier not ready. Cannot classify without LLM.")
        
        start_time = time.time()
        
        # First line of defense: obvious automated email patterns
        obvious_result = self._check_obvious_automated(from_email)
        if obvious_result:
            logger.debug(f"🚫 Obvious automated email detected: {from_email}")
            obvious_result.processing_time_ms = (time.time() - start_time) * 1000
            return obvious_result
        
        # Create classification prompt with correspondence intelligence
        prompt = self._create_classification_prompt(from_email, subject, body)
        
        # Generate response using LLM Python library
        try:
            response = self.model.prompt(prompt)
            response_text = response.text().strip()
            
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}")
        
        # Parse response
        parsed_result = self._parse_classification_response(from_email, response_text)
        
        if parsed_result:
            processing_time = (time.time() - start_time) * 1000
            parsed_result.processing_time_ms = processing_time
            
            self.total_classifications += 1
            self.total_time_ms += processing_time
            
            logger.debug(f"📧 LLM classified {from_email}: {parsed_result.category} ({parsed_result.confidence:.2f}) in {processing_time:.1f}ms")
            return parsed_result
        else:
            raise RuntimeError(f"Failed to parse LLM classification response")
    
    def _check_obvious_automated(self, from_email: str) -> Optional[LLMClassificationResult]:
        """First line of defense: check for obvious automated email patterns"""
        from_email_lower = from_email.lower()
        
        # Exact prefix patterns for automated emails
        automated_prefixes = [
            'noreply@', 'no-reply@', 'notifications@', 'notification@',
            'donotreply@', 'do-not-reply@', 'support@', 'help@',
            'alerts@', 'alert@', 'system@', 'admin@', 'postmaster@',
            'outbound@', 'hello@'
        ]
        
        # Check exact prefixes
        for pattern in automated_prefixes:
            if from_email_lower.startswith(pattern):
                return LLMClassificationResult(
                    category="automated",
                    confidence=0.95,
                    processing_time_ms=0.0,
                    reasoning=f"Obvious automated pattern: {pattern}"
                )
        
        # Check contains patterns
        automated_contains = ['noreply', 'invoice']
        
        for pattern in automated_contains:
            if pattern in from_email_lower:
                return LLMClassificationResult(
                    category="automated",
                    confidence=0.95,
                    processing_time_ms=0.0,
                    reasoning=f"Obvious automated pattern: contains '{pattern}'"
                )
        
        return None
    
    def _create_classification_prompt(self, from_email: str, subject: str, body: str) -> str:
        """Create balanced classification prompt with correspondence as context"""
        
        # Get correspondence intelligence as subtle context
        outbound_count = self._get_outbound_email_count(from_email)
        
        # Build subtle correspondence context
        if outbound_count == 0:
            correspondence_note = "Note: I've never sent emails to this address"
        elif outbound_count == 1:
            correspondence_note = f"Note: I've sent {outbound_count} email to this address"
        else:
            correspondence_note = f"Note: I've sent {outbound_count} emails to this address"
        
        prompt = f"""Classify this email into one of three categories:

From: {from_email}
Subject: {subject}
Body: {body[:300]}

{correspondence_note}

Categories:
• PERSONAL: Real people I know personally - friends, family, colleagues I work with directly
• PROMOTIONAL: Marketing emails, newsletters, sales pitches, promotional content
• AUTOMATED: System notifications, alerts, receipts, automated service messages

Consider the sender, subject, and content. Focus on the nature and tone of the message. Is this from a real person reaching out personally, a marketing campaign, or an automated system?

Classification (respond with exactly one word):"""
        
        return prompt
    
    def _get_outbound_email_count(self, to_email: str) -> int:
        """Get how many times I have emailed this address from IMAP Sent mailbox"""
        try:
            from database import get_db_session
            from models import ImapMessage, ImapMailbox
            from sqlalchemy import func, and_
            
            with get_db_session() as session:
                # Count emails in Sent mailbox where I emailed this address
                from sqlalchemy import text
                count = session.query(func.count(ImapMessage.id)).join(
                    ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id
                ).filter(
                    and_(
                        ImapMailbox.name == 'Sent',
                        # Check if to_email appears in the envelope JSON
                        text(f"envelope::text LIKE '%{to_email}%'")
                    )
                ).scalar() or 0
                
                return count
                
        except Exception as e:
            logger.warning(f"Failed to get outbound email count for {to_email}: {e}")
            return 0
    
    def _parse_classification_response(self, from_email: str, response_text: str) -> Optional[LLMClassificationResult]:
        """Parse LLM classification response"""
        try:
            response_text = response_text.strip().lower()
            
            # Simple parsing - just look for the category word
            if "personal" in response_text:
                category = "personal"
                confidence = 0.85
            elif "promotional" in response_text:
                category = "promotional"
                confidence = 0.85
            elif "automated" in response_text:
                category = "automated"
                confidence = 0.85
            else:
                logger.warning(f"Unclear category from LLM: '{response_text}', defaulting to automated")
                category = "automated"
                confidence = 0.60
            
            return LLMClassificationResult(
                category=category,
                confidence=confidence,
                processing_time_ms=0.0,  # Will be set by caller
                reasoning=f"LLM response: {response_text}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM classification response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return None
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        avg_time = self.total_time_ms / self.total_classifications if self.total_classifications > 0 else 0
        
        return {
            "ready": self.ready,
            "model": self.model_name,
            "total_classifications": self.total_classifications,
            "total_time_ms": self.total_time_ms,
            "average_time_ms": avg_time,
            "classifications_per_second": 1000 / avg_time if avg_time > 0 else 0,
            "my_email": self.my_email
        }


# Global classifier instance
_llm_classifier_instance = None

def get_llm_classifier() -> LLMEmailClassifier:
    """Get singleton LLM classifier instance"""
    global _llm_classifier_instance
    if _llm_classifier_instance is None:
        _llm_classifier_instance = LLMEmailClassifier()
    return _llm_classifier_instance

def initialize_llm_classifier() -> bool:
    """Initialize the global LLM classifier"""
    classifier = get_llm_classifier()
    return classifier.initialize()