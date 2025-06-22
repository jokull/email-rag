"""
Configuration for Email Scorer Service
Optimized for rapid Qwen-0.5B scoring and triage
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ScorerConfig:
    """Configuration for email scoring service"""
    
    # Model configuration (lightweight for rapid scoring)
    model_name: str = "qwen-0.5b-scorer"
    model_version: str = "v1.0"
    model_path: str = "/app/models/qwen-0.5b-q4_0.gguf"
    
    # Processing constraints for rapid scoring
    max_tokens: int = 128  # Very short responses for scoring
    context_length: int = 1024  # Smaller context for speed
    temperature: float = 0.05  # Very low for consistent scoring
    
    # Hardware optimization
    use_metal: bool = True
    threads: int = 4  # Use half cores for scoring service
    
    # Memory management (lighter than main AI processor)
    max_concurrent_requests: int = 8  # Higher concurrency for scoring
    keep_model_loaded: bool = True
    runtime_memory_mb: int = 512  # Target 512MB for scoring
    
    # Scoring thresholds
    human_threshold: float = 0.7
    personal_threshold: float = 0.6
    importance_threshold: float = 0.3
    commercial_threshold: float = 0.5
    sentiment_neutral: float = 0.5
    
    # Content processing limits
    max_content_length: int = 2000  # Truncate long emails for scoring
    max_subject_length: int = 200
    
    # Scoring prompts optimized for compact responses
    classification_prompt: str = """Classify this email as: human, promotional, transactional, or automated.

Subject: {subject}
From: {sender}
Content: {content}

Classification:"""

    human_score_prompt: str = """Rate if this email is from a real human (0-1 scale, 1=definitely human):

Subject: {subject}
From: {sender}
Content: {content}

Human score (0-1):"""

    importance_prompt: str = """Rate importance of this email (0-1 scale, 1=very important):

Subject: {subject}
From: {sender}
Content: {content}

Importance (0-1):"""

    sentiment_prompt: str = """Rate sentiment of this email (0-1 scale, 0=negative, 0.5=neutral, 1=positive):

Subject: {subject}
Content: {content}

Sentiment (0-1):"""

    commercial_prompt: str = """Rate if this email is commercial/marketing (0-1 scale, 1=definitely commercial):

Subject: {subject}
From: {sender}
Content: {content}

Commercial score (0-1):"""

    personal_prompt: str = """Rate if this email is personal/relevant to user (0-1 scale, 1=very personal):

Subject: {subject}
From: {sender}
Content: {content}

Personal score (0-1):"""

    # Environment-specific settings
    database_url: str = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@postgres:5432/email_rag")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    processing_interval: int = int(os.getenv("SCORING_INTERVAL", "5"))  # Seconds between queue checks
    batch_size: int = int(os.getenv("SCORING_BATCH_SIZE", "10"))
    
    # Rate limiting
    max_scores_per_minute: int = 100
    max_scores_per_hour: int = 5000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "max_tokens": self.max_tokens,
            "context_length": self.context_length,
            "temperature": self.temperature,
            "runtime_memory_mb": self.runtime_memory_mb,
            "max_concurrent_requests": self.max_concurrent_requests,
            "thresholds": {
                "human": self.human_threshold,
                "personal": self.personal_threshold,
                "importance": self.importance_threshold,
                "commercial": self.commercial_threshold,
                "sentiment_neutral": self.sentiment_neutral
            },
            "limits": {
                "max_content_length": self.max_content_length,
                "max_subject_length": self.max_subject_length,
                "max_scores_per_minute": self.max_scores_per_minute,
                "max_scores_per_hour": self.max_scores_per_hour
            }
        }

# Classification type mapping
CLASSIFICATION_TYPES = {
    "human": "Direct human communication",
    "promotional": "Marketing, newsletters, promotions", 
    "transactional": "Receipts, confirmations, notifications",
    "automated": "System-generated, bots, auto-replies"
}

# Sender pattern analysis for faster pre-filtering
COMMERCIAL_PATTERNS = [
    "noreply", "no-reply", "donotreply", "newsletter", "marketing",
    "promotions", "offers", "deals", "sales", "unsubscribe"
]

AUTOMATED_PATTERNS = [
    "automated", "auto-", "system", "daemon", "robot", "bot",
    "support-", "help-", "service-", "admin", "notifications"
]

def get_config() -> ScorerConfig:
    """Get scorer configuration with environment overrides"""
    return ScorerConfig()