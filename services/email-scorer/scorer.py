"""
Email Scorer
Multi-dimensional email scoring using Qwen-0.5B for rapid triage
"""

import asyncio
import logging
import time
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import json

from config import ScorerConfig, COMMERCIAL_PATTERNS, AUTOMATED_PATTERNS

logger = logging.getLogger(__name__)

@dataclass
class ScoringResult:
    """Result of email scoring"""
    classification: str  # human, promotional, transactional, automated
    confidence: float
    human_score: float
    personal_score: float
    relevance_score: float
    sentiment_score: float
    importance_score: float
    commercial_score: float
    processing_priority: int
    should_process: bool
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "confidence": self.confidence,
            "scores": {
                "human": self.human_score,
                "personal": self.personal_score,
                "relevance": self.relevance_score,
                "sentiment": self.sentiment_score,
                "importance": self.importance_score,
                "commercial": self.commercial_score
            },
            "processing_priority": self.processing_priority,
            "should_process": self.should_process,
            "processing_time_ms": self.processing_time_ms
        }

class EmailScorer:
    """
    Lightweight email scorer using Qwen-0.5B for rapid multi-dimensional scoring
    """
    
    def __init__(self, config: ScorerConfig):
        self.config = config
        self.model_loaded = False
        self.total_scores = 0
        self.total_processing_time = 0.0
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize and warm up the scoring model"""
        try:
            self.logger.info("🔥 Warming up Qwen-0.5B for email scoring...")
            start_time = time.time()
            
            # Test inference to warm the model
            await self._run_inference("Test email for scoring", max_tokens=5)
            
            warm_time = time.time() - start_time
            self.model_loaded = True
            self.logger.info(f"✅ Scorer model warmed in {warm_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize scorer: {e}")
            raise
    
    async def _run_inference(
        self, 
        prompt: str, 
        max_tokens: int = None,
        temperature: float = None
    ) -> Tuple[str, float]:
        """
        Run inference using llama.cpp optimized for rapid scoring
        Returns: (response, processing_time)
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        start_time = time.time()
        
        # Build command for very fast inference
        cmd = [
            "llama-cli",
            "-m", self.config.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.config.threads),
            "--temp", str(temperature),
            "--ctx-size", str(self.config.context_length),
            "--no-display-prompt",
            "--silent-prompt",
            "--simple-io",  # Fastest mode
        ]
        
        # Add Metal acceleration
        if self.config.use_metal:
            cmd.extend(["-ngl", "999"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10  # Fast timeout for scoring
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"llama.cpp error: {result.stderr}")
            
            response = result.stdout.strip()
            processing_time = time.time() - start_time
            
            return response, processing_time
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Scoring inference timeout")
        except Exception as e:
            raise RuntimeError(f"Scoring inference failed: {e}")
    
    def _preprocess_content(self, content: str, max_length: int = None) -> str:
        """Preprocess email content for scoring"""
        if not content:
            return ""
        
        max_length = max_length or self.config.max_content_length
        
        # Basic cleaning
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        content = re.sub(r'http[s]?://\S+', '[URL]', content)  # Replace URLs
        content = re.sub(r'\b\d+\b', '[NUM]', content)  # Replace numbers
        
        # Truncate if too long
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        return content.strip()
    
    def _extract_sender_features(self, sender: str) -> Dict[str, float]:
        """Extract features from sender email for rapid classification"""
        if not sender:
            return {"commercial": 0.0, "automated": 0.0}
        
        sender_lower = sender.lower()
        
        # Check for commercial patterns
        commercial_score = 0.0
        for pattern in COMMERCIAL_PATTERNS:
            if pattern in sender_lower:
                commercial_score = max(commercial_score, 0.8)
        
        # Check for automated patterns
        automated_score = 0.0
        for pattern in AUTOMATED_PATTERNS:
            if pattern in sender_lower:
                automated_score = max(automated_score, 0.8)
        
        # Domain analysis
        if '@' in sender:
            domain = sender.split('@')[1].lower()
            if any(word in domain for word in ['noreply', 'marketing', 'automated']):
                commercial_score = max(commercial_score, 0.7)
        
        return {
            "commercial": min(commercial_score, 1.0),
            "automated": min(automated_score, 1.0)
        }
    
    async def _score_dimension(self, prompt_template: str, **kwargs) -> float:
        """Score a single dimension using the model"""
        try:
            prompt = prompt_template.format(**kwargs)
            response, _ = await self._run_inference(prompt, max_tokens=10)
            
            # Extract numeric score from response
            score_match = re.search(r'(\d*\.?\d+)', response)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to [0,1]
            
            # Fallback: analyze response text
            response_lower = response.lower()
            if any(word in response_lower for word in ['high', 'very', 'definitely', 'yes']):
                return 0.8
            elif any(word in response_lower for word in ['medium', 'somewhat', 'maybe']):
                return 0.5
            elif any(word in response_lower for word in ['low', 'no', 'not', 'unlikely']):
                return 0.2
            
            return 0.5  # Default neutral score
            
        except Exception as e:
            self.logger.warning(f"Dimension scoring failed: {e}")
            return 0.5  # Default to neutral on error
    
    async def score_email(
        self, 
        subject: str,
        content: str,
        sender: str,
        html_content: Optional[str] = None
    ) -> ScoringResult:
        """
        Score an email across all dimensions
        """
        start_time = time.time()
        
        try:
            # Preprocess inputs
            subject = self._preprocess_content(subject, self.config.max_subject_length)
            content = self._preprocess_content(content)
            
            # Use HTML content if available and longer
            if html_content and len(html_content) > len(content):
                # Basic HTML stripping for scoring
                html_clean = re.sub(r'<[^>]+>', ' ', html_content)
                content = self._preprocess_content(html_clean)
            
            # Extract quick features from sender
            sender_features = self._extract_sender_features(sender)
            
            # Prepare scoring inputs
            scoring_inputs = {
                "subject": subject,
                "sender": sender,
                "content": content[:800]  # Limit for scoring speed
            }
            
            # Score all dimensions in parallel for speed
            scoring_tasks = [
                self._score_dimension(self.config.classification_prompt, **scoring_inputs),
                self._score_dimension(self.config.human_score_prompt, **scoring_inputs),
                self._score_dimension(self.config.importance_prompt, **scoring_inputs),
                self._score_dimension(self.config.sentiment_prompt, **scoring_inputs),
                self._score_dimension(self.config.commercial_prompt, **scoring_inputs),
                self._score_dimension(self.config.personal_prompt, **scoring_inputs),
            ]
            
            # Run scoring in parallel
            results = await asyncio.gather(*scoring_tasks, return_exceptions=True)
            
            # Extract scores with fallbacks
            classification_raw = results[0] if not isinstance(results[0], Exception) else 0.5
            human_score = results[1] if not isinstance(results[1], Exception) else 0.5
            importance_score = results[2] if not isinstance(results[2], Exception) else 0.5
            sentiment_score = results[3] if not isinstance(results[3], Exception) else 0.5
            commercial_score_raw = results[4] if not isinstance(results[4], Exception) else 0.5
            personal_score = results[5] if not isinstance(results[5], Exception) else 0.5
            
            # Combine with sender features
            commercial_score = max(commercial_score_raw, sender_features["commercial"])
            
            # Determine primary classification
            if sender_features["automated"] > 0.7 or human_score < 0.3:
                classification = "automated"
                confidence = 0.8
            elif commercial_score > 0.7:
                classification = "promotional"
                confidence = 0.8
            elif human_score > 0.7 and personal_score > 0.5:
                classification = "human"
                confidence = 0.9
            elif importance_score > 0.6:
                classification = "transactional"
                confidence = 0.7
            else:
                classification = "automated"
                confidence = 0.6
            
            # Calculate relevance (combination of personal + importance - commercial)
            relevance_score = max(0.0, min(1.0, 
                (personal_score + importance_score) / 2 - commercial_score * 0.3
            ))
            
            # Calculate processing priority
            priority_base = int(importance_score * 1000)
            priority_human_boost = int(human_score * 500) if human_score > 0.7 else 0
            priority_sentiment_boost = 300 if sentiment_score < 0.3 else 0  # Negative sentiment
            priority_commercial_penalty = int(commercial_score * -200)
            
            processing_priority = max(0, priority_base + priority_human_boost + 
                                   priority_sentiment_boost + priority_commercial_penalty)
            
            # Determine if should process for RAG
            should_process = (
                importance_score >= self.config.importance_threshold or
                human_score >= self.config.human_threshold or
                sentiment_score <= 0.3 or  # Negative sentiment
                (personal_score >= self.config.personal_threshold and 
                 commercial_score <= self.config.commercial_threshold)
            )
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.total_scores += 1
            self.total_processing_time += processing_time
            
            return ScoringResult(
                classification=classification,
                confidence=confidence,
                human_score=human_score,
                personal_score=personal_score,
                relevance_score=relevance_score,
                sentiment_score=sentiment_score,
                importance_score=importance_score,
                commercial_score=commercial_score,
                processing_priority=processing_priority,
                should_process=should_process,
                processing_time_ms=processing_time * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Email scoring failed: {e}")
            processing_time = time.time() - start_time
            
            # Return safe defaults on error
            return ScoringResult(
                classification="automated",
                confidence=0.5,
                human_score=0.5,
                personal_score=0.5,
                relevance_score=0.5,
                sentiment_score=0.5,
                importance_score=0.5,
                commercial_score=0.5,
                processing_priority=100,
                should_process=False,
                processing_time_ms=processing_time * 1000
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scoring statistics"""
        avg_time = (self.total_processing_time / self.total_scores) if self.total_scores > 0 else 0
        
        return {
            "model_loaded": self.model_loaded,
            "total_scores": self.total_scores,
            "average_processing_time_ms": avg_time * 1000,
            "scores_per_second": 1 / avg_time if avg_time > 0 else 0
        }