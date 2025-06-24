"""
Modern LLM interface using llama-cpp-python + PydanticAI
Replaces the old subprocess-based llama.cpp implementation
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import os

from llama_cpp import Llama
import json

from classification_models import (
    EmailClassification, 
    EmailProcessingRequest, 
    ContactHistory,
    create_classification_prompt,
    CLASSIFICATION_SYSTEM_PROMPT
)
from config import ProcessorConfig

logger = logging.getLogger(__name__)


class ModernQwenInterface:
    """
    Modern Qwen interface using llama-cpp-python + PydanticAI
    Provides structured output with automatic Pydantic validation
    """
    
    def __init__(self, llm_config, processor_config=None):
        self.llm_config = llm_config
        self.processor_config = processor_config or llm_config
        self.model: Optional[Llama] = None
        self.ready = False
        
        # Performance metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0
        
        logger.info("🧠 Initializing Modern Qwen Interface")
    
    async def initialize(self) -> bool:
        """Initialize the Qwen model and PydanticAI agent"""
        try:
            # Check for model file
            model_path = await self._ensure_model_available()
            if not model_path:
                logger.error("❌ Failed to locate Qwen model")
                return False
            
            # Initialize llama-cpp-python model
            logger.info(f"📥 Loading Qwen model from {model_path}")
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=2048,  # Context window
                n_batch=8,   # Batch size for processing
                n_threads=8,  # Use available CPU cores
                verbose=False,
                # Optimize for M2 Mac
                n_gpu_layers=0,  # CPU only for now
                use_mmap=True,
                use_mlock=False,
                # Generation settings
                temperature=0.1,  # Low temperature for consistent classification
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
            )
            
            # Model is ready for direct inference
            self.ready = True
            
            logger.info("✅ Qwen model loaded successfully")
            
            # Test the model with a simple classification
            await self._test_model()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qwen model: {e}")
            return False
    
    async def _ensure_model_available(self) -> Optional[Path]:
        """Ensure Qwen model is available, download if necessary"""
        
        # Check if model exists locally
        model_dir = Path("/app/models")
        model_dir.mkdir(exist_ok=True)
        
        # Look for existing GGUF files
        existing_models = list(model_dir.glob("*.gguf"))
        if existing_models:
            logger.info(f"📁 Found existing model: {existing_models[0]}")
            return existing_models[0]
        
        # Download model using huggingface-hub
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info("📥 Downloading Qwen2.5-0.5B-Instruct GGUF model...")
            
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                filename="qwen2.5-0.5b-instruct-q4_0.gguf",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
            
            logger.info(f"✅ Model downloaded to {model_path}")
            return Path(model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            
            # Fallback: try to download with curl
            return await self._download_model_fallback(model_dir)
    
    async def _download_model_fallback(self, model_dir: Path) -> Optional[Path]:
        """Fallback model download using curl"""
        try:
            import subprocess
            
            model_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
            model_path = model_dir / "qwen2.5-0.5b-instruct-q4_0.gguf"
            
            logger.info(f"📥 Downloading model with curl to {model_path}")
            
            process = await asyncio.create_subprocess_exec(
                "curl", "-L", "-o", str(model_path), model_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and model_path.exists():
                logger.info("✅ Model downloaded successfully with curl")
                return model_path
            else:
                logger.error(f"❌ Curl download failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Fallback download failed: {e}")
            return None
    
    async def _test_model(self):
        """Test the model with a simple classification"""
        try:
            test_request = EmailProcessingRequest(
                email_id="test",
                sender="test@example.com",
                subject="Test email",
                content="Thank you for your help with the project. Best regards, John",
                contact_history=ContactHistory(
                    sender_frequency_score=0.5,
                    response_likelihood=0.7,
                    relationship_strength=0.6,
                    total_emails=10,
                    recent_emails=2
                )
            )
            
            start_time = time.time()
            result = await self.classify_email(test_request)
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"🧪 Model test successful: {result.classification} (confidence: {result.confidence:.2f}) in {processing_time:.0f}ms")
            
        except Exception as e:
            logger.warning(f"⚠️ Model test failed: {e}")
    
    async def classify_email(self, request: EmailProcessingRequest) -> EmailClassification:
        """Classify a single email with structured output"""
        if not self.ready:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        
        try:
            # Create classification prompt with JSON output format
            prompt = self._create_json_classification_prompt(request)
            
            # Get response from model
            response = self.model(
                prompt,
                max_tokens=512,
                temperature=0.1,
                stop=["</classification>"],
                echo=False
            )
            
            # Parse JSON response
            response_text = response['choices'][0]['text'].strip()
            
            # Try to extract JSON from response
            classification_data = self._parse_classification_response(response_text)
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            classification_data["processing_time_ms"] = processing_time
            classification_data["content_length"] = len(request.content)
            
            # Create and validate EmailClassification
            result = EmailClassification(**classification_data)
            
            # Update stats
            self.total_requests += 1
            self.total_time += processing_time
            
            logger.debug(f"📧 Classified email {request.email_id}: {result.classification} ({result.confidence:.2f}) in {processing_time:.0f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Classification failed for email {request.email_id}: {e}")
            
            # Return fallback classification
            return self._create_fallback_classification(request, str(e))
    
    async def classify_batch(self, requests: List[EmailProcessingRequest]) -> List[EmailClassification]:
        """Classify multiple emails efficiently"""
        if not self.ready:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"📧 Processing batch of {len(requests)} emails")
        
        # Process emails concurrently (with a reasonable limit)
        semaphore = asyncio.Semaphore(1)  # Limit concurrent requests (reduced for stability)
        
        async def classify_with_semaphore(request: EmailProcessingRequest) -> EmailClassification:
            async with semaphore:
                return await self.classify_email(request)
        
        tasks = [classify_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        classifications = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Batch classification failed for email {requests[i].email_id}: {result}")
                classifications.append(self._create_fallback_classification(requests[i], str(result)))
            else:
                classifications.append(result)
        
        return classifications
    
    def _create_json_classification_prompt(self, request: EmailProcessingRequest) -> str:
        """Create a prompt that requests JSON output from the model"""
        
        # Build context
        context_parts = [f"Sender: {request.sender}"]
        if request.subject:
            context_parts.append(f"Subject: {request.subject}")
        if request.contact_history:
            ch = request.contact_history
            context_parts.append(
                f"Contact History: {ch.total_emails} total emails, "
                f"{ch.recent_emails} recent, relationship strength {ch.relationship_strength:.2f}"
            )
        
        context = "\n".join(context_parts)
        
        # Truncate content to fit context window
        content = request.content
        if len(content) > 1500:  # Conservative limit for context window
            content = content[:1500] + "..."
        
        return f"""You are an expert email classifier. Analyze the email and provide a detailed classification in JSON format.

{context}

Email Content:
{content}

Analyze this email and classify it across multiple dimensions. Respond with a JSON object containing:

<classification>
{{
    "classification": "human",
    "confidence": 0.85,
    "sentiment": "positive", 
    "sentiment_score": 0.3,
    "formality": "formal",
    "formality_score": 0.7,
    "personalization": "somewhat_personal",
    "personalization_score": 0.6,
    "priority": "normal",
    "priority_score": 0.5,
    "should_process": true,
    "processing_priority": 75,
    "reasoning": "Brief explanation of classification"
}}
</classification>"""
    
    def _parse_classification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON classification response from the model"""
        try:
            # Try to find JSON between tags first
            start_tag = "<classification>"
            end_tag = "</classification>"
            
            if start_tag in response_text and end_tag in response_text:
                start_idx = response_text.find(start_tag) + len(start_tag)
                end_idx = response_text.find(end_tag)
                json_text = response_text[start_idx:end_idx].strip()
            else:
                # Try to find JSON object in the response
                json_text = response_text.strip()
                if not json_text.startswith('{'):
                    # Look for first { and last }
                    start_idx = json_text.find('{')
                    end_idx = json_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_text = json_text[start_idx:end_idx]
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Validate required fields and provide defaults
            result = {
                "classification": data.get("classification", "automated"),
                "confidence": float(data.get("confidence", 0.5)),
                "sentiment": data.get("sentiment", "neutral"),
                "sentiment_score": float(data.get("sentiment_score", 0.0)),
                "formality": data.get("formality", "neutral"),
                "formality_score": float(data.get("formality_score", 0.5)),
                "personalization": data.get("personalization", "generic"),
                "personalization_score": float(data.get("personalization_score", 0.3)),
                "priority": data.get("priority", "normal"),
                "priority_score": float(data.get("priority_score", 0.5)),
                "should_process": bool(data.get("should_process", True)),
                "processing_priority": int(data.get("processing_priority", 50)),
                "reasoning": str(data.get("reasoning", "Model classification")),
            }
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️ Failed to parse model response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Return basic fallback
            return {
                "classification": "automated",
                "confidence": 0.5,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "formality": "neutral",
                "formality_score": 0.5,
                "personalization": "generic",
                "personalization_score": 0.3,
                "priority": "normal",
                "priority_score": 0.5,
                "should_process": True,
                "processing_priority": 50,
                "reasoning": f"Fallback due to parse error: {str(e)[:100]}"
            }
    
    def _create_fallback_classification(self, request: EmailProcessingRequest, error: str) -> EmailClassification:
        """Create a fallback classification when the model fails"""
        
        # Simple heuristic classification
        content_lower = request.content.lower()
        sender_lower = request.sender.lower()
        
        # Basic classification
        if any(word in content_lower for word in ["unsubscribe", "promotional", "offer", "deal"]):
            classification = "promotional"
            confidence = 0.7
        elif any(word in content_lower for word in ["receipt", "invoice", "payment", "order"]):
            classification = "transactional"
            confidence = 0.7
        elif any(word in sender_lower for word in ["noreply", "no-reply", "automated"]):
            classification = "automated"
            confidence = 0.8
        else:
            classification = "human"
            confidence = 0.5
        
        # Basic sentiment
        positive_words = ["thank", "great", "awesome", "excellent"]
        negative_words = ["sorry", "problem", "issue", "error"]
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            sentiment_score = 0.5
        elif negative_count > positive_count:
            sentiment = "negative"
            sentiment_score = -0.5
        else:
            sentiment = "neutral"
            sentiment_score = 0.0
        
        return EmailClassification(
            classification=classification,
            confidence=confidence,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            formality="neutral",
            formality_score=0.5,
            personalization="generic",
            personalization_score=0.3,
            priority="normal",
            priority_score=0.5,
            should_process=classification == "human" and confidence > 0.6,
            processing_priority=50,
            reasoning=f"Fallback classification due to model error: {error}",
            content_length=len(request.content),
            processing_time_ms=1.0
        )
    
    async def health_check(self) -> bool:
        """Check if the model is ready and responsive"""
        if not self.ready or not self.model:
            return False
        
        try:
            # Simple test
            test_prompt = "Hello"
            output = self.model(test_prompt, max_tokens=5, echo=False)
            return True
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_time / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "ready": self.ready,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time,
            "average_time_ms": avg_time,
            "requests_per_second": 1000 / avg_time if avg_time > 0 else 0,
            "model_loaded": self.model is not None,
            "inference_ready": self.model is not None and self.ready
        }