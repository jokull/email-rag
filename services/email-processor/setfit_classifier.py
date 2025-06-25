"""
SetFit-based email classifier using lewispons/Email-classifier-v2
Fast, accurate email classification into personal/promotional/automated categories
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from setfit import SetFitModel
    SETFIT_AVAILABLE = True
except ImportError:
    SETFIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of email classification"""
    category: str  # personal/promotional/automated
    confidence: float  # 0.0 to 1.0
    processing_time_ms: float
    raw_prediction: Optional[str] = None  # Original model output


class SetFitEmailClassifier:
    """
    Email classifier using SetFit model
    Maps model predictions to our 3-category system: personal/promotional/automated
    """
    
    def __init__(self, model_name: str = "lewispons/Email-classifier-v2"):
        self.model_name = model_name
        self.model = None
        self.ready = False
        
        # Performance tracking
        self.total_classifications = 0
        self.total_time_ms = 0.0
        
        logger.info(f"🤖 Initializing SetFit classifier with model: {model_name}")
    
    def initialize(self) -> bool:
        """Initialize the SetFit model"""
        if not SETFIT_AVAILABLE:
            logger.error("❌ SetFit library not available. Install with: pip install setfit")
            return False
        
        try:
            logger.info(f"📥 Loading SetFit model: {self.model_name}")
            start_time = time.time()
            
            self.model = SetFitModel.from_pretrained(self.model_name)
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"✅ SetFit model loaded successfully in {load_time:.0f}ms")
            
            self.ready = True
            
            # Test the model with a simple example
            self._test_model()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load SetFit model: {e}")
            return False
    
    def _test_model(self):
        """Test the model with a simple classification"""
        try:
            test_email = "Thank you for your help with the project yesterday. Looking forward to our meeting next week."
            
            start_time = time.time()
            result = self.classify_single(test_email)
            test_time = (time.time() - start_time) * 1000
            
            logger.info(f"🧪 Model test successful: '{result.category}' (confidence: {result.confidence:.2f}) in {test_time:.0f}ms")
            
        except Exception as e:
            logger.warning(f"⚠️ Model test failed: {e}")
    
    def classify_single(self, email_content: str, subject: Optional[str] = None) -> ClassificationResult:
        """Classify a single email"""
        if not self.ready:
            raise RuntimeError("Classifier not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare text for classification
            # Combine subject and content if both available
            if subject and subject.strip():
                text_to_classify = f"Subject: {subject.strip()}\n\n{email_content}"
            else:
                text_to_classify = email_content
            
            # Truncate if too long (SetFit models typically handle ~512 tokens well)
            if len(text_to_classify) > 2000:
                text_to_classify = text_to_classify[:2000] + "..."
            
            # Get prediction from SetFit model
            predictions = self.model([text_to_classify])
            raw_prediction = predictions[0] if predictions else None
            
            # Handle SetFit model output (can be string or tensor)
            if raw_prediction is not None:
                # Convert tensor to string if needed
                if hasattr(raw_prediction, 'item'):
                    # It's a tensor, convert to index
                    class_idx = int(raw_prediction.item())
                    # Map common class indices to email categories
                    category_map = {0: "personal", 1: "promotional", 2: "automated"}
                    category = category_map.get(class_idx, "automated")
                else:
                    # It's already a string
                    category = str(raw_prediction).lower().strip()
            else:
                category = "automated"
            
            confidence = 0.8  # SetFit doesn't provide confidence, use reasonable default
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.total_classifications += 1
            self.total_time_ms += processing_time
            
            result = ClassificationResult(
                category=category,
                confidence=confidence,
                processing_time_ms=processing_time,
                raw_prediction=str(raw_prediction)
            )
            
            logger.debug(f"📧 Classified email: {category} (confidence: {confidence:.2f}) in {processing_time:.1f}ms")
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Classification failed: {e}")
            
            # Return fallback classification
            return ClassificationResult(
                category="automated",  # Safe default
                confidence=0.1,  # Low confidence for errors
                processing_time_ms=processing_time,
                raw_prediction=f"error: {str(e)}"
            )
    
    def classify_batch(self, email_contents: List[str], subjects: Optional[List[str]] = None) -> List[ClassificationResult]:
        """Classify multiple emails efficiently"""
        if not self.ready:
            raise RuntimeError("Classifier not initialized")
        
        results = []
        start_time = time.time()
        
        try:
            # Prepare texts for batch classification
            texts_to_classify = []
            for i, content in enumerate(email_contents):
                subject = subjects[i] if subjects and i < len(subjects) else None
                
                if subject and subject.strip():
                    text = f"Subject: {subject.strip()}\n\n{content}"
                else:
                    text = content
                
                # Truncate if needed
                if len(text) > 2000:
                    text = text[:2000] + "..."
                
                texts_to_classify.append(text)
            
            # Batch prediction
            predictions = self.model(texts_to_classify)
            
            # Process results
            for i, (content, raw_pred) in enumerate(zip(email_contents, predictions)):
                # Handle SetFit model output (can be string or tensor)
                if raw_pred is not None:
                    # Convert tensor to string if needed
                    if hasattr(raw_pred, 'item'):
                        # It's a tensor, convert to index
                        class_idx = int(raw_pred.item())
                        # Map common class indices to email categories
                        category_map = {0: "personal", 1: "promotional", 2: "automated"}
                        category = category_map.get(class_idx, "automated")
                    else:
                        # It's already a string
                        category = str(raw_pred).lower().strip()
                else:
                    category = "automated"
                
                confidence = 0.8  # SetFit doesn't provide confidence, use reasonable default
                
                results.append(ClassificationResult(
                    category=category,
                    confidence=confidence,
                    processing_time_ms=0.0,  # Will be set below
                    raw_prediction=str(raw_pred)
                ))
            
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(email_contents) if email_contents else 0
            
            # Update processing times
            for result in results:
                result.processing_time_ms = avg_time
            
            # Update stats
            self.total_classifications += len(email_contents)
            self.total_time_ms += total_time
            
            logger.info(f"📧 Batch classified {len(email_contents)} emails in {total_time:.1f}ms (avg: {avg_time:.1f}ms/email)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch classification failed: {e}")
            
            # Return fallback classifications for all
            fallback_time = (time.time() - start_time) * 1000 / len(email_contents) if email_contents else 0
            return [
                ClassificationResult(
                    category="automated",
                    confidence=0.1,
                    processing_time_ms=fallback_time,
                    raw_prediction=f"batch_error: {str(e)}"
                )
                for content in email_contents
            ]
    
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        avg_time = self.total_time_ms / self.total_classifications if self.total_classifications > 0 else 0
        
        return {
            "ready": self.ready,
            "total_classifications": self.total_classifications,
            "total_time_ms": self.total_time_ms,
            "average_time_ms": avg_time,
            "classifications_per_second": 1000 / avg_time if avg_time > 0 else 0,
            "model_loaded": self.model is not None
        }
    
    def health_check(self) -> bool:
        """Check if classifier is ready and working"""
        if not self.ready or not self.model:
            return False
        
        try:
            # Quick test
            test_result = self.classify_single("Test email")
            return test_result.category in ['personal', 'promotional', 'automated']
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False


# Global classifier instance
_classifier_instance = None


def get_classifier() -> SetFitEmailClassifier:
    """Get singleton classifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SetFitEmailClassifier()
    return _classifier_instance


def initialize_classifier() -> bool:
    """Initialize the global classifier"""
    classifier = get_classifier()
    return classifier.initialize()