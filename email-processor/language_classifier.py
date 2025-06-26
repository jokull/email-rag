#!/usr/bin/env python3
"""
Language classification for email content with confidence scoring.
Uses langdetect library for fast, reliable language detection.
"""

import logging
from typing import Tuple, Optional
import re

try:
    from langdetect import detect, detect_langs, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)

class LanguageClassifier:
    """
    Classifies email language content with confidence scoring.
    
    Features:
    - Primary language detection with confidence
    - Multi-language content handling
    - Fallback for short/ambiguous content
    - Email-specific preprocessing
    """
    
    def __init__(self):
        if not LANGDETECT_AVAILABLE:
            logger.warning("langdetect not available - language classification disabled")
            return
            
        # Set deterministic seed for consistent results
        DetectorFactory.seed = 0
        
        # Common language mappings (ISO 639-1 to readable names)
        self.language_names = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'is': 'Icelandic',
            'pl': 'Polish',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'tr': 'Turkish',
            'el': 'Greek',
            'he': 'Hebrew',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Filipino',
            'sw': 'Swahili',
            'af': 'Afrikaans',
            'ca': 'Catalan',
            'eu': 'Basque',
            'gl': 'Galician',
            'cy': 'Welsh',
            'ga': 'Irish',
            'mt': 'Maltese',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'et': 'Estonian',
            'sl': 'Slovenian',
            'sk': 'Slovak',
            'hr': 'Croatian',
            'sr': 'Serbian',
            'bg': 'Bulgarian',
            'mk': 'Macedonian',
            'sq': 'Albanian',
            'ro': 'Romanian',
            'uk': 'Ukrainian',
            'be': 'Belarusian'
        }
        
        # Minimum text length for reliable detection
        self.min_text_length = 20
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.85
        self.medium_confidence_threshold = 0.65
        
    def classify_language(self, text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """
        Classify the language of text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Tuple of (language_code, confidence, language_name)
            Returns (None, None, None) if classification fails
        """
        if not LANGDETECT_AVAILABLE:
            return None, None, None
            
        if not text or not text.strip():
            return None, None, None
            
        # Preprocess text for better detection
        cleaned_text = self._preprocess_text(text)
        
        if len(cleaned_text) < self.min_text_length:
            logger.debug(f"Text too short for reliable language detection: {len(cleaned_text)} chars")
            return None, None, None
            
        try:
            # Get detailed language probabilities
            lang_probs = detect_langs(cleaned_text)
            
            if not lang_probs:
                return None, None, None
                
            # Get the most probable language
            top_lang = lang_probs[0]
            language_code = top_lang.lang
            confidence = float(top_lang.prob)
            
            # Fix common misclassifications
            language_code = self._fix_language_misclassifications(language_code, cleaned_text)
            
            # Get readable language name
            language_name = self.language_names.get(language_code, language_code.upper())
            
            logger.debug(f"Detected language: {language_name} ({language_code}) with confidence {confidence:.3f}")
            
            return language_code, confidence, language_name
            
        except LangDetectException as e:
            logger.debug(f"Language detection failed: {e}")
            return None, None, None
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            return None, None, None
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better language detection.
        
        Removes email artifacts that can interfere with detection:
        - Email addresses
        - URLs
        - Excessive whitespace
        - Common email headers/footers
        """
        if not text:
            return ""
            
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove common email artifacts
        text = re.sub(r'\b(mailto:|tel:|fax:)\S+', ' ', text)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _fix_language_misclassifications(self, language_code: str, text: str) -> str:
        """
        Fix common language misclassifications based on known patterns.
        
        Args:
            language_code: Initially detected language code
            text: The text that was analyzed
            
        Returns:
            Corrected language code
        """
        # Fix Hungarian -> Icelandic confusion
        # Hungarian and Icelandic can be confused due to similar character patterns
        if language_code == 'hu':
            # Check for Icelandic-specific patterns
            icelandic_patterns = [
                r'\bþ',  # Icelandic thorn character
                r'\bð',  # Icelandic eth character  
                r'\bá',  # Common in Icelandic
                r'\bé',  # Common in Icelandic
                r'\bí',  # Common in Icelandic
                r'\bó',  # Common in Icelandic
                r'\bú',  # Common in Icelandic
                r'\bý',  # Common in Icelandic
                r'\bæ',  # Common in Icelandic
                r'\bö',  # Common in Icelandic
                r'\bau\b',  # Common Icelandic diphthong
                r'\bei\b',  # Common Icelandic diphthong
                r'\bvið\b',  # Common Icelandic word
                r'\ber\b',   # Common Icelandic word
                r'\bog\b',   # Common Icelandic word
                r'\bí\b',    # Common Icelandic preposition
                r'\bá\b',    # Common Icelandic preposition
                r'\bekki\b', # Common Icelandic word
                r'\bertu\b', # Common Icelandic word
                r'\bhvað\b', # Common Icelandic word
            ]
            
            # Count Icelandic patterns
            icelandic_matches = sum(1 for pattern in icelandic_patterns 
                                   if re.search(pattern, text.lower()))
            
            # If we find Icelandic patterns, switch to Icelandic
            if icelandic_matches >= 2:  # Need at least 2 pattern matches
                logger.debug(f"Correcting Hungarian->Icelandic based on {icelandic_matches} pattern matches")
                return 'is'
        
        return language_code
    
    def get_confidence_level(self, confidence: Optional[float]) -> str:
        """
        Get human-readable confidence level.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            String description of confidence level
        """
        if confidence is None:
            return "unknown"
        elif confidence >= self.high_confidence_threshold:
            return "high"
        elif confidence >= self.medium_confidence_threshold:
            return "medium"
        else:
            return "low"
    
    def classify_email_content(self, subject: str = "", body: str = "") -> dict:
        """
        Classify language of email content (subject + body).
        
        Args:
            subject: Email subject line
            body: Email body text
            
        Returns:
            Dictionary with classification results
        """
        # Combine subject and body, prioritizing body content
        combined_text = ""
        if body:
            combined_text += body
        if subject:
            combined_text += " " + subject
            
        language_code, confidence, language_name = self.classify_language(combined_text)
        
        return {
            'language_code': language_code,
            'language_name': language_name,
            'confidence': confidence,
            'confidence_level': self.get_confidence_level(confidence),
            'text_length': len(combined_text.strip()) if combined_text else 0,
            'detected': language_code is not None
        }
    
    def get_supported_languages(self) -> dict:
        """Get dictionary of all supported language codes and names."""
        return self.language_names.copy()

def test_language_classifier():
    """Test the language classifier with sample texts."""
    classifier = LanguageClassifier()
    
    test_cases = [
        ("Hello, how are you doing today? I hope everything is going well.", "English"),
        ("Bonjour, comment allez-vous? J'espère que tout va bien.", "French"),
        ("Hola, ¿cómo estás? Espero que todo esté bien.", "Spanish"),
        ("Hallo, wie geht es dir? Ich hoffe, alles ist gut.", "German"),
        ("Привет, как дела? Надеюсь, все хорошо.", "Russian"),
        ("你好，你好吗？我希望一切都好。", "Chinese"),
        ("Halló! Hvað segirðu gott? Ég vona að allt sé í lagi.", "Icelandic"),
        ("", "Empty text"),
        ("OK", "Too short"),
        ("Meeting at 3pm", "Short English")
    ]
    
    print("Testing Language Classifier:")
    print("=" * 50)
    
    for text, expected in test_cases:
        result = classifier.classify_email_content(body=text)
        
        print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"Expected: {expected}")
        print(f"Detected: {result['language_name']} ({result['language_code']})")
        confidence_str = f"{result['confidence']:.3f}" if result['confidence'] is not None else "N/A"
        print(f"Confidence: {confidence_str} ({result['confidence_level']})")
        print(f"Text length: {result['text_length']}")
        print("-" * 30)

if __name__ == "__main__":
    test_language_classifier()