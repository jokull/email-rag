"""
Simple rule-based email classifier as fallback when SetFit fails
Focuses on accurately detecting personal emails based on sender patterns
"""

import re
from typing import List, Set
from dataclasses import dataclass

@dataclass
class SimpleClassificationResult:
    category: str  # personal/promotional/automated
    confidence: float
    reason: str  # explanation for classification


class SimpleRuleClassifier:
    """Rule-based email classifier focused on personal email detection"""
    
    def __init__(self):
        # Known personal email domains
        self.personal_domains = {
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com',
            'yahoo.fr', 'yahoo.co.uk', 'live.com', 'msn.com', 'aol.com'
        }
        
        # Known personal indicators in email content
        self.personal_keywords = {
            'honey', 'love you', 'darling', 'sweetie', 'baby', 'dear',
            'family', 'kids', 'children', 'mom', 'dad', 'parent',
            'dinner', 'lunch', 'home', 'vacation', 'holiday',
            'birthday', 'anniversary', 'wedding'
        }
        
        # Promotional indicators
        self.promotional_keywords = {
            'sale', 'discount', 'offer', 'deal', 'promotion', 'coupon',
            'subscribe', 'newsletter', 'unsubscribe', '% off', 'limited time',
            'buy now', 'shop now', 'free shipping', 'exclusive'
        }
        
        # Automated/system indicators
        self.automated_keywords = {
            'noreply', 'no-reply', 'donotreply', 'automated', 'system',
            'notification', 'alert', 'confirmation', 'receipt',
            'shipped', 'delivered', 'invoice', 'statement'
        }
        
        # Known personal email addresses (add specific ones you know)
        self.known_personal_emails = {
            'sunnagunnars@gmail.com',  # Your wife
            'mariehuby@yahoo.fr',      # Personal contact
            'torsteinnkari@gmail.com'  # Personal contact
        }
    
    def classify(self, from_email: str, subject: str = "", body: str = "") -> SimpleClassificationResult:
        """Classify email using simple rules"""
        from_email = (from_email or "").lower().strip()
        subject = (subject or "").lower()
        body = (body or "").lower()
        
        # High confidence personal emails
        if from_email in self.known_personal_emails:
            return SimpleClassificationResult(
                category="personal",
                confidence=0.95,
                reason=f"Known personal email: {from_email}"
            )
        
        # Check domain
        domain = self._extract_domain(from_email)
        if domain in self.personal_domains:
            # Gmail/Yahoo could be personal or promotional, need more signals
            personal_score = self._count_keywords(subject + " " + body, self.personal_keywords)
            promotional_score = self._count_keywords(subject + " " + body, self.promotional_keywords)
            automated_score = self._count_keywords(subject + " " + body, self.automated_keywords)
            
            if personal_score > 0:
                return SimpleClassificationResult(
                    category="personal",
                    confidence=0.85,
                    reason=f"Personal domain ({domain}) + personal keywords"
                )
            elif promotional_score > automated_score:
                return SimpleClassificationResult(
                    category="promotional",
                    confidence=0.75,
                    reason=f"Personal domain ({domain}) but promotional content"
                )
            else:
                # Default personal for personal domains without clear promotional signals
                return SimpleClassificationResult(
                    category="personal",
                    confidence=0.70,
                    reason=f"Personal domain ({domain}) - assuming personal"
                )
        
        # Check for automated patterns
        if self._is_automated_sender(from_email):
            return SimpleClassificationResult(
                category="automated",
                confidence=0.90,
                reason="Automated sender pattern"
            )
        
        # Check content patterns
        promotional_score = self._count_keywords(subject + " " + body, self.promotional_keywords)
        automated_score = self._count_keywords(subject + " " + body, self.automated_keywords)
        
        if promotional_score > 0:
            return SimpleClassificationResult(
                category="promotional",
                confidence=0.80,
                reason="Promotional keywords detected"
            )
        elif automated_score > 0:
            return SimpleClassificationResult(
                category="automated",
                confidence=0.80,
                reason="Automated keywords detected"
            )
        
        # Default to automated for unknown patterns
        return SimpleClassificationResult(
            category="automated",
            confidence=0.60,
            reason="No clear personal/promotional signals"
        )
    
    def _extract_domain(self, email: str) -> str:
        """Extract domain from email address"""
        if '@' in email:
            return email.split('@')[-1]
        return ""
    
    def _count_keywords(self, text: str, keywords: Set[str]) -> int:
        """Count keyword matches in text"""
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)
    
    def _is_automated_sender(self, email: str) -> bool:
        """Check if sender looks automated"""
        email_lower = email.lower()
        automated_patterns = [
            'noreply', 'no-reply', 'donotreply', 'automated', 'system',
            'notification', 'alert', 'admin', 'support', 'help',
            'info@', 'news@', 'updates@', 'marketing@'
        ]
        return any(pattern in email_lower for pattern in automated_patterns)


# Global instance
_simple_classifier = None

def get_simple_classifier() -> SimpleRuleClassifier:
    """Get singleton simple classifier"""
    global _simple_classifier
    if _simple_classifier is None:
        _simple_classifier = SimpleRuleClassifier()
    return _simple_classifier