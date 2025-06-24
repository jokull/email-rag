#!/usr/bin/env python3
"""
Simple Email Classifier Service
Lightweight classification using regex patterns and heuristics
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@postgres:5432/email_rag?sslmode=disable")
PROCESSING_INTERVAL = int(os.getenv("PROCESSING_INTERVAL", "10"))  # seconds
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))

app = FastAPI(title="Simple Email Classifier")

class SimpleClassifier:
    """Simple rule-based email classifier"""
    
    def __init__(self):
        self.commercial_patterns = [
            r'unsubscribe', r'marketing', r'promotion', r'sale', r'discount',
            r'newsletter', r'noreply', r'no-reply', r'donotreply', r'automated',
            r'notification', r'alert', r'system@', r'support@', r'admin@'
        ]
        
        self.human_indicators = [
            r'thanks', r'please', r'regards', r'sincerely', r'best',
            r'meeting', r'call', r'discuss', r'question', r'help'
        ]
    
    def classify_email(self, subject: str, from_email: str, body: str) -> Dict[str, Any]:
        """Simple classification using patterns and heuristics"""
        import re
        
        # Normalize text for analysis
        text = f"{subject} {from_email} {body}".lower()
        
        # Commercial score (0-1, higher = more commercial)
        commercial_score = 0.0
        for pattern in self.commercial_patterns:
            if re.search(pattern, text):
                commercial_score += 0.2
        commercial_score = min(commercial_score, 1.0)
        
        # Human score (0-1, higher = more human)
        human_score = 0.0
        for pattern in self.human_indicators:
            if re.search(pattern, text):
                human_score += 0.15
        human_score = min(human_score, 1.0)
        
        # Personal score based on email patterns
        personal_score = 0.5  # Default
        if '@gmail.com' in from_email or '@yahoo.com' in from_email:
            personal_score += 0.3
        if any(domain in from_email for domain in ['noreply', 'no-reply', 'donotreply']):
            personal_score -= 0.4
            
        personal_score = max(0.0, min(personal_score, 1.0))
        
        # Importance score
        importance_score = 0.5
        if any(word in text for word in ['urgent', 'asap', 'important', 'critical']):
            importance_score += 0.3
        if any(word in text for word in ['meeting', 'call', 'deadline']):
            importance_score += 0.2
            
        importance_score = min(importance_score, 1.0)
        
        # Sentiment score (simple keyword based)
        sentiment_score = 0.5  # Neutral
        positive_words = ['thank', 'great', 'good', 'excellent', 'perfect', 'love']
        negative_words = ['problem', 'issue', 'error', 'fail', 'wrong', 'bad']
        
        for word in positive_words:
            if word in text:
                sentiment_score += 0.1
        for word in negative_words:
            if word in text:
                sentiment_score -= 0.1
                
        sentiment_score = max(0.0, min(sentiment_score, 1.0))
        
        # Determine classification
        if commercial_score > 0.6:
            classification = "promotional"
        elif human_score > 0.6 and personal_score > 0.6:
            classification = "human"
        elif commercial_score > 0.3 and human_score < 0.3:
            classification = "transactional"
        else:
            classification = "automated"
        
        # Should process determination
        should_process = (
            human_score >= 0.5 or 
            importance_score >= 0.7 or 
            (commercial_score <= 0.5 and personal_score >= 0.6)
        )
        
        return {
            "classification": classification,
            "confidence": max(commercial_score, human_score, personal_score),
            "human_score": human_score,
            "personal_score": personal_score,
            "commercial_score": commercial_score,
            "importance_score": importance_score,
            "sentiment_score": sentiment_score,
            "relevance_score": (human_score + personal_score + importance_score) / 3,
            "should_process": should_process,
            "model_used": "simple-heuristic-v1",
            "reasoning": f"Commercial: {commercial_score:.2f}, Human: {human_score:.2f}, Personal: {personal_score:.2f}"
        }

classifier = SimpleClassifier()

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL)

async def process_classification_queue():
    """Process emails in the classification queue"""
    logger.info("Starting classification processing...")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get pending classification tasks
        cur.execute("""
            SELECT pq.id, pq.thread_id, t.subject_normalized, 
                   e.from_email, e.subject, e.body_text
            FROM processing_queue pq
            JOIN threads t ON pq.thread_id = t.id
            JOIN emails e ON e.thread_id = t.id
            WHERE pq.queue_type = 'classification' 
            AND pq.status = 'pending'
            ORDER BY pq.priority DESC, pq.created_at ASC
            LIMIT %s
        """, (BATCH_SIZE,))
        
        tasks = cur.fetchall()
        
        if not tasks:
            logger.debug("No classification tasks pending")
            return
        
        logger.info(f"Processing {len(tasks)} classification tasks")
        
        for task in tasks:
            try:
                # Mark as processing
                cur.execute("""
                    UPDATE processing_queue 
                    SET status = 'processing', processing_started_at = NOW()
                    WHERE id = %s
                """, (task['id'],))
                
                # Get the most recent email from thread for classification
                subject = task['subject'] or ''
                body = task['body_text'] or ''
                from_email = task['from_email'] or ''
                
                # Classify the email
                result = classifier.classify_email(subject, from_email, body)
                
                # Insert classification
                cur.execute("""
                    INSERT INTO classifications (
                        thread_id, classification, confidence, model_used, reasoning,
                        human_score, personal_score, relevance_score, 
                        commercial_score, importance_score, sentiment_score, should_process
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE SET
                        classification = EXCLUDED.classification,
                        confidence = EXCLUDED.confidence,
                        human_score = EXCLUDED.human_score,
                        personal_score = EXCLUDED.personal_score,
                        relevance_score = EXCLUDED.relevance_score,
                        commercial_score = EXCLUDED.commercial_score,
                        importance_score = EXCLUDED.importance_score,
                        sentiment_score = EXCLUDED.sentiment_score,
                        should_process = EXCLUDED.should_process,
                        created_at = NOW()
                """, (
                    task['thread_id'], result['classification'], result['confidence'],
                    result['model_used'], result['reasoning'], result['human_score'],
                    result['personal_score'], result['relevance_score'], 
                    result['commercial_score'], result['importance_score'],
                    result['sentiment_score'], result['should_process']
                ))
                
                # Mark as completed
                cur.execute("""
                    UPDATE processing_queue 
                    SET status = 'completed', processing_completed_at = NOW()
                    WHERE id = %s
                """, (task['id'],))
                
                logger.info(f"Classified thread {task['thread_id']}: {result['classification']} (confidence: {result['confidence']:.2f})")
                
            except Exception as e:
                logger.error(f"Error processing task {task['id']}: {e}")
                # Mark as failed
                cur.execute("""
                    UPDATE processing_queue 
                    SET status = 'failed', error_message = %s, processing_completed_at = NOW()
                    WHERE id = %s
                """, (str(e), task['id']))
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error in classification processing: {e}")

async def classification_worker():
    """Background worker for processing classifications"""
    while True:
        try:
            await process_classification_queue()
            await asyncio.sleep(PROCESSING_INTERVAL)
        except Exception as e:
            logger.error(f"Error in classification worker: {e}")
            await asyncio.sleep(PROCESSING_INTERVAL)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return {"status": "healthy", "service": "simple-classifier"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

@app.get("/stats")
async def get_stats():
    """Get classification statistics"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get queue stats
        cur.execute("""
            SELECT status, COUNT(*) as count 
            FROM processing_queue 
            WHERE queue_type = 'classification'
            GROUP BY status
        """)
        queue_stats = dict(cur.fetchall())
        
        # Get classification stats
        cur.execute("""
            SELECT classification, COUNT(*) as count
            FROM classifications
            GROUP BY classification
        """)
        classification_stats = dict(cur.fetchall())
        
        cur.close()
        conn.close()
        
        return {
            "queue_stats": queue_stats,
            "classification_stats": classification_stats,
            "service": "simple-classifier"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    logger.info("Starting Simple Email Classifier Service")
    asyncio.create_task(classification_worker())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")