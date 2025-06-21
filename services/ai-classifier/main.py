import os
import json
import time
import subprocess
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple

load_dotenv()

class EmailClassifier:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.model = os.getenv('LLM_MODEL', 'llama3.2:3b')
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        self.sleep_interval = int(os.getenv('SLEEP_INTERVAL', '60'))
        
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        self.setup_model()
    
    def setup_model(self):
        """Install and setup the LLM model if not already available"""
        try:
            # Install sentence-transformers plugin for embeddings
            subprocess.run(['llm', 'install', 'llm-sentence-transformers'], check=True)
            print("Installed llm-sentence-transformers plugin")
            
            # Check if model is available
            result = subprocess.run(['llm', 'models', 'list'], 
                                  capture_output=True, text=True, check=True)
            
            # For Qwen models, they're automatically available through Ollama or other providers
            print(f"Using model: {self.model}")
            print("Available models:", result.stdout)
                
        except subprocess.CalledProcessError as e:
            print(f"Error setting up model: {e}")
            raise

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    def get_pending_classification_queue(self, limit: int = None) -> List[Dict]:
        """Get threads from processing queue that need classification"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                SELECT 
                    t.id::text,
                    t.subject_normalized,
                    t.participants,
                    t.message_count,
                    t.first_message_date,
                    t.last_message_date,
                    pq.priority,
                    pq.attempts,
                    array_agg(
                        e.from_email || ': ' || 
                        COALESCE(e.subject, '') || ' | ' || 
                        COALESCE(substring(e.body_text, 1, 500), '')
                        ORDER BY e.date_sent
                    ) as sample_messages,
                    -- Check if any participant is whitelisted/blacklisted
                    (SELECT COUNT(*) FROM sender_rules sr 
                     WHERE sr.rule_type = 'whitelist' AND sr.is_active = TRUE
                     AND EXISTS (SELECT 1 FROM unnest(t.participants) p 
                                WHERE p ILIKE sr.email_pattern OR sr.email_pattern ILIKE '%' || p || '%')
                    ) > 0 as has_whitelisted_sender,
                    (SELECT COUNT(*) FROM sender_rules sr 
                     WHERE sr.rule_type = 'blacklist' AND sr.is_active = TRUE
                     AND EXISTS (SELECT 1 FROM unnest(t.participants) p 
                                WHERE p ILIKE sr.email_pattern OR sr.email_pattern ILIKE '%' || p || '%')
                    ) > 0 as has_blacklisted_sender
                FROM processing_queue pq
                JOIN threads t ON pq.thread_id = t.id
                LEFT JOIN emails e ON t.id = e.thread_id
                WHERE pq.queue_type = 'classification'
                AND pq.status = 'pending'
                AND pq.attempts < pq.max_attempts
                GROUP BY t.id, t.subject_normalized, t.participants, t.message_count, 
                         t.first_message_date, t.last_message_date, pq.priority, pq.attempts
                ORDER BY pq.priority DESC, t.last_message_date DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cur.execute(query)
                return cur.fetchall()

    def classify_thread(self, thread: Dict) -> Dict:
        """Classify a single thread using Qwen with detailed scoring"""
        
        # Pre-check: if blacklisted, skip processing
        if thread.get('has_blacklisted_sender', False):
            return {
                'classification': 'automated',
                'confidence': 1.0,
                'human_score': 0.0,
                'personal_score': 0.0,
                'relevance_score': 0.0,
                'should_process': False,
                'reasoning': 'Sender is blacklisted'
            }
        
        # Prepare the prompt
        sample_messages = thread['sample_messages'][:3] if thread['sample_messages'] else []
        participants_str = ', '.join(thread['participants'])
        
        # Enhanced prompt for detailed scoring
        prompt = f"""
Analyze this email thread and provide detailed scoring for AI processing decisions.

Thread Details:
Subject: {thread['subject_normalized']}
Participants: {participants_str}
Message Count: {thread['message_count']}
Whitelisted Sender: {thread.get('has_whitelisted_sender', False)}

Sample Messages:
{chr(10).join(sample_messages) if sample_messages else 'No message content available'}

Provide scores (0.0 to 1.0) for:
1. human_score: How likely this is genuine human-to-human communication
2. personal_score: How personal/meaningful this conversation is (vs formal/business)
3. relevance_score: How relevant/important this thread seems for the user
4. classification: primary category (human, promotional, transactional, automated)

Consider:
- Personal emails, project discussions, meaningful business correspondence = HIGH scores
- Newsletters, automated reports, system notifications = LOW scores
- Marketing, promotions, spam = VERY LOW scores
- Support tickets, order confirmations = MEDIUM scores

Respond with ONLY a JSON object:
{{
    "human_score": 0.85,
    "personal_score": 0.70,
    "relevance_score": 0.80,
    "classification": "human",
    "confidence": 0.90,
    "reasoning": "Brief explanation of scoring"
}}
"""

        try:
            # Call LLM using subprocess
            result = subprocess.run([
                'llm', 'prompt', prompt, 
                '-m', self.model,
                '--temperature', '0.1'
            ], capture_output=True, text=True, check=True)
            
            response_text = result.stdout.strip()
            
            # Try to parse JSON response
            try:
                response = json.loads(response_text)
                
                # Extract scores with defaults
                human_score = max(0.0, min(1.0, float(response.get('human_score', 0.3))))
                personal_score = max(0.0, min(1.0, float(response.get('personal_score', 0.3))))
                relevance_score = max(0.0, min(1.0, float(response.get('relevance_score', 0.3))))
                classification = response.get('classification', 'automated')
                confidence = max(0.0, min(1.0, float(response.get('confidence', 0.5))))
                reasoning = response.get('reasoning', 'AI classification completed')
                
                # Validate classification
                valid_classifications = ['human', 'promotional', 'transactional', 'automated']
                if classification not in valid_classifications:
                    classification = 'automated'
                    confidence = 0.3
                
                # Boost scores for whitelisted senders
                if thread.get('has_whitelisted_sender', False):
                    human_score = min(1.0, human_score + 0.2)
                    relevance_score = min(1.0, relevance_score + 0.3)
                
                # Determine if thread should be processed for RAG
                should_process = self.should_process_thread(human_score, personal_score, relevance_score, classification)
                
                return {
                    'classification': classification,
                    'confidence': confidence,
                    'human_score': human_score,
                    'personal_score': personal_score,
                    'relevance_score': relevance_score,
                    'should_process': should_process,
                    'reasoning': reasoning
                }
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse JSON response: {response_text}, error: {e}")
                return {
                    'classification': 'automated',
                    'confidence': 0.3,
                    'human_score': 0.3,
                    'personal_score': 0.3,
                    'relevance_score': 0.3,
                    'should_process': False,
                    'reasoning': 'Failed to parse LLM response'
                }
                
        except subprocess.CalledProcessError as e:
            print(f"Error calling LLM: {e}")
            return {
                'classification': 'automated',
                'confidence': 0.3,
                'human_score': 0.3,
                'personal_score': 0.3,
                'relevance_score': 0.3,
                'should_process': False,
                'reasoning': f'LLM call failed: {str(e)}'
            }

    def should_process_thread(self, human_score: float, personal_score: float, 
                             relevance_score: float, classification: str) -> bool:
        """Determine if thread should be processed for RAG based on scores"""
        # Get thresholds from user preferences (with defaults)
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT preference_value FROM user_preferences 
                        WHERE preference_key = 'classification_thresholds'
                    """)
                    result = cur.fetchone()
                    if result:
                        thresholds = result['preference_value']
                        min_human = thresholds.get('min_human_score', 0.7)
                        min_personal = thresholds.get('min_personal_score', 0.6)
                        min_relevance = thresholds.get('min_relevance_score', 0.5)
                    else:
                        min_human, min_personal, min_relevance = 0.7, 0.6, 0.5
        except:
            min_human, min_personal, min_relevance = 0.7, 0.6, 0.5
        
        # Only process if meets thresholds AND is human classification
        return (classification == 'human' and 
                human_score >= min_human and 
                (personal_score >= min_personal or relevance_score >= min_relevance))

    def save_classification(self, thread_id: str, result: Dict):
        """Save classification result to database"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO classifications (
                        thread_id, classification, confidence, human_score, 
                        personal_score, relevance_score, should_process, 
                        model_used, reasoning
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE SET
                        classification = EXCLUDED.classification,
                        confidence = EXCLUDED.confidence,
                        human_score = EXCLUDED.human_score,
                        personal_score = EXCLUDED.personal_score,
                        relevance_score = EXCLUDED.relevance_score,
                        should_process = EXCLUDED.should_process,
                        model_used = EXCLUDED.model_used,
                        reasoning = EXCLUDED.reasoning,
                        created_at = NOW()
                """, (
                    thread_id, 
                    result['classification'], 
                    result['confidence'],
                    result['human_score'],
                    result['personal_score'],
                    result['relevance_score'],
                    result['should_process'],
                    self.model, 
                    result['reasoning']
                ))
                
                # Update processing queue
                cur.execute("""
                    UPDATE processing_queue 
                    SET status = 'completed', 
                        processing_completed_at = NOW(),
                        updated_at = NOW()
                    WHERE thread_id = %s AND queue_type = 'classification'
                """, (thread_id,))
                
                # If should process, add to embedding queue
                if result['should_process']:
                    cur.execute("""
                        SELECT add_to_processing_queue(%s, 'embedding')
                    """, (thread_id,))

    def update_processing_queue_status(self, thread_id: str, status: str, error_msg: str = None):
        """Update processing queue status"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                if status == 'processing':
                    cur.execute("""
                        UPDATE processing_queue 
                        SET status = %s, 
                            processing_started_at = NOW(),
                            attempts = attempts + 1,
                            updated_at = NOW()
                        WHERE thread_id = %s AND queue_type = 'classification'
                    """, (status, thread_id))
                elif status == 'failed':
                    cur.execute("""
                        UPDATE processing_queue 
                        SET status = %s, 
                            error_message = %s,
                            processing_completed_at = NOW(),
                            updated_at = NOW()
                        WHERE thread_id = %s AND queue_type = 'classification'
                    """, (status, error_msg, thread_id))

    def check_processing_budget(self) -> bool:
        """Check if we're within daily processing budget"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Get today's stats
                    cur.execute("""
                        SELECT stat_value FROM processing_stats 
                        WHERE date = CURRENT_DATE AND stat_type = 'classification_tokens'
                    """)
                    result = cur.fetchone()
                    used_tokens = result['stat_value'] if result else 0
                    
                    # Get budget from preferences
                    cur.execute("""
                        SELECT preference_value FROM user_preferences 
                        WHERE preference_key = 'daily_processing_budget'
                    """)
                    result = cur.fetchone()
                    if result:
                        budget = result['preference_value'].get('classification_tokens', 100000)
                    else:
                        budget = 100000
                    
                    return used_tokens < budget
        except:
            return True  # Default to allowing processing

    def process_batch(self):
        """Process a batch of classification queue items"""
        if not self.check_processing_budget():
            print("Daily processing budget exceeded, skipping classification")
            return
            
        threads = self.get_pending_classification_queue(self.batch_size)
        
        if not threads:
            print("No threads in classification queue")
            return
        
        print(f"Processing {len(threads)} threads from classification queue...")
        
        for i, thread in enumerate(threads):
            try:
                print(f"Classifying thread {i+1}/{len(threads)}: {thread['subject_normalized']}")
                
                # Mark as processing
                self.update_processing_queue_status(thread['id'], 'processing')
                
                # Classify thread
                result = self.classify_thread(thread)
                
                # Save results
                self.save_classification(thread['id'], result)
                
                print(f"  -> {result['classification']} (human: {result['human_score']:.2f}, " +
                      f"personal: {result['personal_score']:.2f}, " +
                      f"relevance: {result['relevance_score']:.2f}, " +
                      f"process: {result['should_process']})")
                
                # Update processing stats (rough token estimate)
                estimated_tokens = len(str(thread.get('sample_messages', []))) // 4
                with self.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT update_processing_stats('classification_tokens', %s)", 
                                  (estimated_tokens,))
                
                # Small delay to avoid overwhelming the LLM
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing thread {thread['id']}: {e}")
                self.update_processing_queue_status(thread['id'], 'failed', str(e))

    def run_continuous(self):
        """Run the classifier continuously"""
        print(f"Starting AI classifier with model: {self.model}")
        print(f"Processing {self.batch_size} threads every {self.sleep_interval} seconds")
        
        while True:
            try:
                self.process_batch()
                print(f"Sleeping for {self.sleep_interval} seconds...")
                time.sleep(self.sleep_interval)
                
            except KeyboardInterrupt:
                print("Shutting down classifier...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(self.sleep_interval)

if __name__ == "__main__":
    classifier = EmailClassifier()
    classifier.run_continuous()