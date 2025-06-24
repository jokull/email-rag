#!/usr/bin/env python3
"""
Conversation Thread Viewer
Shows a beautiful markdown-style output of email threads with enhanced classification data
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import DictCursor

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@localhost:5432/email_rag")

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL)

def get_conversations(limit: int = 10, filter_type: str = None, min_messages: int = 2) -> List[Dict[str, Any]]:
    """Get conversations with filtering options"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Build WHERE clause for filtering
            where_conditions = [f"t.message_count >= {min_messages}"]
            
            if filter_type:
                if filter_type == "human":
                    where_conditions.append("EXISTS (SELECT 1 FROM classifications c JOIN emails e ON c.email_id = e.id WHERE e.thread_id = t.id AND c.classification = 'human')")
                elif filter_type == "high_priority":
                    where_conditions.append("EXISTS (SELECT 1 FROM classifications c JOIN emails e ON c.email_id = e.id WHERE e.thread_id = t.id AND c.priority = 'urgent')")
                elif filter_type == "personal":
                    where_conditions.append("EXISTS (SELECT 1 FROM classifications c JOIN emails e ON c.email_id = e.id WHERE e.thread_id = t.id AND c.personalization IN ('highly_personal', 'somewhat_personal'))")
                elif filter_type == "important":
                    where_conditions.append("EXISTS (SELECT 1 FROM classifications c JOIN emails e ON c.email_id = e.id WHERE e.thread_id = t.id AND (c.relationship_strength > 0.5 OR c.priority_score > 0.7))")
                elif filter_type == "with_summary":
                    where_conditions.append("t.summary_oneliner IS NOT NULL")
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT 
                    t.id,
                    t.subject_normalized,
                    t.participants,
                    t.message_count,
                    t.first_message_date,
                    t.last_message_date,
                    -- Enhanced thread metadata
                    t.summary_oneliner,
                    t.key_entities,
                    t.thread_mood,
                    t.action_items,
                    t.last_summary_update,
                    t.summary_version,
                    -- Get classification stats for this thread
                    COUNT(CASE WHEN c.classification = 'human' THEN 1 END) as human_count,
                    COUNT(CASE WHEN c.priority = 'urgent' THEN 1 END) as urgent_count,
                    AVG(c.relationship_strength) as avg_relationship,
                    AVG(c.priority_score) as avg_priority,
                    MAX(c.personalization_score) as max_personalization,
                    AVG(c.sentiment_score) as avg_sentiment,
                    COUNT(CASE WHEN c.formality = 'formal' THEN 1 END) as formal_count,
                    AVG(c.confidence) as avg_confidence
                FROM threads t
                LEFT JOIN emails e ON e.thread_id = t.id
                LEFT JOIN classifications c ON c.email_id = e.id
                WHERE {where_clause}
                GROUP BY t.id, t.subject_normalized, t.participants, t.message_count, 
                         t.first_message_date, t.last_message_date, t.summary_oneliner,
                         t.key_entities, t.thread_mood, t.action_items, t.last_summary_update,
                         t.summary_version
                ORDER BY t.last_message_date DESC
                LIMIT {limit}
            """
            
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]

def get_thread_emails(thread_id: str) -> List[Dict[str, Any]]:
    """Get all emails in a thread with classification data"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = """
                SELECT 
                    e.id,
                    e.from_email,
                    e.from_name,
                    e.subject,
                    e.body_text,
                    e.date_sent,
                    e.date_received,
                    -- Classification data
                    c.classification,
                    c.confidence,
                    c.sentiment,
                    c.sentiment_score,
                    c.formality,
                    c.formality_score,
                    c.personalization,
                    c.personalization_score,
                    c.priority,
                    c.priority_score,
                    c.sender_frequency_score,
                    c.response_likelihood,
                    c.relationship_strength,
                    c.should_process,
                    -- Cleaned content
                    ce.clean_content,
                    ce.cleaning_method,
                    ce.cleaned_length,
                    -- Enhanced embeddings
                    COUNT(ee.id) as embeddings_count
                FROM emails e
                LEFT JOIN classifications c ON c.email_id = e.id
                LEFT JOIN cleaned_emails ce ON ce.email_id = e.id
                LEFT JOIN enhanced_embeddings ee ON ee.email_id = e.id
                WHERE e.thread_id = %s
                GROUP BY e.id, e.from_email, e.from_name, e.subject, e.body_text, e.date_sent, e.date_received,
                         c.classification, c.confidence, c.sentiment, c.sentiment_score, c.formality, c.formality_score,
                         c.personalization, c.personalization_score, c.priority, c.priority_score,
                         c.sender_frequency_score, c.response_likelihood, c.relationship_strength, c.should_process,
                         ce.clean_content, ce.cleaning_method, ce.cleaned_length
                ORDER BY e.date_sent ASC
            """
            
            cur.execute(query, (thread_id,))
            return [dict(row) for row in cur.fetchall()]

def format_classification_badges(email: Dict[str, Any]) -> str:
    """Format classification data as badges"""
    badges = []
    
    # Basic classification
    if email.get('classification'):
        classification = email['classification']
        confidence = email.get('confidence', 0)
        if classification == 'human':
            badges.append(f"🧑 Human ({confidence:.1f})")
        elif classification == 'promotional':
            badges.append(f"📢 Promo ({confidence:.1f})")
        elif classification == 'transactional':
            badges.append(f"💳 Transaction ({confidence:.1f})")
        elif classification == 'automated':
            badges.append(f"🤖 Auto ({confidence:.1f})")
    
    # Priority
    if email.get('priority'):
        priority = email['priority']
        if priority == 'urgent':
            badges.append("🚨 URGENT")
        elif priority == 'low':
            badges.append("⬇️ Low")
    
    # Sentiment
    if email.get('sentiment'):
        sentiment = email['sentiment']
        if sentiment == 'positive':
            badges.append("😊 Positive")
        elif sentiment == 'negative':
            badges.append("😟 Negative")
    
    # Personalization
    if email.get('personalization'):
        personalization = email['personalization']
        if personalization == 'highly_personal':
            badges.append("💌 Personal")
        elif personalization == 'somewhat_personal':
            badges.append("📨 Semi-Personal")
    
    # Relationship strength
    if email.get('relationship_strength') and email['relationship_strength'] > 0.5:
        badges.append(f"🤝 Strong ({email['relationship_strength']:.1f})")
    
    # Processing status
    if email.get('should_process'):
        badges.append("⚡ Will Process")
    
    # Cleaning status
    if email.get('clean_content'):
        badges.append(f"🧹 Cleaned ({email.get('cleaning_method', 'unknown')})")
    
    # Embeddings
    if email.get('embeddings_count', 0) > 0:
        badges.append(f"🎯 {email['embeddings_count']} embeddings")
    
    return " ".join(badges) if badges else "📭 No classification"

def format_email_content(email: Dict[str, Any], show_cleaned: bool = False) -> str:
    """Format email content with proper truncation"""
    content = email.get('clean_content') if show_cleaned and email.get('clean_content') else email.get('body_text', '')
    
    if not content:
        return "*[No content]*"
    
    # Truncate long content
    if len(content) > 500:
        content = content[:497] + "..."
    
    # Replace newlines with proper markdown line breaks
    content = content.replace('\n', '  \n')
    
    return content

def print_conversation(thread_id: str, show_cleaned: bool = False, show_stats: bool = True):
    """Print a beautifully formatted conversation thread"""
    emails = get_thread_emails(thread_id)
    
    if not emails:
        print(f"❌ No emails found for thread: {thread_id}")
        return
    
    # Get thread summary info
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT summary_oneliner, key_entities, thread_mood, action_items, 
                       last_summary_update, summary_version
                FROM threads WHERE id = %s
            """, (thread_id,))
            thread_info = cur.fetchone()
    
    # Header
    first_email = emails[0]
    subject = first_email.get('subject', 'No Subject')
    
    print("=" * 80)
    print(f"📧 **CONVERSATION THREAD**")
    print(f"**Subject:** {subject}")
    
    # Show thread summary if available
    if thread_info and thread_info['summary_oneliner']:
        print(f"🧠 **AI Summary:** {thread_info['summary_oneliner']}")
        
        # Show thread mood with emoji
        mood_emoji = {
            'planning': '📅', 'urgent': '🚨', 'social': '💬', 
            'work': '💼', 'problem_solving': '🔧', 'informational': '📖'
        }
        if thread_info['thread_mood']:
            emoji = mood_emoji.get(thread_info['thread_mood'], '💭')
            print(f"{emoji} **Mood:** {thread_info['thread_mood']}")
        
        # Show key entities
        if thread_info['key_entities']:
            entities_str = ', '.join(thread_info['key_entities'][:5])  # Limit to 5
            print(f"🏷️ **Key Topics:** {entities_str}")
        
        # Show action items
        if thread_info['action_items']:
            print(f"✅ **Action Items:** {', '.join(thread_info['action_items'][:3])}")  # Limit to 3
    
    print(f"**Thread ID:** {thread_id}")
    print(f"**Messages:** {len(emails)}")
    print(f"**Duration:** {first_email.get('date_sent', 'Unknown')} → {emails[-1].get('date_sent', 'Unknown')}")
    print("=" * 80)
    print()
    
    # Thread statistics
    if show_stats:
        print("📊 **THREAD STATISTICS**")
        classifications = [e.get('classification') for e in emails if e.get('classification')]
        priorities = [e.get('priority') for e in emails if e.get('priority')]
        sentiments = [e.get('sentiment') for e in emails if e.get('sentiment')]
        avg_relationship = sum((e.get('relationship_strength') or 0) for e in emails) / len(emails)
        avg_sentiment = sum((e.get('sentiment_score') or 0) for e in emails) / len(emails)
        
        print(f"- **Classifications:** {', '.join(set(classifications)) if classifications else 'None'}")
        print(f"- **Priorities:** {', '.join(set(priorities)) if priorities else 'None'}")
        print(f"- **Sentiments:** {', '.join(set(sentiments)) if sentiments else 'None'}")
        print(f"- **Avg Relationship Strength:** {avg_relationship:.2f}")
        print(f"- **Avg Sentiment Score:** {avg_sentiment:.2f}")
        print(f"- **Processed Emails:** {sum(1 for e in emails if e.get('should_process'))}/{len(emails)}")
        print(f"- **Cleaned Emails:** {sum(1 for e in emails if e.get('clean_content'))}/{len(emails)}")
        print(f"- **With Embeddings:** {sum(1 for e in emails if e.get('embeddings_count', 0) > 0)}/{len(emails)}")
        
        if thread_info and thread_info['last_summary_update']:
            print(f"- **Summary Updated:** {thread_info['last_summary_update']} (v{thread_info['summary_version'] or 1})")
        print()
    
    # Individual emails
    for i, email in enumerate(emails, 1):
        sender = email.get('from_name') or email.get('from_email', 'Unknown')
        date = email.get('date_sent', 'Unknown date')
        
        print(f"## 📨 Message {i}/{len(emails)}")
        print(f"**From:** {sender}")
        print(f"**Date:** {date}")
        print(f"**Badges:** {format_classification_badges(email)}")
        print()
        
        # Content
        content = format_email_content(email, show_cleaned)
        print("**Content:**")
        print(f"> {content}")
        print()
        
        # Separator
        if i < len(emails):
            print("---")
            print()

def main():
    parser = argparse.ArgumentParser(description="View email conversation threads with enhanced classification data")
    parser.add_argument("--list", action="store_true", help="List available conversations")
    parser.add_argument("--thread-id", type=str, help="View specific thread by ID")
    parser.add_argument("--filter", choices=["human", "high_priority", "personal", "important", "with_summary"], 
                       help="Filter conversations by type")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of conversations to show")
    parser.add_argument("--min-messages", type=int, default=2, help="Minimum messages per thread")
    parser.add_argument("--show-cleaned", action="store_true", help="Show cleaned content instead of raw")
    parser.add_argument("--no-stats", action="store_true", help="Hide thread statistics")
    
    args = parser.parse_args()
    
    try:
        if args.thread_id:
            # Show specific thread
            print_conversation(args.thread_id, args.show_cleaned, not args.no_stats)
        else:
            # List conversations
            conversations = get_conversations(args.limit, args.filter, args.min_messages)
            
            if not conversations:
                print("❌ No conversations found matching your criteria")
                return
            
            print(f"📋 **FOUND {len(conversations)} CONVERSATIONS**")
            if args.filter:
                print(f"**Filter:** {args.filter}")
            print("=" * 80)
            print()
            
            for i, conv in enumerate(conversations, 1):
                participants = conv.get('participants', [])
                participant_str = ", ".join(participants[:3])
                if len(participants) > 3:
                    participant_str += f" + {len(participants) - 3} more"
                
                # Thread header with subject or summary
                subject_or_summary = conv.get('summary_oneliner', conv.get('subject_normalized', 'No Subject'))
                print(f"**{i}.** `{conv['id'][:8]}...` - {subject_or_summary}")
                
                # Thread metadata line
                print(f"   📅 {conv.get('last_message_date', 'Unknown')} | 💬 {conv.get('message_count', 0)} msgs | 👥 {participant_str}")
                
                # Enhanced stats line with thread mood
                avg_rel = conv.get('avg_relationship') or 0.0
                avg_sentiment = conv.get('avg_sentiment') or 0.0
                mood_emoji = {'planning': '📅', 'urgent': '🚨', 'social': '💬', 'work': '💼', 'problem_solving': '🔧', 'informational': '📖'}
                mood = conv.get('thread_mood')
                mood_display = f"{mood_emoji.get(mood, '💭')} {mood}" if mood else "💭 unknown"
                
                print(f"   🤖 {conv.get('human_count', 0)} human | 🚨 {conv.get('urgent_count', 0)} urgent | 🤝 {avg_rel:.1f} rel | 😊 {avg_sentiment:.1f} sentiment")
                print(f"   {mood_display} | 🎯 conf: {conv.get('avg_confidence', 0) or 0:.1f}")
                
                # Show key entities if available
                if conv.get('key_entities'):
                    entities = ', '.join(conv['key_entities'][:3])
                    print(f"   🏷️ {entities}")
                
                # Show action items if available
                if conv.get('action_items'):
                    actions = ', '.join(conv['action_items'][:2])
                    print(f"   ✅ {actions}")
                
                print()
            
            print("💡 **TIP:** Use `--thread-id <id>` to view a specific conversation")
            print("💡 **EXAMPLE:** `python conversation_viewer.py --thread-id abc12345 --show-cleaned`")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()