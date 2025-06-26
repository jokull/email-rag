#!/usr/bin/env python3
"""
Re-summarization Tool - Analyze and re-generate conversation summaries
Allows viewing model usage and re-summarizing with updated models/prompts
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, List, Optional

from database import get_db_session
from models import Conversation, Message
from threader import EmailSummarizer
from sqlalchemy import func, text

def print_model_usage_summary():
    """Print summary of models used for summarization"""
    print("📊 Summarization Model Usage Report")
    print("=" * 50)
    
    with get_db_session() as session:
        # Get model usage statistics
        model_stats = session.execute(text("""
            SELECT 
                summary_model_info->>'model' as model_name,
                summary_model_info->>'version' as model_version,
                summary_model_info->>'prompt_version' as prompt_version,
                COUNT(*) as usage_count,
                MIN(summary_generated_at) as first_used,
                MAX(summary_generated_at) as last_used
            FROM conversations 
            WHERE summary_model_info IS NOT NULL 
            AND summary_model_info != '{}'::jsonb
            GROUP BY 
                summary_model_info->>'model',
                summary_model_info->>'version',
                summary_model_info->>'prompt_version'
            ORDER BY usage_count DESC;
        """)).fetchall()
        
        if not model_stats:
            print("❌ No model tracking information found")
            print("💡 This feature was added recently - existing summaries don't have model info")
            return
            
        for stat in model_stats:
            model_name = stat.model_name or "Unknown"
            version = stat.model_version or "Unknown"
            prompt_version = stat.prompt_version or "v1"
            count = stat.usage_count
            first_used = stat.first_used.strftime("%Y-%m-%d") if stat.first_used else "Unknown"
            last_used = stat.last_used.strftime("%Y-%m-%d") if stat.last_used else "Unknown"
            
            print(f"\n🤖 Model: {model_name}")
            print(f"   Version: {version}")
            print(f"   Prompt: {prompt_version}")
            print(f"   Usage: {count:,} summaries")
            print(f"   Period: {first_used} to {last_used}")
        
        # Show legacy summaries without model info
        legacy_count = session.query(func.count(Conversation.id)).filter(
            Conversation.summary.isnot(None),
            func.coalesce(Conversation.summary_model_info, text("'{}'::jsonb")) == text("'{}'::jsonb")
        ).scalar()
        
        if legacy_count > 0:
            print(f"\n📜 Legacy summaries (no model info): {legacy_count:,}")

def list_conversations_by_model(model_filter: Optional[str] = None, limit: int = 20):
    """List conversations filtered by model"""
    print(f"📝 Recent Conversations by Model (limit: {limit})")
    print("=" * 60)
    
    with get_db_session() as session:
        query = session.query(Conversation).filter(
            Conversation.summary.isnot(None)
        )
        
        if model_filter:
            query = query.filter(
                Conversation.summary_model_info['model'].astext == model_filter
            )
            print(f"🔍 Filtering by model: {model_filter}")
        
        conversations = query.order_by(
            Conversation.summary_generated_at.desc()
        ).limit(limit).all()
        
        if not conversations:
            print("❌ No conversations found")
            return
        
        for conv in conversations:
            model_info = conv.summary_model_info or {}
            model_name = model_info.get('model', 'Unknown')
            version = model_info.get('version', 'Unknown')
            
            print(f"\n🧵 ID: {conv.id} | Thread: {conv.thread_id[:20]}...")
            print(f"   📅 Generated: {conv.summary_generated_at.strftime('%Y-%m-%d %H:%M') if conv.summary_generated_at else 'Unknown'}")
            print(f"   🤖 Model: {model_name} ({version})")
            print(f"   📧 Messages: {conv.message_count}")
            print(f"   📝 Summary: \"{conv.summary}\"")

def resummary_by_criteria(
    model_filter: Optional[str] = None,
    before_date: Optional[str] = None,
    thread_ids: Optional[List[str]] = None,
    dry_run: bool = True,
    limit: int = 10
):
    """Re-summarize conversations matching criteria"""
    
    if dry_run:
        print("🔍 DRY RUN - No changes will be made")
    else:
        print("⚠️  LIVE RUN - Summaries will be updated")
    
    print("=" * 50)
    
    with get_db_session() as session:
        query = session.query(Conversation).filter(
            Conversation.summary.isnot(None)
        )
        
        # Apply filters
        if model_filter:
            query = query.filter(
                Conversation.summary_model_info['model'].astext == model_filter
            )
            print(f"🔍 Model filter: {model_filter}")
        
        if before_date:
            query = query.filter(
                Conversation.summary_generated_at < before_date
            )
            print(f"📅 Before date: {before_date}")
            
        if thread_ids:
            query = query.filter(
                Conversation.thread_id.in_(thread_ids)
            )
            print(f"🧵 Specific threads: {len(thread_ids)} threads")
        
        conversations = query.order_by(
            Conversation.summary_generated_at.asc()
        ).limit(limit).all()
        
        if not conversations:
            print("❌ No conversations match the criteria")
            return
        
        print(f"\n📊 Found {len(conversations)} conversations to re-summarize")
        
        if dry_run:
            for conv in conversations:
                model_info = conv.summary_model_info or {}
                old_model = model_info.get('model', 'Unknown')
                print(f"   🧵 {conv.id}: {old_model} → [would update]")
            print(f"\n💡 Use --live to actually perform re-summarization")
            return
        
        # Actual re-summarization
        if not dry_run:
            summarizer = EmailSummarizer()
            updated_count = 0
            
            for conv in conversations:
                try:
                    print(f"\n🔄 Re-summarizing conversation {conv.id}...")
                    
                    # Get all messages for this conversation
                    messages = session.query(Message).filter(
                        Message.thread_id == conv.thread_id
                    ).order_by(Message.date_sent.asc()).all()
                    
                    if not messages:
                        print(f"   ❌ No messages found for thread {conv.thread_id}")
                        continue
                    
                    # Convert to format expected by summarizer
                    message_data = []
                    for msg in messages:
                        message_data.append({
                            'from_email': msg.from_email,
                            'subject': msg.subject,
                            'date_sent': msg.date_sent.isoformat() if msg.date_sent else '',
                            'body_text': msg.body_text or ''
                        })
                    
                    # Generate new summary
                    old_summary = conv.summary
                    new_summary = summarizer.summarize_conversation_thread(message_data)
                    
                    # Track new model information
                    model_info = {
                        "model": summarizer.model_id,
                        "version": "2025-06-26",
                        "method": "llm_library",
                        "max_chars": 120,
                        "prompt_version": "v1"
                    }
                    
                    # Update conversation
                    conv.summary = new_summary
                    conv.summary_generated_at = datetime.now()
                    conv.summary_model_info = model_info
                    
                    print(f"   📝 Old: \"{old_summary}\"")
                    print(f"   📝 New: \"{new_summary}\"")
                    
                    updated_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Failed to re-summarize: {e}")
                    continue
            
            # Commit all changes
            session.commit()
            print(f"\n✅ Successfully re-summarized {updated_count} conversations")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Re-summarization Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Show model usage stats
    stats_parser = subparsers.add_parser('stats', help='Show model usage statistics')
    
    # List conversations
    list_parser = subparsers.add_parser('list', help='List conversations by model')
    list_parser.add_argument('--model', help='Filter by model name')
    list_parser.add_argument('--limit', type=int, default=20, help='Number of conversations to show')
    
    # Re-summarize conversations
    resummary_parser = subparsers.add_parser('resummary', help='Re-summarize conversations')
    resummary_parser.add_argument('--model', help='Filter by current model name')
    resummary_parser.add_argument('--before', help='Re-summarize summaries generated before this date (YYYY-MM-DD)')
    resummary_parser.add_argument('--threads', nargs='+', help='Specific thread IDs to re-summarize')
    resummary_parser.add_argument('--limit', type=int, default=10, help='Maximum conversations to re-summarize')
    resummary_parser.add_argument('--live', action='store_true', help='Actually perform re-summarization (default is dry run)')
    
    args = parser.parse_args()
    
    if args.command == 'stats':
        print_model_usage_summary()
    elif args.command == 'list':
        list_conversations_by_model(args.model, args.limit)
    elif args.command == 'resummary':
        resummary_by_criteria(
            model_filter=args.model,
            before_date=args.before,
            thread_ids=args.threads,
            dry_run=not args.live,
            limit=args.limit
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()