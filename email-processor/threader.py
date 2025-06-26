#!/usr/bin/env python3
"""
Email threading and summarization worker
Threads related messages together and generates conversation summaries using Qwen 3
"""

import logging
import os
import queue
import re
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from database import get_db_session
from models import Message, Conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailSummarizer:
    """Email summarizer using LLM library with Qwen 3 for tight, informal summaries"""
    
    def __init__(self, model_id=None):
        self.model_id = model_id or self._discover_qwen_model()
        self.model = None
        
    def _discover_qwen_model(self) -> str:
        """Discover available Qwen 3 model from installed models"""
        try:
            import llm
            available_models = [model.model_id for model in llm.get_models()]
            
            # Look for Qwen models with priority order - NO FALLBACKS
            qwen_candidates = [
                "mlx-community/Qwen3-8B-4bit",  # Actual available Qwen 3 model
                "qwen-3", "qwen3", "qwen-2.5", "qwen2.5", "qwen"
            ]
            
            for candidate in qwen_candidates:
                if candidate in available_models:
                    logger.info(f"Using Qwen model: {candidate}")
                    return candidate
            
            # FAIL FAST - no fallbacks to remote models
            raise Exception(f"No Qwen model found in available models: {available_models}")
            
        except ImportError:
            raise Exception("LLM library not available")
        
    def _call_llm(self, prompt: str) -> str:
        """Call LLM using llm library directly with reasoning suppression"""
        try:
            import llm
            import signal
            
            if self.model is None:
                self.model = llm.get_model(self.model_id)
            
            # Add instruction to suppress reasoning
            enhanced_prompt = f"Answer directly without showing your thinking process.\n\n{prompt}"
            
            # Set up timeout handler for LLM calls
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM call timed out")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            
            try:
                # Set 30 second timeout for LLM calls
                signal.alarm(30)
                response = self.model.prompt(enhanced_prompt)
                raw_text = response.text().strip()
                signal.alarm(0)  # Cancel timeout
            except KeyboardInterrupt:
                signal.alarm(0)  # Cancel timeout
                raise KeyboardInterrupt("LLM call interrupted by user")
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                raise Exception("LLM call timed out after 30 seconds")
            finally:
                signal.signal(signal.SIGALRM, old_handler)  # Restore original handler
            
            # Remove reasoning tags for Qwen models (they use <think>...</think>)
            # Handle both closed and unclosed thinking tags
            cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'<think>.*', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = cleaned_text.strip()
            
            # If nothing left after removing reasoning, extract any content after reasoning
            if not cleaned_text and '<think>' in raw_text:
                # Try to find content after the thinking block
                parts = raw_text.split('</think>')
                if len(parts) > 1:
                    cleaned_text = parts[-1].strip()
            
            # If still nothing, return original
            if not cleaned_text:
                return raw_text
                
            return cleaned_text
        except Exception as e:
            raise Exception(f"LLM call failed with model {self.model_id}: {e}")
    
    def summarize_conversation_thread(self, messages: List[Dict]) -> str:
        """
        Summarize an email thread with focus on outcome (max 120 chars)
        Examples: 'Marketing strategy decided: 70% social, 30% traditional', 'Team agreed on new hiring timeline'
        """
        # Build conversation history
        conversation = ""
        for i, msg in enumerate(messages, 1):
            conversation += f"Email {i}:\nFrom: {msg.get('from_email', '')}\nSubject: {msg.get('subject', '')}\nDate: {msg.get('date_sent', '')}\n\n{msg.get('body_text', '')}\n\n"
        
        prompt = f"""Summarize this email thread in one tight phrase (max 120 chars). Focus on the outcome. Examples: 'Marketing strategy decided: 70% social, 30% traditional', 'Team agreed on new hiring timeline':

{conversation}"""
        
        return self._call_llm(prompt)

class ThreadingOutput:
    """Rich output formatting for threading worker progress"""

    @staticmethod
    def current_time():
        """Get current time in HH:MM:SS format"""
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def print_batch_header(count: int):
        """Print batch processing header"""
        print(f"{ThreadingOutput.current_time()} │ 🧵 Threading and summarizing {count} conversations:")

    @staticmethod
    def print_conversation_progress(conv_id: int, thread_id: str, msg_count: int, result: Dict, duration_ms: int):
        """Print individual conversation processing progress"""
        status = "✅ summarized" if result.get('success') else "❌ failed"
        summary_len = len(result.get('summary', '')) if result.get('summary') else 0
        summary_preview = (result.get('summary', '') or '')[:80] + ('...' if summary_len > 80 else '')
        
        print(f"{ThreadingOutput.current_time()} │ 🧵 #{conv_id:5} {thread_id[:25]:<25} → {status:<12} ({msg_count} msgs, {summary_len} chars) {duration_ms:4}ms")
        if summary_preview:
            print(f"{ThreadingOutput.current_time()} │    📝 Summary: \"{summary_preview}\"")
        
        # Show email preview if available
        email_preview = result.get('email_preview', '')
        if email_preview:
            print(f"{ThreadingOutput.current_time()} │    📧 Preview: {email_preview}")

    @staticmethod
    def print_threading_progress(msg_id: int, from_email: str, subject: str, thread_action: str, duration_ms: int):
        """Print individual message threading progress"""
        subject_preview = (subject or 'No subject')[:40] + ('...' if len(subject or '') > 40 else '')
        from_preview = from_email[:25] + ('...' if len(from_email) > 25 else '')
        
        print(f"{ThreadingOutput.current_time()} │ 📧 #{msg_id:5} {from_preview:<25} → {thread_action:<15} \"{subject_preview}\" {duration_ms:3}ms")

    @staticmethod
    def print_batch_summary(processed: int, total: int, duration_s: float, extra_stats: Dict = None):
        """Print batch completion summary"""
        rate = processed / duration_s if duration_s > 0 else 0
        total_summaries = extra_stats.get('total_summaries', 0) if extra_stats else 0
        avg_length = extra_stats.get('avg_summary_length', 0) if extra_stats else 0
        print(f"{ThreadingOutput.current_time()} │ ✅ Threading batch complete: {processed}/{total} conversations in {duration_s:.1f}s ({rate:.1f}/s)")
        if total_summaries > 0:
            print(f"{ThreadingOutput.current_time()} │    Generated {total_summaries} summaries (avg {avg_length:.0f} chars)")

    @staticmethod
    def print_stats(stats: Dict):
        """Print threading worker statistics"""
        print(f"{ThreadingOutput.current_time()} │ 📊 Threading Stats:")
        print(f"   Total conversations: {stats.get('total_conversations', 0)}")
        print(f"   Summarized: {stats.get('summarized_conversations', 0)}")
        print(f"   Pending: {stats.get('pending_conversations', 0)}")
        print(f"   Average messages/conversation: {stats.get('avg_messages_per_conversation', 0):.1f}")
        
        # Summary stats
        avg_summary_length = stats.get('avg_summary_length', 0)
        if avg_summary_length > 0:
            print(f"   Average summary length: {avg_summary_length:.0f} chars")

    @staticmethod
    def print_worker_start():
        """Print worker startup message"""
        print(f"{ThreadingOutput.current_time()} │ 🧵 Email Threading & Summarization Worker Starting...")

    @staticmethod
    def print_waiting(next_poll_in: int):
        """Print waiting for next poll"""
        print(f"{ThreadingOutput.current_time()} │ ⏰ No conversations to thread, next poll in {next_poll_in}s...")

class EmailThreader:
    """Email threading and summarization worker"""

    def __init__(self, interactive: bool = False, max_conversations: Optional[int] = None):
        self.summarizer = EmailSummarizer()
        self.running = True
        self.interactive = interactive
        self.max_conversations = max_conversations
        self.processed_count = 0
        self.poll_interval = int(os.getenv("THREAD_POLL_INTERVAL", "60"))  # seconds
        self.batch_size = int(os.getenv("THREAD_BATCH_SIZE", "20"))

        # Interactive mode setup
        self.command_queue = queue.Queue() if interactive else None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        ThreadingOutput.print_worker_start()
        max_info = f", max_conversations={self.max_conversations}" if self.max_conversations else ""
        logger.info(f"Email threader initialized with poll_interval={self.poll_interval}s, batch_size={self.batch_size}{max_info}")
        
        if self.max_conversations:
            print(f"{ThreadingOutput.current_time()} │ 🎯 Will exit after processing {self.max_conversations} conversations")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def run(self):
        """Main threading worker loop"""

        # Start interactive command listener if needed
        if self.interactive:
            command_thread = threading.Thread(target=self._command_listener, daemon=True)
            command_thread.start()

        while self.running:
            try:
                # Check for interactive commands
                if self.interactive and not self.command_queue.empty():
                    self._handle_command(self.command_queue.get())

                # Check if we should still be running before processing
                if not self.running:
                    break

                # Process threading batch
                processed_count, batch_stats = self._process_threading_batch()
                self.processed_count += processed_count

                # Check if we've hit the max limit
                if self.max_conversations and self.processed_count >= self.max_conversations:
                    print(f"{ThreadingOutput.current_time()} │ 🎯 Reached max conversations limit ({self.max_conversations}), shutting down...")
                    break

                if processed_count > 0:
                    # If we processed a full batch, don't wait - there might be more
                    if processed_count >= self.batch_size:
                        continue
                else:
                    # Show waiting message and sleep
                    ThreadingOutput.print_waiting(self.poll_interval)

                # Wait before next poll
                self._sleep_with_interrupt_check(self.poll_interval)

            except KeyboardInterrupt:
                print(f"\n{ThreadingOutput.current_time()} │ 🛑 Received interrupt signal, shutting down gracefully...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in threading worker loop: {e}")
                # Wait a bit longer on error to avoid tight error loops
                self._sleep_with_interrupt_check(min(self.poll_interval * 2, 120))

        print(f"{ThreadingOutput.current_time()} │ 🏁 Threading worker stopped")

    def _process_threading_batch(self) -> Tuple[int, Dict]:
        """Process a batch of messages for threading with on-demand summarization"""
        
        # Process threading and return stats about new messages added to threads
        return self._identify_conversation_threads()

    def _identify_conversation_threads(self) -> Tuple[int, Dict]:
        """Identify and create conversation threads based on email headers - Two phase approach"""
        with get_db_session() as session:
            # Get personal messages that aren't threaded yet (no thread_id) - reverse chronological order
            unthreaded_messages = session.query(Message).filter(
                Message.thread_id.is_(None),
                Message.category == 'personal'  # Only thread personal emails
            ).order_by(Message.date_sent.desc()).limit(self.batch_size * 5).all()  # Process more for threading

            if not unthreaded_messages:
                return 0, {}

            # PHASE 1: Assign thread_id to all messages (no conversation creation)
            threaded_count = 0
            thread_assignments = {}  # track thread_id -> [messages] for phase 2
            new_threads = set()  # track which threads are completely new
            existing_threads = set()  # track which threads got new messages
            
            for message in unthreaded_messages:
                thread_id = self._determine_thread_id(session, message)
                message.thread_id = thread_id
                
                # Track for conversation creation in phase 2
                if thread_id not in thread_assignments:
                    thread_assignments[thread_id] = []
                thread_assignments[thread_id].append(message)
                
                # Check if this is a new thread or existing thread
                is_new_thread = thread_id == message.message_id
                if is_new_thread:
                    new_threads.add(thread_id)
                else:
                    existing_threads.add(thread_id)
                
                threaded_count += 1

            # Commit thread_id assignments first
            session.commit()
            
            # PHASE 2: Create/update conversation records and generate summaries on-demand
            summarized_count = 0
            total_summary_length = 0
            
            if thread_assignments:
                for thread_id, new_messages in thread_assignments.items():
                    # Check if we should still be running
                    if not self.running:
                        logger.info("Threading worker interrupted during conversation processing")
                        break
                        
                    # Update or create conversation record
                    genesis_message = min(new_messages, key=lambda m: m.date_sent or datetime.min)
                    conversation = self._create_or_update_conversation(session, genesis_message, thread_id, len(new_messages))
                    
                    # Generate summary on-demand when thread gets new message(s)
                    if conversation:
                        try:
                            summary, summary_success = self._generate_conversation_summary(session, conversation, thread_id)
                        except KeyboardInterrupt:
                            logger.info("Summary generation interrupted by user")
                            self.running = False
                            break
                        
                        if summary_success and summary:
                            # Log the new message(s) added to thread with summary
                            self._log_thread_update(thread_id, new_messages, summary, thread_id in new_threads)
                            summarized_count += 1
                            total_summary_length += len(summary)
                        else:
                            # Log thread update without summary (error case)
                            self._log_thread_update(thread_id, new_messages, None, thread_id in new_threads)
                
                session.commit()

            return threaded_count, {
                'new_threads': len(new_threads),
                'updated_threads': len(existing_threads), 
                'total_summaries': summarized_count,
                'avg_summary_length': total_summary_length / summarized_count if summarized_count > 0 else 0
            }

    def _determine_thread_id(self, session, message: Message) -> str:
        """Determine thread_id for a message based on headers and subject"""
        
        # Check if this is a reply (has In-Reply-To or References)
        if message.in_reply_to:
            # Look for existing message with this Message-ID
            parent_message = session.query(Message).filter(
                Message.message_id == message.in_reply_to
            ).first()
            
            if parent_message and parent_message.thread_id:
                return parent_message.thread_id

        # Check References header for threading
        if message.email_references:
            reference_ids = [ref.strip('<>') for ref in message.email_references.split()]
            for ref_id in reversed(reference_ids):  # Check most recent first
                ref_message = session.query(Message).filter(
                    Message.message_id == ref_id
                ).first()
                
                if ref_message and ref_message.thread_id:
                    return ref_message.thread_id

        # Check for subject-based threading (normalize subject)
        if message.subject:
            normalized_subject = self._normalize_subject(message.subject)
            
            # Look for existing messages with similar subject from same participants
            similar_messages = session.query(Message).filter(
                Message.subject.ilike(f"%{normalized_subject}%"),
                Message.thread_id.isnot(None),
                Message.id != message.id
            ).order_by(Message.date_sent.desc()).limit(5).all()

            for similar_msg in similar_messages:
                # Check if participants overlap
                if self._participants_overlap(message, similar_msg):
                    return similar_msg.thread_id

        # No existing thread found, create new thread_id based on message_id
        return message.message_id

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject line for threading"""
        if not subject:
            return ""
        
        # Remove Re:, Fwd:, etc. prefixes
        normalized = re.sub(r'^(Re|RE|re|Fwd|FWD|fwd):\s*', '', subject.strip())
        normalized = re.sub(r'^\[.*?\]\s*', '', normalized)  # Remove [tags]
        
        return normalized.strip()

    def _participants_overlap(self, msg1: Message, msg2: Message) -> bool:
        """Check if two messages have overlapping participants"""
        msg1_emails = {msg1.from_email}
        if msg1.to_emails:
            msg1_emails.update(msg1.to_emails)
        if msg1.cc_emails:
            msg1_emails.update(msg1.cc_emails)

        msg2_emails = {msg2.from_email}
        if msg2.to_emails:
            msg2_emails.update(msg2.to_emails)
        if msg2.cc_emails:
            msg2_emails.update(msg2.cc_emails)

        return len(msg1_emails & msg2_emails) > 0

    def _create_or_update_conversation(self, session, genesis_message: Message, thread_id: str, message_count: int) -> Optional[Conversation]:
        """Create or update conversation record using upsert pattern to avoid constraints"""
        try:
            # Try to get existing conversation
            conversation = session.query(Conversation).filter(
                Conversation.thread_id == thread_id
            ).first()
            
            if conversation:
                # Update existing conversation
                conversation.message_count = message_count
                conversation.updated_at = datetime.now()
            else:
                # Create new conversation
                conversation = Conversation(
                    thread_id=thread_id,
                    genesis_message_id=genesis_message.message_id,
                    subject_normalized=self._normalize_subject(genesis_message.subject or ''),
                    participants=[],
                    message_count=message_count,
                    first_message_date=genesis_message.date_sent,
                    last_message_date=genesis_message.date_sent
                )
                session.add(conversation)
                session.flush()  # Get the ID
                
            return conversation
                
        except Exception as e:
            # Handle any remaining constraint violations gracefully
            logger.warning(f"Conversation creation conflict for thread {thread_id}: {e}")
            session.rollback()
            # Try to get the conversation that was created by another process
            conversation = session.query(Conversation).filter(
                Conversation.thread_id == thread_id
            ).first()
            return conversation

    def _update_conversation(self, session, message: Message, thread_id: str):
        """Update or create conversation record for thread"""
        conversation = session.query(Conversation).filter(
            Conversation.thread_id == thread_id
        ).first()

        if not conversation:
            # Create new conversation
            conversation = Conversation(
                thread_id=thread_id,
                genesis_message_id=message.message_id,
                subject_normalized=self._normalize_subject(message.subject or ''),
                participants=[],
                message_count=0,
                first_message_date=message.date_sent,
                last_message_date=message.date_sent
            )
            session.add(conversation)

        # Update conversation metadata
        conversation.message_count += 1
        if message.date_sent:
            if not conversation.first_message_date or message.date_sent < conversation.first_message_date:
                conversation.first_message_date = message.date_sent
            if not conversation.last_message_date or message.date_sent > conversation.last_message_date:
                conversation.last_message_date = message.date_sent

        # Update participants (simple approach - could be more sophisticated)
        participants = set()
        if conversation.participants:
            participants.update([p.get('email') for p in conversation.participants if p.get('email')])
        
        participants.add(message.from_email)
        if message.to_emails:
            participants.update(message.to_emails)

        conversation.participants = [{'email': email} for email in participants]
        conversation.updated_at = datetime.now()

    def _create_email_preview(self, messages: List[Message]) -> str:
        """Create a preview of the email thread for display"""
        if not messages:
            return ""
        
        # Show first and last message info
        first_msg = messages[0]
        last_msg = messages[-1]
        
        if len(messages) == 1:
            # Single message
            body_preview = (first_msg.body_text or '')[:60].replace('\n', ' ')
            return f"{first_msg.from_email}: \"{body_preview}...\""
        else:
            # Multiple messages - show range
            first_body = (first_msg.body_text or '')[:30].replace('\n', ' ')
            last_body = (last_msg.body_text or '')[:30].replace('\n', ' ')
            return f"{first_msg.from_email}: \"{first_body}...\" → {last_msg.from_email}: \"{last_body}...\""

    def _generate_conversation_summary(self, session, conversation: Conversation, thread_id: str) -> Tuple[Optional[str], bool]:
        """Generate summary for conversation on-demand"""
        try:
            # Get all messages for this conversation
            messages = session.query(Message).filter(
                Message.thread_id == thread_id
            ).order_by(Message.date_sent.asc()).all()

            if not messages:
                return None, False

            # Convert messages to format expected by summarizer
            message_data = []
            for msg in messages:
                message_data.append({
                    'from_email': msg.from_email,
                    'subject': msg.subject,
                    'date_sent': msg.date_sent.isoformat() if msg.date_sent else '',
                    'body_text': msg.body_text or ''
                })

            # Generate summary using Qwen
            summary = self.summarizer.summarize_conversation_thread(message_data)
            
            # Track model information
            model_info = {
                "model": self.summarizer.model_id,
                "version": "2025-06-26",
                "method": "llm_library",
                "max_chars": 120,
                "prompt_version": "v1"
            }
            
            # Update conversation with summary and model info
            conversation.summary = summary
            conversation.summary_generated_at = datetime.now()
            conversation.summary_model_info = model_info
            
            return summary, True
            
        except Exception as e:
            logger.error(f"Failed to generate summary for thread {thread_id}: {e}")
            return None, False

    def _log_thread_update(self, thread_id: str, new_messages: List[Message], summary: Optional[str], is_new_thread: bool):
        """Log when a thread gets new message(s) with the summary"""
        action = "🆕 New thread" if is_new_thread else "📧 Thread updated"
        message_count = len(new_messages)
        
        # Get representative message for display
        latest_msg = max(new_messages, key=lambda m: m.date_sent or datetime.min)
        from_email = latest_msg.from_email[:25] + '...' if len(latest_msg.from_email) > 25 else latest_msg.from_email
        subject_preview = (latest_msg.subject or 'No subject')[:40] + '...' if len(latest_msg.subject or '') > 40 else latest_msg.subject or 'No subject'
        
        thread_preview = thread_id[:20] + '...' if len(thread_id) > 20 else thread_id
        
        if summary:
            summary_preview = summary[:80] + '...' if len(summary) > 80 else summary
            print(f"{ThreadingOutput.current_time()} │ {action}: {from_email}")
            print(f"{ThreadingOutput.current_time()} │    📧 Subject: \"{subject_preview}\"")
            print(f"{ThreadingOutput.current_time()} │    🧵 Thread: {thread_preview} ({message_count} msg{'s' if message_count > 1 else ''})")
            print(f"{ThreadingOutput.current_time()} │    📝 Summary: \"{summary_preview}\"")
        else:
            print(f"{ThreadingOutput.current_time()} │ {action}: {from_email} (summary failed)")
            print(f"{ThreadingOutput.current_time()} │    📧 Subject: \"{subject_preview}\"")
            print(f"{ThreadingOutput.current_time()} │    🧵 Thread: {thread_preview} ({message_count} msg{'s' if message_count > 1 else ''})")

    def get_threading_stats(self) -> Dict:
        """Get threading worker statistics"""
        try:
            with get_db_session() as session:
                from sqlalchemy import func

                # Total conversations
                total_conversations = session.query(Conversation).count()
                
                # Summarized conversations
                summarized_conversations = session.query(Conversation).filter(
                    Conversation.summary.isnot(None)
                ).count()
                
                # Pending conversations
                pending_conversations = session.query(Conversation).filter(
                    Conversation.summary.is_(None),
                    Conversation.message_count > 0
                ).count()

                # Average messages per conversation
                avg_result = session.query(func.avg(Conversation.message_count)).filter(
                    Conversation.message_count > 0
                ).scalar()
                avg_messages_per_conversation = float(avg_result) if avg_result else 0

                # Average summary length
                avg_summary_length = 0
                if summarized_conversations > 0:
                    avg_length_result = session.query(func.avg(func.length(Conversation.summary))).filter(
                        Conversation.summary.isnot(None)
                    ).scalar()
                    avg_summary_length = float(avg_length_result) if avg_length_result else 0

                return {
                    'total_conversations': total_conversations,
                    'summarized_conversations': summarized_conversations,
                    'pending_conversations': pending_conversations,
                    'avg_messages_per_conversation': avg_messages_per_conversation,
                    'avg_summary_length': avg_summary_length
                }

        except Exception as e:
            logger.error(f"Failed to get threading stats: {e}")
            return {}

    def _command_listener(self):
        """Listen for interactive commands in separate thread"""
        try:
            while self.running:
                try:
                    command = input().strip()
                    if command:
                        self.command_queue.put(command)
                except (EOFError, KeyboardInterrupt):
                    break
        except Exception as e:
            logger.debug(f"Command listener error: {e}")

    def _handle_command(self, command: str):
        """Handle interactive commands"""
        parts = command.lower().split()
        if not parts:
            return

        cmd = parts[0]

        try:
            if cmd == 'stats':
                self._show_stats()
            elif cmd == 'view' and len(parts) > 1:
                try:
                    conv_id = int(parts[1])
                    self._view_conversation(conv_id)
                except ValueError:
                    print(f"{ThreadingOutput.current_time()} │ ❌ Invalid conversation ID: {parts[1]}")
            elif cmd == 'pause':
                print(f"{ThreadingOutput.current_time()} │ ⏸️  Threading paused (type 'resume' to continue)")
                self.running = False
            elif cmd == 'resume':
                print(f"{ThreadingOutput.current_time()} │ ▶️  Threading resumed")
                self.running = True
            elif cmd == 'help':
                self._show_help()
            elif cmd == 'quit' or cmd == 'exit':
                print(f"{ThreadingOutput.current_time()} │ 👋 Shutting down...")
                self.running = False
            else:
                print(f"{ThreadingOutput.current_time()} │ ❓ Unknown command: {command} (type 'help' for commands)")
        except Exception as e:
            print(f"{ThreadingOutput.current_time()} │ ❌ Command error: {e}")

    def _show_stats(self):
        """Show threading worker statistics"""
        try:
            stats = self.get_threading_stats()
            ThreadingOutput.print_stats(stats)
        except Exception as e:
            print(f"{ThreadingOutput.current_time()} │ ❌ Stats error: {e}")

    def _view_conversation(self, conv_id: int):
        """View conversation details"""
        try:
            with get_db_session() as session:
                conversation = session.query(Conversation).filter(Conversation.id == conv_id).first()
                if conversation:
                    print(f"{ThreadingOutput.current_time()} │ 🧵 Conversation #{conversation.id}")
                    print(f"   Thread ID: {conversation.thread_id}")
                    print(f"   Subject: {conversation.subject_normalized}")
                    print(f"   Messages: {conversation.message_count}")
                    print(f"   Participants: {len(conversation.participants or [])}")
                    print(f"   Date range: {conversation.first_message_date} to {conversation.last_message_date}")
                    if conversation.summary:
                        print(f"   Summary: \"{conversation.summary}\"")
                    else:
                        print(f"   Summary: Not generated yet")
                else:
                    print(f"{ThreadingOutput.current_time()} │ ❌ Conversation #{conv_id} not found")
        except Exception as e:
            print(f"{ThreadingOutput.current_time()} │ ❌ View error: {e}")

    def _show_help(self):
        """Show available commands"""
        help_text = f"""
{ThreadingOutput.current_time()} │ 📋 Available Commands:
   stats          - Show threading statistics
   view <id>      - View conversation details by ID
   pause          - Pause threading
   resume         - Resume threading  
   help           - Show this help
   quit/exit      - Shutdown worker
"""
        print(help_text.strip())

    def _sleep_with_interrupt_check(self, duration: int):
        """Sleep for duration seconds, but check for shutdown signal periodically"""
        start_time = time.time()
        while time.time() - start_time < duration and self.running:
            time.sleep(min(1, duration - (time.time() - start_time)))

    def health_check(self) -> bool:
        """Check if threader is healthy"""
        try:
            with get_db_session() as session:
                # Simple database connection test
                from sqlalchemy import text
                session.execute(text("SELECT 1")).fetchone()
                
                # Test summarizer
                self.summarizer._discover_qwen_model()
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Email threading and summarization worker")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Enable interactive mode with commands (stats, view, etc.)")
    parser.add_argument("--max", type=int, metavar="N",
                       help="Exit after processing N conversations (useful for testing)")
    args = parser.parse_args()

    logger.info("Email threader starting up...")

    # Environment validation (use default if not provided)
    database_url = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@localhost:5433/email_rag")
    logger.info(f"Using database: {database_url.split('@')[1] if '@' in database_url else database_url}")

    # Create and run threader
    threader = EmailThreader(interactive=args.interactive, max_conversations=args.max)

    if args.interactive:
        print(f"{ThreadingOutput.current_time()} │ 🎛️  Interactive mode enabled - type 'help' for commands")

    try:
        # Initial health check
        if not threader.health_check():
            logger.error("Initial health check failed, exiting")
            sys.exit(1)

        # Get initial stats
        stats = threader.get_threading_stats()
        logger.info(f"Threader starting with stats: {stats}")

        # Run the threader
        threader.run()

    except Exception as e:
        logger.error(f"Threader failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()