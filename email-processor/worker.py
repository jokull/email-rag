#!/usr/bin/env python3
"""
Email processing worker - Simple polling loop that processes emails from imap_messages
Supports multiple modes: email-processing (default) and rag-indexing
"""

import logging
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from database import get_db_session
from processor import EmailProcessor

from models import Message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BeautifulOutput:
    """Rich output formatting for worker progress"""

    @staticmethod
    def current_time():
        """Get current time in HH:MM:SS format"""
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def print_batch_header(mode: str, count: int):
        """Print batch processing header"""
        emoji = "📦" if mode == "email-processing" else "🧠"
        task = "emails" if mode == "email-processing" else "personal emails for RAG indexing"
        print(f"{BeautifulOutput.current_time()} │ {emoji} Processing batch of {count} {task}:")

    @staticmethod
    def print_email_progress(mode: str, msg_id: int, from_email: str, result: dict, duration_ms: int):
        """Print individual email processing progress"""
        if mode == "email-processing":
            # For email processing: show basic parsing info
            status = "✅ parsed" if result.get('success') else "❌ failed"
            chars = len(result.get('body_text', '')) if result.get('body_text') else 0

            # Add language info if available
            language_info = ""
            if result.get('language'):
                lang_code = result.get('language')
                confidence = result.get('language_confidence', 0)
                language_info = f" {lang_code}:{confidence:.2f}"

            print(f"{BeautifulOutput.current_time()} │ 📧 #{msg_id:5} {from_email[:25]:<25} → {status:<10} ({chars} chars{language_info}) {duration_ms:4}ms")
        else:
            # For RAG indexing: show chunks and embedding info
            chunks = result.get('chunks', 0)
            chars = result.get('total_chars', 0)
            status = "✅ embedded" if result.get('success') else "❌ failed"
            print(f"{BeautifulOutput.current_time()} │ 📧 #{msg_id:5} {from_email[:25]:<25} → {status:<12} ({chunks} chunks, {chars} chars) {duration_ms:4}ms")

    @staticmethod
    def print_batch_summary(mode: str, processed: int, total: int, duration_s: float, extra_stats: dict = None):
        """Print batch completion summary"""
        rate = processed / duration_s if duration_s > 0 else 0
        if mode == "email-processing":
            total_chars = extra_stats.get('total_chars', 0) if extra_stats else 0
            print(f"{BeautifulOutput.current_time()} │ ✅ Batch complete: {processed}/{total} parsed in {duration_s:.1f}s ({rate:.1f}/s, {total_chars} chars)")
        else:
            total_chunks = extra_stats.get('total_chunks', 0) if extra_stats else 0
            total_chars = extra_stats.get('total_chars', 0) if extra_stats else 0
            print(f"{BeautifulOutput.current_time()} │ ✅ RAG batch complete: {processed}/{total} → {total_chunks} chunks, {total_chars} chars in {duration_s:.1f}s ({rate:.1f}/s)")

    @staticmethod
    def print_stats(mode: str, stats: dict):
        """Print worker statistics"""
        if mode == "email-processing":
            print(f"{BeautifulOutput.current_time()} │ 📊 Processing Stats:")
            print(f"   Processed: {stats.get('total_processed_messages', 0)}")
            print(f"   Pending: {stats.get('pending_messages', 0)}")
            print(f"   Rate: {stats.get('processing_rate', '0/0')}")

            # Language distribution
            lang_dist = stats.get('language_distribution', {})
            if lang_dist:
                print(f"   Languages: {dict(list(lang_dist.items())[:5])}")  # Show top 5

            # Language detection rate
            lang_rate = stats.get('language_detection_rate', '0/0')
            avg_conf = stats.get('avg_language_confidence')
            print(f"   Lang detection: {lang_rate} (avg conf: {avg_conf:.3f})" if avg_conf else f"   Lang detection: {lang_rate}")
        else:
            pending = stats.get('pending_embedding', 0)
            embedded = stats.get('embedded_messages', 0)
            total_chunks = stats.get('total_chunks', 0)
            print(f"{BeautifulOutput.current_time()} │ 📊 RAG stats: {embedded} embedded, {pending} pending, {total_chunks} chunks total")

    @staticmethod
    def print_worker_start(mode: str):
        """Print worker startup message"""
        emoji = "📦" if mode == "email-processing" else "🧠"
        task = "Email Processing" if mode == "email-processing" else "RAG Indexing"
        print(f"{BeautifulOutput.current_time()} │ {emoji} {task} Worker Starting...")

    @staticmethod
    def print_waiting(mode: str, next_poll_in: int):
        """Print waiting for next poll"""
        emoji = "⏰" if mode == "email-processing" else "🔄"
        task = "emails" if mode == "email-processing" else "personal emails"
        print(f"{BeautifulOutput.current_time()} │ {emoji} No {task} to process, next poll in {next_poll_in}s...")

class EmailWorker:
    """Simple email processing worker with polling and beautiful output"""

    def __init__(self, mode: str = "email-processing", interactive: bool = False):
        self.processor = EmailProcessor()
        self.running = True
        self.mode = mode  # "email-processing" or "rag-indexing"
        self.interactive = interactive
        self.poll_interval = int(os.getenv("POLL_INTERVAL", "30"))  # seconds
        self.batch_size = int(os.getenv("BATCH_SIZE", "50" if mode == "email-processing" else "10"))

        # Interactive mode setup
        self.command_queue = queue.Queue() if interactive else None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        BeautifulOutput.print_worker_start(mode)
        logger.info(f"Email worker initialized in {mode} mode with poll_interval={self.poll_interval}s, batch_size={self.batch_size}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def run(self):
        """Main worker loop with beautiful output"""

        # Start interactive command listener if needed
        if self.interactive:
            command_thread = threading.Thread(target=self._command_listener, daemon=True)
            command_thread.start()

        while self.running:
            try:
                # Check for interactive commands
                if self.interactive and not self.command_queue.empty():
                    self._handle_command(self.command_queue.get())

                # Process based on mode
                if self.mode == "email-processing":
                    processed_count, batch_stats = self._process_email_batch()
                elif self.mode == "rag-indexing":
                    processed_count, batch_stats = self._process_rag_batch()
                else:
                    logger.error(f"Unknown worker mode: {self.mode}")
                    break

                if processed_count > 0:
                    # If we processed a full batch, don't wait - there might be more
                    if processed_count >= self.batch_size:
                        continue
                else:
                    # Show waiting message and sleep
                    BeautifulOutput.print_waiting(self.mode, self.poll_interval)

                # Wait before next poll
                self._sleep_with_interrupt_check(self.poll_interval)

            except KeyboardInterrupt:
                print(f"\n{BeautifulOutput.current_time()} │ 🛑 Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                # Wait a bit longer on error to avoid tight error loops
                self._sleep_with_interrupt_check(min(self.poll_interval * 2, 60))

        print(f"{BeautifulOutput.current_time()} │ 🏁 {self.mode} worker stopped")

    def _process_email_batch(self) -> tuple[int, dict]:
        """Process a batch of emails with beautiful output"""
        # Get emails to process
        with get_db_session() as session:
            from models import ImapMailbox, ImapMessage

            target_mailbox = os.getenv("MAILBOX_FILTER", "Inbox")
            query = session.query(ImapMessage).select_from(ImapMessage).join(ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id).filter(
                ~session.query(Message.imap_message_id).filter(
                    Message.imap_message_id == ImapMessage.id
                ).exists()
            )

            if target_mailbox != "ALL":
                query = query.filter(ImapMailbox.name == target_mailbox)

            unprocessed = query.order_by(ImapMessage.created_at.desc()).limit(self.batch_size).all()

        if not unprocessed:
            return 0, {}

        # Print batch header
        BeautifulOutput.print_batch_header(self.mode, len(unprocessed))

        batch_start = time.time()
        processed_count = 0
        total_chars = 0

        for imap_msg in unprocessed:
            start_time = time.time()

            try:
                with get_db_session() as session:
                    success = self.processor.process_single_email(session, imap_msg)

                    # Get the created message for display
                    if success:
                        message = session.query(Message).filter(Message.imap_message_id == imap_msg.id).first()
                        result = {
                            'success': True,
                            'body_text': message.body_text if message else '',
                            'language': message.language if message else None,
                            'language_confidence': message.language_confidence if message else None
                        }
                        total_chars += len(result['body_text'])
                        processed_count += 1
                    else:
                        result = {'success': False, 'body_text': ''}

                duration_ms = int((time.time() - start_time) * 1000)

                # Get from email for display
                from_email = "unknown@domain.com"
                if imap_msg.envelope and isinstance(imap_msg.envelope, dict):
                    from_data = imap_msg.envelope.get('from', [])
                    if from_data and len(from_data) > 0:
                        from_email = from_data[0].get('email', from_email)

                BeautifulOutput.print_email_progress(self.mode, imap_msg.id, from_email, result, duration_ms)

            except Exception as e:
                logger.error(f"Failed to process email {imap_msg.id}: {e}")
                result = {'success': False, 'body_text': ''}
                duration_ms = int((time.time() - start_time) * 1000)
                BeautifulOutput.print_email_progress(self.mode, imap_msg.id, "error@processing", result, duration_ms)

        # Print batch summary
        batch_duration = time.time() - batch_start
        batch_stats = {'total_chars': total_chars}
        BeautifulOutput.print_batch_summary(self.mode, processed_count, len(unprocessed), batch_duration, batch_stats)

        return processed_count, batch_stats

    def _process_rag_batch(self) -> tuple[int, dict]:
        """Process a batch of personal emails for RAG indexing with beautiful output"""
        # Get personal emails to embed - store IDs only to avoid session issues
        with get_db_session() as session:
            unembedded_query = session.query(Message.id, Message.from_email).filter(
                Message.category == 'personal',
                Message.embedded_at.is_(None),
                Message.body_text.isnot(None),
                Message.body_text != ''
            ).order_by(Message.processed_at.asc()).limit(self.batch_size).all()

        if not unembedded_query:
            return 0, {}

        # Print batch header
        BeautifulOutput.print_batch_header(self.mode, len(unembedded_query))

        batch_start = time.time()
        processed_count = 0
        total_chunks = 0
        total_chars = 0

        for message_id, from_email in unembedded_query:
            start_time = time.time()

            try:
                with get_db_session() as session:
                    # Reload the message in the new session
                    message_obj = session.query(Message).filter(Message.id == message_id).first()
                    if not message_obj:
                        continue
                    
                    success = self.processor.embed_single_message(session, message_obj)

                    if success:
                        # Get chunk info
                        from models import MessageChunk
                        chunks = session.query(MessageChunk).filter(MessageChunk.message_id == message_obj.id).all()
                        chunk_count = len(chunks)
                        chars = sum(len(chunk.text_content) for chunk in chunks)

                        result = {
                            'success': True,
                            'chunks': chunk_count,
                            'total_chars': chars
                        }
                        total_chunks += chunk_count
                        total_chars += chars
                        processed_count += 1
                    else:
                        result = {'success': False, 'chunks': 0, 'total_chars': 0}

                    duration_ms = int((time.time() - start_time) * 1000)
                    BeautifulOutput.print_email_progress(self.mode, message_obj.id, message_obj.from_email, result, duration_ms)

            except Exception as e:
                logger.error(f"Failed to embed message {message_id}: {e}")
                result = {'success': False, 'chunks': 0, 'total_chars': 0}
                duration_ms = int((time.time() - start_time) * 1000)
                # Use stored data for error reporting
                BeautifulOutput.print_email_progress(self.mode, message_id, from_email, result, duration_ms)

        # Print batch summary
        batch_duration = time.time() - batch_start
        batch_stats = {'total_chunks': total_chunks, 'total_chars': total_chars}
        BeautifulOutput.print_batch_summary(self.mode, processed_count, len(unembedded_query), batch_duration, batch_stats)

        return processed_count, batch_stats

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
        """Handle interactive commands like 'view {id}', 'stats', etc."""
        parts = command.lower().split()
        if not parts:
            return

        cmd = parts[0]

        try:
            if cmd == 'stats':
                self._show_stats()
            elif cmd == 'view' and len(parts) > 1:
                try:
                    msg_id = int(parts[1])
                    self._view_message(msg_id)
                except ValueError:
                    print(f"{BeautifulOutput.current_time()} │ ❌ Invalid message ID: {parts[1]}")
            elif cmd == 'pause':
                print(f"{BeautifulOutput.current_time()} │ ⏸️  Processing paused (type 'resume' to continue)")
                self.running = False
            elif cmd == 'resume':
                print(f"{BeautifulOutput.current_time()} │ ▶️  Processing resumed")
                self.running = True
            elif cmd == 'help':
                self._show_help()
            elif cmd == 'quit' or cmd == 'exit':
                print(f"{BeautifulOutput.current_time()} │ 👋 Shutting down...")
                self.running = False
            else:
                print(f"{BeautifulOutput.current_time()} │ ❓ Unknown command: {command} (type 'help' for commands)")
        except Exception as e:
            print(f"{BeautifulOutput.current_time()} │ ❌ Command error: {e}")

    def _show_stats(self):
        """Show worker statistics"""
        try:
            if self.mode == "email-processing":
                stats = self.processor.get_processing_stats()
            else:
                stats = self.processor.get_rag_stats()

            BeautifulOutput.print_stats(self.mode, stats)
        except Exception as e:
            print(f"{BeautifulOutput.current_time()} │ ❌ Stats error: {e}")

    def _view_message(self, msg_id: int):
        """View message details"""
        try:
            with get_db_session() as session:
                if self.mode == "email-processing":
                    from models import ImapMailbox, ImapMessage
                    message = session.query(ImapMessage).filter(ImapMessage.id == msg_id).first()
                    if message:
                        print(f"{BeautifulOutput.current_time()} │ 📧 Message #{message.id}")
                        print(f"   From: {message.envelope['from'][0]['email'] if message.envelope and 'from' in message.envelope else 'unknown'}")
                        print(f"   Subject: {message.envelope['subject'] if message.envelope and 'subject' in message.envelope else 'No subject'}")
                        print(f"   Date: {message.date}")
                        print(f"   Size: {len(message.body) if message.body else 0} bytes")
                    else:
                        print(f"{BeautifulOutput.current_time()} │ ❌ IMAP message #{msg_id} not found")
                else:
                    from models import Message
                    message = session.query(Message).filter(Message.id == msg_id).first()
                    if message:
                        print(f"{BeautifulOutput.current_time()} │ 📧 Message #{message.id}")
                        print(f"   From: {message.from_email}")
                        print(f"   Subject: {message.subject}")
                        print(f"   Category: {message.category}")
                        print(f"   Confidence: {message.confidence}")
                        print(f"   Language: {message.language} (confidence: {message.language_confidence:.3f})" if message.language else "   Language: Not detected")
                        print(f"   Body: {message.body_text[:200]}{'...' if len(message.body_text or '') > 200 else ''}")
                        print(f"   Embedded: {'Yes' if message.embedded_at else 'No'}")
                    else:
                        print(f"{BeautifulOutput.current_time()} │ ❌ Message #{msg_id} not found")
        except Exception as e:
            print(f"{BeautifulOutput.current_time()} │ ❌ View error: {e}")

    def _show_help(self):
        """Show available commands"""
        help_text = f"""
{BeautifulOutput.current_time()} │ 📋 Available Commands:
   stats          - Show processing statistics
   view <id>      - View message details by ID
   pause          - Pause processing
   resume         - Resume processing
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
        """Check if worker is healthy"""
        try:
            with get_db_session() as session:
                # Simple database connection test
                from sqlalchemy import text
                session.execute(text("SELECT 1")).fetchone()
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Email processing worker")
    parser.add_argument("mode", nargs="?", default="email-processing",
                       choices=["email-processing", "rag-indexing"],
                       help="Worker mode: email-processing (default) or rag-indexing")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Enable interactive mode with commands (stats, view, etc.)")
    args = parser.parse_args()

    logger.info(f"Email worker starting up in {args.mode} mode...")

    # Environment validation (use default if not provided)
    database_url = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@localhost:5433/email_rag")
    logger.info(f"Using database: {database_url.split('@')[1] if '@' in database_url else database_url}")

    # Create and run worker
    worker = EmailWorker(mode=args.mode, interactive=args.interactive)

    if args.interactive:
        print(f"{worker.processor.__class__.__name__} │ 🎛️  Interactive mode enabled - type 'help' for commands")

    try:
        # Initial health check
        if not worker.health_check():
            logger.error("Initial health check failed, exiting")
            sys.exit(1)

        # Get initial stats based on mode
        if args.mode == "email-processing":
            stats = worker.processor.get_processing_stats()
        elif args.mode == "rag-indexing":
            stats = worker.processor.get_rag_stats()

        logger.info(f"Worker starting with stats: {stats}")

        # Run the worker
        worker.run()

    except Exception as e:
        logger.error(f"Worker failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
