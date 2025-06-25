#!/usr/bin/env python3
"""
Email Classification CLI

Elegant command-line interface for email classification with multiple run modes.

Usage:
    classify.py worker                    # Run continuous worker
    classify.py batch [--limit N]        # Process N emails and exit
    classify.py test [--email EMAIL]     # Test classifier
    classify.py health                    # Health check
    classify.py stats                     # Show statistics

Examples:
    classify.py worker                    # Start worker (Ctrl+C to stop)
    classify.py batch --limit 50         # Process 50 emails
    classify.py test --email test@me.com # Test specific email
"""

import argparse
import signal
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import threading

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from database import get_db_session
from models import Message, ImapMessage, ImapMailbox
from llm_classifier import LLMEmailClassifier, initialize_llm_classifier
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session


class EmailClassifierCLI:
    """Elegant email classification CLI"""
    
    def __init__(self, log_level=logging.INFO):
        self.classifier = None
        self.running = False
        self.shutdown_event = threading.Event()
        self.log_level = log_level
        
        # Configure clean logging with custom formatter
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging with clean format and colors"""
        # Remove any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create custom formatter
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
            }
            RESET = '\033[0m'
            
            def format(self, record):
                # Clean format for different levels
                if self.levelno <= logging.DEBUG:
                    # Verbose: show timestamp and level
                    fmt = '%(asctime)s │ %(levelname)8s │ %(message)s'
                    datefmt = '%H:%M:%S'
                elif self.levelno <= logging.INFO:
                    # Normal: just timestamp and message
                    fmt = '%(asctime)s │ %(message)s'
                    datefmt = '%H:%M:%S'
                else:
                    # Quiet: just message
                    fmt = '%(message)s'
                    datefmt = None
                
                # Set formatter
                self._style = logging.PercentStyle(fmt)
                self.datefmt = datefmt
                
                # Format message
                formatted = super().format(record)
                
                # Add color if level name is shown
                if '│' in formatted and hasattr(record, 'levelname'):
                    color = self.COLORS.get(record.levelname, '')
                    if color:
                        formatted = formatted.replace(record.levelname, f"{color}{record.levelname}{self.RESET}")
                
                return formatted
        
        # Setup handler
        handler = logging.StreamHandler()
        formatter = ColoredFormatter()
        formatter.levelno = self.log_level
        handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger.setLevel(self.log_level)
        root_logger.addHandler(handler)
        
        # Quiet down noisy libraries
        if self.log_level > logging.DEBUG:
            logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
            logging.getLogger('urllib3').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        if self.running:
            print("\n🛑 Graceful shutdown requested...")
            self.running = False
            self.shutdown_event.set()
        else:
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    def _initialize_classifier(self) -> bool:
        """Initialize the LLM classifier"""
        if self.classifier and self.classifier.ready:
            return True
            
        self.logger.info("🤖 Initializing LLM classifier...")
        
        if not initialize_llm_classifier():
            self.logger.error("❌ Failed to initialize LLM classifier")
            return False
        
        self.classifier = LLMEmailClassifier()
        if not self.classifier.initialize():
            self.logger.error("❌ Failed to initialize classifier instance")
            return False
        
        self.logger.info(f"✅ Classifier ready with model: {self.classifier.model_name}")
        return True
    
    def _get_unclassified_emails(self, session: Session, limit: int = 10) -> list:
        """Get unclassified emails from INBOX that have been cleaned"""
        return session.query(Message).join(
            ImapMessage, Message.imap_message_id == ImapMessage.id
        ).join(
            ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id
        ).filter(
            and_(
                ImapMailbox.name == 'INBOX',                # Only INBOX
                Message.category.is_(None),                 # Unclassified
                Message.cleaned_at.isnot(None),            # Must be cleaned
                Message.from_email.isnot(None),            # Has from email
                Message.subject.isnot(None),               # Has subject
                Message.body_text.isnot(None)              # Has cleaned body
            )
        ).order_by(desc(Message.date_sent)).limit(limit).all()
    
    def _classify_single_email(self, session: Session, message: Message) -> bool:
        """Classify a single email"""
        try:
            result = self.classifier.classify_email(
                from_email=message.from_email,
                subject=message.subject or "",
                body=message.body_text or ""
            )
            
            if not result:
                return False
            
            # Update message
            message.category = result.category
            message.confidence = result.confidence
            message.classified_at = datetime.utcnow()
            session.commit()
            
            # Clean logging with message ID
            from_short = message.from_email[:25].ljust(25)
            category_colored = self._colorize_category(result.category)
            time_str = f"{result.processing_time_ms:4.0f}ms"
            id_str = f"#{message.id:5d}"
            
            self.logger.info(f"📧 {id_str} {from_short} → {category_colored} ({result.confidence:.2f}) {time_str}")
            self.logger.debug(f"Classified message {message.id}: {result.category}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Classification failed for {message.id}: {e}")
            session.rollback()
            return False
    
    def _colorize_category(self, category: str) -> str:
        """Add color to category output"""
        colors = {
            'personal': '\033[92m',     # Green
            'promotional': '\033[94m',  # Blue
            'automated': '\033[93m',    # Yellow
        }
        reset = '\033[0m'
        color = colors.get(category, '')
        return f"{color}{category:11}{reset}"
    
    def cmd_worker(self, args) -> int:
        """Run continuous classification worker"""
        if not self._initialize_classifier():
            return 1
        
        self.logger.info("🔄 Starting email classification worker")
        self.logger.info("📧 Processing INBOX emails in batches")
        if self.log_level <= logging.INFO:
            print("Press Ctrl+C to stop gracefully\n")
        
        self.running = True
        batch_size = 10
        sleep_interval = 30
        
        try:
            while self.running:
                batch_start = time.time()
                processed = 0
                
                with get_db_session() as session:
                    emails = self._get_unclassified_emails(session, batch_size)
                    
                    if emails:
                        self.logger.info(f"📦 Processing batch of {len(emails)} emails:")
                        for email in emails:
                            if not self.running:
                                break
                            if self._classify_single_email(session, email):
                                processed += 1
                        
                        batch_time = time.time() - batch_start
                        rate = processed / batch_time if batch_time > 0 else 0
                        self.logger.info(f"✅ Batch complete: {processed}/{len(emails)} in {batch_time:.1f}s ({rate:.1f}/s)")
                        
                        if self.log_level <= logging.INFO:
                            print()  # Add spacing between batches
                    else:
                        self.logger.debug("😴 No unclassified emails found, sleeping...")
                
                # Sleep with interrupt checking
                for _ in range(sleep_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                
        except KeyboardInterrupt:
            pass
        
        self.logger.info("🛑 Worker stopped gracefully")
        return 0
    
    def cmd_batch(self, args) -> int:
        """Process a batch of emails and exit"""
        if not self._initialize_classifier():
            return 1
        
        limit = args.limit
        print(f"📦 Processing batch of up to {limit} emails\n")
        
        processed = 0
        start_time = time.time()
        
        with get_db_session() as session:
            emails = self._get_unclassified_emails(session, limit)
            
            if not emails:
                print("😴 No unclassified emails found")
                return 0
            
            print(f"📧 Found {len(emails)} unclassified emails:")
            
            for email in emails:
                if self._classify_single_email(session, email):
                    processed += 1
        
        total_time = time.time() - start_time
        print(f"\n✅ Batch complete: {processed}/{len(emails)} emails in {total_time:.1f}s")
        
        return 0
    
    def cmd_test(self, args) -> int:
        """Test classifier with sample emails"""
        if not self._initialize_classifier():
            return 1
        
        print("🧪 Testing email classifier\n")
        
        if args.email:
            # Test specific email
            print(f"Testing: {args.email}")
            result = self.classifier.classify_email(
                from_email=args.email,
                subject="Test email",
                body="This is a test email to check classification."
            )
            
            if result:
                category_colored = self._colorize_category(result.category)
                print(f"Result: {category_colored} (confidence: {result.confidence:.2f})")
                print(f"Time: {result.processing_time_ms:.0f}ms")
            else:
                print("❌ Classification failed")
            
            return 0
        
        # Test suite
        test_emails = [
            ('noreply@github.com', 'Should be automated (first-line defense)'),
            ('notifications@vercel.com', 'Should be automated (first-line defense)'),
            ('hello@wonderbly.com', 'Should be automated (first-line defense)'),
            ('invoice-alerts@stripe.com', 'Should be automated (contains invoice)'),
            ('amy@redwoodjs.com', 'Should be promotional (marketing)'),
            ('sunna@solberg.is', 'Should be personal (correspondence)'),
            ('team@fly.io', 'Should be promotional (newsletter)'),
            ('regular@person.com', 'Should be personal (normal person)'),
        ]
        
        for email, description in test_emails:
            result = self.classifier.classify_email(
                from_email=email,
                subject="Test email",
                body="This is a test email to check classification."
            )
            
            if result:
                category_colored = self._colorize_category(result.category)
                time_str = f"{result.processing_time_ms:4.0f}ms"
                print(f"📧 {email:25} → {category_colored} {time_str} │ {description}")
            else:
                print(f"❌ {email:25} → FAILED")
        
        return 0
    
    def cmd_health(self, args) -> int:
        """Check system health"""
        print("🔍 Email Classification Health Check\n")
        
        # Check database
        try:
            with get_db_session() as session:
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                
                recent_count = session.query(func.count(Message.id)).filter(
                    Message.classified_at >= one_hour_ago
                ).scalar() or 0
                
                unclassified_count = session.query(func.count(Message.id)).filter(
                    Message.category.is_(None)
                ).scalar() or 0
                
                print(f"✅ Database accessible")
                print(f"📧 Recent classifications (1h): {recent_count}")
                print(f"⏳ Unclassified emails: {unclassified_count}")
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return 1
        
        # Check classifier
        print("\n🤖 Testing LLM classifier...")
        
        if not self._initialize_classifier():
            return 1
        
        start_time = time.time()
        result = self.classifier.classify_email(
            from_email="health-check@example.com",
            subject="Health check",
            body="This is a health check message"
        )
        response_time = (time.time() - start_time) * 1000
        
        if result:
            print(f"✅ Classifier working")
            print(f"🏷️  Test result: {result.category}")
            print(f"⚡ Response time: {response_time:.0f}ms")
        else:
            print("❌ Classifier test failed")
            return 1
        
        print("\n🎉 All systems healthy!")
        return 0
    
    def cmd_stats(self, args) -> int:
        """Show classification statistics"""
        print("📊 Email Classification Statistics\n")
        
        try:
            with get_db_session() as session:
                # Total counts
                total_emails = session.query(func.count(Message.id)).scalar() or 0
                classified_emails = session.query(func.count(Message.id)).filter(
                    Message.category.isnot(None)
                ).scalar() or 0
                
                print(f"📧 Total emails: {total_emails:,}")
                print(f"🏷️  Classified: {classified_emails:,}")
                print(f"⏳ Unclassified: {total_emails - classified_emails:,}")
                
                if classified_emails > 0:
                    completion = (classified_emails / total_emails) * 100
                    print(f"📈 Completion: {completion:.1f}%")
                
                # Category breakdown
                print("\n📋 Category Breakdown:")
                categories = session.query(
                    Message.category, 
                    func.count(Message.id)
                ).filter(
                    Message.category.isnot(None)
                ).group_by(Message.category).all()
                
                for category, count in categories:
                    percentage = (count / classified_emails) * 100 if classified_emails > 0 else 0
                    category_colored = self._colorize_category(category or 'unknown')
                    print(f"  {category_colored} {count:6,} ({percentage:5.1f}%)")
                
                # Recent activity
                one_day_ago = datetime.utcnow() - timedelta(days=1)
                recent_count = session.query(func.count(Message.id)).filter(
                    Message.classified_at >= one_day_ago
                ).scalar() or 0
                
                print(f"\n⏰ Classified in last 24h: {recent_count:,}")
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return 1
        
        return 0
    
    def cmd_view(self, args) -> int:
        """View detailed email information"""
        message_id = args.message_id
        show_body = args.body
        
        print(f"📧 Email Details - Message ID #{message_id}\n")
        
        try:
            with get_db_session() as session:
                # Get message with IMAP info
                message = session.query(Message).filter(Message.id == message_id).first()
                
                if not message:
                    print(f"❌ Message #{message_id} not found")
                    return 1
                
                # Get IMAP message for additional details
                imap_message = None
                if message.imap_message_id:
                    imap_message = session.query(ImapMessage).filter(
                        ImapMessage.id == message.imap_message_id
                    ).first()
                
                # Get mailbox info
                mailbox_name = "Unknown"
                if imap_message:
                    mailbox = session.query(ImapMailbox).filter(
                        ImapMailbox.id == imap_message.mailbox_id
                    ).first()
                    if mailbox:
                        mailbox_name = mailbox.name
                
                # Header info
                print(f"📋 Message Info:")
                print(f"  ID: #{message.id}")
                print(f"  IMAP ID: #{message.imap_message_id}" if message.imap_message_id else "  IMAP ID: None")
                print(f"  Mailbox: {mailbox_name}")
                print(f"  Message-ID: {message.message_id or 'None'}")
                print()
                
                # Email details
                print(f"📧 Email Details:")
                print(f"  From: {message.from_email}")
                print(f"  To: {', '.join(message.to_emails) if message.to_emails else 'None'}")
                if message.cc_emails:
                    print(f"  CC: {', '.join(message.cc_emails)}")
                if message.reply_to:
                    print(f"  Reply-To: {message.reply_to}")
                print(f"  Subject: {message.subject or 'None'}")
                print(f"  Date: {message.date_sent.isoformat() if message.date_sent else 'None'}")
                print()
                
                # Classification info
                print(f"🏷️  Classification:")
                if message.category:
                    category_colored = self._colorize_category(message.category)
                    print(f"  Category: {category_colored}")
                    print(f"  Confidence: {message.confidence:.2f}" if message.confidence else "  Confidence: None")
                    print(f"  Classified: {message.classified_at.isoformat()}" if message.classified_at else "  Classified: None")
                    
                    # Test first-line defense
                    if self._initialize_classifier():
                        obvious_result = self.classifier._check_obvious_automated(message.from_email)
                        if obvious_result:
                            print(f"  First-line: ✅ {obvious_result.reasoning}")
                        else:
                            print(f"  First-line: 🎯 Passed to LLM")
                        
                        # Show correspondence count
                        outbound_count = self.classifier._get_outbound_email_count(message.from_email)
                        print(f"  Correspondence: {outbound_count} emails sent to this address")
                else:
                    print(f"  Category: ⏳ Unclassified")
                print()
                
                # Threading info
                if message.thread_id:
                    print(f"🧵 Threading:")
                    print(f"  Thread ID: {message.thread_id}")
                    if message.in_reply_to:
                        print(f"  In-Reply-To: {message.in_reply_to}")
                    if message.email_references:
                        print(f"  References: {message.email_references}")
                    print()
                
                # Processing info
                print(f"⚙️  Processing:")
                print(f"  Status: {message.processing_status or 'None'}")
                print(f"  Parsed: {message.parsed_at.isoformat()}" if message.parsed_at else "  Parsed: None")
                print(f"  Cleaned: {message.cleaned_at.isoformat()}" if message.cleaned_at else "  Cleaned: None")
                print(f"  Created: {message.created_at.isoformat()}" if message.created_at else "  Created: None")
                print()
                
                # Body preview or full
                if message.body_text:
                    print(f"📄 Body {'(Full)' if show_body else '(Preview)'}:")
                    if show_body:
                        print(f"  {message.body_text}")
                    else:
                        # Show first 3 lines
                        lines = message.body_text.split('\n')
                        preview_lines = lines[:3]
                        for line in preview_lines:
                            if line.strip():
                                print(f"  {line[:80]}{'...' if len(line) > 80 else ''}")
                        if len(lines) > 3:
                            total_lines = len(lines)
                            print(f"  ... ({total_lines} total lines)")
                        print(f"\n  💡 Use --body flag to see full content")
                else:
                    print(f"📄 Body: None")
                
                return 0
                
        except Exception as e:
            print(f"❌ Error retrieving message: {e}")
            return 1
    
    def run(self):
        """Main CLI entry point"""
        parser = argparse.ArgumentParser(
            description='Email Classification CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__
        )
        
        # Global options
        parser.add_argument('--quiet', '-q', action='store_true',
                          help='Quiet mode (warnings and errors only)')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose mode (debug output)')
        
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # Worker command
        worker_parser = subparsers.add_parser('worker', help='Run continuous worker')
        worker_parser.add_argument('--quiet', '-q', action='store_true',
                                 help='Quiet mode (warnings and errors only)')
        worker_parser.add_argument('--verbose', '-v', action='store_true',
                                 help='Verbose mode (debug output)')
        
        # Batch command
        batch_parser = subparsers.add_parser('batch', help='Process batch and exit')
        batch_parser.add_argument('--limit', type=int, default=50, 
                                help='Number of emails to process (default: 50)')
        batch_parser.add_argument('--quiet', '-q', action='store_true',
                                help='Quiet mode (warnings and errors only)')
        batch_parser.add_argument('--verbose', '-v', action='store_true',
                                help='Verbose mode (debug output)')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Test classifier')
        test_parser.add_argument('--email', type=str, 
                               help='Test specific email address')
        test_parser.add_argument('--quiet', '-q', action='store_true',
                               help='Quiet mode (warnings and errors only)')
        test_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Verbose mode (debug output)')
        
        # Health command
        health_parser = subparsers.add_parser('health', help='Health check')
        health_parser.add_argument('--quiet', '-q', action='store_true',
                                 help='Quiet mode (warnings and errors only)')
        health_parser.add_argument('--verbose', '-v', action='store_true',
                                 help='Verbose mode (debug output)')
        
        # Stats command
        stats_parser = subparsers.add_parser('stats', help='Show statistics')
        stats_parser.add_argument('--quiet', '-q', action='store_true',
                                help='Quiet mode (warnings and errors only)')
        stats_parser.add_argument('--verbose', '-v', action='store_true',
                                help='Verbose mode (debug output)')
        
        # View command
        view_parser = subparsers.add_parser('view', help='View email details')
        view_parser.add_argument('message_id', type=int, help='Message ID to view')
        view_parser.add_argument('--body', '-b', action='store_true',
                               help='Show full email body')
        view_parser.add_argument('--quiet', '-q', action='store_true',
                               help='Quiet mode (warnings and errors only)')
        view_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Verbose mode (debug output)')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return 1
        
        # Route to command handlers
        commands = {
            'worker': self.cmd_worker,
            'batch': self.cmd_batch,
            'test': self.cmd_test,
            'health': self.cmd_health,
            'stats': self.cmd_stats,
            'view': self.cmd_view,
        }
        
        if args.command in commands:
            return commands[args.command](args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1


def main():
    """Entry point"""
    # Parse args first to get log level
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args, _ = parser.parse_known_args()
    
    # Determine log level
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    
    cli = EmailClassifierCLI(log_level)
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())