from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from email_reply_parser import EmailReplyParser
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging
import json

from models import ImapMessage, Message, Contact
from database import get_db_session

# We'll import the Rust email parser once it's built
# from email_parser_py import parse_email_bytes

logger = logging.getLogger(__name__)

class EmailProcessor:
    """Email cleaning and processing service"""
    
    def __init__(self):
        self.reply_parser = EmailReplyParser()
    
    def process_unprocessed_emails(self, limit: int = 100) -> int:
        """Process emails that haven't been parsed yet"""
        processed_count = 0
        
        with get_db_session() as session:
            # Find unprocessed IMAP messages
            unprocessed = session.query(ImapMessage).filter(
                ~session.query(Message.imap_message_id).filter(
                    Message.imap_message_id == ImapMessage.id
                ).exists()
            ).limit(limit).all()
            
            logger.info(f"Found {len(unprocessed)} unprocessed emails")
            
            for imap_msg in unprocessed:
                try:
                    if self.process_single_email(session, imap_msg):
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to process email {imap_msg.id}: {e}")
                    # Mark as failed
                    self._mark_processing_failed(session, imap_msg.id, str(e))
        
        return processed_count
    
    def process_single_email(self, session: Session, imap_msg: ImapMessage) -> bool:
        """Process a single email message"""
        if not imap_msg.raw_message:
            logger.warning(f"No raw message data for IMAP message {imap_msg.id}")
            return False
        
        # Check if already processed
        existing = session.query(Message).filter(Message.imap_message_id == imap_msg.id).first()
        if existing:
            logger.debug(f"Email {imap_msg.id} already processed as message {existing.id}")
            return True
        
        try:
            # Parse email using Rust bindings (placeholder for now)
            parsed_email = self._parse_email_with_rust(imap_msg.raw_message)
            
            # Clean the email body
            cleaned_body = self._clean_email_body(parsed_email.get('body_text', ''))
            
            # Normalize participants and update contacts (in separate transaction)
            participants = self._normalize_participants_safe(parsed_email.get('participants', []))
            
            # Create message record
            message = Message(
                imap_message_id=imap_msg.id,
                message_id=parsed_email.get('message_id', ''),
                subject=parsed_email.get('subject', ''),
                from_email=parsed_email.get('from_email', ''),
                to_emails=parsed_email.get('to_emails', []),
                cc_emails=parsed_email.get('cc_emails', []),
                reply_to=parsed_email.get('reply_to'),
                email_references=parsed_email.get('references'),
                in_reply_to=parsed_email.get('in_reply_to'),
                thread_topic=parsed_email.get('thread_topic'),
                date_sent=self._parse_date(parsed_email.get('date_sent')),
                body_text=cleaned_body,
                body_html=parsed_email.get('body_html'),
                participants=participants,
                parsed_at=datetime.utcnow(),
                cleaned_at=datetime.utcnow(),
                processing_status='completed'
            )
            
            session.add(message)
            session.commit()
            
            logger.info(f"Successfully processed email {imap_msg.id} -> message {message.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing email {imap_msg.id}: {e}")
            session.rollback()
            return False
    
    def _normalize_participants_safe(self, participants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Safely normalize participants using separate transactions for contact updates"""
        normalized = []
        
        for participant in participants:
            email = participant.get('email', '').lower().strip()
            name = participant.get('name', '').strip() if participant.get('name') else None
            
            if not email:
                continue
            
            # Update contacts in separate session to avoid transaction conflicts
            self._upsert_contact(email, name)
            
            normalized.append({
                'email': email,
                'name': name
            })
        
        return normalized
    
    def _upsert_contact(self, email: str, name: Optional[str]):
        """Upsert contact in separate transaction"""
        try:
            with get_db_session() as contact_session:
                contact = contact_session.query(Contact).filter(Contact.email == email).first()
                if contact:
                    # Update existing contact
                    if name and name not in (contact.seen_names or []):
                        seen_names = list(contact.seen_names or [])
                        seen_names.append(name)
                        contact.seen_names = seen_names
                        contact.updated_at = datetime.utcnow()
                        
                        # Update current name if we have a new one
                        if name and (not contact.name or len(name) > len(contact.name or '')):
                            contact.name = name
                else:
                    # Create new contact
                    contact = Contact(
                        email=email,
                        name=name,
                        seen_names=[name] if name else []
                    )
                    contact_session.add(contact)
                # Session auto-commits on context exit
        except Exception as e:
            # Log but don't fail the main processing
            logger.warning(f"Failed to upsert contact {email}: {e}")
    
    def _parse_email_with_rust(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Parse email using Rust bindings"""
        try:
            # Import the Rust email parser
            from email_parser_py import parse_email_bytes
            
            # Parse email using Rust bindings
            parsed = parse_email_bytes(raw_bytes)
            return parsed.to_dict()
            
        except ImportError as e:
            logger.warning(f"Rust email parser not available, falling back to Python: {e}")
            return self._parse_email_with_python_fallback(raw_bytes)
        except Exception as e:
            logger.error(f"Failed to parse email with Rust parser: {e}")
            return self._parse_email_with_python_fallback(raw_bytes)
    
    def _parse_email_with_python_fallback(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Fallback Python email parsing implementation"""
        import email
        from email.utils import parseaddr, getaddresses
        
        try:
            msg = email.message_from_bytes(raw_bytes)
            
            # Extract participants
            participants = []
            
            # From address
            from_addr = parseaddr(msg.get('From', ''))
            if from_addr[1]:  # email exists
                participants.append({'email': from_addr[1], 'name': from_addr[0] or None})
            
            # To addresses
            to_addrs = getaddresses([msg.get('To', '')])
            for name, email_addr in to_addrs:
                if email_addr:
                    participants.append({'email': email_addr, 'name': name or None})
            
            # CC addresses
            cc_addrs = getaddresses([msg.get('Cc', '')])
            for name, email_addr in cc_addrs:
                if email_addr:
                    participants.append({'email': email_addr, 'name': name or None})
            
            # Extract body text
            body_text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_text = payload.decode('utf-8', errors='ignore')
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_text = payload.decode('utf-8', errors='ignore')
            
            return {
                'message_id': msg.get('Message-ID', ''),
                'subject': msg.get('Subject', ''),
                'from_email': from_addr[1] if from_addr[1] else '',
                'to_emails': [addr[1] for addr in to_addrs if addr[1]],
                'cc_emails': [addr[1] for addr in cc_addrs if addr[1]],
                'reply_to': msg.get('Reply-To'),
                'references': msg.get('References'),
                'in_reply_to': msg.get('In-Reply-To'),
                'thread_topic': msg.get('Thread-Topic'),
                'date_sent': msg.get('Date'),
                'body_text': body_text,
                'body_html': None,  # TODO: Extract HTML body
                'participants': participants
            }
        except Exception as e:
            logger.error(f"Failed to parse email with Python fallback parser: {e}")
            return {}
    
    def _clean_email_body(self, body_text: str) -> str:
        """Clean email body using email_reply_parser"""
        if not body_text:
            return ""
        
        try:
            # Remove quoted replies and signatures
            cleaned = EmailReplyParser.parse_reply(body_text)
            return cleaned.strip() if cleaned else body_text
        except Exception as e:
            logger.warning(f"Failed to clean email body: {e}")
            return body_text
    
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse email date string to datetime"""
        if not date_str:
            return None
        
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None
    
    def _mark_processing_failed(self, session: Session, imap_msg_id: int, error: str):
        """Mark an email as failed processing"""
        try:
            # Check if message record exists
            message = session.query(Message).filter(
                Message.imap_message_id == imap_msg_id
            ).first()
            
            if message:
                message.processing_status = 'failed'
                message.updated_at = datetime.utcnow()
            else:
                # Create a failed message record
                message = Message(
                    imap_message_id=imap_msg_id,
                    message_id='',
                    from_email='',
                    date_sent=datetime.utcnow(),
                    processing_status='failed',
                    body_text=f"Processing failed: {error}"
                )
                session.add(message)
            
            session.commit()
        except Exception as e:
            logger.error(f"Failed to mark processing failed: {e}")
            session.rollback()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with get_db_session() as session:
            total_imap = session.query(func.count(ImapMessage.id)).scalar()
            total_processed = session.query(func.count(Message.id)).scalar()
            
            status_counts = dict(
                session.query(Message.processing_status, func.count(Message.id))
                .group_by(Message.processing_status)
                .all()
            )
            
            return {
                'total_imap_messages': total_imap,
                'total_processed_messages': total_processed,
                'pending_messages': total_imap - total_processed,
                'status_breakdown': status_counts,
                'processing_rate': f"{total_processed}/{total_imap}" if total_imap > 0 else "0/0"
            }