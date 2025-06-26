from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from email_reply_parser import EmailReplyParser
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging
import json

from models import ImapMessage, Message, Contact, MessageChunk
from database import get_db_session
from language_classifier import LanguageClassifier
# Removed: LLM classifier and threading service imports (now run on host)

# We'll import the Rust email parser once it's built
# from email_parser_py import parse_email_bytes

logger = logging.getLogger(__name__)

class EmailProcessor:
    """Email cleaning and processing service"""
    
    def __init__(self):
        self.reply_parser = EmailReplyParser()
        self.language_classifier = LanguageClassifier()
        self._embedding_model = None  # Lazy load to avoid startup delays
        # Removed: Threading service and LLM classifier initialization (now run on host)
    
    def _get_embedding_model(self):
        """Get cached embedding model, loading if necessary"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading sentence-transformers model (this may take a moment on first run)")
                self._embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embedding_model
    
    def process_unprocessed_emails(self, limit: int = 100) -> int:
        """Process emails that haven't been parsed yet"""
        import os
        
        processed_count = 0
        
        # Get mailbox filter from environment (default to "Inbox")
        target_mailbox = os.getenv("MAILBOX_FILTER", "Inbox")
        
        with get_db_session() as session:
            # Find unprocessed IMAP messages filtered by mailbox
            from models import ImapMailbox
            
            query = session.query(ImapMessage).select_from(ImapMessage).join(ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id).filter(
                ~session.query(Message.imap_message_id).filter(
                    Message.imap_message_id == ImapMessage.id
                ).exists()
            )
            
            # Apply mailbox filter
            if target_mailbox != "ALL":
                query = query.filter(ImapMailbox.name == target_mailbox)
            
            # Order by most recent first to prioritize new emails
            unprocessed = query.order_by(ImapMessage.created_at.desc()).limit(limit).all()
            
            if target_mailbox == "ALL":
                logger.info(f"Found {len(unprocessed)} unprocessed emails across all mailboxes")
            else:
                logger.info(f"Found {len(unprocessed)} unprocessed emails in '{target_mailbox}' mailbox")
            
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
            # Parse email using Python email library
            parsed_email = self._parse_email_with_python(imap_msg.raw_message)
            
            # Clean the email body
            cleaned_body = self._clean_email_body(parsed_email.get('body_text', ''))
            
            # Classify language
            language_result = self.language_classifier.classify_email_content(
                subject=parsed_email.get('subject', ''),
                body=cleaned_body
            )
            
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
                language=language_result.get('language_code'),
                language_confidence=language_result.get('confidence'),
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
    
    def _parse_email_with_python(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Parse email using Python email library"""
        return self._parse_email_with_python_implementation(raw_bytes)
    
    def _parse_email_with_python_implementation(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Fallback Python email parsing implementation"""
        import email
        from email.utils import parseaddr, getaddresses
        from email.header import decode_header
        
        def decode_mime_header(header_value):
            """Decode MIME-encoded header values like =?UTF-8?Q?...?="""
            if not header_value:
                return header_value
            
            # Handle Header objects by converting to string first
            if hasattr(header_value, '__str__'):
                header_value = str(header_value)
            
            try:
                decoded_parts = decode_header(header_value)
                decoded_string = ""
                for part, encoding in decoded_parts:
                    if isinstance(part, bytes):
                        if encoding and encoding.lower() != 'unknown-8bit':
                            try:
                                decoded_string += part.decode(encoding, errors='ignore')
                            except (LookupError, UnicodeDecodeError):
                                # Fallback to utf-8 for unknown/invalid encodings
                                decoded_string += part.decode('utf-8', errors='ignore')
                        else:
                            # Handle unknown-8bit or no encoding
                            decoded_string += part.decode('utf-8', errors='ignore')
                    else:
                        decoded_string += str(part)
                return decoded_string.strip()
            except Exception as e:
                logger.warning(f"Failed to decode header '{header_value}': {e}")
                return str(header_value) if header_value else ""
        
        def decode_address_header(header_value):
            """Decode address headers with proper MIME decoding"""
            if not header_value:
                return []
            
            try:
                addresses = getaddresses([header_value])
                decoded_addresses = []
                for name, email_addr in addresses:
                    decoded_name = decode_mime_header(name) if name else None
                    decoded_addresses.append((decoded_name, email_addr))
                return decoded_addresses
            except Exception as e:
                logger.warning(f"Failed to decode address header '{header_value}': {e}")
                return []
        
        try:
            msg = email.message_from_bytes(raw_bytes)
            
            # Extract participants with proper decoding
            participants = []
            
            # From address
            from_header = msg.get('From', '')
            from_addrs = decode_address_header(from_header)
            for name, email_addr in from_addrs:
                if email_addr:
                    participants.append({'email': email_addr, 'name': name})
            
            # To addresses
            to_header = msg.get('To', '')
            to_addrs = decode_address_header(to_header)
            for name, email_addr in to_addrs:
                if email_addr:
                    participants.append({'email': email_addr, 'name': name})
            
            # CC addresses
            cc_header = msg.get('Cc', '')
            cc_addrs = decode_address_header(cc_header)
            for name, email_addr in cc_addrs:
                if email_addr:
                    participants.append({'email': email_addr, 'name': name})
            
            # Extract body text
            body_text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_text = payload.decode('utf-8', errors='ignore')
                            # Remove NULL bytes which cause PostgreSQL errors
                            body_text = body_text.replace('\x00', '')
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_text = payload.decode('utf-8', errors='ignore')
                    # Remove NULL bytes which cause PostgreSQL errors
                    body_text = body_text.replace('\x00', '')
            
            # Get first from address for main fields
            main_from_email = from_addrs[0][1] if from_addrs else ''
            main_from_name = from_addrs[0][0] if from_addrs else None
            
            # Extract headers safely
            raw_subject = msg.get('Subject', '')
            raw_reply_to = msg.get('Reply-To', '')
            raw_thread_topic = msg.get('Thread-Topic', '')
            
            return {
                'message_id': msg.get('Message-ID', ''),
                'subject': decode_mime_header(raw_subject) or '',
                'from_email': main_from_email,
                'from_name': main_from_name,
                'to_emails': [addr[1] for addr in to_addrs if addr[1]],
                'cc_emails': [addr[1] for addr in cc_addrs if addr[1]],
                'reply_to': decode_mime_header(raw_reply_to) if raw_reply_to else None,
                'references': msg.get('References'),
                'in_reply_to': msg.get('In-Reply-To'),
                'thread_topic': decode_mime_header(raw_thread_topic) if raw_thread_topic else None,
                'date_sent': msg.get('Date'),
                'body_text': body_text,
                'body_html': None,  # TODO: Extract HTML body
                'participants': participants
            }
        except Exception as e:
            logger.error(f"Failed to parse email with Python parser: {e}")
            return {}
    
    def _clean_email_body(self, body_text: str) -> str:
        """Clean email body using email_reply_parser"""
        if not body_text:
            return ""
        
        try:
            # Remove quoted replies and signatures
            cleaned = EmailReplyParser.parse_reply(body_text)
            result = cleaned.strip() if cleaned else body_text
            
            # Debug: Log when quotes are removed
            if len(result) < len(body_text) * 0.8:  # More than 20% reduction
                logger.info(f"Removed quotes: {len(body_text)} -> {len(result)} chars")
            
            return result
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
        import os
        
        # Get mailbox filter from environment (default to "Inbox")
        target_mailbox = os.getenv("MAILBOX_FILTER", "Inbox")
        
        with get_db_session() as session:
            from models import ImapMailbox
            
            # Count total IMAP messages (filtered by mailbox)
            imap_query = session.query(func.count(ImapMessage.id)).select_from(ImapMessage).join(ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id)
            if target_mailbox != "ALL":
                imap_query = imap_query.filter(ImapMailbox.name == target_mailbox)
            total_imap = imap_query.scalar()
            
            # Count processed messages (all processed, regardless of filter)
            total_processed = session.query(func.count(Message.id)).scalar()
            
            # Count pending messages (total - processed)
            processed_imap_ids = session.query(Message.imap_message_id).subquery()
            pending_query = session.query(func.count(ImapMessage.id)).select_from(ImapMessage).join(ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id).filter(
                ~ImapMessage.id.in_(session.query(processed_imap_ids.c.imap_message_id))
            )
            if target_mailbox != "ALL":
                pending_query = pending_query.filter(ImapMailbox.name == target_mailbox)
            pending_in_target = pending_query.scalar()
            
            status_counts = dict(
                session.query(Message.processing_status, func.count(Message.id))
                .group_by(Message.processing_status)
                .all()
            )
            
            # Language distribution
            language_counts = dict(
                session.query(Message.language, func.count(Message.id))
                .group_by(Message.language)
                .all()
            )
            
            # Language detection stats
            total_with_language = session.query(func.count(Message.id)).filter(Message.language.isnot(None)).scalar()
            avg_language_confidence = session.query(func.avg(Message.language_confidence)).filter(Message.language_confidence.isnot(None)).scalar()
            
            return {
                'mailbox_filter': target_mailbox,
                'total_imap_messages': total_imap,
                'total_processed_messages': total_processed,
                'pending_messages': pending_in_target,
                'status_breakdown': status_counts,
                'processing_rate': f"{total_processed}/{total_imap}" if total_imap > 0 else "0/0",
                'language_distribution': language_counts,
                'language_detection_rate': f"{total_with_language}/{total_processed}" if total_processed > 0 else "0/0",
                'avg_language_confidence': round(avg_language_confidence, 3) if avg_language_confidence else None
            }
    
    # === RAG INDEXING METHODS ===
    
    def process_unembedded_personal_emails(self, limit: int = 10) -> int:
        """Process personal emails that need RAG indexing"""
        processed_count = 0
        
        with get_db_session() as session:
            # Find personal messages that need embedding
            unembedded = session.query(Message).filter(
                Message.category == 'personal',
                Message.embedded_at.is_(None),
                Message.body_text.isnot(None),
                Message.body_text != ''
            ).order_by(Message.processed_at.asc()).limit(limit).all()
            
            logger.info(f"Found {len(unembedded)} personal emails to embed")
            
            for message in unembedded:
                try:
                    if self.embed_single_message(session, message):
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to embed message {message.id}: {e}")
        
        return processed_count
    
    def embed_single_message(self, session: Session, message: Message) -> bool:
        """Generate embeddings for a single message"""
        try:
            # Import RAG dependencies here to avoid startup issues
            import subprocess
            import tempfile
            import json
            import os
            
            # Ensure message is properly attached to this session
            if message not in session:
                message = session.merge(message)
            
            logger.info(f"Embedding message {message.id}: {message.subject[:50]}...")
            
            # Step 1: Chunk the text using Unstructured container
            chunks = self._chunk_text_with_unstructured(message.body_text)
            if not chunks:
                logger.warning(f"No chunks generated for message {message.id}")
                # Still mark as embedded to avoid reprocessing
                message.embedded_at = datetime.now()
                session.commit()
                return True
            
            # Step 2: Generate embeddings using cached model
            model = self._get_embedding_model()
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = model.encode(chunk_texts, convert_to_tensor=False)
            
            # Step 3: Store chunks and embeddings
            chunk_records = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_record = MessageChunk(
                    message_id=message.id,
                    chunk_index=chunk['chunk_index'],
                    text_content=chunk['text'],
                    element_type=chunk.get('element_type', 'NarrativeText'),
                    chunk_metadata=chunk.get('metadata', {}),
                    embedding=embedding.tolist()
                )
                chunk_records.append(chunk_record)
            
            session.add_all(chunk_records)
            
            # Update message embedded_at timestamp
            message.embedded_at = datetime.now()
            session.commit()
            
            logger.info(f"Stored {len(chunk_records)} chunks for message {message.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed message {message.id}: {e}")
            session.rollback()
            return False
    
    def _chunk_text_with_unstructured(self, text: str) -> List[Dict]:
        """Process text with Unstructured library directly (no Docker)"""
        if not text or not text.strip():
            return []
        
        try:
            # Import Unstructured libraries directly
            from unstructured.partition.text import partition_text
            from unstructured.chunking.basic import chunk_elements
            
            # Process the text directly 
            elements = partition_text(text=text)
            chunks = chunk_elements(
                elements, 
                max_characters=1000,
                overlap=200
            )
            
            # Convert to JSON serializable format
            result = []
            for i, chunk in enumerate(chunks):
                result.append({
                    "text": str(chunk),
                    "element_type": chunk.category if hasattr(chunk, 'category') else 'NarrativeText',
                    "metadata": chunk.metadata.to_dict() if hasattr(chunk, 'metadata') and chunk.metadata else {},
                    "chunk_index": i
                })
            
            logger.debug(f"Generated {len(result)} chunks from {len(text)} characters")
            return result
            
        except ImportError as e:
            logger.error(f"Unstructured library not available: {e}")
            logger.error("Install with: uv add 'unstructured[local-inference]'")
            return []
        except Exception as e:
            logger.error(f"Unstructured processing failed: {e}")
            return []
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG indexing statistics"""
        with get_db_session() as session:
            total_personal = session.query(func.count(Message.id)).filter(
                Message.category == 'personal'
            ).scalar()
            
            embedded = session.query(func.count(Message.id)).filter(
                Message.category == 'personal',
                Message.embedded_at.isnot(None)
            ).scalar()
            
            pending = total_personal - embedded
            
            total_chunks = session.query(func.count(MessageChunk.id)).scalar()
            
            return {
                'total_personal_messages': total_personal,
                'embedded_messages': embedded,
                'pending_embedding': pending,
                'total_chunks': total_chunks,
                'embedding_rate': f"{embedded}/{total_personal}" if total_personal > 0 else "0/0"
            }
    
    # Removed: All classification and threading methods (now run on host)
    # - process_unclassified_emails()
    # - classify_single_message()
    # - classify_batch_messages()
    # - _mark_classification_failed()
    # - get_classification_stats()
    # - process_unthreaded_personal_emails()
    # - get_threading_stats()