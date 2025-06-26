from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional, List, Dict, Any

Base = declarative_base()

class ImapMailbox(Base):
    """IMAP mailboxes table - read-only for our service"""
    __tablename__ = 'imap_mailboxes'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    name = Column(String, nullable=False)  # Mailbox name like "Inbox", "Sent", etc.
    uidvalidity = Column(Integer, nullable=False)
    uidnext = Column(Integer, default=1)
    special_use = Column(JSONB)
    created_at = Column(TIMESTAMP, server_default=func.now())

class ImapMessage(Base):
    """IMAP messages table - read-only for our service"""
    __tablename__ = 'imap_messages'
    
    id = Column(Integer, primary_key=True)
    mailbox_id = Column(Integer, nullable=False)
    uid = Column(Integer, nullable=False)
    flags = Column(JSONB, default=list)
    internal_date = Column(TIMESTAMP, nullable=False)
    size = Column(Integer, nullable=False)
    body_structure = Column(JSONB)
    envelope = Column(JSONB)
    raw_message = Column(LargeBinary)  # The raw email bytes we'll parse
    created_at = Column(TIMESTAMP, server_default=func.now())

class Contact(Base):
    """Contacts table for normalized email/name management"""
    __tablename__ = 'contacts'
    
    email = Column(Text, primary_key=True)  # Normalized email address
    name = Column(Text)  # Optional current display name
    seen_names = Column(JSONB, default=list)  # Array of historical names seen
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now())

class Message(Base):
    """Messages table - cleaned and parsed emails from imap_messages"""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    imap_message_id = Column(Integer, ForeignKey('imap_messages.id', ondelete='CASCADE'), nullable=False)
    message_id = Column(Text, nullable=False)  # Email Message-ID header
    thread_id = Column(Text)  # Thread identifier from headers
    subject = Column(Text)
    from_email = Column(Text, nullable=False)
    to_emails = Column(JSONB, default=list)  # Array of email addresses
    cc_emails = Column(JSONB, default=list)
    reply_to = Column(Text)
    
    # Raw threading headers for debugging/reprocessing
    email_references = Column(Text)  # References header (renamed from 'references' to avoid SQL keyword)
    in_reply_to = Column(Text)  # In-Reply-To header
    thread_topic = Column(Text)  # Thread-Topic header if present
    date_sent = Column(TIMESTAMP, nullable=False)
    processed_at = Column(TIMESTAMP, server_default=func.now())
    
    # Cleaned content from mail-parser + email_reply_parser
    body_text = Column(Text)  # Cleaned plaintext body
    body_html = Column(Text)  # Original HTML (may be null)
    
    # Participant information (normalized)
    participants = Column(JSONB, default=list)  # Array of {email, name} objects
    
    # Email classification (from AI step)
    category = Column(Text)  # personal/promotional/automated
    confidence = Column(Float)  # classification confidence score 0.0-1.0
    
    # Language classification
    language = Column(Text)  # ISO 639-1 language code (en, es, fr, etc.)
    language_confidence = Column(Float)  # language detection confidence score 0.0-1.0
    
    # Granular processing pipeline stages
    parsed_at = Column(TIMESTAMP)  # mail-parser completed
    cleaned_at = Column(TIMESTAMP)  # email_reply_parser completed
    classified_at = Column(TIMESTAMP)  # AI categorization completed
    embedded_at = Column(TIMESTAMP)  # Unstructured + embeddings completed
    processing_status = Column(Text, default='pending')  # pending/processing/completed/failed
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now())

class Conversation(Base):
    """Conversations table - threaded email conversations"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Text, nullable=False, unique=True)  # Unique thread identifier
    genesis_message_id = Column(Text, nullable=False)  # Message-ID of first message in thread
    subject_normalized = Column(Text, nullable=False)  # Normalized subject (Re:, Fwd: removed)
    
    # Conversation metadata
    participants = Column(JSONB, default=list)  # Denormalized participant list
    message_count = Column(Integer, default=0)
    first_message_date = Column(TIMESTAMP)
    last_message_date = Column(TIMESTAMP)
    
    # AI-generated summary (updated on each new message)
    summary = Column(Text)  # One-liner summary of the conversation
    key_topics = Column(JSONB, default=list)  # Array of key topics/entities
    summary_model_info = Column(JSONB, default=dict)  # Tracks model, version, and parameters used for summary
    
    # Processing status
    summary_generated_at = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now())

class MessageChunk(Base):
    """Message chunks table - for RAG embeddings storage"""
    __tablename__ = 'message_chunks'
    
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey('messages.id', ondelete='CASCADE'), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order within the message
    text_content = Column(Text, nullable=False)  # The chunked text
    
    # Unstructured.io metadata
    element_type = Column(Text)  # Title, NarrativeText, etc.
    chunk_metadata = Column(JSONB, default=dict)  # Additional unstructured metadata
    
    # Vector embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding = Column(Vector(384))
    
    # Processing metadata
    processed_at = Column(TIMESTAMP, server_default=func.now())
    created_at = Column(TIMESTAMP, server_default=func.now())