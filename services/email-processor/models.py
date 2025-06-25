from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any

Base = declarative_base()

class ImapMessage(Base):
    """IMAP messages table - read-only for our service"""
    __tablename__ = 'imap_messages'
    
    id = Column(Integer, primary_key=True)
    mailbox_id = Column(Integer, nullable=False)
    uid = Column(Integer, nullable=False)
    flags = Column(JSONB, default=list)
    internal_date = Column(TIMESTAMP(timezone=True), nullable=False)
    size = Column(Integer, nullable=False)
    body_structure = Column(JSONB)
    envelope = Column(JSONB)
    raw_message = Column(LargeBinary)  # The raw email bytes we'll parse
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

class Contact(Base):
    """Contacts table for normalized email/name management"""
    __tablename__ = 'contacts'
    
    email = Column(Text, primary_key=True)  # Normalized email address
    name = Column(Text)  # Optional current display name
    seen_names = Column(JSONB, default=list)  # Array of historical names seen
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

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
    date_sent = Column(TIMESTAMP(timezone=True), nullable=False)
    date_received = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Cleaned content from mail-parser + email_reply_parser
    body_text = Column(Text)  # Cleaned plaintext body
    body_html = Column(Text)  # Original HTML (may be null)
    
    # Participant information (normalized)
    participants = Column(JSONB, default=list)  # Array of {email, name} objects
    
    # Email classification (from future AI step)
    category = Column(Text)  # personal/promotion/automated
    
    # Granular processing pipeline stages
    parsed_at = Column(TIMESTAMP(timezone=True))  # mail-parser completed
    cleaned_at = Column(TIMESTAMP(timezone=True))  # email_reply_parser completed
    classified_at = Column(TIMESTAMP(timezone=True))  # AI categorization completed
    embedded_at = Column(TIMESTAMP(timezone=True))  # Unstructured + embeddings completed
    processing_status = Column(Text, default='pending')  # pending/processing/completed/failed
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

class Conversation(Base):
    """Conversations table - threaded email conversations"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Text, nullable=False)  # From threading logic
    genesis_message_id = Column(Text)  # The first message that started the thread
    subject_normalized = Column(Text)  # Normalized subject line
    
    # Conversation metadata
    participants = Column(JSONB, default=list)  # Denormalized participant list
    message_count = Column(Integer, default=0)
    first_message_date = Column(TIMESTAMP(timezone=True))
    last_message_date = Column(TIMESTAMP(timezone=True))
    
    # AI-generated summary (from future step)
    summary = Column(Text)  # One-liner summary of the conversation
    
    # Processing status
    summary_generated_at = Column(TIMESTAMP(timezone=True))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())