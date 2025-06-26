#!/usr/bin/env python3

from models import Message
from database import get_db_session
from sqlalchemy import text

# Test database access
print("Testing database column access:")

with get_db_session() as session:
    # Try a simple query first
    try:
        result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'messages' ORDER BY ordinal_position")).fetchall()
        print("Database columns in messages table:")
        for row in result:
            print(f"  {row[0]}")
    except Exception as e:
        print(f"Database query failed: {e}")
    
    # Try to query messages with SQLAlchemy ORM
    try:
        message = session.query(Message).first()
        print(f"\nFirst message: {message}")
    except Exception as e:
        print(f"\nMessage query failed: {e}")
        print(f"Error type: {type(e)}")
        
    # Try to query existing message by ID
    try:
        existing = session.query(Message).filter(Message.imap_message_id == 50).first()
        print(f"\nExisting message with imap_message_id=50: {existing}")
    except Exception as e:
        print(f"\nExisting message query failed: {e}")
        print(f"Error type: {type(e)}")
        # Show the actual SQL being generated
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        import os
        engine = create_engine(os.getenv('DATABASE_URL'), echo=True)
        Session = sessionmaker(bind=engine)
        debug_session = Session()
        try:
            debug_session.query(Message).filter(Message.imap_message_id == 50).first()
        except Exception as debug_e:
            print(f"Debug error: {debug_e}")
        finally:
            debug_session.close()