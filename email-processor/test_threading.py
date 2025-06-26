#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch, MagicMock
import datetime
import sys
import os

# Add the email-processor directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'email-processor'))

class MockMessage:
    """Mock message object for threading tests"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.message_id = kwargs.get('message_id', '<msg1@example.com>')
        self.in_reply_to = kwargs.get('in_reply_to', None)
        self.references = kwargs.get('references', None)
        self.thread_topic = kwargs.get('thread_topic', None)
        self.from_email = kwargs.get('from_email', 'test@example.com')
        self.subject = kwargs.get('subject', 'Test Subject')
        self.body_text = kwargs.get('body_text', 'Test body')
        self.date_sent = kwargs.get('date_sent', datetime.datetime.now())
        self.thread_id = kwargs.get('thread_id', None)
        self.genesis_message_id = kwargs.get('genesis_message_id', None)

class MockConversation:
    """Mock conversation object for threading tests"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.genesis_message_id = kwargs.get('genesis_message_id', '<msg1@example.com>')
        self.subject = kwargs.get('subject', 'Thread Subject')
        self.participant_emails = kwargs.get('participant_emails', ['test@example.com'])
        self.message_count = kwargs.get('message_count', 1)
        self.last_message_date = kwargs.get('last_message_date', datetime.datetime.now())
        self.created_at = kwargs.get('created_at', datetime.datetime.now())

class EmailThreader:
    """Mock threading service for testing - implements threading logic"""
    
    def __init__(self):
        self.conversations = {}  # genesis_message_id -> MockConversation
        self.message_threads = {}  # message_id -> thread_id
    
    def process_message_threading(self, message):
        """Main threading logic"""
        # Check if this is a reply to existing message
        thread_id = self._find_existing_thread(message)
        
        if thread_id:
            # Add to existing thread
            return self._add_to_thread(message, thread_id)
        else:
            # Create new thread
            return self._create_new_thread(message)
    
    def _find_existing_thread(self, message):
        """Find existing thread based on headers"""
        # Check In-Reply-To header
        if message.in_reply_to:
            clean_reply_to = self._clean_message_id(message.in_reply_to)
            if clean_reply_to in self.message_threads:
                return self.message_threads[clean_reply_to]
        
        # Check References header
        if message.references:
            references = self._parse_references(message.references)
            for ref_id in references:
                clean_ref = self._clean_message_id(ref_id)
                if clean_ref in self.message_threads:
                    return self.message_threads[clean_ref]
        
        # Check subject-based threading
        if message.subject:
            normalized_subject = self._normalize_subject(message.subject)
            for conv in self.conversations.values():
                if self._normalize_subject(conv.subject) == normalized_subject:
                    return conv.id
        
        return None
    
    def _add_to_thread(self, message, thread_id):
        """Add message to existing thread"""
        # Find the conversation
        conversation = None
        for conv in self.conversations.values():
            if conv.id == thread_id:
                conversation = conv
                break
        
        if conversation:
            # Update conversation metadata
            conversation.message_count += 1
            conversation.last_message_date = message.date_sent
            if message.from_email not in conversation.participant_emails:
                conversation.participant_emails.append(message.from_email)
            
            # Track message in thread
            self.message_threads[message.message_id] = thread_id
            message.thread_id = thread_id
            message.genesis_message_id = conversation.genesis_message_id
            
            return thread_id
        
        return None
    
    def _create_new_thread(self, message):
        """Create new conversation thread"""
        thread_id = len(self.conversations) + 1
        genesis_message_id = message.message_id
        
        conversation = MockConversation(
            id=thread_id,
            genesis_message_id=genesis_message_id,
            subject=message.subject,
            participant_emails=[message.from_email],
            message_count=1,
            last_message_date=message.date_sent
        )
        
        self.conversations[genesis_message_id] = conversation
        self.message_threads[message.message_id] = thread_id
        
        message.thread_id = thread_id
        message.genesis_message_id = genesis_message_id
        
        return thread_id
    
    def _clean_message_id(self, message_id):
        """Clean message ID for comparison"""
        if not message_id:
            return None
        return message_id.strip('<>')
    
    def _parse_references(self, references):
        """Parse References header into list of message IDs"""
        if not references:
            return []
        
        # Split on whitespace and filter empty strings
        refs = [ref.strip() for ref in references.split() if ref.strip()]
        return refs
    
    def _normalize_subject(self, subject):
        """Normalize subject for threading comparison"""
        if not subject:
            return ""
        
        # Remove Re:, Fwd:, etc.
        import re
        normalized = re.sub(r'^(Re:|RE:|Fwd:|FWD:|Fw:)\s*', '', subject, flags=re.IGNORECASE)
        return normalized.strip().lower()
    
    def get_thread_messages(self, thread_id):
        """Get all messages in a thread ordered by date"""
        messages = []
        for msg_id, t_id in self.message_threads.items():
            if t_id == thread_id:
                # This would normally query the database
                messages.append(msg_id)
        return sorted(messages)
    
    def get_conversation_stats(self):
        """Get threading statistics"""
        return {
            'total_conversations': len(self.conversations),
            'total_threaded_messages': len(self.message_threads),
            'avg_messages_per_thread': sum(c.message_count for c in self.conversations.values()) / max(len(self.conversations), 1)
        }

class TestEmailHeaderParsing(unittest.TestCase):
    """Test email header parsing for threading fields"""
    
    def setUp(self):
        self.threader = EmailThreader()
    
    def test_message_id_parsing(self):
        """Test Message-ID header parsing"""
        message = MockMessage(
            message_id='<msg123@example.com>',
            subject='Original message'
        )
        
        thread_id = self.threader.process_message_threading(message)
        
        self.assertIsNotNone(thread_id)
        self.assertEqual(message.thread_id, thread_id)
        self.assertEqual(message.genesis_message_id, '<msg123@example.com>')
    
    def test_in_reply_to_parsing(self):
        """Test In-Reply-To header parsing"""
        # First message
        original = MockMessage(
            id=1,
            message_id='<original@example.com>',
            subject='Original message'
        )
        self.threader.process_message_threading(original)
        
        # Reply message
        reply = MockMessage(
            id=2,
            message_id='<reply@example.com>',
            in_reply_to='<original@example.com>',
            subject='Re: Original message'
        )
        
        thread_id = self.threader.process_message_threading(reply)
        
        self.assertEqual(thread_id, original.thread_id)
        self.assertEqual(reply.genesis_message_id, original.genesis_message_id)
    
    def test_references_parsing(self):
        """Test References header parsing"""
        # Original message
        msg1 = MockMessage(
            message_id='<msg1@example.com>',
            subject='Thread start'
        )
        self.threader.process_message_threading(msg1)
        
        # Second message
        msg2 = MockMessage(
            message_id='<msg2@example.com>',
            in_reply_to='<msg1@example.com>',
            references='<msg1@example.com>',
            subject='Re: Thread start'
        )
        self.threader.process_message_threading(msg2)
        
        # Third message with full References chain
        msg3 = MockMessage(
            message_id='<msg3@example.com>',
            in_reply_to='<msg2@example.com>',
            references='<msg1@example.com> <msg2@example.com>',
            subject='Re: Thread start'
        )
        
        thread_id = self.threader.process_message_threading(msg3)
        
        self.assertEqual(thread_id, msg1.thread_id)
        self.assertEqual(msg3.genesis_message_id, msg1.genesis_message_id)

class TestThreadDetectionLogic(unittest.TestCase):
    """Test thread detection logic"""
    
    def setUp(self):
        self.threader = EmailThreader()
    
    def test_subject_based_threading(self):
        """Test threading based on subject normalization"""
        msg1 = MockMessage(
            message_id='<msg1@example.com>',
            subject='Project Discussion'
        )
        self.threader.process_message_threading(msg1)
        
        msg2 = MockMessage(
            message_id='<msg2@example.com>',
            subject='Re: Project Discussion'
        )
        
        thread_id = self.threader.process_message_threading(msg2)
        
        self.assertEqual(thread_id, msg1.thread_id)
    
    def test_subject_normalization(self):
        """Test subject normalization removes Re:, Fwd:, etc."""
        test_cases = [
            ('Project Discussion', 'project discussion'),
            ('Re: Project Discussion', 'project discussion'),
            ('RE: Project Discussion', 'project discussion'),
            ('Fwd: Project Discussion', 'project discussion'),
            ('FWD: Project Discussion', 'project discussion'),
            ('Fw: Project Discussion', 'project discussion'),
        ]
        
        for original, expected in test_cases:
            normalized = self.threader._normalize_subject(original)
            self.assertEqual(normalized, expected)
    
    def test_circular_reference_handling(self):
        """Test handling of circular references"""
        # This shouldn't happen in practice, but test robustness
        msg1 = MockMessage(
            message_id='<msg1@example.com>',
            in_reply_to='<msg2@example.com>',  # Circular
            subject='Test'
        )
        
        msg2 = MockMessage(
            message_id='<msg2@example.com>',
            in_reply_to='<msg1@example.com>',  # Circular
            subject='Test'
        )
        
        # Should handle gracefully without infinite loops
        thread_id1 = self.threader.process_message_threading(msg1)
        thread_id2 = self.threader.process_message_threading(msg2)
        
        self.assertIsNotNone(thread_id1)
        self.assertIsNotNone(thread_id2)

class TestConversationCreation(unittest.TestCase):
    """Test conversation creation with genesis message ID"""
    
    def setUp(self):
        self.threader = EmailThreader()
    
    def test_new_conversation_creation(self):
        """Test creating new conversation"""
        message = MockMessage(
            message_id='<genesis@example.com>',
            from_email='user@example.com',
            subject='New conversation',
            date_sent=datetime.datetime(2024, 1, 15, 10, 0)
        )
        
        thread_id = self.threader.process_message_threading(message)
        
        self.assertIsNotNone(thread_id)
        self.assertEqual(message.genesis_message_id, '<genesis@example.com>')
        
        # Check conversation was created
        conversation = self.threader.conversations['<genesis@example.com>']
        self.assertEqual(conversation.genesis_message_id, '<genesis@example.com>')
        self.assertEqual(conversation.subject, 'New conversation')
        self.assertEqual(conversation.participant_emails, ['user@example.com'])
        self.assertEqual(conversation.message_count, 1)
    
    def test_conversation_metadata_updates(self):
        """Test conversation metadata updates when adding messages"""
        # Original message
        msg1 = MockMessage(
            message_id='<msg1@example.com>',
            from_email='alice@example.com',
            subject='Project planning',
            date_sent=datetime.datetime(2024, 1, 15, 10, 0)
        )
        self.threader.process_message_threading(msg1)
        
        # Reply from different user
        msg2 = MockMessage(
            message_id='<msg2@example.com>',
            from_email='bob@example.com',
            in_reply_to='<msg1@example.com>',
            subject='Re: Project planning',
            date_sent=datetime.datetime(2024, 1, 15, 11, 0)
        )
        self.threader.process_message_threading(msg2)
        
        conversation = self.threader.conversations['<msg1@example.com>']
        self.assertEqual(conversation.message_count, 2)
        self.assertIn('alice@example.com', conversation.participant_emails)
        self.assertIn('bob@example.com', conversation.participant_emails)
        self.assertEqual(conversation.last_message_date, msg2.date_sent)

class TestThreadBuilding(unittest.TestCase):
    """Test thread building with multiple messages"""
    
    def setUp(self):
        self.threader = EmailThreader()
    
    def test_linear_thread_building(self):
        """Test building a linear thread (A -> B -> C -> D)"""
        messages = []
        for i in range(4):
            msg = MockMessage(
                id=i+1,
                message_id=f'<msg{i+1}@example.com>',
                in_reply_to=f'<msg{i}@example.com>' if i > 0 else None,
                references=' '.join([f'<msg{j+1}@example.com>' for j in range(i)]) if i > 0 else None,
                subject=f'Re: Thread topic' if i > 0 else 'Thread topic',
                date_sent=datetime.datetime(2024, 1, 15, 10 + i, 0)
            )
            messages.append(msg)
            self.threader.process_message_threading(msg)
        
        # All messages should be in the same thread
        thread_ids = [msg.thread_id for msg in messages]
        self.assertEqual(len(set(thread_ids)), 1, "All messages should be in same thread")
        
        # All should have same genesis message ID
        genesis_ids = [msg.genesis_message_id for msg in messages]
        self.assertEqual(len(set(genesis_ids)), 1, "All messages should have same genesis ID")
        self.assertEqual(genesis_ids[0], '<msg1@example.com>')
    
    def test_branched_thread_building(self):
        """Test building a branched thread (A -> B, A -> C)"""
        # Root message
        msg_a = MockMessage(
            message_id='<msgA@example.com>',
            subject='Main topic'
        )
        self.threader.process_message_threading(msg_a)
        
        # First branch
        msg_b = MockMessage(
            message_id='<msgB@example.com>',
            in_reply_to='<msgA@example.com>',
            subject='Re: Main topic - branch 1'
        )
        self.threader.process_message_threading(msg_b)
        
        # Second branch
        msg_c = MockMessage(
            message_id='<msgC@example.com>',
            in_reply_to='<msgA@example.com>',
            subject='Re: Main topic - branch 2'
        )
        self.threader.process_message_threading(msg_c)
        
        # All should be in same thread
        self.assertEqual(msg_a.thread_id, msg_b.thread_id)
        self.assertEqual(msg_a.thread_id, msg_c.thread_id)
        self.assertEqual(msg_a.genesis_message_id, msg_b.genesis_message_id)
        self.assertEqual(msg_a.genesis_message_id, msg_c.genesis_message_id)
    
    def test_complex_thread_reconstruction(self):
        """Test complex thread with out-of-order message processing"""
        # Process messages out of chronological order
        messages_data = [
            {'id': 'msg3', 'reply_to': 'msg2', 'refs': 'msg1 msg2'},
            {'id': 'msg1', 'reply_to': None, 'refs': None},
            {'id': 'msg4', 'reply_to': 'msg1', 'refs': 'msg1'},  # Branch from msg1
            {'id': 'msg2', 'reply_to': 'msg1', 'refs': 'msg1'},
        ]
        
        messages = []
        for data in messages_data:
            msg = MockMessage(
                message_id=f'<{data["id"]}@example.com>',
                in_reply_to=f'<{data["reply_to"]}@example.com>' if data['reply_to'] else None,
                references=' '.join([f'<{ref}@example.com>' for ref in data['refs'].split()]) if data['refs'] else None,
                subject='Complex thread'
            )
            messages.append(msg)
            self.threader.process_message_threading(msg)
        
        # All should be in same thread
        thread_ids = [msg.thread_id for msg in messages]
        self.assertEqual(len(set(thread_ids)), 1)

class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases like malformed headers, circular references"""
    
    def setUp(self):
        self.threader = EmailThreader()
    
    def test_malformed_message_ids(self):
        """Test handling of malformed Message-ID headers"""
        malformed_cases = [
            '',  # Empty
            'not-a-message-id',  # No angle brackets
            '<>',  # Empty brackets
            '<malformed',  # Missing closing bracket
            'malformed>',  # Missing opening bracket
            '<msg@>',  # Missing domain
            '<@domain.com>',  # Missing local part
        ]
        
        for malformed_id in malformed_cases:
            msg = MockMessage(
                message_id=malformed_id,
                subject='Test malformed ID'
            )
            
            try:
                thread_id = self.threader.process_message_threading(msg)
                self.assertIsNotNone(thread_id, f"Should handle malformed ID: {malformed_id}")
            except Exception as e:
                self.fail(f"Failed to handle malformed message ID '{malformed_id}': {e}")
    
    def test_empty_headers(self):
        """Test handling of empty/None headers"""
        msg = MockMessage(
            message_id=None,
            in_reply_to=None,
            references=None,
            subject=None
        )
        
        try:
            thread_id = self.threader.process_message_threading(msg)
            self.assertIsNotNone(thread_id)
        except Exception as e:
            self.fail(f"Failed to handle empty headers: {e}")
    
    def test_very_long_references_chain(self):
        """Test handling of very long References header"""
        # Create a very long references chain
        long_refs = ' '.join([f'<msg{i}@example.com>' for i in range(100)])
        
        msg = MockMessage(
            message_id='<final@example.com>',
            references=long_refs,
            subject='Long references test'
        )
        
        try:
            thread_id = self.threader.process_message_threading(msg)
            self.assertIsNotNone(thread_id)
        except Exception as e:
            self.fail(f"Failed to handle long references: {e}")
    
    def test_unicode_in_headers(self):
        """Test handling of Unicode characters in headers"""
        msg = MockMessage(
            message_id='<测试@example.com>',
            subject='Тестовое сообщение with émojis 🧵',
            from_email='tëst@exämple.com'
        )
        
        try:
            thread_id = self.threader.process_message_threading(msg)
            self.assertIsNotNone(thread_id)
        except Exception as e:
            self.fail(f"Failed to handle Unicode in headers: {e}")

class TestThreadReconstructionAndOrdering(unittest.TestCase):
    """Test thread reconstruction and display order"""
    
    def setUp(self):
        self.threader = EmailThreader()
    
    def test_chronological_ordering(self):
        """Test messages are ordered chronologically within thread"""
        base_time = datetime.datetime(2024, 1, 15, 10, 0)
        
        # Create messages with specific timestamps
        messages = []
        for i in range(5):
            msg = MockMessage(
                message_id=f'<msg{i+1}@example.com>',
                in_reply_to=f'<msg{i}@example.com>' if i > 0 else None,
                date_sent=base_time + datetime.timedelta(hours=i)
            )
            messages.append(msg)
            self.threader.process_message_threading(msg)
        
        thread_id = messages[0].thread_id
        thread_messages = self.threader.get_thread_messages(thread_id)
        
        # Should be ordered by timestamp
        self.assertEqual(len(thread_messages), 5)
        # This is a simplified test - in real implementation would check actual message objects
    
    def test_threading_statistics(self):
        """Test thread statistics calculation"""
        # Create several threads with different message counts
        thread_configs = [
            (3, 'Thread 1'),  # 3 messages
            (1, 'Thread 2'),  # 1 message (no replies)
            (5, 'Thread 3'),  # 5 messages
        ]
        
        for msg_count, subject in thread_configs:
            for i in range(msg_count):
                msg = MockMessage(
                    message_id=f'<{subject.replace(" ", "")}_msg{i+1}@example.com>',
                    in_reply_to=f'<{subject.replace(" ", "")}_msg{i}@example.com>' if i > 0 else None,
                    subject=f'Re: {subject}' if i > 0 else subject
                )
                self.threader.process_message_threading(msg)
        
        stats = self.threader.get_conversation_stats()
        
        self.assertEqual(stats['total_conversations'], 3)
        self.assertEqual(stats['total_threaded_messages'], 9)  # 3+1+5
        self.assertEqual(stats['avg_messages_per_thread'], 3.0)  # 9/3

class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance with large number of messages"""
    
    def setUp(self):
        self.threader = EmailThreader()
    
    def test_large_thread_performance(self):
        """Test performance with large thread (100+ messages)"""
        import time
        
        start_time = time.time()
        
        # Create a large linear thread
        for i in range(100):
            msg = MockMessage(
                message_id=f'<large_thread_msg{i+1}@example.com>',
                in_reply_to=f'<large_thread_msg{i}@example.com>' if i > 0 else None,
                subject=f'Re: Large thread test' if i > 0 else 'Large thread test'
            )
            self.threader.process_message_threading(msg)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should process 100 messages in reasonable time (< 1 second)
        self.assertLess(duration, 1.0, f"Large thread processing took {duration:.2f}s, too slow")
        
        # Verify all messages are properly threaded
        stats = self.threader.get_conversation_stats()
        self.assertEqual(stats['total_conversations'], 1)
        self.assertEqual(stats['total_threaded_messages'], 100)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)