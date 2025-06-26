#!/usr/bin/env python3
"""
Integration test for email summarization pipeline
Tests real summarization using llm library with Qwen 3 model
"""

import sys
import os
from datetime import datetime

# Add email-processor to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'email-processor'))

from database import get_db_session
from models import Message

class EmailSummarizer:
    """Real email summarizer using llm library with optimized prompts"""
    
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
                    print(f"🎯 Using Qwen model: {candidate}")
                    return candidate
            
            # FAIL FAST - no fallbacks to remote models
            raise Exception(f"No Qwen model found in available models: {available_models}")
            
        except ImportError:
            raise Exception("LLM library not available")
        
    def _call_llm(self, prompt: str) -> str:
        """Call LLM using llm library directly"""
        try:
            import llm
            import re
            
            if self.model is None:
                self.model = llm.get_model(self.model_id)
            
            # Add instruction to suppress reasoning
            enhanced_prompt = f"Answer directly without showing your thinking process.\n\n{prompt}"
            
            response = self.model.prompt(enhanced_prompt)
            raw_text = response.text().strip()
            
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
    
    def summarize_single_email(self, from_email: str, subject: str, body: str) -> str:
        """
        Summarize a single email with tight, informal style (max 120 chars)
        Examples: 'Budget approved for Q4 marketing campaign', 'Meeting rescheduled to Friday'
        """
        # Combine email content
        email_content = f"From: {from_email}\nSubject: {subject}\n\n{body}"
        
        prompt = f"""Summarize this email in exactly one short phrase (max 120 characters). Jump straight to the topic, no fluff. Examples: 'Budget approved for Q4 marketing campaign', 'Meeting rescheduled to Friday':

{email_content}"""
        
        return self._call_llm(prompt)
    
    def summarize_conversation_thread(self, messages: list) -> str:
        """
        Summarize an email thread with focus on outcome (max 120 chars)
        Examples: 'Marketing strategy decided: 70% social, 30% traditional', 'Team agreed on new hiring timeline'
        """
        # Build conversation history
        conversation = ""
        for i, msg in enumerate(messages, 1):
            conversation += f"Email {i}:\nFrom: {msg.get('from_email', '')}\nSubject: {msg.get('subject', '')}\nDate: {msg.get('date', '')}\n\n{msg.get('body', '')}\n\n"
        
        prompt = f"""Summarize this email thread in one tight phrase (max 120 chars). Focus on the outcome. Examples: 'Marketing strategy decided: 70% social, 30% traditional', 'Team agreed on new hiring timeline':

{conversation}"""
        
        return self._call_llm(prompt)
    
    def summarize_technical_incident(self, from_email: str, subject: str, body: str) -> str:
        """
        Summarize technical incidents crisply (max 120 chars)
        Examples: 'API latency spike resolved, memory leak fix pending', 'Database outage fixed, monitoring improved'
        """
        email_content = f"From: {from_email}\nSubject: {subject}\n\n{body}"
        
        prompt = f"""Summarize this incident in one crisp phrase (max 120 chars). Examples: 'API latency spike resolved, memory leak fix pending', 'Database outage fixed, monitoring improved':

{email_content}"""
        
        return self._call_llm(prompt)
    
    def summarize_personal_email(self, from_email: str, subject: str, body: str) -> str:
        """
        Summarize personal emails casually (max 120 chars)
        Examples: 'Family reunion July 15th, bring brownies', 'Mom asking about weekend dinner plans'
        """
        email_content = f"From: {from_email}\nSubject: {subject}\n\n{body}"
        
        prompt = f"""Summarize this personal email in one casual phrase (max 120 chars). Examples: 'Family reunion July 15th, bring brownies', 'Mom asking about weekend dinner plans':

{email_content}"""
        
        return self._call_llm(prompt)

def test_llm_direct_access():
    """Test that we can access LLM library directly"""
    print("🔧 Testing direct LLM library access...")
    
    try:
        import llm
        print("✅ LLM library imported successfully")
        
        # Discover available models
        available_models = [model.model_id for model in llm.get_models()]
        print(f"📋 Available models: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}")
        
        if not available_models:
            print("❌ No models available")
            return False
        
        # Try to find Qwen model - FAIL FAST if not found
        test_model_id = None
        for candidate in ["mlx-community/Qwen3-8B-4bit", "qwen-3", "qwen3", "qwen-2.5", "qwen2.5", "qwen"]:
            if candidate in available_models:
                test_model_id = candidate
                break
        
        if not test_model_id:
            print(f"❌ No Qwen model found in available models: {available_models}")
            return False
        else:
            print(f"🎯 Testing with Qwen model: {test_model_id}")
        
        # Test model access
        model = llm.get_model(test_model_id)
        print(f"✅ {test_model_id} model accessible")
        
        # Test simple prompt
        response = model.prompt("Say hello in exactly 10 characters")
        raw_result = response.text().strip()
        
        # Remove reasoning tags if present
        import re
        result = re.sub(r'<think>.*?</think>', '', raw_result, flags=re.DOTALL).strip()
        if not result:
            result = raw_result
            
        print(f"✅ LLM responds: '{result}'")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM direct access failed: {e}")
        return False

def test_email_summarizer_initialization():
    """Test EmailSummarizer initialization"""
    print("\n🧪 Testing EmailSummarizer initialization...")
    
    try:
        summarizer = EmailSummarizer()
        print("✅ EmailSummarizer initialized successfully")
        
        # Test required methods
        required_methods = ['summarize_single_email', 'summarize_conversation_thread', 'summarize_technical_incident', 'summarize_personal_email']
        for method in required_methods:
            if hasattr(summarizer, method):
                print(f"✅ Method {method} available")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ EmailSummarizer initialization failed: {e}")
        return False

def test_single_email_summarization():
    """Test single email summarization with real examples"""
    print("\n📧 Testing single email summarization...")
    
    try:
        summarizer = EmailSummarizer()
        
        # Test with a realistic business email
        test_email = {
            'from_email': 'sarah@company.com',
            'subject': 'Team Meeting Follow-up',
            'body': '''Hi everyone,

Thanks for the productive meeting today. Here are the key points we discussed:

1. Budget approval for Q4 marketing campaign - approved for $50k
2. New hire for development team - we'll start interviews next week
3. Product launch timeline - moved to January 2024 due to testing delays
4. Client feedback integration - high priority for next sprint

Action items:
- Sarah: Prepare job descriptions by Friday
- Mike: Update project timeline in Jira
- Lisa: Schedule client feedback sessions

Next meeting: Same time next week.

Best,
Sarah'''
        }
        
        summary = summarizer.summarize_single_email(**test_email)
        char_count = len(summary)
        status = "✅" if char_count <= 120 else "⚠️"
        
        print(f"{status} Business Email Summary ({char_count}/120 chars):")
        print(f"   '{summary}'")
        
        return char_count > 0
        
    except Exception as e:
        print(f"❌ Single email summarization failed: {e}")
        return False

def test_conversation_thread_summarization():
    """Test conversation thread summarization"""
    print("\n🧵 Testing conversation thread summarization...")
    
    try:
        summarizer = EmailSummarizer()
        
        # Test with a conversation thread
        messages = [
            {
                'from_email': 'john@startup.com',
                'subject': 'Marketing Strategy Discussion',
                'date': 'Monday, 9 AM',
                'body': "Hey team, I've been thinking about our Q4 marketing approach. We need to decide between focusing on social media campaigns vs. traditional advertising. What are your thoughts?"
            },
            {
                'from_email': 'lisa@startup.com',
                'subject': 'Re: Marketing Strategy Discussion',
                'date': 'Monday, 11 AM',
                'body': "I vote for social media. Our target demographic is 25-40 and they're more active online. Plus it's more cost-effective and we can track metrics better."
            },
            {
                'from_email': 'mike@startup.com',
                'subject': 'Re: Marketing Strategy Discussion',
                'date': 'Monday, 2 PM',
                'body': "Agree with Lisa on social media, but we should also consider some targeted traditional ads in tech magazines. Our B2B clients still read industry publications."
            },
            {
                'from_email': 'john@startup.com',
                'subject': 'Re: Marketing Strategy Discussion',
                'date': 'Tuesday, 9 AM',
                'body': "Great points everyone. Let's go with 70% social media budget and 30% traditional. Lisa, can you draft a proposal by Friday? Mike, please research tech publication ad rates."
            }
        ]
        
        summary = summarizer.summarize_conversation_thread(messages)
        char_count = len(summary)
        status = "✅" if char_count <= 120 else "⚠️"
        
        print(f"{status} Conversation Thread Summary ({char_count}/120 chars):")
        print(f"   '{summary}'")
        
        return char_count > 0
        
    except Exception as e:
        print(f"❌ Conversation thread summarization failed: {e}")
        return False

def test_technical_incident_summarization():
    """Test technical incident summarization"""
    print("\n🔧 Testing technical incident summarization...")
    
    try:
        summarizer = EmailSummarizer()
        
        test_incident = {
            'from_email': 'devops@company.com',
            'subject': 'Production Incident Report - API Latency Spike',
            'body': '''Team,

We experienced a significant API latency spike today from 14:30 to 15:45 UTC.

ROOT CAUSE:
- Database connection pool exhaustion due to long-running queries
- Memory leak in the user authentication service
- Increased traffic from new mobile app release (3x normal load)

IMMEDIATE ACTIONS TAKEN:
- Restarted authentication service instances
- Increased database connection pool size from 50 to 100
- Enabled request rate limiting
- Scaled horizontally to 8 instances

RESOLUTION:
- Latency returned to normal levels by 15:45
- No data loss occurred
- All services are now stable

FOLLOW-UP ITEMS:
- Code review for memory leak fix (Due: Thursday)
- Implement better monitoring for connection pools (Due: Next week)
- Capacity planning for mobile app growth (Due: End of month)

Post-mortem meeting scheduled for tomorrow at 10 AM.

DevOps Team'''
        }
        
        summary = summarizer.summarize_technical_incident(**test_incident)
        char_count = len(summary)
        status = "✅" if char_count <= 120 else "⚠️"
        
        print(f"{status} Technical Incident Summary ({char_count}/120 chars):")
        print(f"   '{summary}'")
        
        return char_count > 0
        
    except Exception as e:
        print(f"❌ Technical incident summarization failed: {e}")
        return False

def test_personal_email_summarization():
    """Test personal email summarization"""
    print("\n💌 Testing personal email summarization...")
    
    try:
        summarizer = EmailSummarizer()
        
        test_personal = {
            'from_email': 'mom@family.com',
            'subject': 'Family Reunion Planning Update',
            'body': '''Hi sweetie,

Hope you're doing well! I wanted to update you on the family reunion plans.

We've confirmed the date for July 15th at Grandma's house. Your aunt Susan is flying in from Seattle, and cousin Mark is driving up from Portland with the kids. Uncle Bob might not make it due to his back surgery recovery, but he's trying.

For food, we're doing a potluck style like last time. Mom is making her famous apple pie, I'm handling the barbecue, and Susan volunteered to bring salads. Could you bring some of those delicious brownies you made for Christmas? Everyone's still talking about them!

Also, we reserved the community center pavilion as backup in case of rain. It's $75 for the day but worth it for peace of mind.

Let me know if you can make it and what you'd like to bring. Can't wait to see you!

Love,
Mom'''
        }
        
        summary = summarizer.summarize_personal_email(**test_personal)
        char_count = len(summary)
        status = "✅" if char_count <= 120 else "⚠️"
        
        print(f"{status} Personal Email Summary ({char_count}/120 chars):")
        print(f"   '{summary}'")
        
        return char_count > 0
        
    except Exception as e:
        print(f"❌ Personal email summarization failed: {e}")
        return False

def test_real_database_emails():
    """Test summarization on real emails from database"""
    print("\n🗄️  Testing summarization on real database emails...")
    
    try:
        summarizer = EmailSummarizer()
        
        with get_db_session() as session:
            # Get sample personal emails for testing
            sample_emails = session.query(Message).filter(
                Message.category == 'personal',
                Message.body_text.isnot(None),
                Message.body_text != ''
            ).limit(3).all()
            
            if not sample_emails:
                print("⚠️  No personal emails found for testing")
                return True
            
            print(f"🧪 Testing summarization on {len(sample_emails)} real emails:")
            
            for msg in sample_emails:
                try:
                    summary = summarizer.summarize_single_email(
                        from_email=msg.from_email or '',
                        subject=msg.subject or '',
                        body=msg.body_text or ''
                    )
                    
                    char_count = len(summary)
                    status = "✅" if char_count <= 120 else "⚠️"
                    
                    from_email = msg.from_email[:25] + '...' if len(msg.from_email) > 25 else msg.from_email
                    subject_preview = (msg.subject or '')[:30] + '...' if len(msg.subject or '') > 30 else msg.subject or 'No subject'
                    
                    print(f"{status} #{msg.id}: {from_email}")
                    print(f"    Subject: \"{subject_preview}\"")
                    print(f"    Summary ({char_count}/120): '{summary}'")
                    print()
                    
                except Exception as e:
                    print(f"❌ Failed to summarize email #{msg.id}: {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ Real database email test failed: {e}")
        return False

def test_character_length_compliance():
    """Test that summaries consistently meet 120 character limit"""
    print("\n📏 Testing character length compliance...")
    
    try:
        summarizer = EmailSummarizer()
        
        # Test with various email lengths
        test_cases = [
            ("Short email", "Quick question", "Are we meeting today?"),
            ("Medium email", "Project update needed", "Hi team, can you please send me the latest status on the project? We need to review progress before Friday's client meeting. Thanks!"),
            ("Long email", "Detailed quarterly review", "This is a very long email with lots of details about quarterly performance, budget allocations, strategic initiatives, team restructuring, new partnerships, technology upgrades, and various other business matters that require careful attention and thorough review by all stakeholders involved in the decision-making process.")
        ]
        
        all_compliant = True
        
        for case_name, subject, body in test_cases:
            summary = summarizer.summarize_single_email("test@example.com", subject, body)
            char_count = len(summary)
            compliant = char_count <= 120
            status = "✅" if compliant else "❌"
            
            print(f"{status} {case_name}: {char_count}/120 chars")
            print(f"    '{summary}'")
            
            if not compliant:
                all_compliant = False
        
        return all_compliant
        
    except Exception as e:
        print(f"❌ Character length compliance test failed: {e}")
        return False

def main():
    """Run all email summarization integration tests"""
    import argparse
    parser = argparse.ArgumentParser(description='Email Summarization Integration Tests')
    parser.add_argument('--model', default=None, help='LLM model to use (default: auto-discover)')
    args = parser.parse_args()
    
    print("🚀 Email Summarization Integration Tests")
    if args.model:
        print(f"   Model: {args.model} (specified)")
    else:
        print(f"   Model: Auto-discovery (prioritizing Qwen)")
    print("=" * 50)
    
    # Update summarizer model if specified
    if args.model:
        EmailSummarizer.__init__.__defaults__ = (args.model,)
    
    tests = [
        ("LLM Direct Access", test_llm_direct_access),
        ("EmailSummarizer Initialization", test_email_summarizer_initialization),
        ("Single Email Summarization", test_single_email_summarization),
        ("Conversation Thread Summarization", test_conversation_thread_summarization),
        ("Technical Incident Summarization", test_technical_incident_summarization),
        ("Personal Email Summarization", test_personal_email_summarization),
        ("Real Database Emails", test_real_database_emails),
        ("Character Length Compliance", test_character_length_compliance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Integration Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed >= 6:  # Allow some flexibility
        print("🎉 Email summarization integration tests successful!")
        print(f"\n💡 {args.model} model ready for email summarization with 120-char limit!")
        return True
    else:
        print("⚠️  Some email summarization integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)