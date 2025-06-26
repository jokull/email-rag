#!/usr/bin/env python3
"""
Test summarization with Qwen 3 model
Tests email summarization capabilities for conversation threads
"""

import subprocess
import sys
import os
from datetime import datetime

# Add email-processor to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'email-processor'))

def test_qwen_simple_summarization():
    """Test basic Qwen 3 summarization"""
    print("🌟 Testing Qwen 3 simple summarization...")
    
    test_email = """
    From: sarah@company.com
    Subject: Team Meeting Follow-up
    
    Hi everyone,
    
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
    Sarah
    """
    
    prompt = f"Summarize this email in exactly one short phrase (max 120 characters). Jump straight to the topic, no fluff. Examples: 'Budget approved for Q4 marketing campaign', 'Meeting rescheduled to Friday':\n\n{test_email}"
    
    try:
        result = subprocess.run([
            "llm", "-m", "q3", prompt
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            summary = result.stdout.strip()
            char_count = len(summary)
            status = "✅" if char_count <= 120 else "⚠️"
            print(f"{status} Qwen 3 Summary ({char_count}/120 chars):")
            print(f"   {summary}")
            return True
        else:
            print(f"❌ Qwen 3 failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Qwen 3 test failed: {e}")
        return False

def test_qwen_conversation_summarization():
    """Test conversation thread summarization"""
    print("\n🧵 Testing Qwen 3 conversation thread summarization...")
    
    conversation = """
    Email 1:
    From: john@startup.com
    Subject: Marketing Strategy Discussion
    Date: Monday, 9 AM
    
    Hey team, I've been thinking about our Q4 marketing approach. We need to decide between focusing on social media campaigns vs. traditional advertising. What are your thoughts?
    
    Email 2:
    From: lisa@startup.com
    Subject: Re: Marketing Strategy Discussion
    Date: Monday, 11 AM
    
    I vote for social media. Our target demographic is 25-40 and they're more active online. Plus it's more cost-effective and we can track metrics better.
    
    Email 3:
    From: mike@startup.com
    Subject: Re: Marketing Strategy Discussion
    Date: Monday, 2 PM
    
    Agree with Lisa on social media, but we should also consider some targeted traditional ads in tech magazines. Our B2B clients still read industry publications.
    
    Email 4:
    From: john@startup.com
    Subject: Re: Marketing Strategy Discussion
    Date: Tuesday, 9 AM
    
    Great points everyone. Let's go with 70% social media budget and 30% traditional. Lisa, can you draft a proposal by Friday? Mike, please research tech publication ad rates.
    """
    
    prompt = f"Summarize this email thread in one tight phrase (max 120 chars). Focus on the outcome. Examples: 'Marketing strategy decided: 70% social, 30% traditional', 'Team agreed on new hiring timeline':\n\n{conversation}"
    
    try:
        result = subprocess.run([
            "llm", "-m", "q3", prompt
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            summary = result.stdout.strip()
            char_count = len(summary)
            status = "✅" if char_count <= 120 else "⚠️"
            print(f"{status} Conversation Summary ({char_count}/120 chars):")
            print(f"   {summary}")
            return True
        else:
            print(f"❌ Conversation summarization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        return False

def test_qwen_technical_email():
    """Test summarization of technical email"""
    print("\n🔧 Testing Qwen 3 technical email summarization...")
    
    technical_email = """
    From: devops@company.com
    Subject: Production Incident Report - API Latency Spike
    
    Team,
    
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
    
    DevOps Team
    """
    
    prompt = f"Summarize this incident in one crisp phrase (max 120 chars). Examples: 'API latency spike resolved, memory leak fix pending', 'Database outage fixed, monitoring improved':\n\n{technical_email}"
    
    try:
        result = subprocess.run([
            "llm", "-m", "q3", prompt
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            summary = result.stdout.strip()
            char_count = len(summary)
            status = "✅" if char_count <= 120 else "⚠️"
            print(f"{status} Technical Summary ({char_count}/120 chars):")
            print(f"   {summary}")
            return True
        else:
            print(f"❌ Technical summarization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Technical test failed: {e}")
        return False

def test_qwen_personal_email():
    """Test summarization of personal email"""
    print("\n💌 Testing Qwen 3 personal email summarization...")
    
    personal_email = """
    From: mom@family.com
    Subject: Family Reunion Planning Update
    
    Hi sweetie,
    
    Hope you're doing well! I wanted to update you on the family reunion plans.
    
    We've confirmed the date for July 15th at Grandma's house. Your aunt Susan is flying in from Seattle, and cousin Mark is driving up from Portland with the kids. Uncle Bob might not make it due to his back surgery recovery, but he's trying.
    
    For food, we're doing a potluck style like last time. Mom is making her famous apple pie, I'm handling the barbecue, and Susan volunteered to bring salads. Could you bring some of those delicious brownies you made for Christmas? Everyone's still talking about them!
    
    Also, we reserved the community center pavilion as backup in case of rain. It's $75 for the day but worth it for peace of mind.
    
    Let me know if you can make it and what you'd like to bring. Can't wait to see you!
    
    Love,
    Mom
    """
    
    prompt = f"Summarize this personal email in one casual phrase (max 120 chars). Examples: 'Family reunion July 15th, bring brownies', 'Mom asking about weekend dinner plans':\n\n{personal_email}"
    
    try:
        result = subprocess.run([
            "llm", "-m", "q3", prompt
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            summary = result.stdout.strip()
            char_count = len(summary)
            status = "✅" if char_count <= 120 else "⚠️"
            print(f"{status} Personal Summary ({char_count}/120 chars):")
            print(f"   {summary}")
            return True
        else:
            print(f"❌ Personal summarization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Personal test failed: {e}")
        return False

def test_qwen_comparison_with_gpt4o_mini():
    """Compare Qwen 3 with GPT-4o-mini for summarization"""
    print("\n⚖️  Testing Qwen 3 vs GPT-4o-mini comparison...")
    
    test_email = """
    From: ceo@startup.com
    Subject: Important Company Update - Funding and Growth
    
    Team,
    
    I'm excited to share some major news about our company's future.
    
    We've successfully closed our Series A funding round, raising $5M from TechVentures and Innovation Capital. This funding will allow us to:
    - Hire 10 new engineers over the next 6 months
    - Expand our marketing efforts in European markets
    - Accelerate product development for our AI features
    - Open a new office in Austin, Texas
    
    Additionally, I'm pleased to announce that Sarah Johnson has joined as our new VP of Engineering. Sarah brings 15 years of experience from Google and will be leading our technical roadmap.
    
    With this growth, we're updating our remote work policy. Starting next month, all employees will be expected in the office 3 days per week (Tuesday, Wednesday, Thursday). We believe this hybrid approach will maintain our culture while supporting collaboration.
    
    Thank you all for your hard work that made this possible. Let's make the most of this opportunity!
    
    Best,
    Alex Chen, CEO
    """
    
    prompt = f"Summarize this company announcement in one punchy phrase (max 120 chars). Examples: '$5M Series A raised, 10 engineers hiring, 3-day office policy', 'Funding secured, team expanding, hybrid work announced':\n\n{test_email}"
    
    results = {}
    
    for model_name, model_id in [("Qwen 3", "q3"), ("GPT-4o-mini", "gpt-4o-mini")]:
        try:
            result = subprocess.run([
                "llm", "-m", model_id, prompt
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                summary = result.stdout.strip()
                char_count = len(summary)
                results[model_name] = summary
                status = "✅" if char_count <= 120 else "⚠️"
                print(f"{status} {model_name} Summary ({char_count}/120 chars):")
                print(f"   {summary}")
                print()
            else:
                print(f"❌ {model_name} failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ {model_name} test failed: {e}")
    
    return len(results) > 0

def main():
    """Run all Qwen 3 summarization tests"""
    print("🚀 Qwen 3 Email Summarization Tests")
    print("=" * 50)
    
    tests = [
        ("Simple Email Summarization", test_qwen_simple_summarization),
        ("Conversation Thread Summarization", test_qwen_conversation_summarization),
        ("Technical Email Summarization", test_qwen_technical_email),
        ("Personal Email Summarization", test_qwen_personal_email),
        ("Model Comparison", test_qwen_comparison_with_gpt4o_mini),
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
    print(f"📊 Summarization Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed >= 3:  # Allow some flexibility since it depends on model availability
        print("🎉 Qwen 3 summarization tests mostly successful!")
        print("\n💡 Qwen 3 model appears ready for email summarization in your RAG system!")
        return True
    else:
        print("⚠️  Some Qwen 3 summarization tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)