#!/usr/bin/env python3

from email_reply_parser import EmailReplyParser

# Test with the actual email content we found
test_email = """Það lítur út fyrir að það vanti í löngu plötuna (hún á að vera 241,5 plús   
fræsing, en er bara 242cm). Það var greinilega ekki gert ráð fyrir          
fræsingunni. Við verðum að fá senda nýja plötu sem fyrst, erum með smið sem 
bíður eftir réttri plötu.                                                   

S 6161339 ef þú vilt heyra í mér.                                           

kv Jökull                                                                   

On Thu, Jan 26, 2017 at 2:20 PM Jökull Sólberg Auðunsson <jokull@solberg.is>
wrote:"""

print("Original email:")
print(repr(test_email))
print("\nLength:", len(test_email))

print("\n" + "="*50)

cleaned = EmailReplyParser.parse_reply(test_email)
print("Cleaned email:")
print(repr(cleaned))
print("\nLength:", len(cleaned))

print("\n" + "="*50)
print("Was quote removed?", "On Thu, Jan 26, 2017" not in cleaned)