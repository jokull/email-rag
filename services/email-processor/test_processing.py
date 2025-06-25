#!/usr/bin/env python3

from processor import EmailProcessor
import logging

# Set up logging to see what happens
logging.basicConfig(level=logging.INFO)

# Create processor instance
processor = EmailProcessor()

# Test email with quoted content
test_body = """Hi there!

This is my reply to your email.

Best regards,
John

On Thu, Jan 26, 2017 at 2:20 PM, Jane Doe <jane@example.com> wrote:
> This is the original message that should be removed.
> It contains quoted content.
> 
> Thanks,
> Jane"""

print("Original email body:")
print(repr(test_body))
print(f"Length: {len(test_body)}")
print()

# Test the cleaning function
cleaned = processor._clean_email_body(test_body)

print("Cleaned email body:")
print(repr(cleaned))
print(f"Length: {len(cleaned)}")
print()

print("Cleaning worked:", "Jane Doe" not in cleaned and "original message" not in cleaned)