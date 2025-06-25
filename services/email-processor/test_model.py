#!/usr/bin/env python3

from models import Message
from sqlalchemy import inspect

# Test the model definition
print("Message model columns:")
mapper = inspect(Message)
for column in mapper.columns:
    print(f"  {column.name}: {column.type}")

print("\nColumn names in Message:")
column_names = [col.name for col in mapper.columns]
print(column_names)

if 'date_received' in column_names:
    print("\n❌ ERROR: date_received still in model!")
else:
    print("\n✅ Good: date_received not in model")

if 'processed_at' in column_names:
    print("✅ Good: processed_at is in model")
else:
    print("❌ ERROR: processed_at missing from model!")