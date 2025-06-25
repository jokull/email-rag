#!/bin/bash

# Email Processing Trigger Script
# Manually triggers processing of emails

set -e

# Configuration
API_URL="${EMAIL_PROCESSOR_URL:-http://localhost:8080}"
BATCH_SIZE="${1:-50}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Triggering Email Processing"
echo "============================="
echo

# Check if API is healthy first
echo "🏥 Checking API health..."
if ! curl -s "$API_URL/health" >/dev/null 2>&1; then
    echo "❌ API is not responding at $API_URL"
    exit 1
fi
echo -e "  ${GREEN}✅ API is healthy${NC}"

echo

# Get current status
echo "📊 Current Status..."
if status_before=$(curl -s "$API_URL/status" 2>/dev/null); then
    pending_before=$(echo "$status_before" | grep -o '"pending_messages":[0-9]*' | cut -d':' -f2)
    processed_before=$(echo "$status_before" | grep -o '"total_processed_messages":[0-9]*' | cut -d':' -f2)
    echo "  ⏳ Pending messages: $pending_before"
    echo "  ✅ Already processed: $processed_before"
else
    echo "❌ Could not get current status"
    exit 1
fi

echo

# Trigger processing
echo "🔄 Triggering processing (batch size: $BATCH_SIZE)..."
if process_response=$(curl -s -X POST "$API_URL/process" 2>/dev/null); then
    processed_count=$(echo "$process_response" | grep -o '"processed_count":[0-9]*' | cut -d':' -f2)
    
    if [ "$processed_count" -gt 0 ]; then
        echo -e "  ${GREEN}✅ Successfully processed $processed_count emails${NC}"
    else
        echo -e "  ${YELLOW}⚠️  No emails were processed (might be none pending)${NC}"
    fi
else
    echo "❌ Failed to trigger processing"
    exit 1
fi

echo

# Get updated status
echo "📈 Updated Status..."
if status_after=$(curl -s "$API_URL/status" 2>/dev/null); then
    pending_after=$(echo "$status_after" | grep -o '"pending_messages":[0-9]*' | cut -d':' -f2)
    processed_after=$(echo "$status_after" | grep -o '"total_processed_messages":[0-9]*' | cut -d':' -f2)
    echo "  ⏳ Pending messages: $pending_after"
    echo "  ✅ Total processed: $processed_after"
    
    # Calculate change
    processed_change=$((processed_after - processed_before))
    if [ "$processed_change" -gt 0 ]; then
        echo -e "  🎯 Change: ${GREEN}+$processed_change processed${NC}"
    fi
else
    echo "❌ Could not get updated status"
fi

echo
echo "✨ Processing trigger complete!"