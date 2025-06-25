#!/bin/bash

# Email Processor Monitoring Script
# Checks health and status of the email processing system

set -e

# Configuration
API_URL="${EMAIL_PROCESSOR_URL:-http://localhost:8080}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "📧 Email Processor Status Monitor"
echo "================================="
echo

# Health Check
echo "🏥 Health Check..."
if health_response=$(curl -s "$API_URL/health" 2>/dev/null); then
    if echo "$health_response" | grep -q '"status":"healthy"'; then
        echo -e "  ${GREEN}✅ API is healthy${NC}"
        echo "  📊 Database: $(echo "$health_response" | grep -o '"database":"[^"]*"' | cut -d'"' -f4)"
    else
        echo -e "  ${RED}❌ API responded but not healthy${NC}"
        echo "  Response: $health_response"
        exit 1
    fi
else
    echo -e "  ${RED}❌ API is not responding${NC}"
    exit 1
fi

echo

# Processing Status
echo "📈 Processing Status..."
if status_response=$(curl -s "$API_URL/status" 2>/dev/null); then
    total_imap=$(echo "$status_response" | grep -o '"total_imap_messages":[0-9]*' | cut -d':' -f2)
    total_processed=$(echo "$status_response" | grep -o '"total_processed_messages":[0-9]*' | cut -d':' -f2)
    pending=$(echo "$status_response" | grep -o '"pending_messages":[0-9]*' | cut -d':' -f2)
    rate=$(echo "$status_response" | grep -o '"processing_rate":"[^"]*"' | cut -d'"' -f4)
    
    echo "  📧 Total IMAP Messages: $total_imap"
    echo "  ✅ Processed Messages: $total_processed"
    echo "  ⏳ Pending Messages: $pending"
    echo "  📊 Processing Rate: $rate"
    
    # Calculate percentage
    if [ "$total_imap" -gt 0 ]; then
        percentage=$((total_processed * 100 / total_imap))
        if [ "$percentage" -ge 80 ]; then
            echo -e "  🎯 Progress: ${GREEN}${percentage}%${NC}"
        elif [ "$percentage" -ge 50 ]; then
            echo -e "  🎯 Progress: ${YELLOW}${percentage}%${NC}"
        else
            echo -e "  🎯 Progress: ${RED}${percentage}%${NC}"
        fi
    fi
else
    echo -e "  ${RED}❌ Could not get processing status${NC}"
fi

echo

# Queue Status
echo "📋 Queue Status..."
if queue_response=$(curl -s "$API_URL/queue" 2>/dev/null); then
    processed_today=$(echo "$queue_response" | grep -o '"processed_today":[0-9]*' | cut -d':' -f2)
    echo "  📅 Processed Today: $processed_today"
else
    echo -e "  ${RED}❌ Could not get queue status${NC}"
fi

echo

# Container Status
echo "🐳 Container Status..."
if command -v docker-compose >/dev/null 2>&1; then
    if docker-compose ps email-processor email-processor-worker >/dev/null 2>&1; then
        api_status=$(docker-compose ps -q email-processor | xargs docker inspect --format='{{.State.Status}}' 2>/dev/null || echo "unknown")
        worker_status=$(docker-compose ps -q email-processor-worker | xargs docker inspect --format='{{.State.Status}}' 2>/dev/null || echo "unknown")
        
        if [ "$api_status" = "running" ]; then
            echo -e "  ${GREEN}✅ API Container: $api_status${NC}"
        else
            echo -e "  ${RED}❌ API Container: $api_status${NC}"
        fi
        
        if [ "$worker_status" = "running" ]; then
            echo -e "  ${GREEN}✅ Worker Container: $worker_status${NC}"
        else
            echo -e "  ${RED}❌ Worker Container: $worker_status${NC}"
        fi
    else
        echo -e "  ${YELLOW}⚠️  Could not check container status${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠️  docker-compose not available${NC}"
fi

echo
echo "🕐 Last checked: $(date)"
echo "✨ Monitor complete!"