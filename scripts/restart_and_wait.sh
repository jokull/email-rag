#!/bin/bash

# Email RAG Docker Restart Script with Log-based Readiness Detection
# This script restarts Docker services and waits for them to be ready using log output heuristics
# Usage: ./restart_and_wait.sh [timeout] [service_name]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
if [[ $# -eq 1 && ! "$1" =~ ^[0-9]+$ ]]; then
    # Single argument that's not a number = service name
    MAX_WAIT_TIME=300
    TARGET_SERVICE="$1"
elif [[ $# -eq 2 ]]; then
    # Two arguments = timeout and service
    MAX_WAIT_TIME="$1"
    TARGET_SERVICE="$2"
elif [[ $# -eq 1 ]]; then
    # Single numeric argument = timeout only
    MAX_WAIT_TIME="$1"
    TARGET_SERVICE=""
else
    # No arguments or too many
    MAX_WAIT_TIME=300
    TARGET_SERVICE=""
fi

CHECK_INTERVAL=2   # Check every 2 seconds
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}📦 Email RAG Docker Restart Script${NC}"
echo "Project Directory: $PROJECT_DIR"
echo "Max Wait Time: ${MAX_WAIT_TIME}s"
if [[ -n "$TARGET_SERVICE" ]]; then
    echo "Target Service: $TARGET_SERVICE"
fi
echo

# Function to print timestamped messages
log() {
    echo -e "[$(date '+%H:%M:%S')] $1"
}

# Function to check if a service is ready based on log patterns
check_service_ready() {
    local service_name=$1
    local ready_pattern=$2
    
    # Get recent logs (last 50 lines)
    local logs=$(docker-compose logs --tail=50 "$service_name" 2>/dev/null || echo "")
    
    if echo "$logs" | grep -q "$ready_pattern"; then
        return 0  # Ready
    else
        return 1  # Not ready
    fi
}

# Function to check overall system health
check_system_health() {
    local health_response=$(curl -s http://localhost:8080/health 2>/dev/null || echo "")
    
    if echo "$health_response" | grep -q '"healthy":true'; then
        return 0  # Healthy
    else
        return 1  # Not healthy
    fi
}

# Change to project directory
cd "$PROJECT_DIR"

# Step 1: Stop services gracefully
if [[ -n "$TARGET_SERVICE" ]]; then
    log "${YELLOW}🛑 Stopping Docker service: $TARGET_SERVICE...${NC}"
    docker-compose stop "$TARGET_SERVICE" || true
    docker-compose rm -f "$TARGET_SERVICE" || true
else
    log "${YELLOW}🛑 Stopping Docker services...${NC}"
    docker-compose down || true
fi

# Step 2: Start services
if [[ -n "$TARGET_SERVICE" ]]; then
    log "${YELLOW}🚀 Starting Docker service: $TARGET_SERVICE...${NC}"
    docker-compose up -d "$TARGET_SERVICE"
else
    log "${YELLOW}🚀 Starting Docker services...${NC}"
    docker-compose up -d
fi

# Step 3: Wait for services to be ready
log "${YELLOW}⏳ Waiting for services to be ready...${NC}"

# Function to get readiness pattern for a service
get_readiness_pattern() {
    case "$1" in
        "postgres")
            echo "ready to accept connections"
            ;;
        "imap-sync")
            echo "IMAP sync service started\|Sync completed successfully"
            ;;
        "ai-processor")
            echo "LIFESPAN: setup_services completed successfully\|Application startup complete"
            ;;
        "content-processor")
            echo "Content processor ready\|Application startup complete"
            ;;
        "simple-classifier")
            echo "Simple classifier ready\|Application startup complete"
            ;;
        *)
            echo "Application startup complete"
            ;;
    esac
}

start_time=$(date +%s)

if [[ -n "$TARGET_SERVICE" ]]; then
    # Single service mode
    log "${YELLOW}⏳ Waiting for $TARGET_SERVICE to be ready...${NC}"
    service_ready=false
    pattern=$(get_readiness_pattern "$TARGET_SERVICE")
    
    while [ $service_ready = false ]; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $MAX_WAIT_TIME ]; then
            log "${RED}❌ Timeout waiting for $TARGET_SERVICE to be ready after ${MAX_WAIT_TIME}s${NC}"
            echo
            echo "Service Status:"
            echo "  $TARGET_SERVICE: ❌ Not Ready"
            
            # Show recent logs for debugging
            echo
            echo "Recent logs from $TARGET_SERVICE:"
            docker-compose logs --tail=10 "$TARGET_SERVICE" 2>/dev/null || echo "No logs available"
            exit 1
        fi
        
        # Check service readiness
        if check_service_ready "$TARGET_SERVICE" "$pattern"; then
            service_ready=true
            log "${GREEN}✅ $TARGET_SERVICE is ready${NC}"
        else
            log "${BLUE}📊 Waiting for $TARGET_SERVICE... (${elapsed}s elapsed)${NC}"
            sleep $CHECK_INTERVAL
        fi
    done
else
    # Multi-service mode (original logic)
    # Track service status using simple variables
    postgres_ready=false
    imap_sync_ready=false
    ai_processor_ready=false
    content_processor_ready=false
    simple_classifier_ready=false
    
    all_ready=false
    
    while [ $all_ready = false ]; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $MAX_WAIT_TIME ]; then
            log "${RED}❌ Timeout waiting for services to be ready after ${MAX_WAIT_TIME}s${NC}"
            echo
            echo "Service Status:"
            echo "  postgres: $([ $postgres_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
            echo "  imap-sync: $([ $imap_sync_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
            echo "  ai-processor: $([ $ai_processor_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
            echo "  content-processor: $([ $content_processor_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
            echo "  simple-classifier: $([ $simple_classifier_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
            exit 1
        fi
        
        # Check each service
        ready_count=0
        
        # Check postgres
        if [ $postgres_ready = false ]; then
            if check_service_ready "postgres" "ready to accept connections"; then
                postgres_ready=true
                log "${GREEN}✅ postgres is ready${NC}"
            fi
        fi
        [ $postgres_ready = true ] && ready_count=$((ready_count + 1))
        
        # Check imap-sync
        if [ $imap_sync_ready = false ]; then
            if check_service_ready "imap-sync" "IMAP sync service started\|Sync completed successfully"; then
                imap_sync_ready=true
                log "${GREEN}✅ imap-sync is ready${NC}"
            fi
        fi
        [ $imap_sync_ready = true ] && ready_count=$((ready_count + 1))
        
        # Check ai-processor
        if [ $ai_processor_ready = false ]; then
            if check_service_ready "ai-processor" "LIFESPAN: setup_services completed successfully\|Application startup complete"; then
                ai_processor_ready=true
                log "${GREEN}✅ ai-processor is ready${NC}"
            fi
        fi
        [ $ai_processor_ready = true ] && ready_count=$((ready_count + 1))
        
        # Check content-processor
        if [ $content_processor_ready = false ]; then
            if check_service_ready "content-processor" "Content processor ready\|Application startup complete"; then
                content_processor_ready=true
                log "${GREEN}✅ content-processor is ready${NC}"
            fi
        fi
        [ $content_processor_ready = true ] && ready_count=$((ready_count + 1))
        
        # Check simple-classifier
        if [ $simple_classifier_ready = false ]; then
            if check_service_ready "simple-classifier" "Simple classifier ready\|Application startup complete"; then
                simple_classifier_ready=true
                log "${GREEN}✅ simple-classifier is ready${NC}"
            fi
        fi
        [ $simple_classifier_ready = true ] && ready_count=$((ready_count + 1))
        
        # Check if all services are ready
        total_services=5
        if [ $ready_count -eq $total_services ]; then
            log "${YELLOW}🔍 All services report ready, checking system health...${NC}"
            
            # Wait a moment for final initialization
            sleep 3
            
            # Check AI processor health endpoint
            if check_system_health; then
                log "${GREEN}🎉 All services are healthy and ready!${NC}"
                all_ready=true
            else
                log "${YELLOW}⏳ Services starting but health check not passing yet...${NC}"
                # Reset ai-processor to force re-check
                ai_processor_ready=false
                ready_count=$((ready_count - 1))
            fi
        fi
        
        # Show progress
        progress=$((ready_count * 100 / total_services))
        log "${BLUE}📊 Progress: $ready_count/$total_services services ready (${progress}%) - ${elapsed}s elapsed${NC}"
        
        if [ $all_ready = false ]; then
            sleep $CHECK_INTERVAL
        fi
    done
fi

# Final validation
total_time=$(($(date +%s) - start_time))
log "${GREEN}✅ Email RAG system is ready! (took ${total_time}s)${NC}"

# Show final service status
echo
if [[ -n "$TARGET_SERVICE" ]]; then
    echo "📋 Service Status:"
    echo "  $TARGET_SERVICE: ✅ Ready"
else
    echo "📋 Final Service Status:"
    echo "  postgres: $([ $postgres_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
    echo "  imap-sync: $([ $imap_sync_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
    echo "  ai-processor: $([ $ai_processor_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
    echo "  content-processor: $([ $content_processor_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
    echo "  simple-classifier: $([ $simple_classifier_ready = true ] && echo "✅ Ready" || echo "❌ Not Ready")"
fi

echo
echo "🔗 Available Services:"
echo "  - AI Processor Health: http://localhost:8080/health"
echo "  - AI Processor API: http://localhost:8080/docs"
echo "  - Content Processor: http://localhost:8082/health"
echo "  - Simple Classifier: http://localhost:8083/health"
echo "  - PostgreSQL: localhost:5433"
echo "  - Zero Cache: http://localhost:4848"

echo
echo "🧪 Test Endpoints:"
echo "  - RAG Pipeline Test: POST http://localhost:8080/test/rag-pipeline"
echo "  - Embedding Pipeline: POST http://localhost:8080/test/embedding-pipeline"
echo "  - Chunking Visualization: POST http://localhost:8080/test/chunking-visualization"

echo
log "${GREEN}🎯 System ready for use!${NC}"

exit 0