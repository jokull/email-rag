#!/bin/bash

# Deploy Email Classifier on macOS using launchd
# This script sets up the email classification worker as a macOS background service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_FILE="$SCRIPT_DIR/com.emailrag.classifier.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
SERVICE_NAME="com.emailrag.classifier"

echo "🍎 Deploying Email Classifier on macOS"
echo "📁 Script directory: $SCRIPT_DIR"
echo ""

# Check if plist file exists
if [ ! -f "$PLIST_FILE" ]; then
    echo "❌ Plist file not found: $PLIST_FILE"
    exit 1
fi

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"

# Stop service if it's already running
echo "🛑 Stopping existing service (if running)..."
launchctl unload "$LAUNCH_AGENTS_DIR/$SERVICE_NAME.plist" 2>/dev/null || true

# Copy plist file to LaunchAgents
echo "📋 Installing service plist..."
cp "$PLIST_FILE" "$LAUNCH_AGENTS_DIR/"

# Load and start the service
echo "🚀 Loading and starting service..."
launchctl load "$LAUNCH_AGENTS_DIR/$SERVICE_NAME.plist"

# Check if service is running
sleep 2
if launchctl list | grep -q "$SERVICE_NAME"; then
    echo "✅ Service started successfully"
    echo ""
    echo "📊 Service status:"
    launchctl list | grep "$SERVICE_NAME"
else
    echo "❌ Service failed to start"
    echo ""
    echo "📋 Check logs:"
    echo "  stdout: $SCRIPT_DIR/email_classifier_stdout.log"
    echo "  stderr: $SCRIPT_DIR/email_classifier_stderr.log"
    exit 1
fi

echo ""
echo "🎉 Email Classifier deployed successfully!"
echo ""
echo "📚 Management commands:"
echo "  Stop:    launchctl unload ~/Library/LaunchAgents/$SERVICE_NAME.plist"
echo "  Start:   launchctl load ~/Library/LaunchAgents/$SERVICE_NAME.plist"
echo "  Status:  launchctl list | grep $SERVICE_NAME"
echo "  Logs:    tail -f $SCRIPT_DIR/email_classifier_stdout.log"
echo ""
echo "🔍 Health check: $SCRIPT_DIR/check_classifier_health.py"