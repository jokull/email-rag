#!/usr/bin/env python3
"""
Threading Worker - Background process for email threading
Processes personal emails to create conversation threads with AI summaries
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from threading import Event

from processor import EmailProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThreadingWorker:
    """Background worker for email threading"""
    
    def __init__(self, batch_size: int = 50, sleep_interval: int = 10):
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval
        self.processor = EmailProcessor()
        self.shutdown_event = Event()
        
        # Statistics
        self.total_threaded = 0
        self.start_time = datetime.utcnow()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📛 Received signal {signum}, shutting down gracefully...")
        self.shutdown_event.set()
    
    async def run(self):
        """Main worker loop"""
        logger.info("🧵 Threading worker starting...")
        logger.info(f"📊 Batch size: {self.batch_size}, Sleep interval: {self.sleep_interval}s")
        
        # Get initial stats
        stats = self.processor.get_threading_stats()
        logger.info(f"📈 Initial threading stats: {stats}")
        
        while not self.shutdown_event.is_set():
            try:
                # Process batch of unthreaded personal emails
                threaded_count = self.processor.process_unthreaded_personal_emails(self.batch_size)
                
                if threaded_count > 0:
                    self.total_threaded += threaded_count
                    logger.info(f"🧵 Threaded {threaded_count} messages (total: {self.total_threaded})")
                    
                    # Log updated stats periodically
                    if self.total_threaded % 100 == 0:
                        stats = self.processor.get_threading_stats()
                        logger.info(f"📊 Threading progress: {stats}")
                else:
                    # No messages to process, sleep longer
                    logger.debug("😴 No unthreaded messages found, sleeping...")
                
                # Sleep between batches
                await asyncio.sleep(self.sleep_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in threading worker: {e}")
                await asyncio.sleep(self.sleep_interval * 2)  # Sleep longer on error
        
        # Final stats
        runtime = datetime.utcnow() - self.start_time
        logger.info(f"📊 Threading worker finished: {self.total_threaded} messages threaded in {runtime}")

async def main():
    """Main entry point"""
    import os
    
    # Configuration from environment
    batch_size = int(os.getenv('THREADING_BATCH_SIZE', '50'))
    sleep_interval = int(os.getenv('THREADING_SLEEP_INTERVAL', '10'))
    
    logger.info(f"🚀 Starting threading worker (batch_size={batch_size}, sleep={sleep_interval}s)")
    
    worker = ThreadingWorker(batch_size=batch_size, sleep_interval=sleep_interval)
    
    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("⚡ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Threading worker failed: {e}")
        return 1
    
    logger.info("👋 Threading worker stopped")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))