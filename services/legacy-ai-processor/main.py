"""
Consolidated AI Processor for Email RAG System
Handles all LLM tasks in a single service optimized for Qwen-0.5B on Mac mini M2

This service consolidates:
- Email classification (human vs promotional/transactional/automated)
- Thread detection and conversation building  
- Content cleaning and chunking
- Embedding generation for RAG pipeline
- All ML-based processing tasks

Optimized for:
- Mac mini M2 16GB RAM
- Qwen-0.5B (620M params, Q4_0 GGUF quantized)
- Low memory footprint (~1-2GB total)
- Fast inference with Metal acceleration
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime, timedelta
import numpy as np

# Import our consolidated modules
from config import get_config, LLMConfig, ProcessorConfig
from modern_llm import ModernQwenInterface
from email_processor import EmailProcessor, ProcessingResult
from embedding_service import EmbeddingService
from health_monitor import HealthMonitor

# Global state - will be stored in app.state
class AppState:
    llm_interface: Optional[ModernQwenInterface] = None
    email_processor: Optional[EmailProcessor] = None
    embedding_service: Optional[EmbeddingService] = None
    health_monitor: Optional[HealthMonitor] = None
    database_engine = None
    SessionLocal = None
    llm_config: Optional[LLMConfig] = None
    processor_config: Optional[ProcessorConfig] = None

async def setup_services(app: FastAPI):
    """Initialize all AI processing services and store in app.state"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 SETUP: Starting Email Threading & Cleaning Service (with Qwen)")
    
    try:
        # Load configuration
        logger.info("🔧 SETUP: Loading configuration...")
        llm_config, processor_config = get_config()
        app.state.llm_config = llm_config
        app.state.processor_config = processor_config
        logger.info(f"✅ SETUP: Configuration loaded - batch_size: {processor_config.batch_size}")
        
        # Setup database connection
        logger.info("🗄️ SETUP: Setting up database connection...")
        database_engine = create_engine(processor_config.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=database_engine)
        app.state.database_engine = database_engine
        app.state.SessionLocal = SessionLocal
        logger.info("✅ SETUP: Database connection established")
        
        # Try to initialize LLM interface
        fallback_mode = os.getenv("FALLBACK_TO_BASIC_CLASSIFICATION", "false").lower() == "true"
        disable_llm = os.getenv("DISABLE_LLM_INIT", "false").lower() == "true"
        logger.info(f"🧠 SETUP: LLM initialization - disable_llm: {disable_llm}, fallback_mode: {fallback_mode}")
        
        if disable_llm:
            logger.info("🔧 SETUP: LLM initialization disabled for debugging")
            app.state.llm_interface = None
        else:
            try:
                logger.info("🧠 SETUP: Initializing Modern Qwen-0.5B interface...")
                llm_interface = ModernQwenInterface(llm_config, processor_config)
                
                # Initialize the model
                logger.info("🧠 SETUP: Starting LLM model initialization...")
                model_ready = await llm_interface.initialize()
                logger.info(f"🧠 SETUP: LLM initialization result: {model_ready}")
                
                if model_ready:
                    logger.info("✅ SETUP: Modern Qwen-0.5B interface loaded and ready")
                    app.state.llm_interface = llm_interface
                else:
                    raise RuntimeError("Model failed to initialize")
                    
            except Exception as e:
                if fallback_mode:
                    logger.warning(f"⚠️ SETUP: Qwen model failed to load: {e}")
                    logger.info("📧 SETUP: Falling back to basic classification (threading + Talon cleaning only)")
                    app.state.llm_interface = None
                else:
                    logger.error(f"❌ SETUP: Failed to load Qwen-0.5B model: {e}")
                    raise RuntimeError(f"Failed to load Qwen-0.5B model: {e}")
        
        # Initialize other services
        logger.info("🎯 SETUP: Initializing embedding service...")
        try:
            embedding_service = EmbeddingService(processor_config)
            await embedding_service._load_model()  # Load model during startup
            app.state.embedding_service = embedding_service
            logger.info("✅ SETUP: Embedding service initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ SETUP: Embedding service failed to initialize: {e}")
            logger.exception("Embedding service error details:")
            app.state.embedding_service = None
            
        logger.info("📧 SETUP: Initializing email processor...")
        email_processor = EmailProcessor(app.state.llm_interface, app.state.embedding_service, processor_config, SessionLocal)
        logger.info("📊 SETUP: Initializing health monitor...")
        health_monitor = HealthMonitor(app.state.llm_interface, email_processor, app.state.embedding_service)
        
        app.state.email_processor = email_processor
        app.state.health_monitor = health_monitor
        
        logger.info("🎯 SETUP: All AI services initialized successfully")
        logger.info(f"🎯 SETUP: Final service status - LLM: {app.state.llm_interface is not None}, Embedding: {app.state.embedding_service is not None}, Email: {app.state.email_processor is not None}")
        
    except Exception as e:
        logger.error(f"❌ SETUP: Failed to initialize services: {e}")
        logger.exception("Full setup error traceback:")
        raise

async def shutdown_services(app: FastAPI):
    """Clean shutdown of all services"""
    logging.info("🛑 Shutting down AI Processor...")
    
    try:
        # Stop background processing
        if hasattr(app.state, 'email_processor') and app.state.email_processor:
            app.state.email_processor.stop_processing()
        
        # Close database connections
        if hasattr(app.state, 'database_engine') and app.state.database_engine:
            app.state.database_engine.dispose()
        
        logging.info("✅ AI Processor shutdown complete")
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager - proper FastAPI pattern"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 LIFESPAN: Starting lifespan context manager")
    
    try:
        # Startup
        logger.info("🚀 LIFESPAN: About to call setup_services")
        await setup_services(app)
        logger.info("✅ LIFESPAN: setup_services completed successfully")
        
        yield
        
        # Shutdown
        logger.info("🛑 LIFESPAN: Starting shutdown process")
        await shutdown_services(app)
        logger.info("✅ LIFESPAN: Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ LIFESPAN: Error in lifespan context manager: {e}")
        logger.exception("Full traceback:")
        raise

# Create FastAPI app with lifespan
app = FastAPI(
    title="Email RAG AI Processor",
    description="Consolidated AI processing service for email RAG system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint for Docker and monitoring"""
    if not hasattr(request.app.state, 'health_monitor') or not request.app.state.health_monitor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    health_status = await request.app.state.health_monitor.get_health_status()
    
    if not health_status["healthy"]:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

# Metrics endpoint
@app.get("/metrics")
async def get_metrics(request: Request):
    """Get performance metrics"""
    if not hasattr(request.app.state, 'health_monitor') or not request.app.state.health_monitor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return await request.app.state.health_monitor.get_detailed_metrics()

# Email cleaning test endpoint (using existing metrics route pattern)
@app.post("/metrics/clean")
async def test_email_cleaning(request: Request, content: Dict[str, Any]):
    """Test email cleaning functionality"""
    if not hasattr(request.app.state, 'email_processor') or not request.app.state.email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    raw_content = content.get("content", "")
    if not raw_content:
        raise HTTPException(status_code=400, detail="Content required")
    
    try:
        # Test our tiered cleaning approach
        cleaned = await request.app.state.email_processor._clean_with_tiered_approach(raw_content)
        method_used = getattr(request.app.state.email_processor, '_last_cleaning_method', 'unknown')
        
        return {
            "success": True,
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned),
            "cleaned_content": cleaned,
            "cleaning_method": method_used,
            "reduction_ratio": round((len(raw_content) - len(cleaned)) / len(raw_content), 2) if len(raw_content) > 0 else 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original_length": len(raw_content)
        }

# Manual processing trigger (useful for debugging)
@app.post("/process/trigger")
async def trigger_processing(request: Request, background_tasks: BackgroundTasks):
    """Manually trigger email processing cycle"""
    if not hasattr(request.app.state, 'email_processor') or not request.app.state.email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    background_tasks.add_task(request.app.state.email_processor.process_pending_emails)
    return {"status": "processing_triggered", "timestamp": datetime.utcnow()}

# Get processing status
@app.get("/process/status")
async def get_processing_status(request: Request):
    """Get current processing status and queue information"""
    if not hasattr(request.app.state, 'email_processor') or not request.app.state.email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    return await request.app.state.email_processor.get_processing_status()

# Classification endpoint (for external API access)
@app.post("/classify")
async def classify_content(request: Request, content: Dict[str, Any]):
    """Classify email content via API"""
    if not hasattr(request.app.state, 'llm_interface') or not request.app.state.llm_interface:
        raise HTTPException(status_code=503, detail="LLM interface not ready")
    
    email_content = content.get("content", "")
    if not email_content:
        raise HTTPException(status_code=400, detail="Content required")
    
    try:
        # Create a simple processing request for classification
        from classification_models import EmailProcessingRequest
        classification_request = EmailProcessingRequest(
            email_id="api_request",
            sender="unknown@example.com",
            content=email_content
        )
        
        result = await request.app.state.llm_interface.classify_email(classification_request)
        return {
            "classification": result.classification,
            "confidence": result.confidence,
            "sentiment": result.sentiment,
            "sentiment_score": result.sentiment_score,
            "formality": result.formality,
            "personalization": result.personalization,
            "priority": result.priority,
            "should_process": result.should_process,
            "processing_time_ms": result.processing_time_ms,
            "reasoning": result.reasoning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Embedding endpoint
@app.post("/embed")
async def generate_embedding(request: Request, content: Dict[str, Any]):
    """Generate embedding for text content"""
    if not hasattr(request.app.state, 'embedding_service') or not request.app.state.embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not ready")
    
    text = content.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text required")
    
    try:
        embedding = await request.app.state.embedding_service.generate_embedding(text)
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "model": request.app.state.processor_config.embedding_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

# Email cleaning endpoint (for testing our tiered approach)
@app.post("/clean")
async def clean_email_content(request: Request, content: Dict[str, Any]):
    """Clean email content using our tiered approach"""
    if not hasattr(request.app.state, 'email_processor') or not request.app.state.email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    raw_content = content.get("content", "")
    if not raw_content:
        raise HTTPException(status_code=400, detail="Content required")
    
    try:
        cleaned = await request.app.state.email_processor._clean_with_tiered_approach(raw_content)
        method_used = getattr(request.app.state.email_processor, '_last_cleaning_method', 'unknown')
        
        return {
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned),
            "cleaned_content": cleaned,
            "cleaning_method": method_used,
            "reduction_ratio": round((len(raw_content) - len(cleaned)) / len(raw_content), 2) if len(raw_content) > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email cleaning failed: {str(e)}")

# Thread summary endpoint
@app.post("/summarize/thread")
async def generate_thread_summary(request: Request, data: Dict[str, Any]):
    """Generate or update thread summary"""
    if not hasattr(request.app.state, 'email_processor') or not request.app.state.email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    thread_id = data.get("thread_id")
    force_update = data.get("force_update", False)
    
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    
    try:
        summary = await request.app.state.email_processor.generate_thread_summary(thread_id, force_update)
        
        if summary:
            return {
                "success": True,
                "summary": summary.dict(),
                "message": f"Generated summary: {summary.summary_oneliner}"
            }
        else:
            raise HTTPException(status_code=404, detail="Thread not found or summary generation failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

# Test endpoint for incremental development
@app.get("/test1")
async def test_endpoint():
    """Simple test endpoint"""
    return {"status": "auto-reload working!", "timestamp": "2025-06-23", "reload": True}

@app.get("/test2")
async def test_endpoint2():
    """Another test endpoint"""
    return {"test": "working"}

# Comprehensive embedding pipeline test endpoint
@app.post("/test/embedding-pipeline")
async def test_embedding_pipeline(request: Request, data: Dict[str, Any]):
    """
    Test the full embedding pipeline with email retrieval by ID
    
    This endpoint exercises the complete workflow:
    1. Fetch email by ID from database
    2. Clean email content using tiered approach
    3. Generate embeddings using the embedding service
    4. Test classification if LLM is available
    5. Return comprehensive results to expose any bugs
    """
    if not hasattr(request.app.state, 'email_processor') or not request.app.state.email_processor:
        raise HTTPException(status_code=503, detail="Email processor not ready")
    
    email_id = data.get("email_id")
    if not email_id:
        raise HTTPException(status_code=400, detail="email_id required")
    
    try:
        # Step 1: Fetch email from database
        session = request.app.state.SessionLocal()
        try:
            email_query = session.execute(text("""
                SELECT 
                    e.id,
                    e.from_email,
                    e.from_name, 
                    e.subject,
                    e.body_text,
                    e.body_html,
                    e.date_sent,
                    e.date_received,
                    e.thread_id,
                    -- Get existing classification if any
                    c.classification,
                    c.confidence,
                    c.sentiment,
                    c.priority,
                    c.should_process,
                    -- Get existing embeddings count
                    (SELECT COUNT(*) FROM enhanced_embeddings ee WHERE ee.email_id = e.id) as existing_embeddings_count,
                    -- Get thread info
                    t.subject_normalized as thread_subject,
                    t.message_count as thread_message_count
                FROM emails e
                LEFT JOIN classifications c ON c.email_id = e.id
                LEFT JOIN threads t ON t.id = e.thread_id
                WHERE e.id = :email_id
            """), {"email_id": email_id})
            
            email_data = email_query.fetchone()
            if not email_data:
                raise HTTPException(status_code=404, detail=f"Email {email_id} not found")
            
            # Convert to dict for easier handling
            email_dict = dict(email_data._mapping)
            
            # Step 2: Test email cleaning
            raw_content = email_dict.get('body_text', '') or email_dict.get('body_html', '')
            if not raw_content:
                raise HTTPException(status_code=400, detail="Email has no content to process")
            
            cleaned_content = await request.app.state.email_processor._clean_with_tiered_approach(raw_content)
            cleaning_method = getattr(request.app.state.email_processor, '_last_cleaning_method', 'unknown')
            
            # Step 3: Test embedding generation
            embeddings_result = None
            if request.app.state.embedding_service:
                try:
                    embedding_vector = await request.app.state.embedding_service.generate_embedding(cleaned_content)
                    embeddings_result = {
                        "success": True,
                        "dimension": len(embedding_vector),
                        "model": request.app.state.processor_config.embedding_model,
                        "vector_sample": embedding_vector[:5].tolist() if len(embedding_vector) >= 5 else embedding_vector.tolist()
                    }
                except Exception as e:
                    embeddings_result = {
                        "success": False,
                        "error": str(e)
                    }
            else:
                embeddings_result = {"success": False, "error": "Embedding service not available"}
            
            # Step 4: Test classification if LLM is available
            classification_result = None
            if request.app.state.llm_interface:
                try:
                    from classification_models import EmailProcessingRequest
                    classification_request = EmailProcessingRequest(
                        email_id=email_id,
                        sender=email_dict.get('from_email', 'unknown@example.com'),
                        content=cleaned_content
                    )
                    
                    classification = await request.app.state.llm_interface.classify_email(classification_request)
                    classification_result = {
                        "success": True,
                        "classification": classification.classification,
                        "confidence": classification.confidence,
                        "sentiment": classification.sentiment,
                        "priority": classification.priority,
                        "should_process": classification.should_process,
                        "processing_time_ms": classification.processing_time_ms
                    }
                except Exception as e:
                    classification_result = {
                        "success": False,
                        "error": str(e)
                    }
            else:
                classification_result = {"success": False, "error": "LLM interface not available"}
            
            # Step 5: Test thread context (if part of a thread)
            thread_context = None
            if email_dict.get('thread_id'):
                try:
                    thread_emails = session.execute(text("""
                        SELECT COUNT(*) as total_emails,
                               COUNT(CASE WHEN c.classification IS NOT NULL THEN 1 END) as classified_emails
                        FROM emails e
                        LEFT JOIN classifications c ON c.email_id = e.id
                        WHERE e.thread_id = :thread_id
                    """), {"thread_id": email_dict['thread_id']})
                    
                    thread_stats = thread_emails.fetchone()
                    thread_context = {
                        "thread_id": email_dict['thread_id'],
                        "thread_subject": email_dict.get('thread_subject'),
                        "thread_message_count": email_dict.get('thread_message_count'),
                        "total_emails": dict(thread_stats._mapping)['total_emails'],
                        "classified_emails": dict(thread_stats._mapping)['classified_emails']
                    }
                except Exception as e:
                    thread_context = {"error": str(e)}
            
            # Compile comprehensive test results
            test_results = {
                "email_info": {
                    "id": email_dict['id'],
                    "from_email": email_dict['from_email'],
                    "from_name": email_dict['from_name'],
                    "subject": email_dict['subject'],
                    "date_sent": str(email_dict['date_sent']) if email_dict['date_sent'] else None,
                    "existing_classification": email_dict.get('classification'),
                    "existing_embeddings_count": email_dict['existing_embeddings_count']
                },
                
                "content_analysis": {
                    "original_length": len(raw_content),
                    "cleaned_length": len(cleaned_content),
                    "cleaning_method": cleaning_method,
                    "reduction_ratio": round((len(raw_content) - len(cleaned_content)) / len(raw_content), 2) if len(raw_content) > 0 else 0,
                    "cleaned_preview": cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content
                },
                
                "embedding_test": embeddings_result,
                "classification_test": classification_result,
                "thread_context": thread_context,
                
                "service_status": {
                    "email_processor_ready": request.app.state.email_processor is not None,
                    "embedding_service_ready": request.app.state.embedding_service is not None,
                    "llm_interface_ready": request.app.state.llm_interface is not None,
                    "database_connected": True  # If we got here, DB is working
                },
                
                "test_summary": {
                    "total_steps": 5,
                    "successful_steps": sum([
                        1,  # Email fetch (we got here)
                        1,  # Content cleaning (always works)
                        1 if embeddings_result.get("success") else 0,
                        1 if classification_result.get("success") else 0,
                        1 if thread_context and not thread_context.get("error") else 0
                    ]),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            return test_results
            
        finally:
            session.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding pipeline test failed: {str(e)}")

# Chunking visualization endpoint
@app.post("/test/chunking-visualization")
async def visualize_chunking(request: Request, data: Dict[str, Any]):
    """
    Visualize email chunking with clear boundaries and vector embedding proof
    
    This endpoint:
    1. Retrieves email by ID
    2. Shows content cleaning process
    3. Demonstrates chunking with visible boundaries  
    4. Generates embeddings for each chunk
    5. Proves each chunk has its unique vector
    6. Shows similarity calculations between chunks
    """
    if not hasattr(request.app.state, 'embedding_service') or not request.app.state.embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not ready")
    
    email_id = data.get("email_id")
    chunk_size = data.get("chunk_size", 500)  # Characters per chunk
    overlap_size = data.get("overlap_size", 50)  # Character overlap
    
    if not email_id:
        raise HTTPException(status_code=400, detail="email_id required")
    
    try:
        # Step 1: Fetch email content
        session = request.app.state.SessionLocal()
        try:
            email_query = session.execute(text("""
                SELECT id, subject, body_text, body_html, from_email, from_name
                FROM emails 
                WHERE id = :email_id
            """), {"email_id": email_id})
            
            email_data = email_query.fetchone()
            if not email_data:
                raise HTTPException(status_code=404, detail=f"Email {email_id} not found")
            
            email_dict = dict(email_data._mapping)
            
            # Step 2: Get raw content and clean it
            raw_content = email_dict.get('body_text', '') or email_dict.get('body_html', '')
            if not raw_content:
                raise HTTPException(status_code=400, detail="Email has no content")
            
            # Clean content using our existing processor
            cleaned_content = await request.app.state.email_processor._clean_with_tiered_approach(raw_content)
            cleaning_method = getattr(request.app.state.email_processor, '_last_cleaning_method', 'unknown')
            
            # Step 3: Create chunks with visible boundaries
            def create_chunks_with_boundaries(text: str, chunk_size: int, overlap_size: int):
                """Create overlapping chunks with boundary markers"""
                chunks = []
                start = 0
                chunk_index = 0
                
                while start < len(text):
                    # Calculate end position
                    end = min(start + chunk_size, len(text))
                    
                    # Extract chunk text
                    chunk_text = text[start:end]
                    
                    # Find word boundaries to avoid cutting words
                    if end < len(text) and chunk_text:
                        # Look for the last space within chunk
                        last_space = chunk_text.rfind(' ')
                        if last_space > chunk_size * 0.8:  # If space is reasonably close to end
                            end = start + last_space
                            chunk_text = text[start:end]
                    
                    # Create chunk info
                    chunk_info = {
                        "index": chunk_index,
                        "text": chunk_text,
                        "start_pos": start,
                        "end_pos": end,
                        "length": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                    }
                    
                    chunks.append(chunk_info)
                    
                    # Move start position for next chunk (with overlap)
                    next_start = end - overlap_size
                    if next_start <= start:  # Prevent infinite loop
                        start = end
                    else:
                        start = next_start
                    
                    chunk_index += 1
                    
                    # Safety limit
                    if chunk_index > 50:  # Max 50 chunks
                        break
                
                return chunks
            
            chunks = create_chunks_with_boundaries(cleaned_content, chunk_size, overlap_size)
            
            # Step 4: Generate embeddings for each chunk
            chunk_embeddings = []
            embedding_stats = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = await request.app.state.embedding_service.generate_embedding(chunk["text"])
                    chunk_embeddings.append(embedding)
                    
                    # Calculate embedding statistics
                    vector_norm = float(np.linalg.norm(embedding))
                    vector_mean = float(np.mean(embedding))
                    vector_std = float(np.std(embedding))
                    
                    embedding_stats.append({
                        "chunk_index": i,
                        "vector_dimension": len(embedding),
                        "vector_norm": vector_norm,
                        "vector_mean": vector_mean,
                        "vector_std": vector_std,
                        "vector_sample": embedding[:5].tolist(),  # First 5 values as proof
                        "embedding_generated": True
                    })
                    
                except Exception as e:
                    chunk_embeddings.append(None)
                    embedding_stats.append({
                        "chunk_index": i,
                        "embedding_generated": False,
                        "error": str(e)
                    })
            
            # Step 5: Calculate similarity matrix between chunks
            similarity_matrix = []
            if len(chunk_embeddings) > 1:
                for i, emb1 in enumerate(chunk_embeddings):
                    row = []
                    for j, emb2 in enumerate(chunk_embeddings):
                        if emb1 is not None and emb2 is not None:
                            similarity = await request.app.state.embedding_service.compute_similarity(emb1, emb2)
                            row.append(round(float(similarity), 3))
                        else:
                            row.append(None)
                    similarity_matrix.append(row)
            
            # Step 6: Create visual representation with boundaries
            def create_visual_representation(text: str, chunks: List[Dict], max_display_chars: int = 2000):
                """Create visual representation with chunk boundaries marked"""
                if len(text) <= max_display_chars:
                    display_text = text
                    truncated = False
                else:
                    display_text = text[:max_display_chars]
                    truncated = True
                
                # Add boundary markers
                visual_chunks = []
                for chunk in chunks:
                    start = chunk["start_pos"]
                    end = chunk["end_pos"]
                    
                    # Only process chunks that are visible in display
                    if start < len(display_text):
                        actual_end = min(end, len(display_text))
                        visual_chunks.append({
                            "index": chunk["index"],
                            "start": start,
                            "end": actual_end,
                            "text": display_text[start:actual_end],
                            "boundary_marker": f"[CHUNK_{chunk['index']}_START]",
                            "end_marker": f"[CHUNK_{chunk['index']}_END]"
                        })
                
                return {
                    "display_text": display_text,
                    "truncated": truncated,
                    "visual_chunks": visual_chunks,
                    "total_display_length": len(display_text)
                }
            
            visual_rep = create_visual_representation(cleaned_content, chunks)
            
            # Step 7: Compile comprehensive results
            results = {
                "email_info": {
                    "id": email_dict["id"],
                    "subject": email_dict["subject"],
                    "from_email": email_dict["from_email"],
                    "from_name": email_dict["from_name"]
                },
                
                "content_analysis": {
                    "original_length": len(raw_content),
                    "cleaned_length": len(cleaned_content),
                    "cleaning_method": cleaning_method,
                    "reduction_ratio": round((len(raw_content) - len(cleaned_content)) / len(raw_content), 3) if len(raw_content) > 0 else 0
                },
                
                "chunking_parameters": {
                    "chunk_size": chunk_size,
                    "overlap_size": overlap_size,
                    "total_chunks": len(chunks)
                },
                
                "chunks": chunks,
                
                "embedding_proof": {
                    "model_used": request.app.state.processor_config.embedding_model,
                    "vector_dimension": request.app.state.processor_config.vector_dimension,
                    "embeddings_generated": len([e for e in embedding_stats if e.get("embedding_generated")]),
                    "embedding_stats": embedding_stats
                },
                
                "similarity_analysis": {
                    "similarity_matrix": similarity_matrix,
                    "interpretation": {
                        "diagonal_values": "Should be 1.0 (chunk similarity to itself)",
                        "high_similarity": "Values > 0.8 indicate very similar content",
                        "medium_similarity": "Values 0.5-0.8 indicate related content", 
                        "low_similarity": "Values < 0.5 indicate different content"
                    }
                },
                
                "visualization": visual_rep,
                
                "verification": {
                    "unique_embeddings": len(set(tuple(e.tolist()) if e is not None else () for e in chunk_embeddings)),
                    "expected_unique": len([e for e in chunk_embeddings if e is not None]),
                    "all_embeddings_unique": len(set(tuple(e.tolist()) if e is not None else () for e in chunk_embeddings)) == len([e for e in chunk_embeddings if e is not None]),
                    "embedding_dimensions_consistent": len(set(len(e) for e in chunk_embeddings if e is not None)) <= 1
                },
                
                "summary": {
                    "chunks_created": len(chunks),
                    "embeddings_generated": len([e for e in embedding_stats if e.get("embedding_generated")]),
                    "processing_successful": len([e for e in embedding_stats if e.get("embedding_generated")]) == len(chunks),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            return results
            
        finally:
            session.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking visualization failed: {str(e)}")

# RAG Testing Endpoint for Qwen-approved emails
@app.post("/test/rag-pipeline")
async def test_rag_pipeline(request: Request, data: Dict[str, Any]):
    """
    Test the RAG pipeline for emails that have been classified by Qwen
    
    This endpoint:
    1. Validates email exists and has Qwen classification
    2. Checks if email meets RAG processing criteria  
    3. Shows detailed chunking with boundaries for debugging
    4. Generates embeddings for each chunk
    5. Demonstrates RAG retrieval with similarity scoring
    6. Provides debugging output for fine-tuning chunking strategy
    """
    if not hasattr(request.app.state, 'embedding_service') or not request.app.state.embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not ready")
    
    email_id = data.get("email_id")
    chunk_size = data.get("chunk_size", 400)  # Adjustable for fine-tuning
    overlap_ratio = data.get("overlap_ratio", 0.1)  # 10% overlap
    rag_query = data.get("query", "What is this email about?")  # Test query for RAG
    
    if not email_id:
        raise HTTPException(status_code=400, detail="email_id required")
    
    try:
        # Step 1: Fetch email with classification data
        session = request.app.state.SessionLocal()
        try:
            email_query = session.execute(text("""
                SELECT 
                    e.id, e.subject, e.body_text, e.body_html, e.from_email, e.from_name, e.date_sent,
                    -- Classification data from Qwen
                    c.classification, c.confidence, c.sentiment, c.priority, 
                    c.should_process, c.relationship_strength, c.priority_score,
                    c.sentiment_score, c.formality, c.personalization,
                    c.created_at as classified_at
                FROM emails e 
                LEFT JOIN classifications c ON c.email_id = e.id 
                WHERE e.id = :email_id
            """), {"email_id": email_id})
            
            email_data = email_query.fetchone()
            if not email_data:
                raise HTTPException(status_code=404, detail=f"Email {email_id} not found")
            
            email_dict = dict(email_data._mapping)
            
            # Step 2: Validate RAG readiness
            rag_readiness = {
                "has_classification": email_dict.get('classification') is not None,
                "classification_confidence": email_dict.get('confidence', 0),
                "should_process": email_dict.get('should_process', False),
                "is_human_conversation": email_dict.get('classification') == 'human',
                "has_content": bool(email_dict.get('body_text') or email_dict.get('body_html')),
                "classification_date": str(email_dict.get('classified_at')) if email_dict.get('classified_at') else None
            }
            
            # Determine RAG eligibility
            rag_eligible = (
                rag_readiness["has_classification"] and 
                rag_readiness["has_content"] and
                rag_readiness["classification_confidence"] >= 0.5
            )
            
            if not rag_readiness["has_classification"]:
                return {
                    "error": "Email not yet classified by Qwen",
                    "email_id": email_id,
                    "rag_readiness": rag_readiness,
                    "recommendation": "Run email through Qwen classification first"
                }
            
            if not rag_eligible:
                return {
                    "error": "Email not eligible for RAG processing",
                    "email_id": email_id,
                    "rag_readiness": rag_readiness,
                    "recommendation": "Email may be too low quality or confidence for RAG"
                }
            
            # Step 3: Get and clean content
            raw_content = email_dict.get('body_text', '') or email_dict.get('body_html', '')
            cleaned_content = await request.app.state.email_processor._clean_with_tiered_approach(raw_content)
            cleaning_method = getattr(request.app.state.email_processor, '_last_cleaning_method', 'unknown')
            
            # Step 4: Create RAG-optimized chunks with debugging info
            def create_rag_chunks(text: str, chunk_size: int, overlap_ratio: float):
                """Create overlapping chunks optimized for RAG retrieval"""
                chunks = []
                overlap_size = int(chunk_size * overlap_ratio)
                start = 0
                chunk_index = 0
                
                while start < len(text):
                    # Calculate end position
                    end = min(start + chunk_size, len(text))
                    
                    # Extract chunk text
                    chunk_text = text[start:end].strip()
                    
                    # Find sentence boundaries to avoid cutting sentences
                    if end < len(text) and chunk_text:
                        # Look for sentence endings near the end of chunk
                        sentence_ends = [chunk_text.rfind('. '), chunk_text.rfind('! '), chunk_text.rfind('? ')]
                        valid_ends = [pos for pos in sentence_ends if pos > chunk_size * 0.7]
                        best_end = max(valid_ends) if valid_ends else -1
                        
                        if best_end > 0:
                            end = start + best_end + 2  # Include the period and space
                            chunk_text = text[start:end].strip()
                    
                    if len(chunk_text.strip()) < 20:  # Skip very short chunks
                        start = end
                        continue
                    
                    # Calculate chunk quality metrics
                    word_count = len(chunk_text.split())
                    sentence_count = len([s for s in chunk_text.split('.') if s.strip()])
                    
                    chunk_info = {
                        "index": chunk_index,
                        "text": chunk_text,
                        "start_pos": start,
                        "end_pos": end,
                        "length": len(chunk_text),
                        "word_count": word_count,
                        "sentence_count": sentence_count,
                        "preview": chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text,
                        "boundary_markers": {
                            "start": f"[CHUNK_{chunk_index}_START]",
                            "end": f"[CHUNK_{chunk_index}_END]"
                        },
                        "quality_score": min(1.0, word_count / 50)  # Quality based on word count
                    }
                    
                    chunks.append(chunk_info)
                    
                    # Move to next chunk with overlap
                    next_start = end - overlap_size
                    if next_start <= start:
                        start = end
                    else:
                        start = next_start
                    
                    chunk_index += 1
                    
                    if chunk_index > 20:  # Reasonable limit for RAG
                        break
                
                return chunks
            
            chunks = create_rag_chunks(cleaned_content, chunk_size, overlap_ratio)
            
            # Step 5: Generate embeddings and test RAG retrieval
            chunk_embeddings = []
            embedding_metadata = []
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = await request.app.state.embedding_service.generate_embedding(chunk["text"])
                    chunk_embeddings.append(embedding)
                    
                    embedding_metadata.append({
                        "chunk_index": i,
                        "embedding_generated": True,
                        "vector_dimension": len(embedding),
                        "vector_norm": float(np.linalg.norm(embedding)),
                        "vector_preview": embedding[:3].tolist()  # First 3 values
                    })
                except Exception as e:
                    chunk_embeddings.append(None)
                    embedding_metadata.append({
                        "chunk_index": i,
                        "embedding_generated": False,
                        "error": str(e)
                    })
            
            # Step 6: Test RAG query against chunks
            query_embedding = None
            retrieval_results = []
            
            try:
                query_embedding = await request.app.state.embedding_service.generate_embedding(rag_query)
                
                # Calculate similarities
                for i, chunk_embedding in enumerate(chunk_embeddings):
                    if chunk_embedding is not None:
                        similarity = await request.app.state.embedding_service.compute_similarity(
                            query_embedding, chunk_embedding
                        )
                        retrieval_results.append({
                            "chunk_index": i,
                            "similarity_score": round(float(similarity), 4),
                            "chunk_preview": chunks[i]["preview"],
                            "word_count": chunks[i]["word_count"],
                            "quality_score": chunks[i]["quality_score"]
                        })
                
                # Sort by similarity (best matches first)
                retrieval_results.sort(key=lambda x: x["similarity_score"], reverse=True)
                
            except Exception as e:
                retrieval_results = [{"error": f"RAG query failed: {str(e)}"}]
            
            # Step 7: Create visual representation with chunk boundaries
            def create_boundary_visualization(text: str, chunks: List[Dict], max_chars: int = 1500):
                """Create text with visible chunk boundaries for debugging"""
                if len(text) > max_chars:
                    display_text = text[:max_chars] + "\n\n[... TRUNCATED FOR DISPLAY ...]"
                    truncated = True
                else:
                    display_text = text
                    truncated = False
                
                # Insert boundary markers
                marked_text = display_text
                offset = 0
                
                for chunk in chunks:
                    if chunk["start_pos"] < len(display_text):
                        start_marker = f"\n\n{chunk['boundary_markers']['start']}\n"
                        end_pos = min(chunk["end_pos"], len(display_text))
                        end_marker = f"\n{chunk['boundary_markers']['end']}\n\n"
                        
                        # Insert end marker first (to preserve positions)
                        if end_pos + offset < len(marked_text):
                            marked_text = (marked_text[:end_pos + offset] + 
                                         end_marker + 
                                         marked_text[end_pos + offset:])
                            offset += len(end_marker)
                        
                        # Insert start marker
                        start_pos = chunk["start_pos"] + offset
                        marked_text = (marked_text[:start_pos] + 
                                     start_marker + 
                                     marked_text[start_pos:])
                        offset += len(start_marker)
                
                return {
                    "marked_text": marked_text,
                    "truncated": truncated,
                    "total_display_length": len(display_text)
                }
            
            boundary_viz = create_boundary_visualization(cleaned_content, chunks)
            
            # Step 8: Compile comprehensive RAG test results
            results = {
                "email_info": {
                    "id": email_dict["id"],
                    "subject": email_dict["subject"],
                    "from_email": email_dict["from_email"],
                    "from_name": email_dict["from_name"],
                    "date_sent": str(email_dict["date_sent"]) if email_dict["date_sent"] else None
                },
                
                "qwen_classification": {
                    "classification": email_dict.get("classification"),
                    "confidence": email_dict.get("confidence"),
                    "sentiment": email_dict.get("sentiment"),
                    "priority": email_dict.get("priority"),
                    "should_process": email_dict.get("should_process"),
                    "relationship_strength": email_dict.get("relationship_strength"),
                    "formality": email_dict.get("formality"),
                    "personalization": email_dict.get("personalization"),
                    "classified_at": str(email_dict.get("classified_at")) if email_dict.get("classified_at") else None
                },
                
                "rag_readiness": rag_readiness,
                "rag_eligible": rag_eligible,
                
                "content_processing": {
                    "original_length": len(raw_content),
                    "cleaned_length": len(cleaned_content),
                    "cleaning_method": cleaning_method,
                    "reduction_ratio": round((len(raw_content) - len(cleaned_content)) / len(raw_content), 3) if len(raw_content) > 0 else 0
                },
                
                "chunking_strategy": {
                    "chunk_size": chunk_size,
                    "overlap_ratio": overlap_ratio,
                    "overlap_size": int(chunk_size * overlap_ratio),
                    "total_chunks": len(chunks),
                    "avg_chunk_length": sum(c["length"] for c in chunks) / len(chunks) if chunks else 0,
                    "avg_word_count": sum(c["word_count"] for c in chunks) / len(chunks) if chunks else 0
                },
                
                "chunks": chunks,
                "embeddings": embedding_metadata,
                
                "rag_testing": {
                    "query": rag_query,
                    "query_embedding_generated": query_embedding is not None,
                    "retrieval_results": retrieval_results[:5],  # Top 5 matches
                    "total_matches": len(retrieval_results)
                },
                
                "debugging": {
                    "boundary_visualization": boundary_viz,
                    "chunking_quality": {
                        "all_chunks_valid": all(c["length"] > 20 for c in chunks),
                        "size_variance": np.var([c["length"] for c in chunks]) if chunks else 0,
                        "avg_quality_score": sum(c["quality_score"] for c in chunks) / len(chunks) if chunks else 0
                    }
                },
                
                "recommendations": {
                    "chunking": "Adjust chunk_size and overlap_ratio based on similarity scores",
                    "processing": f"Email is {'ready for' if rag_eligible else 'not ready for'} RAG processing",
                    "optimization": "Monitor retrieval results to fine-tune chunking strategy"
                },
                
                "summary": {
                    "rag_test_successful": rag_eligible and len(retrieval_results) > 0,
                    "chunks_with_embeddings": len([e for e in embedding_metadata if e.get("embedding_generated")]),
                    "best_similarity_score": retrieval_results[0]["similarity_score"] if retrieval_results else 0,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            return results
            
        finally:
            session.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline test failed: {str(e)}")

async def main_processing_loop():
    """Main background processing loop"""
    logging.info("🔄 Starting main processing loop...")
    
    while True:
        try:
            if email_processor and health_monitor:
                # Check if we should process
                health_status = await health_monitor.get_health_status()
                
                if health_status["healthy"]:
                    # Process pending emails
                    await email_processor.process_pending_emails()
                else:
                    logging.warning("⚠️ Skipping processing due to unhealthy status")
                
                # Sleep for the configured interval
                await asyncio.sleep(processor_config.processing_interval)
            else:
                # Services not ready yet
                await asyncio.sleep(5)
                
        except Exception as e:
            logging.error(f"❌ Error in main processing loop: {e}")
            await asyncio.sleep(30)  # Wait longer on error

def setup_logging():
    """Configure logging for the service"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("/app/logs/ai-processor.log") if os.path.exists("/app/logs") else logging.NullHandler()
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logging.info(f"📡 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if we should use uvicorn directly for reload
    use_reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    
    # Always use uvicorn directly for now (disable background processing during startup issues)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,  # Force disable reload to prevent infinite loop
        log_level="info"
    )