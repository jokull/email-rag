"""
Configuration for the thread processor service optimized for Qwen-0.5B on Mac mini M2
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    """Configuration for Qwen-0.5B LLM optimized for Mac mini M2 16GB RAM"""
    
    # Model configuration
    model_name: str = "qwen-0.5b"
    model_path: str = "./models/qwen2.5-0.5b-instruct-q4_0.gguf"  # 4-bit quantized GGUF format
    model_size_mb: int = 395  # Disk size
    runtime_memory_mb: int = 1024  # Max RAM usage including context buffers
    
    # Inference settings optimized for Qwen-0.5B
    max_tokens: int = 512  # Keep responses concise
    context_length: int = 2048  # Conservative context window
    temperature: float = 0.1  # Low temperature for consistent classification
    batch_size: int = 8  # Batch processing for efficiency
    
    # Hardware acceleration
    use_metal: bool = True  # Apple Metal GPU acceleration
    use_neural_engine: bool = True  # Apple Neural Engine when available
    threads: int = 8  # M2 has 8 performance cores
    
    # Memory management
    keep_model_loaded: bool = True  # Keep warm for low latency
    max_concurrent_requests: int = 4  # Prevent memory overflow
    
    # Processing constraints aligned with Qwen-0.5B capabilities
    chunk_size_tokens: int = 600  # Optimal chunk size (500-800 range)
    chunk_overlap_ratio: float = 0.1  # 10% overlap following Unstructured.io best practices
    max_email_length: int = 4000  # Skip extremely long emails
    classification_only: bool = True  # Focus on classification, not generation
    
    # Prompt templates optimized for compact responses
    classification_prompt: str = """Classify this email. Return ONLY one word: human, promotional, transactional, or automated.

Email: {content}

Answer:"""
    
    threading_prompt: str = """Are these emails part of the same conversation thread? Answer: yes/no
Email 1: {email1}
Email 2: {email2}
Answer:"""
    
    chunking_prompt: str = """Split this email into 2-3 semantic chunks. Return only the chunks, no explanation.
Email: {content}
Chunks:"""

@dataclass
class ProcessorConfig:
    """Main thread processor configuration"""
    
    # Database connection
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/emailrag")
    
    # Processing settings
    batch_size: int = 5  # Process emails in batches
    processing_interval: int = 60  # Seconds between processing runs
    max_retries: int = 3
    
    # Email filtering (pre-LLM heuristics to reduce load)
    min_email_length: int = 50  # Skip very short emails
    max_email_length: int = 4000  # Skip extremely long emails
    skip_auto_replies: bool = True  # Skip obvious auto-replies
    skip_large_attachments: bool = True  # Skip emails with large attachments
    
    # Threading configuration
    use_jwz_algorithm: bool = True  # Use JWZ threading as baseline
    ml_threading_confidence_threshold: float = 0.8
    semantic_similarity_threshold: float = 0.7
    
    # Classification thresholds
    human_classification_threshold: float = 0.7
    personal_classification_threshold: float = 0.5
    relevance_classification_threshold: float = 0.6
    
    # RAG pipeline
    enable_embedding_generation: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embedding model
    vector_dimension: int = 384  # MiniLM embedding dimension
    
    # Performance monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # Content cleaning
    use_talon: bool = True  # Use Talon for signature/quote removal
    clean_html: bool = True
    remove_forwarded_content: bool = True
    
def get_config() -> tuple[LLMConfig, ProcessorConfig]:
    """Get configuration instances with environment variable overrides"""
    
    llm_config = LLMConfig()
    processor_config = ProcessorConfig()
    
    # Environment variable overrides
    if model_path := os.getenv("QWEN_MODEL_PATH"):
        llm_config.model_path = model_path
    
    if batch_size := os.getenv("LLM_BATCH_SIZE"):
        llm_config.batch_size = int(batch_size)
    
    if chunk_size := os.getenv("CHUNK_SIZE_TOKENS"):
        llm_config.chunk_size_tokens = int(chunk_size)
    
    # Validate configuration
    if not os.path.exists(llm_config.model_path):
        print(f"Warning: Model file not found at {llm_config.model_path}")
        print("Run: llm install qwen-0.5b-q4_0 to download the model")
    
    return llm_config, processor_config

# Usage constants based on the guide
RECOMMENDED_USE_CASES = [
    "inbox_classification",
    "reply_chain_detection", 
    "message_chunking",
    "embedding_generation",
    "heuristic_labeling"
]

AVOID_USE_CASES = [
    "full_document_summarization",
    "long_form_generation",
    "multi_modal_processing",
    "streaming_conversations",
    "complex_reasoning"
]

# Qwen-0.5B specific prompts optimized for token efficiency
CLASSIFICATION_PROMPTS = {
    "human": "Human conversation? yes/no: {content}",
    "promotional": "Marketing email? yes/no: {content}", 
    "transactional": "Transaction/receipt? yes/no: {content}",
    "automated": "Automated message? yes/no: {content}"
}

# Memory usage estimates
MEMORY_ESTIMATES = {
    "model_base": 512,  # MB - Base model memory
    "context_buffer": 256,  # MB - Context and processing buffers
    "batch_processing": 128,  # MB - Batch processing overhead
    "embeddings_cache": 64,  # MB - Embedding cache
    "total_estimated": 960  # MB - Total estimated usage
}