"""
Configuration for Content Processor Service  
Optimized for Unstructured.io email processing and embeddings
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class ProcessorConfig:
    """Configuration for content processing service"""
    
    # Unstructured configuration
    use_unstructured: bool = True
    unstructured_strategy: str = "by_title"  # or "basic", "by_similarity"
    include_page_breaks: bool = False  # Not relevant for emails
    max_characters: int = 500  # For chunking
    new_after_n_chars: int = 400  # Soft limit
    overlap: int = 50  # Character overlap between chunks
    overlap_all: bool = True
    
    # Email processing settings
    prefer_html_content: bool = True  # Use HTML over text when available
    max_content_length: int = 50000  # Skip extremely long emails
    include_headers: bool = False  # Don't include email headers in elements
    process_attachments: bool = False  # Skip attachments for now
    
    # Element filtering
    min_element_length: int = 20  # Skip very short elements
    element_types_to_process: List[str] = None  # Process all types by default
    
    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # MiniLM dimension
    embedding_batch_size: int = 16
    normalize_embeddings: bool = True
    
    # Chunking quality settings
    min_chunk_words: int = 10
    max_chunk_words: int = 200
    target_chunk_words: int = 100
    semantic_coherence_threshold: float = 0.7
    
    # Processing constraints
    max_concurrent_requests: int = 4  # Lower for content processing
    processing_timeout: int = 120  # 2 minutes per email
    max_retries: int = 2
    
    # Quality scoring
    quality_weights: Dict[str, float] = None
    
    # Environment settings
    database_url: str = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@postgres:5432/email_rag")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    processing_interval: int = int(os.getenv("PROCESSING_INTERVAL", "10"))  # Seconds
    batch_size: int = int(os.getenv("PROCESSING_BATCH_SIZE", "5"))
    
    # Resource limits
    max_memory_mb: int = 2048  # 2GB limit
    cpu_limit_percent: int = 80
    
    def __post_init__(self):
        """Set defaults after initialization"""
        if self.element_types_to_process is None:
            self.element_types_to_process = [
                "Title", "NarrativeText", "ListItem", "Text", 
                "UncategorizedText", "Table", "Address"
            ]
        
        if self.quality_weights is None:
            self.quality_weights = {
                "element_extraction": 0.3,
                "chunk_coherence": 0.3,
                "embedding_quality": 0.2,
                "structure_preservation": 0.2
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "unstructured": {
                "strategy": self.unstructured_strategy,
                "max_characters": self.max_characters,
                "new_after_n_chars": self.new_after_n_chars,
                "overlap": self.overlap,
                "overlap_all": self.overlap_all
            },
            "email_processing": {
                "prefer_html_content": self.prefer_html_content,
                "max_content_length": self.max_content_length,
                "include_headers": self.include_headers,
                "process_attachments": self.process_attachments
            },
            "embedding": {
                "model": self.embedding_model,
                "dimension": self.embedding_dimension,
                "batch_size": self.embedding_batch_size,
                "normalize": self.normalize_embeddings
            },
            "chunking": {
                "min_words": self.min_chunk_words,
                "max_words": self.max_chunk_words,
                "target_words": self.target_chunk_words,
                "coherence_threshold": self.semantic_coherence_threshold
            },
            "performance": {
                "max_concurrent_requests": self.max_concurrent_requests,
                "processing_timeout": self.processing_timeout,
                "max_memory_mb": self.max_memory_mb,
                "cpu_limit_percent": self.cpu_limit_percent
            },
            "quality": {
                "weights": self.quality_weights,
                "element_types": self.element_types_to_process
            }
        }

# Unstructured element type mapping
ELEMENT_TYPE_DESCRIPTIONS = {
    "Title": "Email subject or section headers",
    "NarrativeText": "Main email body content",
    "ListItem": "Bulleted or numbered lists",
    "Text": "General text content",
    "UncategorizedText": "Text that doesn't fit other categories",
    "Table": "Tabular data in emails",
    "Address": "Email addresses, physical addresses",
    "Header": "Email headers (if included)",
    "Footer": "Email footers or signatures"
}

# Email content type priorities
CONTENT_TYPE_PRIORITY = {
    "html": 1,      # Prefer HTML content
    "text": 2,      # Fallback to plain text
    "multipart": 3  # Complex multipart emails
}

def get_config() -> ProcessorConfig:
    """Get processor configuration with environment overrides"""
    return ProcessorConfig()