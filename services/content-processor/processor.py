"""
Content Processor - API Consumer
Lightweight email content processing using Unstructured API + local embeddings
"""

import asyncio
import logging
import time
import tempfile
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path

# Email cleaning imports  
try:
    from email_reply_parser import EmailReplyParser
    EMAIL_CLEANING_AVAILABLE = True
except ImportError as e:
    logging.error(f"❌ Email cleaning libraries not available: {e}")
    EMAIL_CLEANING_AVAILABLE = False

# Embedding imports
from sentence_transformers import SentenceTransformer

from config import ProcessorConfig, ELEMENT_TYPE_DESCRIPTIONS

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of content processing"""
    email_id: str
    success: bool
    elements_extracted: int
    chunks_created: int
    embeddings_created: int
    quality_score: float
    processing_time_ms: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "email_id": self.email_id,
            "success": self.success,
            "elements_extracted": self.elements_extracted,
            "chunks_created": self.chunks_created,
            "embeddings_created": self.embeddings_created,
            "quality_score": self.quality_score,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
            "metadata": self.metadata or {}
        }

@dataclass
class EmailElement:
    """Structured email element from Unstructured API"""
    element_id: str
    element_type: str
    content: str
    metadata: Dict[str, Any]
    coordinates: Optional[Dict[str, Any]] = None
    page_number: int = 1
    sequence_number: int = 0
    parent_id: Optional[str] = None

@dataclass
class EmailChunk:
    """Processed email chunk with embeddings"""
    chunk_id: str
    text: str
    element_type: str
    element_ids: List[str]
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    quality_score: float = 1.0

class UnstructuredLibraryClient:
    """Docker exec client for Unstructured library"""
    
    def __init__(self, container_name: str = "email-rag-unstructured-1"):
        self.container_name = container_name
    
    async def health_check(self) -> bool:
        """Check if Unstructured container is available"""
        try:
            result = subprocess.run([
                "docker", "exec", self.container_name, 
                "python", "-c", "import unstructured; print('OK')"
            ], capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and "OK" in result.stdout
        except Exception as e:
            logger.error(f"Unstructured container health check failed: {e}")
            return False
    
    async def partition_email(self, email_content: str, strategy: str = "auto") -> List[Dict[str, Any]]:
        """Partition email content using Unstructured library via Docker exec"""
        try:
            # Create temporary email file in container's /tmp
            temp_filename = f"/tmp/email_{int(time.time())}.eml"
            
            # Write email content to container
            echo_result = subprocess.run([
                "docker", "exec", "-i", self.container_name,
                "bash", "-c", f"cat > {temp_filename}"
            ], input=email_content, text=True, capture_output=True, timeout=30)
            
            if echo_result.returncode != 0:
                logger.error(f"Failed to write email file: {echo_result.stderr}")
                return []
            
            # Create Python script to process the email
            python_script = f'''
import json
from unstructured.partition.email import partition_email
from unstructured.staging.base import elements_to_json

try:
    elements = partition_email(filename="{temp_filename}")
    result = []
    for element in elements:
        result.append({{
            "type": element.category,
            "text": str(element),
            "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {{}}
        }})
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''
            
            # Execute Python script in container
            exec_result = subprocess.run([
                "docker", "exec", self.container_name,
                "python", "-c", python_script
            ], capture_output=True, text=True, timeout=60)
            
            # Clean up temp file
            subprocess.run([
                "docker", "exec", self.container_name,
                "rm", "-f", temp_filename
            ], capture_output=True)
            
            if exec_result.returncode != 0:
                logger.error(f"Unstructured processing failed: {exec_result.stderr}")
                return []
            
            try:
                result = json.loads(exec_result.stdout)
                if isinstance(result, dict) and "error" in result:
                    logger.error(f"Unstructured processing error: {result['error']}")
                    return []
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Unstructured output: {e}")
                logger.error(f"Raw output: {exec_result.stdout}")
                return []
                    
        except Exception as e:
            logger.error(f"Error calling Unstructured library: {e}")
            return []

class ContentProcessor:
    """
    Lightweight content processor using Unstructured API for document processing
    and local sentence-transformers for embeddings
    """
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.ready = False
        self.embedding_model = None
        self.total_processed = 0
        self.total_processing_time = 0.0
        self.logger = logging.getLogger(__name__)
        self.last_processing_data = {}  # Store last processed data for database storage
        
        # Initialize Unstructured library client
        self.unstructured_client = UnstructuredLibraryClient()
        
    async def initialize(self):
        """Initialize the content processor"""
        try:
            self.logger.info("🔧 Initializing content processor...")
            
            # Check Unstructured container availability
            container_available = await self.unstructured_client.health_check()
            if not container_available:
                self.logger.warning("⚠️ Unstructured container not available - will retry during processing")
            else:
                self.logger.info("✅ Unstructured container connection verified")
                
            # Initialize embedding model
            self.logger.info(f"📥 Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.logger.info("✅ Embedding model loaded")
            
            self.ready = True
            self.logger.info("✅ Content processor ready")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize content processor: {e}")
            raise
    
    def _clean_email_content(self, content: str, html_content: Optional[str] = None, sender: str = "") -> Tuple[str, Dict[str, Any]]:
        """
        Clean email content by removing replies, signatures, and quoted text
        Returns: (cleaned_content, cleaning_metadata)
        """
        if not EMAIL_CLEANING_AVAILABLE:
            return content, {"method": "none", "cleaned": False}
        
        try:
            # Choose best content to clean
            content_to_clean = html_content if html_content and self.config.prefer_html_content else content
            original_length = len(content_to_clean)
            
            # Use email-reply-parser for basic cleaning
            try:
                cleaned_content = EmailReplyParser.parse_reply(content_to_clean)
                self.logger.debug(f"📧 Email reply parser: {original_length} → {len(cleaned_content)} chars")
            except Exception as e:
                self.logger.warning(f"Email reply parser failed: {e}")
                cleaned_content = content_to_clean
            
            # Basic HTML cleaning if needed
            if html_content and "<" in cleaned_content:
                import re
                # Remove HTML tags but preserve structure
                cleaned_content = re.sub(r'<script[^>]*>.*?</script>', '', cleaned_content, flags=re.DOTALL)
                cleaned_content = re.sub(r'<style[^>]*>.*?</style>', '', cleaned_content, flags=re.DOTALL)
                # Keep line breaks from HTML
                cleaned_content = re.sub(r'<br[^>]*>', '\n', cleaned_content)
                cleaned_content = re.sub(r'<div[^>]*>', '\n', cleaned_content)
                cleaned_content = re.sub(r'</div>', '\n', cleaned_content)
                cleaned_content = re.sub(r'<p[^>]*>', '\n', cleaned_content)
                cleaned_content = re.sub(r'</p>', '\n', cleaned_content)
            
            # Final cleanup
            import re
            cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)  # Multiple newlines
            cleaned_content = re.sub(r'[ \t]+', ' ', cleaned_content)  # Multiple spaces
            cleaned_content = cleaned_content.strip()
            
            final_length = len(cleaned_content)
            reduction_ratio = (original_length - final_length) / original_length if original_length > 0 else 0
            
            # Validate cleaning didn't remove too much content
            if final_length < original_length * 0.1 and original_length > 100:
                self.logger.warning(f"⚠️ Cleaning removed {reduction_ratio:.1%} of content, reverting to original")
                cleaned_content = content_to_clean
                reduction_ratio = 0
            
            cleaning_metadata = {
                "method": "email-reply-parser",
                "cleaned": True,
                "original_length": original_length,
                "final_length": final_length,
                "reduction_ratio": reduction_ratio,
                "html_processed": html_content is not None
            }
            
            self.logger.info(f"✨ Email cleaning complete: {reduction_ratio:.1%} reduction ({original_length}→{final_length} chars)")
            return cleaned_content, cleaning_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Email cleaning failed: {e}")
            return content, {"method": "failed", "cleaned": False, "error": str(e)}
    
    def _create_email_file_content(self, content: str, html_content: Optional[str] = None) -> str:
        """Create email file content for Unstructured API processing"""
        # Create a basic .eml structure
        email_lines = []
        
        # Basic headers
        email_lines.append("MIME-Version: 1.0")
        
        if html_content and self.config.prefer_html_content:
            email_lines.append("Content-Type: text/html; charset=utf-8")
            email_lines.append("")  # Empty line after headers
            email_lines.append(html_content)
        else:
            email_lines.append("Content-Type: text/plain; charset=utf-8")
            email_lines.append("")  # Empty line after headers
            email_lines.append(content)
        
        return '\n'.join(email_lines)
    
    async def _extract_elements_via_library(self, email_content: str) -> List[EmailElement]:
        """Extract elements from email using Unstructured library"""
        try:
            # Call Unstructured library via Docker exec
            library_response = await self.unstructured_client.partition_email(
                email_content, 
                strategy="auto"
            )
            
            if not library_response:
                self.logger.warning("No response from Unstructured library")
                return []
            
            elements = []
            for i, element_data in enumerate(library_response):
                # Extract element information from API response
                element_type = element_data.get('type', 'UncategorizedText')
                element_text = element_data.get('text', '')
                element_metadata = element_data.get('metadata', {})
                
                # Filter by element type if specified
                if (self.config.element_types_to_process and 
                    element_type not in self.config.element_types_to_process):
                    continue
                
                # Filter by length
                if len(element_text) < self.config.min_element_length:
                    continue
                
                # Create structured element
                email_element = EmailElement(
                    element_id=element_data.get('element_id', f"element_{i}"),
                    element_type=element_type,
                    content=element_text,
                    metadata=element_metadata,
                    sequence_number=i
                )
                
                # Add coordinates if available
                if 'coordinates' in element_metadata:
                    email_element.coordinates = element_metadata['coordinates']
                
                elements.append(email_element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Element extraction via library failed: {e}")
            return []
    
    def _chunk_elements_simple(self, elements: List[EmailElement]) -> List[EmailChunk]:
        """Simple chunking strategy for elements"""
        if not elements:
            return []
        
        chunks = []
        current_text = ""
        current_elements = []
        chunk_index = 0
        
        for element in elements:
            # Add element to current chunk
            if current_text:
                current_text += " " + element.content
            else:
                current_text = element.content
            current_elements.append(element.element_id)
            
            # Check if chunk is large enough
            word_count = len(current_text.split())
            if word_count >= self.config.target_chunk_words:
                chunk = EmailChunk(
                    chunk_id=f"chunk_{chunk_index}",
                    text=current_text,
                    element_type=element.element_type,
                    element_ids=current_elements.copy(),
                    quality_score=0.8,
                    metadata={
                        "word_count": word_count,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "simple"
                    }
                )
                chunks.append(chunk)
                
                # Reset for next chunk
                current_text = ""
                current_elements = []
                chunk_index += 1
        
        # Add remaining content as final chunk
        if current_text and len(current_text.split()) >= self.config.min_chunk_words:
            chunk = EmailChunk(
                chunk_id=f"chunk_{chunk_index}",
                text=current_text,
                element_type="NarrativeText",
                element_ids=current_elements,
                quality_score=0.8,
                metadata={
                    "word_count": len(current_text.split()),
                    "chunk_index": chunk_index,
                    "chunking_strategy": "simple"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[EmailChunk]) -> List[EmailChunk]:
        """Generate embeddings for chunks"""
        if not self.embedding_model or not chunks:
            return chunks
        
        try:
            # Extract texts for batch processing
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(texts), self.config.embedding_batch_size):
                batch_texts = texts[i:i + self.config.embedding_batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return chunks
    
    def _calculate_quality_score(
        self, 
        elements: List[EmailElement], 
        chunks: List[EmailChunk], 
        processing_time: float
    ) -> float:
        """Calculate overall processing quality score"""
        scores = {}
        
        # Element extraction quality
        if elements:
            avg_element_length = sum(len(e.content) for e in elements) / len(elements)
            element_type_diversity = len(set(e.element_type for e in elements)) / len(elements)
            scores["element_extraction"] = min(1.0, (avg_element_length / 100) * element_type_diversity)
        else:
            scores["element_extraction"] = 0.0
        
        # Chunk coherence quality
        if chunks:
            avg_chunk_quality = sum(c.quality_score for c in chunks) / len(chunks)
            chunk_size_variance = np.var([len(c.text.split()) for c in chunks])
            scores["chunk_coherence"] = avg_chunk_quality * (1 - min(0.5, chunk_size_variance / 1000))
        else:
            scores["chunk_coherence"] = 0.0
        
        # Embedding quality (presence of embeddings)
        embeddings_generated = sum(1 for c in chunks if c.embedding is not None)
        scores["embedding_quality"] = embeddings_generated / len(chunks) if chunks else 0.0
        
        # Structure preservation (element types preserved)
        element_types_preserved = len(set(c.element_type for c in chunks))
        max_possible_types = len(set(e.element_type for e in elements)) if elements else 1
        scores["structure_preservation"] = element_types_preserved / max_possible_types
        
        # Calculate weighted average
        total_score = 0.0
        for metric, weight in self.config.quality_weights.items():
            total_score += scores.get(metric, 0.0) * weight
        
        return min(1.0, max(0.0, total_score))
    
    async def process_email(
        self,
        email_id: str,
        subject: str,
        content: str,
        html_content: Optional[str] = None,
        sender: str = ""
    ) -> ProcessingResult:
        """Process email content using Unstructured API + local embeddings"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 Processing email {email_id}")
            
            # Step 1: Clean email content (remove replies, signatures)
            cleaned_content, cleaning_metadata = self._clean_email_content(
                content, html_content, sender
            )
            
            # Check content length after cleaning
            if len(cleaned_content) > self.config.max_content_length:
                self.logger.warning(f"Email {email_id} too long after cleaning ({len(cleaned_content)} chars), truncating")
                cleaned_content = cleaned_content[:self.config.max_content_length]
            
            # Step 2: Create email file content for API
            email_file_content = self._create_email_file_content(cleaned_content, None)
            
            # Step 3: Extract elements using Unstructured library
            elements = await self._extract_elements_via_library(email_file_content)
            self.logger.info(f"📋 Extracted {len(elements)} elements from email {email_id}")
            
            # Step 4: Chunk elements (simple strategy)
            chunks = self._chunk_elements_simple(elements)
            self.logger.info(f"🧩 Created {len(chunks)} chunks from email {email_id}")
            
            # Step 5: Generate embeddings locally
            chunks_with_embeddings = await self._generate_embeddings(chunks)
            embeddings_created = sum(1 for c in chunks_with_embeddings if c.embedding is not None)
            self.logger.info(f"🎯 Generated {embeddings_created} embeddings for email {email_id}")
            
            # Store processed data for database storage
            self.last_processing_data = {
                'elements': elements,
                'chunks': chunks_with_embeddings
            }
            
            processing_time = time.time() - start_time
            quality_score = self._calculate_quality_score(elements, chunks_with_embeddings, processing_time)
            
            # Update stats
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            # Enhanced metadata including cleaning results
            processing_metadata = {
                "element_types": list(set(e.element_type for e in elements)),
                "avg_chunk_length": sum(len(c.text.split()) for c in chunks_with_embeddings) / len(chunks_with_embeddings) if chunks_with_embeddings else 0,
                "html_processed": html_content is not None,
                "content_length_original": len(html_content or content),
                "content_length_cleaned": len(cleaned_content),
                "cleaning_metadata": cleaning_metadata,
                "processing_method": "unstructured_library"
            }
            
            result = ProcessingResult(
                email_id=email_id,
                success=True,
                elements_extracted=len(elements),
                chunks_created=len(chunks_with_embeddings),
                embeddings_created=embeddings_created,
                quality_score=quality_score,
                processing_time_ms=processing_time * 1000,
                metadata=processing_metadata
            )
            
            self.logger.info(f"✅ Processed email {email_id} in {processing_time:.2f}s (quality: {quality_score:.2f}, {cleaning_metadata.get('reduction_ratio', 0):.1%} content reduction)")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Failed to process email {email_id}: {e}")
            
            return ProcessingResult(
                email_id=email_id,
                success=False,
                elements_extracted=0,
                chunks_created=0,
                embeddings_created=0,
                quality_score=0.0,
                processing_time_ms=processing_time * 1000,
                error_message=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_time = (self.total_processing_time / self.total_processed) if self.total_processed > 0 else 0
        
        return {
            "ready": self.ready,
            "unstructured_available": True,  # We check this via Docker exec
            "embedding_model_loaded": self.embedding_model is not None,
            "total_processed": self.total_processed,
            "average_processing_time_ms": avg_time * 1000,
            "emails_per_second": 1 / avg_time if avg_time > 0 else 0,
            "config": self.config.to_dict()
        }