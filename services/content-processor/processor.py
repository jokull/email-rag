"""
Content Processor
Advanced email content processing using Unstructured.io and embeddings
"""

import asyncio
import logging
import time
import tempfile
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path

# Unstructured imports
try:
    from unstructured.partition.email import partition_email
    from unstructured.chunking.title import chunk_by_title
    from unstructured.chunking.basic import chunk_elements
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    logging.error(f"❌ Unstructured not available: {e}")
    UNSTRUCTURED_AVAILABLE = False

# Email cleaning imports
try:
    import talon
    from talon import quotations
    from email_reply_parser import EmailReplyParser
    EMAIL_CLEANING_AVAILABLE = True
except ImportError as e:
    logging.error(f"❌ Email cleaning libraries not available: {e}")
    EMAIL_CLEANING_AVAILABLE = False

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.error(f"❌ SentenceTransformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
    """Structured email element from Unstructured"""
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

class ContentProcessor:
    """
    Advanced content processor using Unstructured.io for email parsing
    and sentence-transformers for embeddings
    """
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.ready = False
        self.unstructured_available = UNSTRUCTURED_AVAILABLE
        self.embedding_model = None
        self.total_processed = 0
        self.total_processing_time = 0.0
        self.logger = logging.getLogger(__name__)
        self.last_processing_data = {}  # Store last processed data for database storage
        
    async def initialize(self):
        """Initialize the content processor"""
        try:
            self.logger.info("🔧 Initializing content processor...")
            
            # Check Unstructured availability
            if not self.unstructured_available:
                raise RuntimeError("Unstructured.io not available")
                
            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.logger.info(f"📥 Loading embedding model: {self.config.embedding_model}")
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                self.logger.info("✅ Embedding model loaded")
            else:
                self.logger.warning("⚠️ SentenceTransformers not available - embeddings disabled")
            
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
            
            # Step 1: Remove quoted/reply content using Talon
            try:
                cleaned_content = quotations.extract_from_plain(content_to_clean)
                self.logger.debug(f"📧 Talon quotation removal: {original_length} → {len(cleaned_content)} chars")
            except Exception as e:
                self.logger.warning(f"Talon quotation removal failed: {e}")
                cleaned_content = content_to_clean
            
            # Step 2: Remove signatures using Talon
            try:
                cleaned_content, signature = talon.signature.extract(cleaned_content, sender)
                self.logger.debug(f"✂️ Talon signature removal: extracted {len(signature)} char signature")
            except Exception as e:
                self.logger.warning(f"Talon signature removal failed: {e}")
                signature = ""
            
            # Step 3: Alternative cleaning with email-reply-parser for comparison
            try:
                reply_parser_result = EmailReplyParser.parse_reply(content_to_clean)
                if len(reply_parser_result) < len(cleaned_content) and len(reply_parser_result) > 50:
                    # Use reply parser result if it's more aggressive but still substantial
                    cleaned_content = reply_parser_result
                    self.logger.debug(f"🔧 Used email-reply-parser result instead")
            except Exception as e:
                self.logger.warning(f"Email-reply-parser failed: {e}")
            
            # Step 4: Basic HTML cleaning if needed
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
            
            # Step 5: Final cleanup
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
                "method": "talon+email-reply-parser",
                "cleaned": True,
                "original_length": original_length,
                "final_length": final_length,
                "reduction_ratio": reduction_ratio,
                "signature_length": len(signature),
                "html_processed": html_content is not None
            }
            
            self.logger.info(f"✨ Email cleaning complete: {reduction_ratio:.1%} reduction ({original_length}→{final_length} chars)")
            return cleaned_content, cleaning_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Email cleaning failed: {e}")
            return content, {"method": "failed", "cleaned": False, "error": str(e)}
    
    def _convert_elements_to_markdown(self, elements: List[EmailElement]) -> List[Tuple[EmailElement, str]]:
        """
        Convert Unstructured elements to clean markdown format
        Returns: List of (element, markdown_content) tuples
        """
        markdown_elements = []
        
        for element in elements:
            try:
                markdown_content = self._element_to_markdown(element)
                markdown_elements.append((element, markdown_content))
            except Exception as e:
                self.logger.warning(f"Failed to convert element to markdown: {e}")
                # Fallback to plain text
                markdown_elements.append((element, element.content))
        
        return markdown_elements
    
    def _element_to_markdown(self, element: EmailElement) -> str:
        """
        Convert a single element to markdown format
        """
        content = element.content.strip()
        element_type = element.element_type
        
        if element_type == "Title":
            # Email subjects or section headers
            return f"# {content}\n\n"
        
        elif element_type == "NarrativeText":
            # Main email content - clean paragraphs
            # Format URLs as links
            import re
            content = re.sub(
                r'(https?://[^\s]+)', 
                r'[\1](\1)', 
                content
            )
            # Format email addresses  
            content = re.sub(
                r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
                r'`\1`',
                content
            )
            return f"{content}\n\n"
        
        elif element_type == "ListItem":
            # Preserve list structure
            return f"- {content}\n"
        
        elif element_type == "Table":
            # Convert table to markdown table (simplified)
            return f"\n{content}\n\n"  # TODO: Implement proper table conversion
        
        elif element_type == "Address":
            # Format addresses in code blocks
            return f"```\n{content}\n```\n\n"
        
        elif element_type == "UncategorizedText":
            # Treat as regular text
            return f"{content}\n\n"
        
        else:
            # Default formatting
            return f"{content}\n\n"

    def _create_temp_email_file(self, content: str, html_content: Optional[str] = None) -> str:
        """Create temporary .eml file for Unstructured processing"""
        # Create a basic .eml structure
        email_content = []
        
        # Basic headers
        email_content.append("MIME-Version: 1.0")
        email_content.append("Content-Type: text/plain; charset=utf-8")
        email_content.append("")  # Empty line after headers
        
        # Use HTML content if available and preferred
        if html_content and self.config.prefer_html_content:
            email_content = [
                "MIME-Version: 1.0",
                "Content-Type: text/html; charset=utf-8",
                "",
                html_content
            ]
        else:
            email_content.append(content)
        
        # Write to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.eml', delete=False, encoding='utf-8')
        temp_file.write('\n'.join(email_content))
        temp_file.close()
        
        return temp_file.name
    
    def _extract_elements(self, email_path: str) -> List[EmailElement]:
        """Extract elements from email using Unstructured"""
        try:
            # Partition the email
            elements = partition_email(
                filename=email_path,
                include_headers=self.config.include_headers,
                process_attachments=self.config.process_attachments,
                content_source="text/html" if self.config.prefer_html_content else "text/plain"
            )
            
            processed_elements = []
            
            for i, element in enumerate(elements):
                # Filter by element type if specified
                if (self.config.element_types_to_process and 
                    element.category not in self.config.element_types_to_process):
                    continue
                
                # Filter by length
                element_text = str(element)
                if len(element_text) < self.config.min_element_length:
                    continue
                
                # Create structured element
                email_element = EmailElement(
                    element_id=getattr(element, 'id', f"element_{i}"),
                    element_type=element.category,
                    content=element_text,
                    metadata=element.metadata.to_dict() if hasattr(element, 'metadata') else {},
                    sequence_number=i
                )
                
                # Add coordinates if available
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'coordinates'):
                    email_element.coordinates = element.metadata.coordinates
                
                processed_elements.append(email_element)
            
            return processed_elements
            
        except Exception as e:
            self.logger.error(f"Element extraction failed: {e}")
            return []
    
    def _chunk_elements(self, elements: List[EmailElement]) -> List[EmailChunk]:
        """Chunk elements using Unstructured's chunking strategies"""
        if not elements:
            return []
        
        try:
            # Convert back to Unstructured elements for chunking
            unstructured_elements = []
            for elem in elements:
                # Create a minimal element object for chunking
                # This is a simplified approach - in practice you'd want to reconstruct proper Element objects
                class MockElement:
                    def __init__(self, text, category, metadata):
                        self.text = text
                        self.category = category
                        self.metadata = metadata
                    
                    def __str__(self):
                        return self.text
                
                mock_elem = MockElement(elem.content, elem.element_type, elem.metadata)
                unstructured_elements.append(mock_elem)
            
            # Apply chunking strategy
            if self.config.unstructured_strategy == "by_title":
                chunks = chunk_by_title(
                    unstructured_elements,
                    max_characters=self.config.max_characters,
                    new_after_n_chars=self.config.new_after_n_chars,
                    overlap=self.config.overlap,
                    overlap_all=self.config.overlap_all
                )
            else:  # basic strategy
                chunks = chunk_elements(
                    unstructured_elements,
                    max_characters=self.config.max_characters,
                    new_after_n_chars=self.config.new_after_n_chars,
                    overlap=self.config.overlap,
                    overlap_all=self.config.overlap_all
                )
            
            # Convert to EmailChunk objects
            email_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_text = str(chunk)
                
                # Quality check
                word_count = len(chunk_text.split())
                if word_count < self.config.min_chunk_words:
                    continue
                
                # Calculate quality score based on word count and coherence
                quality_score = min(1.0, max(0.1, 
                    (word_count - self.config.min_chunk_words) / 
                    (self.config.target_chunk_words - self.config.min_chunk_words)
                ))
                
                # Get element type from original elements
                element_type = "NarrativeText"  # Default
                element_ids = []
                
                # Try to map back to original elements
                for elem in elements:
                    if elem.content in chunk_text:
                        element_type = elem.element_type
                        element_ids.append(elem.element_id)
                        break
                
                email_chunk = EmailChunk(
                    chunk_id=f"chunk_{i}",
                    text=chunk_text,
                    element_type=element_type,
                    element_ids=element_ids,
                    quality_score=quality_score,
                    metadata={
                        "word_count": word_count,
                        "chunk_index": i,
                        "chunking_strategy": self.config.unstructured_strategy
                    }
                )
                
                email_chunks.append(email_chunk)
            
            return email_chunks
            
        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            # Fallback: create simple chunks from elements
            return self._fallback_chunking(elements)
    
    def _fallback_chunking(self, elements: List[EmailElement]) -> List[EmailChunk]:
        """Fallback chunking when Unstructured chunking fails"""
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
                    chunk_id=f"fallback_chunk_{chunk_index}",
                    text=current_text,
                    element_type=element.element_type,
                    element_ids=current_elements.copy(),
                    quality_score=0.7,  # Lower quality for fallback
                    metadata={
                        "word_count": word_count,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "fallback"
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
                chunk_id=f"fallback_chunk_{chunk_index}",
                text=current_text,
                element_type="NarrativeText",
                element_ids=current_elements,
                quality_score=0.7,
                metadata={
                    "word_count": len(current_text.split()),
                    "chunk_index": chunk_index,
                    "chunking_strategy": "fallback"
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
        """Process email content using enhanced cleaning + Unstructured + embeddings"""
        start_time = time.time()
        temp_file = None
        
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
            
            # Step 2: Create temporary email file with cleaned content
            temp_file = self._create_temp_email_file(cleaned_content, None)  # Use cleaned content only
            
            # Step 3: Extract elements using Unstructured
            elements = self._extract_elements(temp_file)
            self.logger.info(f"📋 Extracted {len(elements)} clean elements from email {email_id}")
            
            # Step 4: Convert elements to markdown
            markdown_elements = self._convert_elements_to_markdown(elements)
            self.logger.info(f"📝 Converted {len(markdown_elements)} elements to markdown")
            
            # Add markdown content to elements
            for element, markdown_content in markdown_elements:
                element.markdown_content = markdown_content
            
            # Step 5: Chunk elements
            chunks = self._chunk_elements(elements)
            self.logger.info(f"🧩 Created {len(chunks)} chunks from email {email_id}")
            
            # Step 6: Generate embeddings
            chunks_with_embeddings = await self._generate_embeddings(chunks)
            embeddings_created = sum(1 for c in chunks_with_embeddings if c.embedding is not None)
            self.logger.info(f"🎯 Generated {embeddings_created} embeddings for email {email_id}")
            
            # Store processed data for database storage
            self.last_processing_data = {
                'elements': elements,
                'chunks': chunks_with_embeddings,
                'markdown_elements': markdown_elements
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
                "markdown_elements_created": len(markdown_elements),
                "has_markdown": True
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
            
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_time = (self.total_processing_time / self.total_processed) if self.total_processed > 0 else 0
        
        return {
            "ready": self.ready,
            "unstructured_available": self.unstructured_available,
            "embedding_model_loaded": self.embedding_model is not None,
            "total_processed": self.total_processed,
            "average_processing_time_ms": avg_time * 1000,
            "emails_per_second": 1 / avg_time if avg_time > 0 else 0,
            "config": self.config.to_dict()
        }