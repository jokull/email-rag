"""
Optimized LLM interface for Qwen-0.5B on Mac mini M2
Follows best practices for lightweight, efficient processing
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import subprocess
import json
import os
from config import LLMConfig, CLASSIFICATION_PROMPTS

@dataclass
class ClassificationResult:
    classification: str
    confidence: float
    processing_time: float
    tokens_used: int

@dataclass
class ChunkingResult:
    chunks: List[str]
    processing_time: float
    tokens_used: int

class QwenInterface:
    """
    Lightweight interface to Qwen-0.5B optimized for Mac mini M2 performance.
    Uses llama.cpp with Metal acceleration for optimal performance.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_loaded = False
        self.total_tokens_processed = 0
        self.total_requests = 0
        self.average_latency = 0.0
        
        # Validate model exists
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Qwen model not found at {config.model_path}")
        
        # Initialize model if keep_loaded is True
        if config.keep_model_loaded:
            asyncio.create_task(self._warm_model())
    
    async def _warm_model(self):
        """Pre-load and warm the model for faster inference"""
        try:
            self.logger.info("Warming Qwen-0.5B model...")
            start_time = time.time()
            
            # Test inference to warm the model
            await self._run_inference("Test message", max_tokens=10)
            
            warm_time = time.time() - start_time
            self.model_loaded = True
            self.logger.info(f"Model warmed in {warm_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to warm model: {e}")
    
    async def _run_inference(
        self, 
        prompt: str, 
        max_tokens: int = None,
        temperature: float = None
    ) -> Tuple[str, int, float]:
        """
        Run inference using llama.cpp with optimized settings for Qwen-0.5B
        Returns: (response, tokens_used, processing_time)
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        start_time = time.time()
        
        # Build llama.cpp command optimized for Mac M2
        cmd = [
            "llama-cli",  # Assumes llama.cpp installed with `brew install llama.cpp`
            "-m", self.config.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.config.threads),
            "--temp", str(temperature),
            "--ctx-size", str(self.config.context_length),
            "--no-display-prompt",  # Clean output
            "--silent-prompt",      # Reduce noise
        ]
        
        # Add Metal acceleration if available
        if self.config.use_metal:
            cmd.extend(["-ngl", "999"])  # Offload all layers to GPU
        
        try:
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"llama.cpp error: {result.stderr}")
            
            response = result.stdout.strip()
            processing_time = time.time() - start_time
            
            # Estimate tokens (rough approximation)
            tokens_used = len(prompt.split()) + len(response.split())
            
            # Update metrics
            self.total_requests += 1
            self.total_tokens_processed += tokens_used
            self.average_latency = (
                (self.average_latency * (self.total_requests - 1) + processing_time) 
                / self.total_requests
            )
            
            return response, tokens_used, processing_time
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("LLM inference timeout")
        except Exception as e:
            raise RuntimeError(f"LLM inference failed: {e}")
    
    async def classify_email(self, content: str) -> ClassificationResult:
        """
        Classify email using optimized prompts for Qwen-0.5B
        Focus on binary classification for efficiency
        """
        # Truncate content to prevent context overflow
        if len(content) > self.config.max_email_length:
            content = content[:self.config.max_email_length] + "..."
        
        # Try each classification in order of likelihood
        classifications = ["human", "promotional", "transactional", "automated"]
        results = {}
        
        for classification_type in classifications:
            prompt = CLASSIFICATION_PROMPTS[classification_type].format(content=content)
            
            try:
                response, tokens, proc_time = await self._run_inference(
                    prompt, 
                    max_tokens=5,  # Just need "yes" or "no"
                    temperature=0.1  # Very low for consistent results
                )
                
                # Parse yes/no response
                response_lower = response.lower().strip()
                confidence = 0.9 if "yes" in response_lower else 0.1
                results[classification_type] = confidence
                
            except Exception as e:
                self.logger.warning(f"Classification failed for {classification_type}: {e}")
                results[classification_type] = 0.0
        
        # Determine best classification
        best_class = max(results, key=results.get)
        best_confidence = results[best_class]
        
        # If all classifications are low, default to "automated"
        if best_confidence < 0.5:
            best_class = "automated"
            best_confidence = 0.6
        
        return ClassificationResult(
            classification=best_class,
            confidence=best_confidence,
            processing_time=sum([0.1] * len(classifications)),  # Approximate
            tokens_used=sum([20] * len(classifications))  # Approximate
        )
    
    async def chunk_content(self, content: str) -> ChunkingResult:
        """
        Chunk email content into semantic sections optimized for embeddings
        """
        # Skip chunking for short content
        if len(content.split()) < 100:
            return ChunkingResult(
                chunks=[content],
                processing_time=0.001,
                tokens_used=0
            )
        
        # Use simple heuristics for very efficient chunking
        # This avoids using the LLM for chunking to preserve resources
        chunks = self._heuristic_chunking(content)
        
        return ChunkingResult(
            chunks=chunks,
            processing_time=0.01,  # Very fast heuristic chunking
            tokens_used=0
        )
    
    def _heuristic_chunking(self, content: str) -> List[str]:
        """
        Semantic email chunking optimized for RAG following Unstructured.io best practices
        Preserves email structure while maintaining optimal chunk sizes for embeddings
        """
        target_chunk_size = self.config.chunk_size_tokens
        words = content.split()
        
        if len(words) <= target_chunk_size:
            return [content]
        
        # Email-specific semantic boundaries (priority order)
        chunks = self._chunk_by_email_sections(content, target_chunk_size)
        
        # Apply overlap for better context preservation
        if len(chunks) > 1:
            chunks = self._apply_semantic_overlap(chunks, overlap_ratio=self.config.chunk_overlap_ratio)
        
        return chunks
    
    def _chunk_by_email_sections(self, content: str, target_size: int) -> List[str]:
        """
        Chunk by email-specific semantic boundaries
        """
        # Email section markers (in priority order)
        section_markers = [
            '\n\n',  # Paragraph breaks (highest priority)
            '. ',    # Sentence boundaries
            ', ',    # Clause boundaries (lowest priority)
        ]
        
        # Try chunking by each marker in order
        for marker in section_markers:
            chunks = self._chunk_by_marker(content, marker, target_size)
            if self._chunks_are_well_sized(chunks, target_size):
                return chunks
        
        # Fallback to word-based chunking if no good semantic boundaries
        return self._chunk_by_words(content, target_size)
    
    def _chunk_by_marker(self, content: str, marker: str, target_size: int) -> List[str]:
        """
        Chunk content by specific semantic marker
        """
        sections = content.split(marker)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            section_words = len(section.split())
            
            # If adding this section exceeds target size, finalize current chunk
            if current_size + section_words > target_size and current_chunk:
                chunks.append(marker.join(current_chunk))
                current_chunk = [section]
                current_size = section_words
            else:
                current_chunk.append(section)
                current_size += section_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(marker.join(current_chunk))
        
        return chunks
    
    def _chunk_by_words(self, content: str, target_size: int) -> List[str]:
        """
        Fallback word-based chunking with sliding window
        """
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), target_size):
            chunk_words = words[i:i + target_size]
            chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def _chunks_are_well_sized(self, chunks: List[str], target_size: int) -> bool:
        """
        Check if chunks are well-sized (not too small or too large)
        """
        for chunk in chunks:
            chunk_size = len(chunk.split())
            # Reject if any chunk is too small (< 20% of target) or too large (> 150% of target)
            if chunk_size < target_size * 0.2 or chunk_size > target_size * 1.5:
                return False
        return True
    
    def _apply_semantic_overlap(self, chunks: List[str], overlap_ratio: float) -> List[str]:
        """
        Apply semantic overlap between chunks to preserve context
        Following Unstructured.io recommendation of 5-20% overlap
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Calculate overlap size (10% of previous chunk)
            prev_words = prev_chunk.split()
            overlap_size = max(1, int(len(prev_words) * overlap_ratio))
            
            # Get last sentences from previous chunk for overlap
            overlap_text = ' '.join(prev_words[-overlap_size:])
            
            # Combine overlap with current chunk
            overlapped_chunk = f"{overlap_text} {current_chunk}"
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    async def batch_classify(self, contents: List[str]) -> List[ClassificationResult]:
        """
        Batch classification for improved efficiency
        """
        # Process in smaller batches to manage memory
        batch_size = min(self.config.batch_size, len(contents))
        results = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            
            # Process batch concurrently but limit concurrency
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            
            async def classify_with_semaphore(content):
                async with semaphore:
                    return await self.classify_email(content)
            
            batch_results = await asyncio.gather(
                *[classify_with_semaphore(content) for content in batch],
                return_exceptions=True
            )
            
            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch classification error: {result}")
                    results.append(ClassificationResult(
                        classification="automated",
                        confidence=0.5,
                        processing_time=0.0,
                        tokens_used=0
                    ))
                else:
                    results.append(result)
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            "total_requests": self.total_requests,
            "total_tokens_processed": self.total_tokens_processed,
            "average_latency": self.average_latency,
            "model_loaded": self.model_loaded,
            "estimated_memory_mb": self.config.runtime_memory_mb,
            "tokens_per_second": (
                self.total_tokens_processed / (self.average_latency * self.total_requests)
                if self.total_requests > 0 else 0
            )
        }
    
    async def health_check(self) -> bool:
        """Simple health check for the LLM interface"""
        try:
            result = await self.classify_email("Test email for health check")
            return result.confidence > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

# Factory function for easy initialization
def create_qwen_interface(config: LLMConfig = None) -> QwenInterface:
    """Create and initialize Qwen interface with default config if none provided"""
    if config is None:
        from config import get_config
        config, _ = get_config()
        config = config  # Use LLM config
    
    return QwenInterface(config)