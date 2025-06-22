"""
Embedding Service for Email RAG
Optimized for lightweight operation alongside Qwen-0.5B
"""

import asyncio
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from config import ProcessorConfig

class EmbeddingService:
    """
    Lightweight embedding service optimized for Mac mini M2
    Uses sentence-transformers with MiniLM for efficient embeddings
    """
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model: Optional[SentenceTransformer] = None
        self.model_loaded = False
        
        # Initialize model in background
        asyncio.create_task(self._load_model())
    
    async def _load_model(self):
        """Load the embedding model"""
        try:
            self.logger.info(f"📥 Loading embedding model: {self.config.embedding_model}")
            
            # Use CPU to save GPU memory for Qwen-0.5B
            device = "cpu"  # Keep GPU free for Qwen
            
            self.model = SentenceTransformer(self.config.embedding_model, device=device)
            
            # Test the model with a simple embedding
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.logger.info(f"✅ Embedding model loaded - dimension: {len(test_embedding)}")
            
            self.model_loaded = True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {e}")
            self.model_loaded = False
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.model_loaded or not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Clean and prepare text
            text = self._prepare_text(text)
            
            # Generate embedding
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(text, convert_to_numpy=True)
            )
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            raise
    
    async def generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts efficiently"""
        if not self.model_loaded or not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Clean and prepare texts
            prepared_texts = [self._prepare_text(text) for text in texts]
            
            # Generate embeddings in batch
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(prepared_texts, convert_to_numpy=True, batch_size=8)
            )
            
            return [emb for emb in embeddings]
            
        except Exception as e:
            self.logger.error(f"❌ Batch embedding generation failed: {e}")
            raise
    
    def _prepare_text(self, text: str) -> str:
        """Prepare text for embedding generation"""
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (MiniLM has 512 token limit)
        max_chars = 2000  # Conservative limit
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text
    
    async def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"❌ Similarity computation failed: {e}")
            return 0.0
    
    async def find_similar_chunks(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        threshold: float = 0.7
    ) -> List[tuple]:
        """Find similar chunks based on embedding similarity"""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.compute_similarity(query_embedding, candidate)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model_loaded or not self.model:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.config.embedding_model,
            "dimension": self.config.vector_dimension,
            "device": str(self.model.device),
            "max_seq_length": getattr(self.model, 'max_seq_length', 512)
        }
    
    async def health_check(self) -> bool:
        """Check if embedding service is healthy"""
        if not self.model_loaded or not self.model:
            return False
        
        try:
            # Test embedding generation
            test_embedding = await self.generate_embedding("health check test")
            return len(test_embedding) == self.config.vector_dimension
        except Exception:
            return False