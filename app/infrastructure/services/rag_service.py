import json
import time
import logging
from typing import List
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.domain import IRAGService, RAGContext
from app.infrastructure.config import Settings


logger = logging.getLogger(__name__)


class FAISSRAGService(IRAGService):
    """
    FAISS-based implementation of the RAG service for similar case retrieval.
    
    DESIGN RATIONALE:
    -----------------
    This service provides semantic similarity search for health records using:
    
    1. EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2
       - 384-dimensional embeddings
       - Fast inference (~10ms per embedding)
       - Good quality for short-to-medium text (our health descriptions)
       - Trade-off: Less accurate than larger models, but sufficient for MVP
    
    2. FAISS INDEX: IndexFlatL2 (exact L2/Euclidean search)
       - Why L2 over cosine? MiniLM embeddings are NOT normalized by default.
         Cosine similarity = dot product of normalized vectors.
         L2 distance is equivalent when comparing relative distances.
       - Why Flat over IVF? Dataset size (~1000 records) doesn't require
         approximate search. Flat gives exact results. For 100K+ records,
         consider IndexIVFFlat for faster approximate search.
    
    3. SIMILARITY SCORE CONVERSION:
       - L2 distance: 0 = identical, larger = more different
       - Convert to similarity: similarity = 1 / (1 + sqrt(distance))
       - Note: sqrt because FAISS returns SQUARED L2 distances
       - Range: (0, 1] where 1 = identical
    
    4. NORMALIZATION OPTION:
       - For cosine similarity, could normalize embeddings and use IndexFlatIP
       - Current approach: Use L2 with conversion, simpler and works well
    
    LIMITATIONS:
    - Exact search scales O(n) - for large datasets, use IndexIVFFlat
    - Memory: 384 dims * 4 bytes * n records = ~1.5KB per record
    - No persistence: Index rebuilt on startup (acceptable for ~1000 records)
    
    USAGE:
    - retrieve_similar(): Get matching text strings only
    - retrieve_similar_with_context(): Get full records with scores and timing
    """

    # Model configuration
    EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension

    def __init__(self, settings: Settings, data_path: Path | None = None):
        self._settings = settings
        self._model = SentenceTransformer(settings.embed_model)
        self._dimension = self.EMBEDDING_DIMENSION
        self._index = faiss.IndexFlatL2(self._dimension)
        self._texts: List[str] = []
        self._records: List[dict] = []  # Store full records for context
        
        if data_path and data_path.exists():
            self._load_data(data_path)

    def _load_data(self, data_path: Path) -> None:
        """Load data from JSON file and build index."""
        try:
            with open(data_path) as f:
                records = json.load(f)
                texts = [rec["free_text_details"] for rec in records]
                self._records = records
                self.add_to_index(texts)
                logger.info(f"Loaded {len(records)} records into RAG index from {data_path}")
        except Exception as e:
            logger.error(f"Failed to load RAG data: {e}")

    def _l2_distance_to_similarity(self, squared_l2_distance: float) -> float:
        """
        Convert FAISS squared L2 distance to a similarity score in range (0, 1].
        
        Math explanation:
        - FAISS IndexFlatL2 returns SQUARED Euclidean distances (not raw L2)
        - squared_l2 = sum((a_i - b_i)^2) for embedding vectors a, b
        - We want: 0 = identical → 1.0, far apart → 0.0
        
        Formula: similarity = 1 / (1 + sqrt(squared_l2))
        
        Why sqrt? FAISS stores squared distances for efficiency.
        Why 1/(1+d)? Monotonic decay, bounded (0, 1], simple.
        
        Alternative: exp(-d) would give faster decay for large distances.
        We use 1/(1+d) for interpretability.
        
        Args:
            squared_l2_distance: Squared L2 distance from FAISS search
        
        Returns:
            Similarity score in range (0, 1] where 1 = identical
        """
        # Handle edge case: negative distances (shouldn't happen, but defensive)
        if squared_l2_distance < 0:
            logger.warning(f"Negative L2 distance: {squared_l2_distance}, clamping to 0")
            squared_l2_distance = 0
        
        l2_distance = np.sqrt(squared_l2_distance)
        return float(1.0 / (1.0 + l2_distance))

    def retrieve_similar(self, text: str, top_k: int = 5) -> List[str]:
        """
        Retrieve similar text entries from the FAISS index.
        
        Args:
            text: Query text to find similar entries for
            top_k: Number of similar entries to retrieve
        
        Returns:
            List of similar text strings (free_text_details)
        """
        if self._index.ntotal == 0:
            return []
            
        embedding = self._model.encode([text]).astype('float32')
        distances, indices = self._index.search(embedding, min(top_k, self._index.ntotal))
        return [self._texts[i] for i in indices[0] if i < len(self._texts)]

    def retrieve_similar_with_context(self, text: str, top_k: int = 5) -> RAGContext:
        """
        Retrieve similar cases with full context and timing metrics.
        
        This is the main method for RAG-enhanced analysis. Returns full records
        (not just text) so the LLM can use predefined_health from similar cases
        to inform its decision.
        
        Args:
            text: Query text to find similar cases for
            top_k: Number of similar cases to retrieve (default: 5)
        
        Returns:
            RAGContext containing:
            - similar_cases: List of full record dicts
            - similarity_scores: Corresponding similarity scores (0-1)
            - retrieval_time_ms: Time taken for retrieval
        """
        start_time = time.time()
        
        if self._index.ntotal == 0:
            return RAGContext(
                similar_cases=[],
                similarity_scores=[],
                retrieval_time_ms=0.0
            )
        
        # Encode query text to embedding
        embedding = self._model.encode([text]).astype('float32')
        
        # Search FAISS index
        distances, indices = self._index.search(embedding, min(top_k, self._index.ntotal))
        
        similar_cases = []
        similarity_scores = []
        
        for idx, squared_dist in zip(indices[0], distances[0]):
            if idx < len(self._records):
                similar_cases.append(self._records[idx])
                similarity = self._l2_distance_to_similarity(squared_dist)
                similarity_scores.append(similarity)
        
        retrieval_time = (time.time() - start_time) * 1000
        
        logger.debug(
            f"RAG retrieval: {len(similar_cases)} cases in {retrieval_time:.2f}ms, "
            f"top similarity: {similarity_scores[0]:.3f}" if similarity_scores else ""
        )
        
        return RAGContext(
            similar_cases=similar_cases,
            similarity_scores=similarity_scores,
            retrieval_time_ms=retrieval_time
        )

    def add_to_index(self, texts: List[str]) -> None:
        """
        Add new texts to the FAISS index.
        
        Note: Embeddings are not normalized. Using L2 distance.
        For cosine similarity, normalize embeddings and use IndexFlatIP.
        
        Args:
            texts: List of text strings to add
        """
        if not texts:
            return
            
        self._texts.extend(texts)
        embeddings = self._model.encode(texts)
        vectors = np.array(embeddings).astype('float32')
        
        # Verify embedding dimension
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {vectors.shape[1]}"
            )
        
        self._index.add(vectors)
        logger.info(f"Added {len(texts)} texts to RAG index. Total: {self._index.ntotal}")

    def add_records_to_index(self, records: List[dict]) -> None:
        """Add full records to the index for context retrieval."""
        if not records:
            return
        
        texts = [r.get("free_text_details", "") for r in records if r.get("free_text_details")]
        self._records.extend(records)
        self.add_to_index(texts)

    def get_index_size(self) -> int:
        """Get the current size of the index."""
        return self._index.ntotal

    def clear_index(self) -> None:
        """Clear the index and reset."""
        self._index = faiss.IndexFlatL2(self._dimension)
        self._texts = []
        self._records = []
        logger.info("RAG index cleared")
