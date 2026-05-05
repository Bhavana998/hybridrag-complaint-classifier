# core/hybrid_retriever.py - Complete Working Version
import numpy as np
import faiss
import pickle
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import logging
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    text: str
    score: float
    semantic_score: float
    keyword_score: float
    metadata: Dict = field(default_factory=dict)

class HybridRetriever:
    """
    Hybrid retriever combining semantic search (FAISS) and keyword search (TF-IDF)
    """
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the hybrid retriever
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.vector_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Keyword search components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        self.tfidf_matrix = None
        
        self.is_indexed = False
        
        # Load embedding model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"✅ Embedding model loaded. Dimension: {self.vector_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Index documents for both semantic and keyword search
        
        Args:
            documents: List of complaint texts
            metadata: Optional list of metadata dictionaries for each document
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        self.metadata = metadata if metadata else [{}] * len(documents)
        
        # 1. Semantic search with FAISS
        logger.info("Creating semantic embeddings...")
        try:
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
            embeddings = embeddings.astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.vector_dim)
            self.index.add(embeddings)
            logger.info(f"✅ FAISS index created with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to create semantic index: {str(e)}")
            raise
        
        # 2. Keyword search with TF-IDF
        logger.info("Building TF-IDF matrix...")
        try:
            # Preprocess documents for TF-IDF
            processed_docs = self._preprocess_for_tfidf(documents)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_docs)
            logger.info(f"✅ TF-IDF matrix created with shape {self.tfidf_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Failed to create TF-IDF matrix: {str(e)}")
            raise
        
        self.is_indexed = True
        logger.info(f"🎉 Indexing complete! {len(documents)} documents ready for search.")
    
    def _preprocess_for_tfidf(self, documents: List[str]) -> List[str]:
        """
        Simple preprocessing for TF-IDF
        
        Args:
            documents: List of document strings
        
        Returns:
            List of preprocessed strings
        """
        import re
        
        processed = []
        for doc in documents:
            # Convert to lowercase
            doc = doc.lower()
            # Remove special characters and digits
            doc = re.sub(r'[^a-zA-Z\s]', ' ', doc)
            # Remove extra whitespace
            doc = re.sub(r'\s+', ' ', doc).strip()
            processed.append(doc)
        
        return processed
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform semantic search only
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (index, score) tuples
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call index_documents first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.documents):
                # Convert cosine similarity from [-1,1] to [0,1]
                normalized_score = float((score + 1) / 2)
                results.append((int(idx), normalized_score))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform keyword search only
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (index, score) tuples
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call index_documents first.")
        
        # Preprocess query
        query_processed = self._preprocess_for_tfidf([query])[0]
        query_tfidf = self.tfidf_vectorizer.transform([query_processed])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[RetrievalResult]:
        """
        Hybrid search combining semantic and keyword retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for semantic search (0-1). Higher = more semantic.
        
        Returns:
            List of RetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call index_documents first.")
        
        # Get semantic results (get more for better combination)
        semantic_results = dict(self.semantic_search(query, top_k=top_k * 2))
        
        # Get keyword results
        keyword_results = dict(self.keyword_search(query, top_k=top_k * 2))
        
        # Combine scores
        all_indices = set(semantic_results.keys()) | set(keyword_results.keys())
        combined_scores = {}
        
        for idx in all_indices:
            sem_score = semantic_results.get(idx, 0)
            kw_score = keyword_results.get(idx, 0)
            
            # Hybrid combination
            combined = alpha * sem_score + (1 - alpha) * kw_score
            combined_scores[idx] = combined
        
        # Normalize scores
        if combined_scores:
            max_score = max(combined_scores.values())
            if max_score > 0:
                combined_scores = {k: v/max_score for k, v in combined_scores.items()}
        
        # Get top-k
        sorted_indices = sorted(combined_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:top_k]
        
        # Create results
        results = []
        for idx, score in sorted_indices:
            results.append(RetrievalResult(
                text=self.documents[idx],
                score=score,
                semantic_score=semantic_results.get(idx, 0),
                keyword_score=keyword_results.get(idx, 0),
                metadata=self.metadata[idx] if idx < len(self.metadata) else {}
            ))
        
        return results
    
    def search_by_category(self, category: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Search documents by category (uses metadata if available)
        
        Args:
            category: Category to filter by
            top_k: Number of results to return
        
        Returns:
            List of RetrievalResult objects
        """
        results = []
        for idx, meta in enumerate(self.metadata):
            if meta.get('category', '').lower() == category.lower():
                results.append(RetrievalResult(
                    text=self.documents[idx],
                    score=1.0,
                    semantic_score=1.0,
                    keyword_score=1.0,
                    metadata=meta
                ))
        
        return results[:top_k]
    
    def save_index(self, path_prefix: str = "hybrid_index"):
        """
        Save index to disk
        
        Args:
            path_prefix: Prefix for saved files
        """
        if not self.is_indexed:
            raise ValueError("No index to save")
        
        # Create directory if needed
        save_dir = Path("data/indexes")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = save_dir / f"{path_prefix}_faiss.index"
        faiss.write_index(self.index, str(faiss_path))
        logger.info(f"✅ FAISS index saved to {faiss_path}")
        
        # Save other components
        data_path = save_dir / f"{path_prefix}_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'embedding_model_name': self.embedding_model_name
            }, f)
        logger.info(f"✅ Data saved to {data_path}")
    
    def load_index(self, path_prefix: str = "hybrid_index"):
        """
        Load index from disk
        
        Args:
            path_prefix: Prefix of saved files
        """
        save_dir = Path("data/indexes")
        faiss_path = save_dir / f"{path_prefix}_faiss.index"
        data_path = save_dir / f"{path_prefix}_data.pkl"
        
        if not faiss_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"Index {path_prefix} not found in {save_dir}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(faiss_path))
        
        # Load other components
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.tfidf_vectorizer = data['tfidf_vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.embedding_model_name = data['embedding_model_name']
        
        # Reload embedding model
        self._load_embedding_model()
        
        self.is_indexed = True
        logger.info(f"✅ Index loaded from {faiss_path}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the index
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "is_indexed": self.is_indexed,
            "num_documents": len(self.documents),
            "vector_dimension": self.vector_dim,
            "embedding_model": self.embedding_model_name,
            "tfidf_features": self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            "faiss_vectors": self.index.ntotal if self.index is not None else 0
        }
    
    def get_document(self, index: int) -> Optional[Dict]:
        """
        Get document by index
        
        Args:
            index: Document index
        
        Returns:
            Dictionary with document text and metadata
        """
        if 0 <= index < len(self.documents):
            return {
                "text": self.documents[index],
                "metadata": self.metadata[index] if index < len(self.metadata) else {}
            }
        return None


# Test function
if __name__ == "__main__":
    print("="*50)
    print("Testing HybridRetriever")
    print("="*50)
    
    # Sample documents
    test_docs = [
        "I was charged twice for my subscription",
        "The app crashes every time I open it",
        "My package was supposed to arrive yesterday",
        "Customer support was very rude to me",
        "The product broke after 2 days of use"
    ]
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Index documents
    retriever.index_documents(test_docs)
    
    # Test search
    print("\n🔍 Testing search for: 'billing error double charge'")
    results = retriever.hybrid_search("billing error double charge", top_k=3)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Text: {result.text}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Semantic: {result.semantic_score:.3f}")
        print(f"  Keyword: {result.keyword_score:.3f}")
    
    # Print stats
    print("\n📊 Statistics:")
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ HybridRetriever test complete!")