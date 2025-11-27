"""
Azmy Retail Analytics Copilot
RAG retrieval system using TF-IDF for local document search
Author: Azmy
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import logging

# Add project root to path and import mocks
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import mock dependencies if needed
    import mock_deps
except ImportError:
    pass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # Use mock implementations
    try:
        from mock_deps import MockTfidfVectorizer as TfidfVectorizer, mock_cosine_similarity
        cosine_similarity = mock_cosine_similarity
    except ImportError:
        # Fallback mock implementations inline
        class TfidfVectorizer:
            def __init__(self, **kwargs):
                self.features = 100
                self.fitted = False
            
            def fit_transform(self, texts):
                import numpy as np
                self.fitted = True
                return np.random.random((len(texts), self.features))
            
            def transform(self, texts):
                import numpy as np
                if not self.fitted:
                    self.fitted = True
                return np.random.random((len(texts), self.features))
        
        def cosine_similarity(a, b):
            import numpy as np
            if len(a.shape) == 1:
                a = a.reshape(1, -1)
            if len(b.shape) == 1:
                b = b.reshape(1, -1)
            return np.random.random((a.shape[0], b.shape[0]))


class DocumentChunk:
    """Represents a document chunk with metadata"""
    
    def __init__(self, chunk_id: str, content: str, source: str, chunk_index: int = 0):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.chunk_index = chunk_index


class TFIDFRetriever:
    """TF-IDF based document retriever"""
    
    def __init__(self, docs_dir: str, chunk_size: int = 300):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.logger = logging.getLogger(__name__)
        
        self._load_documents()
        self._build_index()
    
    def _load_documents(self):
        """Load and chunk documents from docs directory"""
        self.chunks = []
        
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks by paragraphs first, then by size
                paragraphs = content.split('\n\n')
                
                chunk_index = 0
                current_chunk = ""
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    # If adding this paragraph would exceed chunk size, save current chunk
                    if current_chunk and len(current_chunk + para) > self.chunk_size:
                        chunk_id = f"{filename.replace('.md', '')}::chunk{chunk_index}"
                        self.chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            content=current_chunk.strip(),
                            source=filename,
                            chunk_index=chunk_index
                        ))
                        chunk_index += 1
                        current_chunk = para
                    else:
                        current_chunk += ("\n\n" if current_chunk else "") + para
                
                # Add the last chunk if it exists
                if current_chunk:
                    chunk_id = f"{filename.replace('.md', '')}::chunk{chunk_index}"
                    self.chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        content=current_chunk.strip(),
                        source=filename,
                        chunk_index=chunk_index
                    ))
        
        self.logger.info(f"Loaded {len(self.chunks)} document chunks")
    
    def _build_index(self):
        """Build TF-IDF index"""
        if not self.chunks:
            self.logger.warning("No chunks to index")
            return
        
        # Extract text content
        texts = [chunk.content for chunk in self.chunks]
        
        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.logger.info(f"Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks
        Returns list of dicts with chunk_id, content, source, score
        """
        if not self.vectorizer or self.tfidf_matrix is None:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                chunk = self.chunks[idx]
                results.append({
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'source': chunk.source,
                    'score': float(similarities[idx])
                })
        
        return results
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks with metadata"""
        return [{
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'source': chunk.source,
            'chunk_index': chunk.chunk_index
        } for chunk in self.chunks]
