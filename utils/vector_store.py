import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from typing import Dict, List
import os

logger = logging.getLogger(__name__)

class VectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        try:
            # Persistent storage path in your project folder
            save_path = os.path.join(os.getcwd(), "academic_db")
            
            # Initialize ChromaDB Client
            self.client = chromadb.PersistentClient(path=save_path)
            
            # Initialize Embedding Model
            self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Create/Get collections
            self.papers = self.client.get_or_create_collection(
                name="academic_papers",
                metadata={"hnsw:space": "cosine"}
            )
            self.citations = self.client.get_or_create_collection(
                name="citations",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector Store initialized successfully.")
        except Exception as e:
            logger.error(f"Vector Store initialization failed: {str(e)}")
            self.client = None
            self.encoder = None

    def query_similar_papers(self, text: str, n_results: int = 5) -> Dict:
        """Find semantically similar papers"""
        if not self.client or not self.encoder:
            return {}
            
        try:
            embedding = self.encoder.encode(text).tolist()
            return self.papers.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {}

    def add_paper(self, paper_id: str, text: str, metadata: Dict):
        """Add a paper to the vector database"""
        if not self.client or not self.encoder:
            return
            
        try:
            embedding = self.encoder.encode(text).tolist()
            self.papers.add(
                ids=[str(paper_id)],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            # Also add to citations collection for quick lookup
            self.citations.add(
                ids=[str(paper_id)],
                documents=[metadata.get('title', 'Unknown')],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Add paper failed: {str(e)}")

    def check_citation_exists(self, title: str) -> bool:
        """
        Check if a citation (by title) exists in the database.
        Returns True if a close match is found.
        """
        if not self.client or not self.encoder:
            return False
            
        try:
            # Query the citations collection
            embedding = self.encoder.encode(title).tolist()
            results = self.citations.query(
                query_embeddings=[embedding],
                n_results=1
            )
            
            # Check if we got results and distance is low (high similarity)
            # Distance < 0.2 usually implies a very strong match or exact match
            if results and results.get('distances') and len(results['distances'][0]) > 0:
                distance = results['distances'][0][0]
                return distance < 0.2
                
            return False
            
        except Exception as e:
            logger.error(f"Citation existence check failed: {str(e)}")
            return False