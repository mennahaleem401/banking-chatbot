"""
Vector Store Management for Banking Chatbot
Handles ChromaDB vector store initialization and operations
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os


class BankingVectorStore:
    """Manages vector store operations for banking data"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist vector store
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedder = None
        
    def initialize_embedder(self):
        """Initialize sentence transformer embedder"""
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedder initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing embedder: {e}")
            raise
    
    def initialize_client(self, force_recreate: bool = False):
        """Initialize ChromaDB client"""
        try:
            if force_recreate and os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            print("✅ ChromaDB client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB client: {e}")
            raise
    
    def create_collection(self, collection_name: str = "banking_documents"):
        """Create or get collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Banking documents and user data"}
            )
            print("✅ Collection created/retrieved successfully")
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        if not self.embedder:
            self.initialize_embedder()
            
        try:
            contents = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Generate embeddings
            embeddings = self.embedder.encode(contents).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added {len(documents)} documents to vector store")
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise
    
    def search_similar(self, 
                      query: str, 
                      user_id: Optional[int] = None,
                      n_results: int = 5,
                      filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            user_id: Filter by user ID
            n_results: Number of results to return
            filters: Additional filters
            
        Returns:
            List of similar documents with metadata
        """
        if not self.embedder:
            self.initialize_embedder()
            
        try:
            # Prepare where clause
            where_clause = {}
            if user_id is not None:
                where_clause["user_id"] = user_id
            if filters:
                where_clause.update(filters)
            
            # Generate query embedding
            query_embedding = self.embedder.encode([query]).tolist()[0]
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching vector store: {e}")
            return []
    
    def get_user_documents(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all documents for a specific user"""
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                limit=limit,
                include=["metadatas", "documents"]
            )
            
            formatted_results = []
            for i, doc in enumerate(results['documents']):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error getting user documents: {e}")
            return []


def initialize_vector_store(documents: List[Dict[str, Any]],
                          persist_directory: str = "./chroma_db",
                          collection_name: str = "banking_documents",
                          force_recreate: bool = False) -> BankingVectorStore:
    """
    Initialize and populate vector store
    
    Args:
        documents: Documents to add to vector store
        persist_directory: Directory for persistence
        collection_name: Name of the collection
        force_recreate: Whether to recreate the vector store
        
    Returns:
        Initialized BankingVectorStore instance
    """
    print("🔄 Initializing vector store...")
    
    vector_store = BankingVectorStore(persist_directory=persist_directory)
    
    # Initialize components
    vector_store.initialize_embedder()
    vector_store.initialize_client(force_recreate=force_recreate)
    vector_store.create_collection(collection_name=collection_name)
    
    # Add documents if provided
    if documents:
        vector_store.add_documents(documents)
    
    print("✅ Vector store initialized successfully")
    return vector_store


if __name__ == "__main__":
    # Test the vector store
    from data_processor import create_sample_data, BankingDataProcessor
    
    # Create sample data
    create_sample_data()
    
    # Process data
    processor = BankingDataProcessor(
        accounts_path="uploads/accounts.csv",
        transactions_path="uploads/transactions.csv", 
        users_path="uploads/users.csv",
        user_financials_path="uploads/user_financials.csv"
    )
    
    documents = processor.process_all()
    
    # Initialize vector store
    vector_store = initialize_vector_store(documents, force_recreate=True)
    
    # Test search
    results = vector_store.search_similar("account balance", user_id=1)
    print(f"Found {len(results)} results")