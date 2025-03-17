from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional

class RedisCache:
    def __init__(self, redis_url: str, index_name: str, embeddings):
        """
        Initialize the Redis cache with the given URL and index name.
        
        Args:
            redis_url: The Redis URL.
            index_name: The name of the index to store documents in.
            embeddings: The embeddings object to use for embedding documents.
        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.embeddings = embeddings
        
        self.config = RedisConfig(
            index_name=index_name,
            redis_url=redis_url,
            metadata_schema=[
                {"name": "category", "type": "tag"},
                {"name": "source", "type": "tag"},
                {"name": "chunk_id", "type": "numeric"},
                {"name": "chunk_count", "type": "numeric"},
            ],
        )
        
        self.vector_store = RedisVectorStore(embeddings, config=self.config)
    
    def add_texts(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add texts to the cache.
        
        Args:
            texts: The texts to add.
            metadata: Optional metadata for the texts.
            
        Returns:
            The IDs of the added texts.
        """
        return self.vector_store.add_texts(texts, metadata)
    
    def delete_keys(self, keys: List[str]) -> int:
        """
        Delete keys from the cache.
        
        Args:
            keys: The keys to delete.
            
        Returns:
            The number of keys deleted.
        """
        return self.vector_store.index.drop_keys(keys)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The query to search for.
            k: The number of documents to return.
            
        Returns:
            A list of documents similar to the query.
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Search for documents similar to the query and return scores.
        
        Args:
            query: The query to search for.
            k: The number of documents to return.
            
        Returns:
            A list of tuples containing documents and their similarity scores.
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Create a retriever from the vector store.
        
        Args:
            search_type: The type of search to perform.
            search_kwargs: Optional arguments for the search.
            
        Returns:
            A retriever object.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)