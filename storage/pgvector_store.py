from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from typing import List, Dict, Any, Optional

class PGVectorStore:
    def __init__(self, connection_string: str, collection_name: str, embeddings):
        """
        Initialize the PGVector store with the given connection string and collection name.
        
        Args:
            connection_string: The PostgreSQL connection string.
            collection_name: The name of the collection to store documents in.
            embeddings: The embeddings object to use for embedding documents.
        """
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True,
        )
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: The documents to add.
            ids: Optional IDs for the documents.
            
        Returns:
            The IDs of the added documents.
        """
        return self.vector_store.add_documents(documents, ids=ids)
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: The IDs of the documents to delete.
        """
        self.vector_store.delete(ids=ids)
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The query to search for.
            k: The number of documents to return.
            filter: Optional filter to apply to the search.
            
        Returns:
            A list of documents similar to the query.
        """
        return self.vector_store.similarity_search(query, k=k, filter=filter)
    
    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        Search for documents similar to the query and return scores.
        
        Args:
            query: The query to search for.
            k: The number of documents to return.
            filter: Optional filter to apply to the search.
            
        Returns:
            A list of tuples containing documents and their similarity scores.
        """
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
    
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