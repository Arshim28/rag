from typing import List, Dict, Any, Tuple
from pathlib import Path
import os
import yaml

from data_processing.mistral_processor import MistralProcessor
from data_processing.document_processor import DocumentProcessor
from embedding.google_embeddings import GoogleEmbeddings
from storage.pgvector_store import PGVectorStore
from storage.redis_cache import RedisCache
from retrieval.gemini_flash import GeminiFlash

class RAGPipeline:
    def __init__(self, config_path: str = None):
        """
        Initialize the RAG pipeline with the given configuration path.
        
        Args:
            config_path: Path to the configuration file.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Override with environment variables if present
        if os.getenv("MISTRAL_API_KEY"):
            self.config["mistral"]["api_key"] = os.getenv("MISTRAL_API_KEY")
        
        if os.getenv("GOOGLE_API_KEY"):
            self.config["google"]["api_key"] = os.getenv("GOOGLE_API_KEY")
        
        if os.getenv("DB_CONNECTION_STRING"):
            self.config["database"]["connection_string"] = os.getenv("DB_CONNECTION_STRING")
        
        if os.getenv("REDIS_URL"):
            self.config["redis"]["url"] = os.getenv("REDIS_URL")
        
        mistral_config = self.config["mistral"]
        google_config = self.config["google"]
        db_config = self.config["database"]
        redis_config = self.config["redis"]
        chunking_config = self.config["chunking"]
        
        self.mistral_processor = MistralProcessor(
            api_key=mistral_config["api_key"],
            ocr_model=mistral_config["ocr_model"],
        )
        
        self.document_processor = DocumentProcessor(
            mistral_processor=self.mistral_processor,
            max_chunk_size=chunking_config["max_chunk_size"],
            chunk_overlap=chunking_config["chunk_overlap"],
        )
        
        self.embeddings = GoogleEmbeddings(
            api_key=google_config["api_key"],
            model_name=google_config["embedding_model"],
        )
        
        self.vector_store = PGVectorStore(
            connection_string=db_config["connection_string"],
            collection_name=db_config["collection_name"],
            embeddings=self.embeddings,
        )
        
        self.cache = RedisCache(
            redis_url=redis_config["url"],
            index_name=redis_config["index_name"],
            embeddings=self.embeddings,
        )
        
        self.llm = GeminiFlash(
            api_key=google_config["api_key"],
            model_name=google_config["generation_model"],
        )
    
    def process_document(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[str]:
        """
        Process a document and store it in the vector store and cache.
        
        Args:
            pdf_path: Path to the PDF file.
            metadata: Optional metadata to attach to the documents.
            
        Returns:
            The IDs of the added documents.
        """
        if metadata is None:
            metadata = {"source": Path(pdf_path).name}
        
        text = self.document_processor.process_pdf(pdf_path)
        documents = self.document_processor.create_documents(text, metadata)
        
        doc_ids = self.vector_store.add_documents(documents)
        
        texts = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]
        self.cache.add_texts(texts, metadata_list)
        
        return doc_ids
    
    def query(self, question: str, k: int = 4) -> str:
        """
        Query the RAG pipeline.
        
        Args:
            question: The question to ask.
            k: The number of documents to retrieve.
            
        Returns:
            The answer to the question.
        """
        documents = self.cache.similarity_search(question, k=k)
        
        if not documents:
            documents = self.vector_store.similarity_search(question, k=k)
        
        context = "\n\n".join([doc.page_content for doc in documents])
        
        messages = [
            ("system", f"You are a helpful assistant that answers questions based on the provided context. Context: {context}"),
            ("human", question),
        ]
        answer = self.llm.generate(messages)
        
        return answer