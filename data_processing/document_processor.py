from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document

from data_processing.mistral_processor import MistralProcessor
from utils.text_chunker import chunk_text

class DocumentProcessor:
    def __init__(self, mistral_processor: MistralProcessor, max_chunk_size: int = 8000, chunk_overlap: int = 200):
        """
        Initialize the document processor with a Mistral processor.
        
        Args:
            mistral_processor: The Mistral processor to use for OCR.
            max_chunk_size: Maximum size of text chunks.
            chunk_overlap: Overlap between chunks.
        """
        self.mistral_processor = mistral_processor
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pdf(self, pdf_path: str) -> str:
        """
        Process a PDF and return the extracted text.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            The extracted text.
        """
        self.mistral_processor.upload_pdf(pdf_path)
        ocr_response = self.mistral_processor.process_ocr()
        text = self.mistral_processor.extract_text_from_ocr(ocr_response)
        return text
    
    def create_documents(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Create documents from text, chunking as necessary.
        
        Args:
            text: The text to create documents from.
            metadata: Optional metadata to attach to the documents.
            
        Returns:
            A list of documents.
        """
        if metadata is None:
            metadata = {}
        
        chunks = chunk_text(text, self.max_chunk_size, self.chunk_overlap)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata["chunk_id"] = i
            doc_metadata["chunk_count"] = len(chunks)
            
            document = Document(
                page_content=chunk,
                metadata=doc_metadata,
            )
            documents.append(document)
        
        return documents