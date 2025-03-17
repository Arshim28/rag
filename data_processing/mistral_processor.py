from pathlib import Path
from mistralai import Mistral
from typing import Optional, Dict, Any

class MistralProcessor:
    def __init__(self, api_key: str, ocr_model: str):
        """
        Initialize the Mistral Processor with the given API key and OCR model.
        
        Args:
            api_key: The Mistral API key.
            ocr_model: The OCR model to use.
        """
        self.client = Mistral(api_key=api_key)
        self.ocr_model = ocr_model
        self.file_id = None
        self.signed_url = None
    
    def upload_pdf(self, pdf_path: str) -> str:
        """
        Upload a PDF to Mistral.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            The file ID of the uploaded PDF.
        """
        with open(pdf_path, "rb") as f:
            uploaded_pdf = self.client.files.upload(
                file={
                    "file_name": Path(pdf_path).name,
                    "content": f,
                },
                purpose="ocr"
            )
        self.file_id = uploaded_pdf.id
        return self.file_id

    def get_signed_url(self) -> str:
        """
        Get a signed URL for the uploaded PDF.
        
        Returns:
            The signed URL.
        """
        self.client.files.retrieve(file_id=self.file_id)
        self.signed_url = self.client.files.get_signed_url(file_id=self.file_id)
        return self.signed_url.url

    def process_ocr(self) -> Dict[str, Any]:
        """
        Process OCR on the uploaded PDF.
        
        Returns:
            The OCR response.
        """
        url = self.get_signed_url()
        ocr_response = self.client.ocr.process(
            model=self.ocr_model,
            document={
                "type": "document_url",
                "document_url": url,
                "document_name": self.file_id
            },
            include_image_base64=False
        )
        return ocr_response
    
    def extract_text_from_ocr(self, ocr_response: Dict[str, Any]) -> str:
        """
        Extract text from OCR response.
        
        Args:
            ocr_response: The OCR response from Mistral.
            
        Returns:
            The extracted text.
        """
        text = ""
        for page in ocr_response.get("pages", []):
            # Each page contains markdown text that we can extract directly
            markdown_text = page.get("markdown", "")
            text += markdown_text + "\n\n"
        
        return text