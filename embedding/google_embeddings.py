from google import genai
from typing import List, Dict, Any

class GoogleEmbeddings:
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the Google Embeddings client with the given API key and model name.
        
        Args:
            api_key: The Google API key.
            model_name: The embedding model name.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector.
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return result.embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: The texts to embed.
            
        Returns:
            A list of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings