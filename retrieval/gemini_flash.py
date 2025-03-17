from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Tuple

class GeminiFlash:
    def __init__(self, api_key: str, model_name: str, temperature: float = 0):
        """
        Initialize the Gemini Flash model with the given API key and model name.
        
        Args:
            api_key: The Google API key.
            model_name: The model name to use.
            temperature: The temperature to use for generation.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    
    def generate(self, messages: List[Tuple[str, str]]) -> str:
        """
        Generate a response from the Gemini Flash model.
        
        Args:
            messages: A list of tuples containing the role and content of each message.
            
        Returns:
            The generated response.
        """
        return self.llm.invoke(messages)