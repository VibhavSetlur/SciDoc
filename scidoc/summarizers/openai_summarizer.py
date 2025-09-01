"""
OpenAI summarizer for SciDoc.

This module provides summarization using OpenAI's API.
"""

import os
from typing import Any, Dict, Optional

from .base_summarizer import BaseSummarizer


class OpenAISummarizer(BaseSummarizer):
    """OpenAI-based summarizer using GPT models."""
    
    def __init__(self, **kwargs):
        """Initialize OpenAI summarizer."""
        super().__init__(**kwargs)
        
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = kwargs.get("model", "gpt-3.5-turbo")
        self.client = None
        
        if self.api_key:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            print("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
    
    def summarize(self, text: str, **kwargs) -> str:
        """
        Generate a summary using OpenAI API.
        
        Args:
            text: Text to summarize
            **kwargs: Additional arguments
            
        Returns:
            Generated summary as string
        """
        if not self._validate_input(text):
            return "No valid text to summarize."
        
        if not self.client:
            return "OpenAI client not available. Check API key configuration."
        
        # Clean and prepare text
        text = self._clean_text(text)
        
        # Use custom max_length if provided
        max_length = kwargs.get("max_length", self.max_length)
        
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(text, "summarize")
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=self.temperature,
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content.strip()
            else:
                return "Failed to generate summary."
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"Summarization failed: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if the OpenAI API is available."""
        return self.client is not None and self.api_key is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model."""
        info = super().get_model_info()
        info.update({
            "model": self.model,
            "api_available": self.is_available(),
        })
        return info
