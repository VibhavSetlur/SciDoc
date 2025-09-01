"""
Base summarizer class for SciDoc.

This module defines the abstract base class that all summarizer
implementations must implement to be compatible with SciDoc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseSummarizer(ABC):
    """Abstract base class for summarizers."""
    
    def __init__(self, **kwargs):
        """Initialize summarizer with configuration."""
        self.config = kwargs
        self.max_length = kwargs.get("max_length", 200)
        self.temperature = kwargs.get("temperature", 0.7)
        self.model_path = kwargs.get("model_path", None)
    
    @abstractmethod
    def summarize(self, text: str, **kwargs) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            **kwargs: Additional arguments for summarization
            
        Returns:
            Generated summary as string
        """
        pass
    
    def summarize_file_content(self, content: str, file_type: str = "unknown", **kwargs) -> str:
        """
        Summarize file content with context about file type.
        
        Args:
            content: File content to summarize
            file_type: Type of file being summarized
            **kwargs: Additional arguments
            
        Returns:
            Generated summary as string
        """
        # Add file type context to the text
        context = f"File type: {file_type}\n\nContent:\n{content}"
        return self.summarize(context, **kwargs)
    
    def summarize_changes(self, changes: List[Dict[str, Any]], **kwargs) -> str:
        """
        Summarize a list of file changes.
        
        Args:
            changes: List of change dictionaries
            **kwargs: Additional arguments
            
        Returns:
            Generated summary as string
        """
        # Format changes for summarization
        change_text = "File changes:\n"
        for change in changes:
            change_type = change.get("change_type", "unknown")
            filename = change.get("filename", "unknown")
            change_text += f"- {change_type}: {filename}\n"
        
        return self.summarize(change_text, **kwargs)
    
    def summarize_project(self, project_info: Dict[str, Any], **kwargs) -> str:
        """
        Summarize project information.
        
        Args:
            project_info: Project information dictionary
            **kwargs: Additional arguments
            
        Returns:
            Generated summary as string
        """
        # Format project info for summarization
        project_text = f"Project: {project_info.get('name', 'Unknown')}\n"
        project_text += f"Description: {project_info.get('description', 'No description')}\n"
        project_text += f"Total files: {project_info.get('total_files', 0)}\n"
        project_text += f"Total size: {project_info.get('total_size', 0)} bytes\n"
        
        # Add file type breakdown
        file_types = project_info.get('file_types', {})
        if file_types:
            project_text += "\nFile types:\n"
            for file_type, count in file_types.items():
                project_text += f"- {file_type}: {count} files\n"
        
        return self.summarize(project_text, **kwargs)
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[str]:
        """
        Summarize multiple texts in batch.
        
        Args:
            texts: List of texts to summarize
            **kwargs: Additional arguments
            
        Returns:
            List of generated summaries
        """
        summaries = []
        for text in texts:
            try:
                summary = self.summarize(text, **kwargs)
                summaries.append(summary)
            except Exception as e:
                # If summarization fails for one text, add error message
                summaries.append(f"Summarization failed: {str(e)}")
        
        return summaries
    
    def _truncate_text(self, text: str, max_tokens: int = 1000) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        # Simple word-based truncation (can be improved with proper tokenization)
        words = text.split()
        if len(words) <= max_tokens:
            return text
        
        # Keep beginning and end, truncate middle
        half_tokens = max_tokens // 2
        beginning = " ".join(words[:half_tokens])
        end = " ".join(words[-half_tokens:])
        
        return f"{beginning} ... [truncated] ... {end}"
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for summarization.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _validate_input(self, text: str) -> bool:
        """
        Validate input text for summarization.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text is valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        if len(text.strip()) == 0:
            return False
        
        # Check for minimum content
        if len(text.strip()) < 10:
            return False
        
        return True
    
    def _prepare_prompt(self, text: str, task: str = "summarize") -> str:
        """
        Prepare text with task-specific prompt.
        
        Args:
            text: Text to summarize
            task: Type of summarization task
            
        Returns:
            Formatted prompt
        """
        prompts = {
            "summarize": f"Please provide a concise summary of the following text:\n\n{text}",
            "analyze": f"Please analyze the following content and provide key insights:\n\n{text}",
            "extract": f"Please extract the main points from the following text:\n\n{text}",
            "describe": f"Please describe what this content is about:\n\n{text}",
        }
        
        return prompts.get(task, prompts["summarize"])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the summarizer model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "type": self.__class__.__name__,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "model_path": self.model_path,
        }
    
    def is_available(self) -> bool:
        """
        Check if the summarizer is available and ready to use.
        
        Returns:
            True if available, False otherwise
        """
        try:
            # Try to perform a simple test summarization
            test_text = "This is a test text for summarization."
            self.summarize(test_text)
            return True
        except Exception:
            return False
