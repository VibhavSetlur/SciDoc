"""
AI summarizers for SciDoc.

This package contains summarizer implementations that use different
AI models to generate summaries of file content and changes.
"""

from typing import Dict, List, Type

from .base_summarizer import BaseSummarizer
from .huggingface_summarizer import HuggingFaceSummarizer, FileAnalysis
from .openai_summarizer import OpenAISummarizer

# Registry of available summarizers
SUMMARIZER_REGISTRY: Dict[str, Type[BaseSummarizer]] = {
    "huggingface": HuggingFaceSummarizer,
    "openai": OpenAISummarizer,
}


def get_summarizer(summarizer_type: str) -> Type[BaseSummarizer]:
    """Get summarizer class for a type."""
    return SUMMARIZER_REGISTRY.get(summarizer_type.lower(), HuggingFaceSummarizer)


def get_available_summarizers() -> List[str]:
    """Get list of available summarizer names."""
    return list(SUMMARIZER_REGISTRY.keys())


def register_summarizer(name: str, summarizer_class: Type[BaseSummarizer]) -> None:
    """Register a new summarizer."""
    SUMMARIZER_REGISTRY[name.lower()] = summarizer_class


def create_summarizer(summarizer_type: str, **kwargs) -> BaseSummarizer:
    """Create a summarizer instance for a type."""
    summarizer_class = get_summarizer(summarizer_type)
    return summarizer_class(**kwargs)


__all__ = [
    "BaseSummarizer",
    "HuggingFaceSummarizer",
    "OpenAISummarizer",
    "FileAnalysis",
    "SUMMARIZER_REGISTRY",
    "get_summarizer",
    "get_available_summarizers",
    "register_summarizer",
    "create_summarizer",
]
