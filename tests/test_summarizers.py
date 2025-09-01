"""
Tests for SciDoc summarizers.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from scidoc.summarizers import (
    get_summarizer, get_available_summarizers, create_summarizer,
    HuggingFaceSummarizer, OpenAISummarizer
)
from scidoc.models import FileMetadata


class TestSummarizerRegistry:
    """Test summarizer registry functionality."""
    
    def test_get_available_summarizers(self):
        """Test that we can get available summarizers."""
        summarizers = get_available_summarizers()
        assert isinstance(summarizers, dict)
        assert len(summarizers) > 0
        assert "huggingface" in summarizers
    
    def test_get_summarizer(self):
        """Test getting a specific summarizer."""
        summarizer = get_summarizer("huggingface")
        assert summarizer is not None
        assert isinstance(summarizer, type)
    
    def test_get_summarizer_invalid(self):
        """Test getting an invalid summarizer."""
        with pytest.raises(ValueError):
            get_summarizer("invalid_summarizer")
    
    def test_create_summarizer(self):
        """Test creating a summarizer instance."""
        summarizer = create_summarizer("huggingface")
        assert isinstance(summarizer, HuggingFaceSummarizer)


class TestHuggingFaceSummarizer:
    """Test the HuggingFace summarizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.summarizer = HuggingFaceSummarizer()
    
    def test_initialization(self):
        """Test summarizer initialization."""
        assert self.summarizer is not None
        assert hasattr(self.summarizer, 'model_name')
        assert hasattr(self.summarizer, 'model_path')
    
    @patch('transformers.pipeline')
    def test_summarize_text(self, mock_pipeline):
        """Test text summarization."""
        # Mock the pipeline
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "This is a test summary."}]
        mock_pipeline.return_value = mock_pipe
        
        text = "This is a very long text that needs to be summarized. " * 10
        summary = self.summarizer.summarize(text)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "test summary" in summary.lower()
    
    @patch('transformers.pipeline')
    def test_summarize_file_content(self, mock_pipeline):
        """Test file content summarization."""
        # Mock the pipeline
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "File content summary."}]
        mock_pipeline.return_value = mock_pipe
        
        content = "This is file content that needs summarization. " * 5
        summary = self.summarizer.summarize_file_content(content)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @patch('transformers.pipeline')
    def test_summarize_changes(self, mock_pipeline):
        """Test change summarization."""
        # Mock the pipeline
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "Changes summary."}]
        mock_pipeline.return_value = mock_pipe
        
        changes = ["Added file1.txt", "Modified file2.py", "Deleted file3.csv"]
        summary = self.summarizer.summarize_changes(changes)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @patch('transformers.pipeline')
    def test_summarize_project(self, mock_pipeline):
        """Test project summarization."""
        # Mock the pipeline
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "Project summary."}]
        mock_pipeline.return_value = mock_pipe
        
        files = [
            FileMetadata(filename="file1.py", file_type="PYTHON", size=1000),
            FileMetadata(filename="file2.csv", file_type="CSV", size=2000),
            FileMetadata(filename="file3.md", file_type="MARKDOWN", size=500)
        ]
        summary = self.summarizer.summarize_project(files)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_batch_summarize(self):
        """Test batch summarization."""
        texts = [
            "First text to summarize.",
            "Second text to summarize.",
            "Third text to summarize."
        ]
        
        with patch.object(self.summarizer, 'summarize') as mock_summarize:
            mock_summarize.side_effect = ["Summary 1", "Summary 2", "Summary 3"]
            
            summaries = self.summarizer.batch_summarize(texts)
            
            assert isinstance(summaries, list)
            assert len(summaries) == 3
            assert all(isinstance(s, str) for s in summaries)
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.summarizer.get_model_info()
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "model_path" in info
        assert "backend" in info
        assert info["backend"] == "huggingface"
    
    def test_is_available(self):
        """Test availability check."""
        # Should be available if transformers is installed
        assert self.summarizer.is_available() in [True, False]
    
    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "A" * 1000
        truncated = self.summarizer._truncate_text(long_text, max_length=100)
        assert len(truncated) <= 100
    
    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "  \n\n  This is dirty text  \n\n  "
        cleaned = self.summarizer._clean_text(dirty_text)
        assert cleaned == "This is dirty text"
    
    def test_validate_input(self):
        """Test input validation."""
        # Valid input
        assert self.summarizer._validate_input("valid text") is True
        
        # Invalid input
        with pytest.raises(ValueError):
            self.summarizer._validate_input("")
        
        with pytest.raises(ValueError):
            self.summarizer._validate_input(None)


class TestOpenAISummarizer:
    """Test the OpenAI summarizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.summarizer = OpenAISummarizer()
    
    def test_initialization(self):
        """Test summarizer initialization."""
        assert self.summarizer is not None
        assert hasattr(self.summarizer, 'model_name')
        assert hasattr(self.summarizer, 'api_key')
    
    @patch('openai.OpenAI')
    def test_summarize_text(self, mock_openai):
        """Test text summarization."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is an OpenAI summary."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        text = "This is a very long text that needs to be summarized. " * 10
        summary = self.summarizer.summarize(text)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "OpenAI summary" in summary
    
    @patch('openai.OpenAI')
    def test_summarize_file_content(self, mock_openai):
        """Test file content summarization."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "File content summary from OpenAI."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        content = "This is file content that needs summarization. " * 5
        summary = self.summarizer.summarize_file_content(content)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @patch('openai.OpenAI')
    def test_summarize_changes(self, mock_openai):
        """Test change summarization."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Changes summary from OpenAI."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        changes = ["Added file1.txt", "Modified file2.py", "Deleted file3.csv"]
        summary = self.summarizer.summarize_changes(changes)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @patch('openai.OpenAI')
    def test_summarize_project(self, mock_openai):
        """Test project summarization."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Project summary from OpenAI."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        files = [
            FileMetadata(filename="file1.py", file_type="PYTHON", size=1000),
            FileMetadata(filename="file2.csv", file_type="CSV", size=2000),
            FileMetadata(filename="file3.md", file_type="MARKDOWN", size=500)
        ]
        summary = self.summarizer.summarize_project(files)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.summarizer.get_model_info()
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "backend" in info
        assert info["backend"] == "openai"
    
    def test_is_available(self):
        """Test availability check."""
        # Should check for API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            assert self.summarizer.is_available() is True
        
        with patch.dict(os.environ, {}, clear=True):
            assert self.summarizer.is_available() is False
    
    def test_prepare_prompt(self):
        """Test prompt preparation."""
        text = "Test text"
        prompt = self.summarizer._prepare_prompt(text)
        assert isinstance(prompt, str)
        assert text in prompt
        assert "summarize" in prompt.lower()


class TestSummarizerIntegration:
    """Test summarizer integration scenarios."""
    
    def test_summarizer_fallback(self):
        """Test fallback behavior when primary summarizer fails."""
        # Test with invalid model path
        summarizer = HuggingFaceSummarizer(model_path="/invalid/path")
        
        # Should fall back to extractive summarization
        text = "This is a test text. " * 20
        summary = summarizer.summarize(text)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_summarizer_error_handling(self):
        """Test error handling in summarizers."""
        summarizer = HuggingFaceSummarizer()
        
        # Test with empty text
        with pytest.raises(ValueError):
            summarizer.summarize("")
        
        # Test with None text
        with pytest.raises(ValueError):
            summarizer.summarize(None)
    
    def test_summarizer_performance(self):
        """Test summarizer performance with different text lengths."""
        summarizer = HuggingFaceSummarizer()
        
        # Short text
        short_text = "Short text."
        with patch.object(summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Short summary"
            summary = summarizer.summarize(short_text)
            assert summary == "Short summary"
        
        # Long text
        long_text = "Long text. " * 1000
        with patch.object(summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Long summary"
            summary = summarizer.summarize(long_text)
            assert summary == "Long summary"


class TestSummarizerConfiguration:
    """Test summarizer configuration options."""
    
    def test_huggingface_configuration(self):
        """Test HuggingFace summarizer configuration."""
        config = {
            "model_name": "google/flan-t5-base",
            "model_path": "./models/flan-t5-base",
            "max_length": 150,
            "min_length": 30
        }
        
        summarizer = HuggingFaceSummarizer(**config)
        assert summarizer.model_name == config["model_name"]
        assert summarizer.model_path == config["model_path"]
    
    def test_openai_configuration(self):
        """Test OpenAI summarizer configuration."""
        config = {
            "model_name": "gpt-4",
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        summarizer = OpenAISummarizer(**config)
        assert summarizer.model_name == config["model_name"]
    
    def test_custom_prompt_templates(self):
        """Test custom prompt templates."""
        summarizer = HuggingFaceSummarizer()
        
        # Test custom prompt for file content
        custom_prompt = "Summarize this file content: {text}"
        summarizer.file_content_prompt = custom_prompt
        
        with patch.object(summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Custom summary"
            summary = summarizer.summarize_file_content("test content")
            assert summary == "Custom summary"
