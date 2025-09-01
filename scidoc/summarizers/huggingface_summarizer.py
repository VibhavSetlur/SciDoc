"""
HuggingFace summarizer for SciDoc.

This module provides summarization using HuggingFace transformer models,
supporting both local models and models from the HuggingFace Hub.
"""

import os
import re
import ast
import json
import yaml
import csv
import io
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from .base_summarizer import BaseSummarizer


@dataclass
class FileAnalysis:
    """Analysis results for a file."""
    file_type: str
    content_summary: str
    structure_info: Dict[str, Any]
    key_insights: List[str]
    technical_details: Dict[str, Any]
    recommendations: List[str]


class HuggingFaceSummarizer(BaseSummarizer):
    """HuggingFace-based summarizer using transformer models with enhanced file analysis."""
    
    def __init__(self, **kwargs):
        """Initialize HuggingFace summarizer."""
        super().__init__(**kwargs)
        
        self.model_path = kwargs.get("model_path", "google/flan-t5-base")
        self.device = kwargs.get("device", "auto")
        self.model = None
        self.tokenizer = None
        self.summarizer_pipeline = None
        
        # File type patterns for enhanced analysis
        self.file_patterns = {
            'python': r'\.py$',
            'jupyter': r'\.ipynb$',
            'markdown': r'\.(md|markdown)$',
            'csv': r'\.csv$',
            'json': r'\.json$',
            'yaml': r'\.(yaml|yml)$',
            'fastq': r'\.(fastq|fq)$',
            'fasta': r'\.(fasta|fa)$',
            'vcf': r'\.vcf$',
            'bam': r'\.bam$',
            'sam': r'\.sam$',
            'txt': r'\.txt$',
            'log': r'\.log$',
            'config': r'\.(conf|config|ini)$',
            'shell': r'\.(sh|bash|zsh)$',
            'r': r'\.r$',
            'r_markdown': r'\.rmd$',
            'matlab': r'\.m$',
            'cpp': r'\.(cpp|cc|cxx)$',
            'c': r'\.c$',
            'java': r'\.java$',
            'javascript': r'\.js$',
            'typescript': r'\.ts$',
            'html': r'\.html?$',
            'xml': r'\.xml$',
            'sql': r'\.sql$',
            'latex': r'\.tex$',
            'bibtex': r'\.bib$',
        }
        
        # Initialize model
        self._load_model()
    
    def analyze_file(self, file_path: str, content: str) -> FileAnalysis:
        """
        Analyze a file and generate comprehensive understanding.
        
        Args:
            file_path: Path to the file
            content: File content as string
            
        Returns:
            FileAnalysis object with comprehensive analysis
        """
        file_type = self._detect_file_type(file_path)
        
        # Get file-specific analysis
        structure_info = self._analyze_structure(file_type, content, file_path)
        key_insights = self._extract_key_insights(file_type, content, structure_info)
        technical_details = self._get_technical_details(file_type, content, structure_info)
        recommendations = self._generate_recommendations(file_type, structure_info, technical_details)
        
        # Generate content summary
        content_summary = self._generate_content_summary(file_type, content, structure_info)
        
        return FileAnalysis(
            file_type=file_type,
            content_summary=content_summary,
            structure_info=structure_info,
            key_insights=key_insights,
            technical_details=technical_details,
            recommendations=recommendations
        )
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type based on extension and content."""
        file_path_lower = file_path.lower()
        
        for file_type, pattern in self.file_patterns.items():
            if re.search(pattern, file_path_lower):
                return file_type
        
        return 'unknown'
    
    def _analyze_structure(self, file_type: str, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze file structure based on type."""
        structure_info = {
            'file_type': file_type,
            'file_size': len(content),
            'line_count': len(content.splitlines()),
            'character_count': len(content),
        }
        
        if file_type == 'python':
            structure_info.update(self._analyze_python_structure(content))
        elif file_type == 'jupyter':
            structure_info.update(self._analyze_jupyter_structure(content))
        elif file_type == 'markdown':
            structure_info.update(self._analyze_markdown_structure(content))
        elif file_type == 'csv':
            structure_info.update(self._analyze_csv_structure(content))
        elif file_type == 'json':
            structure_info.update(self._analyze_json_structure(content))
        elif file_type == 'yaml':
            structure_info.update(self._analyze_yaml_structure(content))
        elif file_type == 'fastq':
            structure_info.update(self._analyze_fastq_structure(content))
        elif file_type == 'fasta':
            structure_info.update(self._analyze_fasta_structure(content))
        elif file_type == 'vcf':
            structure_info.update(self._analyze_vcf_structure(content))
        elif file_type == 'txt':
            structure_info.update(self._analyze_text_structure(content))
        elif file_type == 'log':
            structure_info.update(self._analyze_log_structure(content))
        
        return structure_info
    
    def _analyze_python_structure(self, content: str) -> Dict[str, Any]:
        """Analyze Python file structure."""
        try:
            tree = ast.parse(content)
            imports = [node.name for node in ast.walk(tree) if isinstance(node, ast.Import)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            return {'imports': imports, 'functions': functions, 'classes': classes}
        except Exception:
            return {}
    
    def _analyze_jupyter_structure(self, content: str) -> Dict[str, Any]:
        """Analyze Jupyter notebook structure."""
        try:
            # This is a simplified check. A full Jupyter notebook parser would be needed for more details.
            if '```python' in content or '```' in content:
                return {'is_code_cell': True}
            return {'is_code_cell': False}
        except Exception:
            return {}
    
    def _analyze_markdown_structure(self, content: str) -> Dict[str, Any]:
        """Analyze Markdown file structure."""
        try:
            # This is a simplified check. A full Markdown parser would be needed for more details.
            if '```' in content:
                return {'is_code_block': True}
            return {'is_code_block': False}
        except Exception:
            return {}
    
    def _analyze_csv_structure(self, content: str) -> Dict[str, Any]:
        """Analyze CSV file structure."""
        try:
            # This is a simplified check. A full CSV parser would be needed for more details.
            if ',' in content or '\t' in content:
                return {'is_tabular': True}
            return {'is_tabular': False}
        except Exception:
            return {}
    
    def _analyze_json_structure(self, content: str) -> Dict[str, Any]:
        """Analyze JSON file structure."""
        try:
            json.loads(content)
            return {'is_json': True}
        except json.JSONDecodeError:
            return {'is_json': False}
        except Exception:
            return {}
    
    def _analyze_yaml_structure(self, content: str) -> Dict[str, Any]:
        """Analyze YAML file structure."""
        try:
            yaml.safe_load(content)
            return {'is_yaml': True}
        except yaml.YAMLError:
            return {'is_yaml': False}
        except Exception:
            return {}
    
    def _analyze_fastq_structure(self, content: str) -> Dict[str, Any]:
        """Analyze FASTQ file structure."""
        try:
            # This is a simplified check. A full FASTQ parser would be needed for more details.
            if '@' in content and '+' in content:
                return {'is_fastq': True}
            return {'is_fastq': False}
        except Exception:
            return {}
    
    def _analyze_fasta_structure(self, content: str) -> Dict[str, Any]:
        """Analyze FASTA file structure."""
        try:
            # This is a simplified check. A full FASTA parser would be needed for more details.
            if '>' in content:
                return {'is_fasta': True}
            return {'is_fasta': False}
        except Exception:
            return {}
    
    def _analyze_vcf_structure(self, content: str) -> Dict[str, Any]:
        """Analyze VCF file structure."""
        try:
            # This is a simplified check. A full VCF parser would be needed for more details.
            if '@' in content and '##' in content:
                return {'is_vcf': True}
            return {'is_vcf': False}
        except Exception:
            return {}
    
    def _analyze_text_structure(self, content: str) -> Dict[str, Any]:
        """Analyze text file structure."""
        try:
            # This is a simplified check. A full text parser would be needed for more details.
            return {'is_text': True}
        except Exception:
            return {}
    
    def _analyze_log_structure(self, content: str) -> Dict[str, Any]:
        """Analyze log file structure."""
        try:
            # This is a simplified check. A full log parser would be needed for more details.
            return {'is_log': True}
        except Exception:
            return {}
    
    def _extract_key_insights(self, file_type: str, content: str, structure_info: Dict[str, Any]) -> List[str]:
        """Extract key insights from file content."""
        insights = []
        
        if file_type == 'python':
            insights.append("This is a Python file.")
            if 'import' in structure_info:
                insights.append(f"Contains {len(structure_info['imports'])} imports.")
            if 'functions' in structure_info:
                insights.append(f"Contains {len(structure_info['functions'])} functions.")
            if 'classes' in structure_info:
                insights.append(f"Contains {len(structure_info['classes'])} classes.")
        elif file_type == 'jupyter':
            insights.append("This is a Jupyter notebook.")
            if 'is_code_cell' in structure_info:
                insights.append("Contains code cells.")
        elif file_type == 'markdown':
            insights.append("This is a Markdown file.")
            if 'is_code_block' in structure_info:
                insights.append("Contains code blocks.")
        elif file_type == 'csv':
            insights.append("This is a CSV file.")
            if 'is_tabular' in structure_info:
                insights.append("Contains tabular data.")
        elif file_type == 'json':
            insights.append("This is a JSON file.")
            if 'is_json' in structure_info:
                insights.append("Contains JSON data.")
        elif file_type == 'yaml':
            insights.append("This is a YAML file.")
            if 'is_yaml' in structure_info:
                insights.append("Contains YAML data.")
        elif file_type == 'fastq':
            insights.append("This is a FASTQ file.")
            if 'is_fastq' in structure_info:
                insights.append("Contains FASTQ data.")
        elif file_type == 'fasta':
            insights.append("This is a FASTA file.")
            if 'is_fasta' in structure_info:
                insights.append("Contains FASTA data.")
        elif file_type == 'vcf':
            insights.append("This is a VCF file.")
            if 'is_vcf' in structure_info:
                insights.append("Contains VCF data.")
        elif file_type == 'txt':
            insights.append("This is a text file.")
            if 'is_text' in structure_info:
                insights.append("Contains plain text.")
        elif file_type == 'log':
            insights.append("This is a log file.")
            if 'is_log' in structure_info:
                insights.append("Contains log data.")
        
        return insights
    
    def _get_technical_details(self, file_type: str, content: str, structure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get technical details about the file."""
        details = {}
        
        if file_type == 'python':
            details['language'] = 'Python'
            details['version'] = 'N/A' # Cannot determine version from content
            details['dependencies'] = 'N/A' # Cannot determine dependencies from content
        elif file_type == 'jupyter':
            details['language'] = 'Python'
            details['version'] = 'N/A' # Cannot determine version from content
            details['dependencies'] = 'N/A' # Cannot determine dependencies from content
        elif file_type == 'markdown':
            details['language'] = 'Markdown'
            details['version'] = 'N/A' # Cannot determine version from content
            details['dependencies'] = 'N/A' # Cannot determine dependencies from content
        elif file_type == 'csv':
            details['format'] = 'CSV'
            details['delimiter'] = 'N/A' # Cannot determine delimiter from content
            details['has_header'] = 'N/A' # Cannot determine header from content
        elif file_type == 'json':
            details['format'] = 'JSON'
            details['structure'] = 'N/A' # Cannot determine structure from content
        elif file_type == 'yaml':
            details['format'] = 'YAML'
            details['structure'] = 'N/A' # Cannot determine structure from content
        elif file_type == 'fastq':
            details['format'] = 'FASTQ'
            details['sequence_type'] = 'N/A' # Cannot determine sequence type from content
        elif file_type == 'fasta':
            details['format'] = 'FASTA'
            details['sequence_type'] = 'N/A' # Cannot determine sequence type from content
        elif file_type == 'vcf':
            details['format'] = 'VCF'
            details['version'] = 'N/A' # Cannot determine version from content
            details['header_info'] = 'N/A' # Cannot determine header info from content
        elif file_type == 'txt':
            details['encoding'] = 'N/A' # Cannot determine encoding from content
            details['line_ending'] = 'N/A' # Cannot determine line ending from content
        elif file_type == 'log':
            details['log_level'] = 'N/A' # Cannot determine log level from content
        
        return details
    
    def _generate_recommendations(self, file_type: str, structure_info: Dict[str, Any], technical_details: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if file_type == 'python':
            if 'imports' in structure_info and len(structure_info['imports']) > 0:
                recommendations.append("Consider organizing imports into sections (e.g., standard library, third-party, local).")
            if 'functions' in structure_info and len(structure_info['functions']) > 0:
                recommendations.append("Consider breaking down large functions into smaller, more manageable ones.")
            if 'classes' in structure_info and len(structure_info['classes']) > 0:
                recommendations.append("Consider organizing classes into logical groups.")
        elif file_type == 'jupyter':
            if 'is_code_cell' in structure_info:
                recommendations.append("Consider organizing code cells into logical sections.")
        elif file_type == 'markdown':
            if 'is_code_block' in structure_info:
                recommendations.append("Consider organizing code blocks into logical sections.")
        elif file_type == 'csv':
            if 'is_tabular' in structure_info:
                recommendations.append("Consider adding column headers for better readability.")
        elif file_type == 'json':
            if 'is_json' in structure_info:
                recommendations.append("Consider adding comments for better code readability.")
        elif file_type == 'yaml':
            if 'is_yaml' in structure_info:
                recommendations.append("Consider adding comments for better code readability.")
        elif file_type == 'fastq':
            if 'is_fastq' in structure_info:
                recommendations.append("Consider adding sequence names for better readability.")
        elif file_type == 'fasta':
            if 'is_fasta' in structure_info:
                recommendations.append("Consider adding sequence names for better readability.")
        elif file_type == 'vcf':
            if 'is_vcf' in structure_info:
                recommendations.append("Consider adding header information for better readability.")
        elif file_type == 'txt':
            if 'is_text' in structure_info:
                recommendations.append("Consider adding line breaks for better readability.")
        elif file_type == 'log':
            if 'is_log' in structure_info:
                recommendations.append("Consider adding timestamps for better readability.")
        
        return recommendations
    
    def _generate_content_summary(self, file_type: str, content: str, structure_info: Dict[str, Any]) -> str:
        """Generate a summary of the file content."""
        summary = f"This file is a {file_type} file."
        
        if file_type == 'python':
            if 'imports' in structure_info and len(structure_info['imports']) > 0:
                summary += f" It contains {len(structure_info['imports'])} imports."
            if 'functions' in structure_info and len(structure_info['functions']) > 0:
                summary += f" It contains {len(structure_info['functions'])} functions."
            if 'classes' in structure_info and len(structure_info['classes']) > 0:
                summary += f" It contains {len(structure_info['classes'])} classes."
        elif file_type == 'jupyter':
            if 'is_code_cell' in structure_info:
                summary += " It contains code cells."
        elif file_type == 'markdown':
            if 'is_code_block' in structure_info:
                summary += " It contains code blocks."
        elif file_type == 'csv':
            if 'is_tabular' in structure_info:
                summary += " It contains tabular data."
        elif file_type == 'json':
            if 'is_json' in structure_info:
                summary += " It contains JSON data."
        elif file_type == 'yaml':
            if 'is_yaml' in structure_info:
                summary += " It contains YAML data."
        elif file_type == 'fastq':
            if 'is_fastq' in structure_info:
                summary += " It contains FASTQ data."
        elif file_type == 'fasta':
            if 'is_fasta' in structure_info:
                summary += " It contains FASTA data."
        elif file_type == 'vcf':
            if 'is_vcf' in structure_info:
                summary += " It contains VCF data."
        elif file_type == 'txt':
            if 'is_text' in structure_info:
                summary += " It contains plain text."
        elif file_type == 'log':
            if 'is_log' in structure_info:
                summary += " It contains log data."
        
        return summary
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        try:
            # Suppress TensorFlow and CUDA warnings
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
            os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Disable CUDA to avoid warnings
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN warnings
            
            # Suppress warnings
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"Loading model from: {self.model_path}")
            print(f"Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            
            # Move model to device
            self.model.to(self.device)
            
            # Create summarization pipeline
            self.summarizer_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=self.max_length,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=self.temperature,
            )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to basic text processing...")
            self.model = None
            self.tokenizer = None
            self.summarizer_pipeline = None
    
    def summarize(self, text: str, **kwargs) -> str:
        """
        Generate a summary using HuggingFace model.
        
        Args:
            text: Text to summarize
            **kwargs: Additional arguments
            
        Returns:
            Generated summary as string
        """
        if not self._validate_input(text):
            return "No valid text to summarize."
        
        # Clean and prepare text
        text = self._clean_text(text)
        
        # Use custom max_length if provided
        max_length = kwargs.get("max_length", self.max_length)
        min_length = kwargs.get("min_length", 30)
        
        try:
            if self.summarizer_pipeline:
                # Use the summarization pipeline
                result = self.summarizer_pipeline(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=self.temperature,
                    truncation=True,
                )
                
                if result and len(result) > 0:
                    return result[0]["summary_text"]
                else:
                    return "Failed to generate summary."
            
            elif self.model and self.tokenizer:
                # Manual summarization using model and tokenizer
                return self._manual_summarize(text, max_length, min_length)
            
            else:
                # Fallback to basic summarization
                return self._basic_summarize(text, max_length)
                
        except Exception as e:
            print(f"Summarization error: {e}")
            return self._basic_summarize(text, max_length)
    
    def _manual_summarize(self, text: str, max_length: int, min_length: int) -> str:
        """
        Manual summarization using model and tokenizer directly.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Generated summary
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"Manual summarization error: {e}")
            return self._basic_summarize(text, max_length)
    
    def _basic_summarize(self, text: str, max_length: int) -> str:
        """
        Basic summarization fallback when model is not available.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            Basic summary
        """
        # Simple extractive summarization
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        # Take first few sentences as summary
        summary_sentences = sentences[:3]
        summary = '. '.join(summary_sentences) + '.'
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _truncate_for_model(self, text: str, max_tokens: int = 512) -> str:
        """
        Truncate text to fit within model's token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        if not self.tokenizer:
            return self._truncate_text(text, max_tokens)
        
        try:
            # Tokenize and truncate
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:
            return self._truncate_text(text, max_tokens)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        
        if self.model:
            info.update({
                "model_name": self.model_path,
                "device": self.device,
                "model_type": type(self.model).__name__,
                "tokenizer_type": type(self.tokenizer).__name__ if self.tokenizer else None,
                "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            })
        
        return info
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.model is not None and self.tokenizer is not None
    
    def save_model(self, save_path: str) -> None:
        """
        Save the model and tokenizer to a local path.
        
        Args:
            save_path: Path to save the model
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("No model loaded to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a model from a local path or HuggingFace Hub.
        
        Args:
            model_path: Path to model or model name
        """
        self.model_path = model_path
        self._load_model()
