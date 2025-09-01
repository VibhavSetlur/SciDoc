"""
Tests for SciDoc parsers.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from scidoc.parsers import (
    get_parser, get_available_parsers, create_parser,
    GenericParser, PythonParser, FastqParser, BamParser, VcfParser,
    CsvParser, JsonParser, YamlParser, JupyterParser, MarkdownParser,
    ImageParser, PdfParser
)
from scidoc.models import FileMetadata


class TestParserRegistry:
    """Test parser registry functionality."""
    
    def test_get_available_parsers(self):
        """Test that we can get available parsers."""
        parsers = get_available_parsers()
        assert isinstance(parsers, dict)
        assert len(parsers) > 0
        assert "generic" in parsers
    
    def test_get_parser(self):
        """Test getting a specific parser."""
        parser = get_parser("generic")
        assert parser is not None
        assert isinstance(parser, type)
    
    def test_get_parser_invalid(self):
        """Test getting an invalid parser."""
        with pytest.raises(ValueError):
            get_parser("invalid_parser")
    
    def test_create_parser(self):
        """Test creating a parser instance."""
        parser = create_parser("generic")
        assert isinstance(parser, GenericParser)


class TestGenericParser:
    """Test the generic parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = GenericParser()
    
    def test_can_parse_any_file(self):
        """Test that generic parser can parse any file."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            f.write(b"test content")
            f.flush()
            assert self.parser.can_parse(f.name)
    
    def test_parse_basic_file(self):
        """Test parsing a basic text file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w") as f:
            f.write("test content")
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.filename == os.path.basename(f.name)
            assert metadata.file_type == "TEXT"
            assert metadata.size > 0
    
    def test_parse_binary_file(self):
        """Test parsing a binary file."""
        with tempfile.NamedTemporaryFile(suffix=".bin", mode="wb") as f:
            f.write(b"\x00\x01\x02\x03")
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.is_binary


class TestPythonParser:
    """Test the Python parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_can_parse_python_file(self):
        """Test that Python parser can parse .py files."""
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            assert self.parser.can_parse(f.name)
    
    def test_cannot_parse_non_python_file(self):
        """Test that Python parser cannot parse non-Python files."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            assert not self.parser.can_parse(f.name)
    
    def test_parse_simple_python_file(self):
        """Test parsing a simple Python file."""
        python_code = '''
def hello():
    """Simple function."""
    print("Hello, World!")

class TestClass:
    def __init__(self):
        self.value = 42
'''
        
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as f:
            f.write(python_code)
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "PYTHON"
            assert "functions" in metadata.content_metadata
            assert "classes" in metadata.content_metadata
            assert metadata.content_metadata["functions"] == 1
            assert metadata.content_metadata["classes"] == 1


class TestFastqParser:
    """Test the FASTQ parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = FastqParser()
    
    def test_can_parse_fastq_file(self):
        """Test that FASTQ parser can parse .fastq files."""
        with tempfile.NamedTemporaryFile(suffix=".fastq") as f:
            assert self.parser.can_parse(f.name)
        
        with tempfile.NamedTemporaryFile(suffix=".fq") as f:
            assert self.parser.can_parse(f.name)
    
    def test_parse_fastq_file(self):
        """Test parsing a FASTQ file."""
        fastq_content = """@seq1
ACGTACGTACGT
+
IIIIIIIIIIII
@seq2
GCTAGCTAGCTA
+
IIIIIIIIIIII
"""
        
        with tempfile.NamedTemporaryFile(suffix=".fastq", mode="w") as f:
            f.write(fastq_content)
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "FASTQ"
            assert "read_count" in metadata.content_metadata
            assert metadata.content_metadata["read_count"] == 2


class TestCsvParser:
    """Test the CSV parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CsvParser()
    
    def test_can_parse_csv_file(self):
        """Test that CSV parser can parse .csv files."""
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            assert self.parser.can_parse(f.name)
        
        with tempfile.NamedTemporaryFile(suffix=".tsv") as f:
            assert self.parser.can_parse(f.name)
    
    def test_parse_csv_file(self):
        """Test parsing a CSV file."""
        csv_content = """name,age,city
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago
"""
        
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w") as f:
            f.write(csv_content)
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "CSV"
            assert "columns" in metadata.content_metadata
            assert "estimated_rows" in metadata.content_metadata
            assert metadata.content_metadata["columns"] == 3
            assert metadata.content_metadata["estimated_rows"] == 3


class TestJsonParser:
    """Test the JSON parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = JsonParser()
    
    def test_can_parse_json_file(self):
        """Test that JSON parser can parse .json files."""
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            assert self.parser.can_parse(f.name)
    
    def test_parse_json_file(self):
        """Test parsing a JSON file."""
        json_content = '''{
    "name": "test",
    "data": {
        "values": [1, 2, 3],
        "nested": {
            "key": "value"
        }
    }
}'''
        
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            f.write(json_content)
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "JSON"
            assert "structure" in metadata.content_metadata
            assert "max_depth" in metadata.content_metadata
            assert metadata.content_metadata["structure"] == "object"


class TestYamlParser:
    """Test the YAML parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = YamlParser()
    
    def test_can_parse_yaml_file(self):
        """Test that YAML parser can parse .yaml files."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            assert self.parser.can_parse(f.name)
        
        with tempfile.NamedTemporaryFile(suffix=".yml") as f:
            assert self.parser.can_parse(f.name)
    
    def test_parse_yaml_file(self):
        """Test parsing a YAML file."""
        yaml_content = """
name: test
data:
  values: [1, 2, 3]
  nested:
    key: value
"""
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w") as f:
            f.write(yaml_content)
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "YAML"
            assert "structure" in metadata.content_metadata
            assert metadata.content_metadata["structure"] == "object"


class TestJupyterParser:
    """Test the Jupyter parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = JupyterParser()
    
    def test_can_parse_jupyter_file(self):
        """Test that Jupyter parser can parse .ipynb files."""
        with tempfile.NamedTemporaryFile(suffix=".ipynb") as f:
            assert self.parser.can_parse(f.name)
    
    def test_parse_jupyter_file(self):
        """Test parsing a Jupyter notebook."""
        notebook_content = '''{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Title"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["print('Hello')"],
            "outputs": []
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}'''
        
        with tempfile.NamedTemporaryFile(suffix=".ipynb", mode="w") as f:
            f.write(notebook_content)
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "JUPYTER"
            assert "total_cells" in metadata.content_metadata
            assert "code_cells" in metadata.content_metadata
            assert "markdown_cells" in metadata.content_metadata
            assert metadata.content_metadata["total_cells"] == 2
            assert metadata.content_metadata["code_cells"] == 1
            assert metadata.content_metadata["markdown_cells"] == 1


class TestMarkdownParser:
    """Test the Markdown parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MarkdownParser()
    
    def test_can_parse_markdown_file(self):
        """Test that Markdown parser can parse .md files."""
        with tempfile.NamedTemporaryFile(suffix=".md") as f:
            assert self.parser.can_parse(f.name)
        
        with tempfile.NamedTemporaryFile(suffix=".markdown") as f:
            assert self.parser.can_parse(f.name)
    
    def test_parse_markdown_file(self):
        """Test parsing a Markdown file."""
        markdown_content = """# Title

## Subtitle

This is a paragraph with [a link](http://example.com).

- List item 1
- List item 2

```python
print("code block")
```

![image](image.png)
"""
        
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w") as f:
            f.write(markdown_content)
            f.flush()
            
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "MARKDOWN"
            assert "headings" in metadata.content_metadata
            assert "links" in metadata.content_metadata
            assert "images" in metadata.content_metadata
            assert "code_blocks" in metadata.content_metadata
            assert "lists" in metadata.content_metadata
            assert metadata.content_metadata["headings"] == 2
            assert metadata.content_metadata["links"] == 1
            assert metadata.content_metadata["images"] == 1
            assert metadata.content_metadata["code_blocks"] == 1
            assert metadata.content_metadata["lists"] == 1


class TestImageParser:
    """Test the Image parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ImageParser()
    
    def test_can_parse_image_file(self):
        """Test that Image parser can parse image files."""
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            assert self.parser.can_parse(f.name)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            assert self.parser.can_parse(f.name)
    
    @patch('PIL.Image.open')
    def test_parse_image_file(self, mock_image_open):
        """Test parsing an image file."""
        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.size = (800, 600)
        mock_image.format = "PNG"
        mock_image.mode = "RGB"
        mock_image_open.return_value = mock_image
        
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "IMAGE"
            assert "width" in metadata.content_metadata
            assert "height" in metadata.content_metadata
            assert "format" in metadata.content_metadata
            assert "mode" in metadata.content_metadata
            assert metadata.content_metadata["width"] == 800
            assert metadata.content_metadata["height"] == 600
            assert metadata.content_metadata["format"] == "PNG"
            assert metadata.content_metadata["mode"] == "RGB"


class TestPdfParser:
    """Test the PDF parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PdfParser()
    
    def test_can_parse_pdf_file(self):
        """Test that PDF parser can parse .pdf files."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            assert self.parser.can_parse(f.name)
    
    @patch('PyPDF2.PdfReader')
    def test_parse_pdf_file(self, mock_pdf_reader):
        """Test parsing a PDF file."""
        # Mock PyPDF2 PdfReader
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()] * 5  # 5 pages
        mock_reader.metadata = {
            '/Title': 'Test Document',
            '/Author': 'Test Author',
            '/Subject': 'Test Subject'
        }
        mock_pdf_reader.return_value = mock_reader
        
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            metadata = self.parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "PDF"
            assert "pages" in metadata.content_metadata
            assert "title" in metadata.content_metadata
            assert "author" in metadata.content_metadata
            assert "subject" in metadata.content_metadata
            assert metadata.content_metadata["pages"] == 5
            assert metadata.content_metadata["title"] == "Test Document"
            assert metadata.content_metadata["author"] == "Test Author"
            assert metadata.content_metadata["subject"] == "Test Subject"


class TestBioinformaticsParsers:
    """Test bioinformatics-specific parsers."""
    
    def test_bam_parser(self):
        """Test BAM parser."""
        parser = BamParser()
        with tempfile.NamedTemporaryFile(suffix=".bam") as f:
            assert parser.can_parse(f.name)
            
            metadata = parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "BAM"
            assert "bioinformatics" in metadata.tags
    
    def test_vcf_parser(self):
        """Test VCF parser."""
        parser = VcfParser()
        with tempfile.NamedTemporaryFile(suffix=".vcf") as f:
            assert parser.can_parse(f.name)
            
            metadata = parser.parse(f.name)
            assert isinstance(metadata, FileMetadata)
            assert metadata.file_type == "VCF"
            assert "bioinformatics" in metadata.tags
