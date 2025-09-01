"""
File parsers for SciDoc.

This package contains parsers for different file types that extract
metadata and content for analysis and summarization.
"""

from typing import Dict, List, Type

from .base_parser import BaseParser
from .fastq_parser import FastqParser
from .bam_parser import BamParser
from .vcf_parser import VcfParser
from .csv_parser import CsvParser
from .json_parser import JsonParser
from .yaml_parser import YamlParser
from .python_parser import PythonParser
from .jupyter_parser import JupyterParser
from .markdown_parser import MarkdownParser
from .image_parser import ImageParser
from .pdf_parser import PdfParser
from .generic_parser import GenericParser

# Registry of available parsers
PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {
    "fastq": FastqParser,
    "bam": BamParser,
    "vcf": VcfParser,
    "csv": CsvParser,
    "json": JsonParser,
    "yaml": YamlParser,
    "python": PythonParser,
    "jupyter": JupyterParser,
    "markdown": MarkdownParser,
    "image": ImageParser,
    "pdf": PdfParser,
    "generic": GenericParser,
}


def get_parser(file_type: str) -> Type[BaseParser]:
    """Get parser class for a file type."""
    return PARSER_REGISTRY.get(file_type.lower(), GenericParser)


def get_available_parsers() -> List[str]:
    """Get list of available parser names."""
    return list(PARSER_REGISTRY.keys())


def register_parser(name: str, parser_class: Type[BaseParser]) -> None:
    """Register a new parser."""
    PARSER_REGISTRY[name.lower()] = parser_class


def create_parser(file_type: str, **kwargs) -> BaseParser:
    """Create a parser instance for a file type."""
    parser_class = get_parser(file_type)
    return parser_class(**kwargs)


__all__ = [
    "BaseParser",
    "FastqParser",
    "BamParser",
    "VcfParser",
    "CsvParser",
    "JsonParser",
    "YamlParser",
    "PythonParser",
    "JupyterParser",
    "MarkdownParser",
    "ImageParser",
    "PdfParser",
    "GenericParser",
    "PARSER_REGISTRY",
    "get_parser",
    "get_available_parsers",
    "register_parser",
    "create_parser",
]
