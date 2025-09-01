"""Markdown file parser for SciDoc."""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class MarkdownParser(BaseParser):
    """Parser for Markdown files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in ['.md', '.markdown']
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse a Markdown file and extract metadata."""
        file_path = Path(file_path)
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Read content for analysis
        content = self._read_text_content(file_path)
        
        # Extract Markdown-specific metadata
        markdown_metadata = self._extract_markdown_metadata(content)
        
        metadata = {
            "mime_type": "text/markdown",
            "is_binary": False,
            "extension": file_path.suffix.lower(),
            "stem": file_path.stem,
            **markdown_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.MARKDOWN,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["markdown", "documentation"]
        )
    
    def _extract_markdown_metadata(self, content: str) -> dict:
        """Extract Markdown-specific metadata."""
        import re
        
        metadata = {
            "headings": 0,
            "links": 0,
            "images": 0,
            "code_blocks": 0,
            "lists": 0,
        }
        
        lines = content.split('\n')
        
        for line in lines:
            # Count headings
            if re.match(r'^#{1,6}\s', line):
                metadata["headings"] += 1
            
            # Count links
            metadata["links"] += len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line))
            
            # Count images
            metadata["images"] += len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', line))
            
            # Count code blocks
            if line.strip().startswith('```'):
                metadata["code_blocks"] += 1
            
            # Count lists
            if re.match(r'^[\s]*[-*+]\s', line):
                metadata["lists"] += 1
        
        return metadata
