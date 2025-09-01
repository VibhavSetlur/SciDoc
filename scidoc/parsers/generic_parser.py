"""
Generic parser for SciDoc.

This parser provides basic metadata extraction for any file type
that doesn't have a specific parser implementation.
"""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class GenericParser(BaseParser):
    """Generic parser for any file type."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Generic parser can handle any file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Always True for generic parser
        """
        return True
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """
        Parse a file and extract basic metadata.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            FileMetadata object with basic information
        """
        file_path = Path(file_path)
        
        # Get basic metadata
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Determine file type
        file_type = self.get_file_type(file_path)
        
        # Extract additional metadata
        additional_metadata = self._extract_generic_metadata(file_path)
        
        # Try to read content for text-based analysis
        content = ""
        dependencies = []
        tags = []
        
        if file_type in [FileType.TEXT, FileType.MARKDOWN, FileType.PYTHON, FileType.JSON, FileType.YAML]:
            try:
                content = self._read_text_content(file_path, max_size=1024 * 1024)  # 1MB limit
                dependencies = self._extract_dependencies(content)
                tags = self._extract_tags(content, str(file_path))
            except Exception:
                # If text reading fails, continue with basic metadata
                pass
        
        # Combine metadata
        metadata = {
            "mime_type": self._get_mime_type(file_path),
            "encoding": self._detect_encoding(file_path) if file_type in [FileType.TEXT, FileType.MARKDOWN, FileType.PYTHON] else None,
            "is_binary": file_type in [FileType.BINARY, FileType.IMAGE, FileType.BAM, FileType.SAM],
            "extension": file_path.suffix.lower(),
            "stem": file_path.stem,
            **additional_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=file_type,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=dependencies,
            tags=tags
        )
    
    def _extract_generic_metadata(self, file_path: Path) -> dict:
        """
        Extract generic metadata from file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with generic metadata
        """
        metadata = {}
        
        # File system metadata
        stat = file_path.stat()
        metadata["permissions"] = oct(stat.st_mode)[-3:]  # Octal permissions
        metadata["owner_id"] = stat.st_uid
        metadata["group_id"] = stat.st_gid
        
        # File characteristics
        metadata["is_symlink"] = file_path.is_symlink()
        metadata["is_hidden"] = file_path.name.startswith('.')
        
        # Size categories
        size = stat.st_size
        if size < 1024:
            metadata["size_category"] = "tiny"
        elif size < 1024 * 1024:
            metadata["size_category"] = "small"
        elif size < 10 * 1024 * 1024:
            metadata["size_category"] = "medium"
        elif size < 100 * 1024 * 1024:
            metadata["size_category"] = "large"
        else:
            metadata["size_category"] = "huge"
        
        # Try to detect file content type
        try:
            with open(file_path, "rb") as f:
                header = f.read(512)  # Read first 512 bytes
                
                # Check for common file signatures
                if header.startswith(b'\x89PNG\r\n\x1a\n'):
                    metadata["content_type"] = "PNG image"
                elif header.startswith(b'\xff\xd8\xff'):
                    metadata["content_type"] = "JPEG image"
                elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                    metadata["content_type"] = "GIF image"
                elif header.startswith(b'%PDF'):
                    metadata["content_type"] = "PDF document"
                elif header.startswith(b'PK\x03\x04'):
                    metadata["content_type"] = "ZIP archive"
                elif header.startswith(b'\x1f\x8b'):
                    metadata["content_type"] = "GZIP archive"
                elif header.startswith(b'BAM\x01'):
                    metadata["content_type"] = "BAM file"
                elif header.startswith(b'@'):
                    metadata["content_type"] = "FASTQ file"
                elif b'\x00' in header[:100]:
                    metadata["content_type"] = "Binary file"
                else:
                    metadata["content_type"] = "Text file"
                    
        except Exception:
            metadata["content_type"] = "Unknown"
        
        # Extract filename patterns
        name = file_path.name.lower()
        if any(pattern in name for pattern in ['readme', 'read_me']):
            metadata["file_purpose"] = "documentation"
        elif any(pattern in name for pattern in ['test', 'spec', 'check']):
            metadata["file_purpose"] = "testing"
        elif any(pattern in name for pattern in ['config', 'conf', 'ini', 'cfg']):
            metadata["file_purpose"] = "configuration"
        elif any(pattern in name for pattern in ['data', 'dataset', 'sample']):
            metadata["file_purpose"] = "data"
        elif any(pattern in name for pattern in ['log', 'error', 'debug']):
            metadata["file_purpose"] = "logging"
        elif any(pattern in name for pattern in ['backup', 'bak', 'old']):
            metadata["file_purpose"] = "backup"
        elif any(pattern in name for pattern in ['temp', 'tmp', 'cache']):
            metadata["file_purpose"] = "temporary"
        else:
            metadata["file_purpose"] = "general"
        
        return metadata
