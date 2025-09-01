"""
Base parser class for SciDoc.

This module defines the abstract base class that all file parsers
must implement to be compatible with SciDoc.
"""

import hashlib
import mimetypes
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..models import FileMetadata, FileType


class BaseParser(ABC):
    """Abstract base class for file parsers."""
    
    def __init__(self, **kwargs):
        """Initialize parser with optional configuration."""
        self.config = kwargs
    
    @abstractmethod
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this parser can parse the file, False otherwise
        """
        pass
    
    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """
        Parse a file and extract metadata.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            FileMetadata object containing extracted information
        """
        pass
    
    def get_file_type(self, file_path: Union[str, Path]) -> FileType:
        """
        Determine the file type for the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileType enum value
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Map extensions to file types
        extension_map = {
            '.fastq': FileType.FASTQ,
            '.fq': FileType.FASTQ,
            '.bam': FileType.BAM,
            '.sam': FileType.SAM,
            '.vcf': FileType.VCF,
            '.fasta': FileType.FASTA,
            '.fa': FileType.FASTA,
            '.csv': FileType.CSV,
            '.tsv': FileType.TSV,
            '.json': FileType.JSON,
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
            '.py': FileType.PYTHON,
            '.ipynb': FileType.JUPYTER,
            '.md': FileType.MARKDOWN,
            '.markdown': FileType.MARKDOWN,
            '.png': FileType.IMAGE,
            '.jpg': FileType.IMAGE,
            '.jpeg': FileType.IMAGE,
            '.gif': FileType.IMAGE,
            '.pdf': FileType.PDF,
            '.log': FileType.LOG,
            '.txt': FileType.TEXT,
        }
        
        return extension_map.get(extension, FileType.UNKNOWN)
    
    def get_basic_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract basic file metadata (size, timestamps, checksum).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing basic metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        # Calculate checksum
        checksum = self._calculate_checksum(file_path)
        
        return {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "checksum": checksum,
        }
    
    def _calculate_checksum(self, file_path: Path, algorithm: str = "md5") -> str:
        """
        Calculate file checksum for change detection.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use (md5, sha1, sha256)
            
        Returns:
            Hexadecimal checksum string
        """
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _read_text_content(self, file_path: Path, max_size: int = 1024 * 1024) -> str:
        """
        Read text content from file with size limit.
        
        Args:
            file_path: Path to the file
            max_size: Maximum size to read in bytes
            
        Returns:
            File content as string
        """
        if file_path.stat().st_size > max_size:
            # For large files, read only the beginning and end
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                start = f.read(max_size // 2)
                f.seek(-max_size // 2, 2)  # Seek from end
                end = f.read()
                return f"{start}\n... [truncated] ...\n{end}"
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    
    def _read_binary_content(self, file_path: Path, max_size: int = 1024 * 1024) -> bytes:
        """
        Read binary content from file with size limit.
        
        Args:
            file_path: Path to the file
            max_size: Maximum size to read in bytes
            
        Returns:
            File content as bytes
        """
        if file_path.stat().st_size > max_size:
            # For large files, read only the beginning
            with open(file_path, "rb") as f:
                return f.read(max_size)
        else:
            with open(file_path, "rb") as f:
                return f.read()
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding
        """
        import chardet
        
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            return result["encoding"] or "utf-8"
    
    def _get_mime_type(self, file_path: Path) -> str:
        """
        Get MIME type of the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """
        Extract file dependencies from content.
        
        Args:
            content: File content as string
            
        Returns:
            List of dependency file paths
        """
        dependencies = []
        
        # Common import patterns
        import re
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
            r'require\s*\(\s*["\']([^"\']+)["\']',
            r'include\s+["\']([^"\']+)["\']',
            r'#include\s+["<]([^">]+)[">]',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_tags(self, content: str, filename: str) -> List[str]:
        """
        Extract tags from file content and name.
        
        Args:
            content: File content as string
            filename: File name
            
        Returns:
            List of tags
        """
        tags = []
        
        # Extract tags from filename
        name_parts = Path(filename).stem.lower().split('_')
        tags.extend(name_parts)
        
        # Extract tags from content (common patterns)
        import re
        
        # Look for TODO, FIXME, etc.
        todo_patterns = [
            r'TODO[:\s]+([^\n]+)',
            r'FIXME[:\s]+([^\n]+)',
            r'NOTE[:\s]+([^\n]+)',
            r'HACK[:\s]+([^\n]+)',
        ]
        
        for pattern in todo_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Extract meaningful words from TODO comments
                words = re.findall(r'\b[a-zA-Z]{3,}\b', match)
                tags.extend(words[:3])  # Limit to first 3 words
        
        # Look for common keywords
        keywords = [
            'data', 'analysis', 'script', 'config', 'test', 'docs',
            'readme', 'license', 'requirements', 'setup', 'build',
            'deploy', 'docker', 'dockerfile', 'makefile', 'gitignore'
        ]
        
        for keyword in keywords:
            if keyword.lower() in filename.lower() or keyword.lower() in content.lower():
                tags.append(keyword)
        
        return list(set(tags))  # Remove duplicates
    
    def _safe_parse(self, file_path: Path, parse_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Safely execute a parsing function with error handling.
        
        Args:
            file_path: Path to the file
            parse_func: Function to execute
            *args: Arguments for parse_func
            **kwargs: Keyword arguments for parse_func
            
        Returns:
            Dictionary with parsing results or error information
        """
        try:
            result = parse_func(*args, **kwargs)
            return {"success": True, "data": result}
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
