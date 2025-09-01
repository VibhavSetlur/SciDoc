"""
FASTQ file parser for SciDoc.

This parser extracts metadata from FASTQ files including read counts,
quality statistics, and sequence information.
"""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class FastqParser(BaseParser):
    """Parser for FASTQ files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in ['.fastq', '.fq']
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse a FASTQ file and extract metadata."""
        file_path = Path(file_path)
        
        # Get basic metadata
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Extract FASTQ-specific metadata
        fastq_metadata = self._extract_fastq_metadata(file_path)
        
        # Combine metadata
        metadata = {
            "mime_type": "text/plain",
            "encoding": "utf-8",
            "is_binary": False,
            "extension": file_path.suffix.lower(),
            "stem": file_path.stem,
            **fastq_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.FASTQ,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["fastq", "sequencing", "bioinformatics"]
        )
    
    def _extract_fastq_metadata(self, file_path: Path) -> dict:
        """Extract FASTQ-specific metadata."""
        metadata = {
            "read_count": 0,
            "avg_read_length": 0,
            "quality_scores": "unknown",
        }
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Count reads (every 4 lines is one read)
            read_count = len(lines) // 4
            metadata["read_count"] = read_count
            
            # Calculate average read length
            total_length = 0
            for i in range(1, len(lines), 4):  # Sequence lines
                if i < len(lines):
                    total_length += len(lines[i].strip())
            
            if read_count > 0:
                metadata["avg_read_length"] = total_length / read_count
            
        except Exception:
            pass
        
        return metadata
