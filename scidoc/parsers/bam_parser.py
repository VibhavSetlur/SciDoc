"""BAM file parser for SciDoc."""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class BamParser(BaseParser):
    """Parser for BAM files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() == '.bam'
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse a BAM file and extract metadata."""
        file_path = Path(file_path)
        basic_metadata = self.get_basic_metadata(file_path)
        
        metadata = {
            "mime_type": "application/octet-stream",
            "is_binary": True,
            "extension": ".bam",
            "stem": file_path.stem,
            "file_format": "BAM",
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.BAM,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["bam", "alignment", "bioinformatics"]
        )
