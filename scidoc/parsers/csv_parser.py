"""CSV file parser for SciDoc."""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class CsvParser(BaseParser):
    """Parser for CSV files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in ['.csv', '.tsv']
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse a CSV file and extract metadata."""
        file_path = Path(file_path)
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Extract CSV-specific metadata
        csv_metadata = self._extract_csv_metadata(file_path)
        
        metadata = {
            "mime_type": "text/csv",
            "is_binary": False,
            "extension": file_path.suffix.lower(),
            "stem": file_path.stem,
            **csv_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.CSV if file_path.suffix.lower() == '.csv' else FileType.TSV,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["csv", "data", "tabular"]
        )
    
    def _extract_csv_metadata(self, file_path: Path) -> dict:
        """Extract CSV-specific metadata."""
        metadata = {
            "columns": 0,
            "rows": 0,
            "delimiter": ",",
        }
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            if lines:
                # Detect delimiter
                first_line = lines[0]
                if '\t' in first_line:
                    metadata["delimiter"] = "\t"
                
                # Count columns
                metadata["columns"] = len(first_line.split(metadata["delimiter"]))
                
                # Count rows (excluding header)
                metadata["rows"] = len(lines) - 1
                
        except Exception:
            pass
        
        return metadata
