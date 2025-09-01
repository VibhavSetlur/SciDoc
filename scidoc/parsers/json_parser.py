"""JSON file parser for SciDoc."""

import json
from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class JsonParser(BaseParser):
    """Parser for JSON files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() == '.json'
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse a JSON file and extract metadata."""
        file_path = Path(file_path)
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Extract JSON-specific metadata
        json_metadata = self._extract_json_metadata(file_path)
        
        metadata = {
            "mime_type": "application/json",
            "is_binary": False,
            "extension": ".json",
            "stem": file_path.stem,
            **json_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.JSON,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["json", "data", "configuration"]
        )
    
    def _extract_json_metadata(self, file_path: Path) -> dict:
        """Extract JSON-specific metadata."""
        metadata = {
            "valid_json": False,
            "structure_type": "unknown",
            "depth": 0,
        }
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            metadata["valid_json"] = True
            
            # Determine structure type
            if isinstance(data, dict):
                metadata["structure_type"] = "object"
                metadata["depth"] = self._get_json_depth(data)
            elif isinstance(data, list):
                metadata["structure_type"] = "array"
                metadata["depth"] = self._get_json_depth(data)
            else:
                metadata["structure_type"] = "primitive"
                
        except Exception:
            pass
        
        return metadata
    
    def _get_json_depth(self, obj, current_depth=1):
        """Calculate the maximum depth of a JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
