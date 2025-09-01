"""YAML file parser for SciDoc."""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class YamlParser(BaseParser):
    """Parser for YAML files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in ['.yaml', '.yml']
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse a YAML file and extract metadata."""
        file_path = Path(file_path)
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Extract YAML-specific metadata
        yaml_metadata = self._extract_yaml_metadata(file_path)
        
        metadata = {
            "mime_type": "text/yaml",
            "is_binary": False,
            "extension": file_path.suffix.lower(),
            "stem": file_path.stem,
            **yaml_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.YAML,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["yaml", "configuration"]
        )
    
    def _extract_yaml_metadata(self, file_path: Path) -> dict:
        """Extract YAML-specific metadata."""
        metadata = {
            "valid_yaml": False,
            "structure_type": "unknown",
        }
        
        try:
            import yaml
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            metadata["valid_yaml"] = True
            
            # Determine structure type
            if isinstance(data, dict):
                metadata["structure_type"] = "object"
            elif isinstance(data, list):
                metadata["structure_type"] = "array"
            else:
                metadata["structure_type"] = "primitive"
                
        except Exception:
            pass
        
        return metadata
