"""Jupyter notebook parser for SciDoc."""

import json
from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class JupyterParser(BaseParser):
    """Parser for Jupyter notebooks."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() == '.ipynb'
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse a Jupyter notebook and extract metadata."""
        file_path = Path(file_path)
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Extract notebook-specific metadata
        notebook_metadata = self._extract_notebook_metadata(file_path)
        
        metadata = {
            "mime_type": "application/x-ipynb+json",
            "is_binary": False,
            "extension": ".ipynb",
            "stem": file_path.stem,
            **notebook_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.JUPYTER,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["jupyter", "notebook", "analysis"]
        )
    
    def _extract_notebook_metadata(self, file_path: Path) -> dict:
        """Extract notebook-specific metadata."""
        metadata = {
            "valid_notebook": False,
            "cells": 0,
            "code_cells": 0,
            "markdown_cells": 0,
            "output_cells": 0,
        }
        
        try:
            with open(file_path, 'r') as f:
                notebook = json.load(f)
            
            if "cells" in notebook:
                metadata["valid_notebook"] = True
                metadata["cells"] = len(notebook["cells"])
                
                for cell in notebook["cells"]:
                    cell_type = cell.get("cell_type", "")
                    if cell_type == "code":
                        metadata["code_cells"] += 1
                    elif cell_type == "markdown":
                        metadata["markdown_cells"] += 1
                    elif cell_type == "output":
                        metadata["output_cells"] += 1
                
        except Exception:
            pass
        
        return metadata
