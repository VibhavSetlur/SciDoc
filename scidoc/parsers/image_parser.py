"""Image file parser for SciDoc."""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class ImageParser(BaseParser):
    """Parser for image files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse an image file and extract metadata."""
        file_path = Path(file_path)
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Extract image-specific metadata
        image_metadata = self._extract_image_metadata(file_path)
        
        metadata = {
            "mime_type": "image/" + file_path.suffix.lower()[1:],
            "is_binary": True,
            "extension": file_path.suffix.lower(),
            "stem": file_path.stem,
            **image_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.IMAGE,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["image", "visual"]
        )
    
    def _extract_image_metadata(self, file_path: Path) -> dict:
        """Extract image-specific metadata."""
        metadata = {
            "width": 0,
            "height": 0,
            "format": file_path.suffix.lower()[1:],
        }
        
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["format"] = img.format
                metadata["mode"] = img.mode
                
        except Exception:
            pass
        
        return metadata
