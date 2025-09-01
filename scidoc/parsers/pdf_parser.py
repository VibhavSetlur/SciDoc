"""PDF file parser for SciDoc."""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class PdfParser(BaseParser):
    """Parser for PDF files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() == '.pdf'
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """Parse a PDF file and extract metadata."""
        file_path = Path(file_path)
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Extract PDF-specific metadata
        pdf_metadata = self._extract_pdf_metadata(file_path)
        
        metadata = {
            "mime_type": "application/pdf",
            "is_binary": True,
            "extension": ".pdf",
            "stem": file_path.stem,
            **pdf_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.PDF,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=[],
            tags=["pdf", "document"]
        )
    
    def _extract_pdf_metadata(self, file_path: Path) -> dict:
        """Extract PDF-specific metadata."""
        metadata = {
            "pages": 0,
            "title": None,
            "author": None,
            "subject": None,
        }
        
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                metadata["pages"] = len(pdf.pages)
                
                if pdf.metadata:
                    metadata["title"] = pdf.metadata.get('/Title')
                    metadata["author"] = pdf.metadata.get('/Author')
                    metadata["subject"] = pdf.metadata.get('/Subject')
                
        except Exception:
            pass
        
        return metadata
