"""
Python file parser for SciDoc.

This parser extracts metadata from Python files including imports,
functions, classes, and code structure.
"""

from pathlib import Path
from typing import Union

from .base_parser import BaseParser
from ..models import FileMetadata, FileType


class PythonParser(BaseParser):
    """Parser for Python files."""
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this parser can parse the file, False otherwise
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() == '.py'
    
    def parse(self, file_path: Union[str, Path]) -> FileMetadata:
        """
        Parse a Python file and extract metadata.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            FileMetadata object containing extracted information
        """
        file_path = Path(file_path)
        
        # Get basic metadata
        basic_metadata = self.get_basic_metadata(file_path)
        
        # Read file content
        content = self._read_text_content(file_path)
        
        # Extract Python-specific metadata
        python_metadata = self._extract_python_metadata(content)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(content)
        
        # Extract tags
        tags = self._extract_tags(content, str(file_path))
        
        # Combine metadata
        metadata = {
            "mime_type": "text/x-python",
            "encoding": "utf-8",
            "is_binary": False,
            "extension": ".py",
            "stem": file_path.stem,
            **python_metadata
        }
        
        return FileMetadata(
            filename=str(file_path),
            file_type=FileType.PYTHON,
            size=basic_metadata["size"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            checksum=basic_metadata["checksum"],
            metadata=metadata,
            dependencies=dependencies,
            tags=tags
        )
    
    def _extract_python_metadata(self, content: str) -> dict:
        """
        Extract Python-specific metadata from file content.
        
        Args:
            content: File content as string
            
        Returns:
            Dictionary with Python metadata
        """
        import ast
        import re
        
        metadata = {
            "functions": [],
            "classes": [],
            "imports": [],
            "docstrings": [],
            "lines_of_code": len(content.splitlines()),
            "complexity": 0,
        }
        
        try:
            # Parse AST
            tree = ast.parse(content)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metadata["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args),
                        "docstring": ast.get_docstring(node),
                    })
                
                elif isinstance(node, ast.ClassDef):
                    metadata["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                        "docstring": ast.get_docstring(node),
                    })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            metadata["imports"].append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            if module:
                                metadata["imports"].append(f"{module}.{alias.name}")
                            else:
                                metadata["imports"].append(alias.name)
            
            # Extract docstrings
            docstring_pattern = r'"""(.*?)"""'
            docstrings = re.findall(docstring_pattern, content, re.DOTALL)
            metadata["docstrings"] = [d.strip() for d in docstrings if d.strip()]
            
            # Calculate complexity (simple metric)
            complexity = 0
            complexity_patterns = [
                r'\bif\b',
                r'\bfor\b',
                r'\bwhile\b',
                r'\band\b',
                r'\bor\b',
                r'\bexcept\b',
                r'\bwith\b',
            ]
            
            for pattern in complexity_patterns:
                complexity += len(re.findall(pattern, content))
            
            metadata["complexity"] = complexity
            
        except SyntaxError:
            # Handle syntax errors gracefully
            metadata["syntax_error"] = True
        
        return metadata
