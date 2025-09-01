#!/usr/bin/env python3
"""
Core SciDoc functionality for scientific document analysis and summarization.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from .config import ProjectConfig
from .models import (
    ProjectMetadata,
    FileMetadata,
    AnalysisResult,
    ChangeLog,
    ValidationResult,
    ProvenanceGraph,
    FileType,
    ChangeType,
)
from .summarizer import DocumentSummarizer
from .scientific_parser import ScientificParser
from .document_generator import SciDocGenerator
from .logger import SciDocLogger


class SciDoc:
    """Main SciDoc class for scientific document analysis and summarization."""
    
    def __init__(self, config: Optional[ProjectConfig] = None):
        """
        Initialize SciDoc with configuration.
        
        Args:
            config: Project configuration (optional)
        """
        self.config = config or ProjectConfig()
        self.logger = SciDocLogger()
        self.summarizer = DocumentSummarizer()
        self.scientific_parser = ScientificParser()
        self.document_generator = SciDocGenerator()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def explore(
        self,
        project_path: Union[str, Path],
        force: bool = False,
        verbose: bool = False
    ) -> ProjectMetadata:
        """
        Explore and analyze a project directory.
        
        Args:
            project_path: Path to the project directory
            force: Force re-analysis of all files
            verbose: Enable verbose output
            
        Returns:
            Project metadata
        """
        project_path = Path(project_path)
        
        if not project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {project_path}")
        
        if verbose:
            self.logger.info(f"Exploring project: {project_path}")
        
        # Scan for files
        files = self._scan_files(project_path, verbose=verbose)
        
        # Parse files with scientific analysis
        file_metadata = []
        scientific_analyses = []
        
        for file_path in files:
            try:
                metadata = self._parse_file(file_path, verbose=verbose)
                if metadata:
                    file_metadata.append(metadata)
                    
                    # Scientific analysis
                    try:
                        analysis_dict = self.scientific_parser.parse_file(file_path)
                        if analysis_dict and 'error' not in analysis_dict:
                            # Convert dictionary to AnalysisResult object
                            analysis = AnalysisResult(
                                file_path=str(file_path),
                                file_type=analysis_dict.get('file_type', 'unknown'),
                                analysis_type='scientific_parsing',
                                content=analysis_dict,
                                insights=analysis_dict.get('recommendations', []),
                                recommendations=analysis_dict.get('recommendations', []),
                                quality_score=None,
                                metadata={}
                            )
                            scientific_analyses.append(analysis)
                    except Exception as e:
                        if verbose:
                            self.logger.warning(f"Failed to analyze {file_path}: {e}")
                            
            except Exception as e:
                if verbose:
                    self.logger.warning(f"Failed to parse {file_path}: {e}")
        
        # Create project metadata
        project_metadata = ProjectMetadata(
            project_path=str(project_path),
            files=file_metadata,
            analysis_results=scientific_analyses,
            change_log=ChangeLog(
                change_type=ChangeType.MODIFIED,
                filename="project_analysis"
            ),
            provenance_graph=ProvenanceGraph(),
            created_at=os.path.getctime(project_path),
            last_modified=os.path.getmtime(project_path)
        )
        
        return project_metadata
    
    def summarize(
        self,
        target: Union[str, Path],
        output_format: str = "text",
        verbose: bool = False
    ) -> str:
        """
        Summarize a file or directory.
        
        Args:
            target: File or directory to summarize
            output_format: Output format (text, markdown, json)
            verbose: Enable verbose output
            
        Returns:
            Summary text
        """
        target_path = Path(target)
        
        if not target_path.exists():
            raise FileNotFoundError(f"Target does not exist: {target}")
        
        if target_path.is_file():
            return self.summarizer.summarize_file(target_path, output_format)
        else:
            return self.summarizer.summarize_directory(target_path, output_format)
    
    def generate_scidoc(
        self,
        directory_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        summary_only: bool = False
    ) -> Path:
        """
        Generate a .scidoc file for a directory.
        
        Args:
            directory_path: Directory to analyze
            output_path: Output path for .scidoc file
            summary_only: Generate summary .scidoc only
            
        Returns:
            Path to generated .scidoc file
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")
        
        if summary_only:
            return self.document_generator.generate_summary_scidoc(directory_path, output_path)
        else:
            return self.document_generator.generate_scidoc(directory_path, output_path)
    
    def _scan_files(self, project_path: Path, verbose: bool = False) -> List[Path]:
        """Scan project directory for files to analyze."""
        files = []
        
        for item in project_path.rglob("*"):
            if item.is_file() and not self._should_skip_file(item):
                files.append(item)
        
        if verbose:
            self.logger.info(f"Found {len(files)} files to analyze")
        
        return files
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            ".git", "__pycache__", ".pyc", ".pyo", ".pyd",
            ".DS_Store", "Thumbs.db", ".cache", ".tmp",
            ".log", ".lock", ".bak", ".swp", ".swo"
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _parse_file(self, file_path: Path, verbose: bool = False) -> Optional[FileMetadata]:
        """Parse a single file and extract metadata."""
        try:
            # Basic file metadata
            stat = file_path.stat()
            
            metadata = FileMetadata(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=stat.st_size,
                file_type=self._get_file_type(file_path),
                created_at=stat.st_ctime,
                last_modified=stat.st_mtime,
                analysis_result=None
            )
            
            # Try scientific parsing
            try:
                analysis = self.scientific_parser.parse_file(str(file_path))
                if analysis:
                    metadata.analysis_result = analysis
            except Exception as e:
                if verbose:
                    self.logger.warning(f"Scientific parsing failed for {file_path}: {e}")
            
            return metadata
            
        except Exception as e:
            if verbose:
                self.logger.error(f"Failed to parse {file_path}: {e}")
            return None
    
    def _get_file_type(self, file_path: Path) -> FileType:
        """Determine the type of a file."""
        suffix = file_path.suffix.lower()
        
        # Scientific file types
        if suffix in ['.fastq', '.fq']:
            return FileType.FASTQ
        elif suffix in ['.fasta', '.fa', '.fas']:
            return FileType.FASTA
        elif suffix in ['.vcf']:
            return FileType.VCF
        elif suffix in ['.bam', '.sam']:
            return FileType.BAM
        elif suffix in ['.csv', '.tsv']:
            return FileType.CSV
        elif suffix in ['.json', '.yaml', '.yml']:
            return FileType.JSON
        elif suffix in ['.py']:
            return FileType.PYTHON
        elif suffix in ['.ipynb']:
            return FileType.JUPYTER
        elif suffix in ['.md', '.txt']:
            return FileType.TEXT
        elif suffix in ['.pdf']:
            return FileType.PDF
        else:
            return FileType.GENERIC
