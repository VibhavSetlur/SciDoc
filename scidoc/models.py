"""
Core data models for SciDoc.

This module defines the data structures used throughout the application
for representing files, metadata, configurations, and results.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class FileType(str, Enum):
    """Supported file types for analysis."""
    
    # Bioinformatics files
    FASTQ = "fastq"
    BAM = "bam"
    SAM = "sam"
    VCF = "vcf"
    FASTA = "fasta"
    
    # Data files
    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    YAML = "yaml"
    EXCEL = "excel"
    
    # Code files
    PYTHON = "python"
    R = "r"
    JUPYTER = "jupyter"
    MARKDOWN = "markdown"
    
    # Other files
    IMAGE = "image"
    PDF = "pdf"
    LOG = "log"
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class ChangeType(str, Enum):
    """Types of file changes."""
    
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class SummarizerBackend(str, Enum):
    """Available summarizer backends."""
    
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    CUSTOM = "custom"


class FileMetadata(BaseModel):
    """Metadata for a single file."""
    
    file_path: str = Field(..., description="File path")
    file_name: str = Field(..., description="File name")
    file_size: int = Field(..., description="File size in bytes")
    file_type: FileType = Field(..., description="Detected file type")
    created_at: float = Field(..., description="File creation time (timestamp)")
    last_modified: float = Field(..., description="File modification time (timestamp)")
    analysis_result: Optional["AnalysisResult"] = Field(None, description="Scientific analysis result")
    
    @property
    def path(self) -> Path:
        """Get file path as Path object."""
        return Path(self.file_path)
    
    @property
    def name(self) -> str:
        """Get just the filename without path."""
        return self.path.name
    
    @property
    def extension(self) -> str:
        """Get file extension."""
        return self.path.suffix.lower()
    
    @property
    def is_binary(self) -> bool:
        """Check if file is binary."""
        return self.file_type in [FileType.BINARY, FileType.IMAGE, FileType.BAM, FileType.SAM]


class ProjectConfig(BaseModel):
    """Configuration for a SciDoc project."""
    
    # Summarizer configuration
    summarizer: SummarizerBackend = Field(
        default=SummarizerBackend.HUGGINGFACE,
        description="Summarizer backend to use"
    )
    model_path: Optional[str] = Field(
        default="./models/flan-t5-base",
        description="Path to local model or model name"
    )
    max_length: int = Field(
        default=200,
        description="Maximum summary length"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation"
    )
    
    # GitHub configuration
    github_enabled: bool = Field(
        default=True,
        description="Enable GitHub integration"
    )
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub API token"
    )
    github_webhook_secret: Optional[str] = Field(
        default=None,
        description="GitHub webhook secret"
    )
    
    # Storage configuration
    metadata_dir: str = Field(
        default=".metadata",
        description="Directory for storing metadata"
    )
    cache_dir: str = Field(
        default=".scidoc_cache",
        description="Directory for caching"
    )
    
    # Parser configuration
    enabled_parsers: List[str] = Field(
        default_factory=lambda: [
            "fastq", "bam", "vcf", "csv", "json", "yaml",
            "python", "jupyter", "markdown", "image", "pdf"
        ],
        description="List of enabled parsers"
    )
    
    # Validation configuration
    validation_rules_file: Optional[str] = Field(
        default=None,
        description="Path to validation rules file"
    )
    auto_fix: bool = Field(
        default=False,
        description="Automatically fix validation issues"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "ProjectConfig":
        """Load configuration from file."""
        import yaml
        
        config_path = Path(config_path)
        if not config_path.exists():
            return cls()
        
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        import yaml
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)


class ValidationResult(BaseModel):
    """Result of a validation check."""
    
    filename: str = Field(..., description="Validated file")
    rule_name: str = Field(..., description="Validation rule name")
    passed: bool = Field(..., description="Whether validation passed")
    message: str = Field(..., description="Validation message")
    severity: str = Field(default="error", description="Severity level")
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for fixing issues"
    )
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error."""
        return self.severity == "error"
    
    @property
    def is_warning(self) -> bool:
        """Check if this is a warning."""
        return self.severity == "warning"
    
    @property
    def is_info(self) -> bool:
        """Check if this is informational."""
        return self.severity == "info"


class AnalysisResult(BaseModel):
    """Result of scientific analysis of a file."""
    
    file_path: str = Field(..., description="Path to analyzed file")
    file_type: str = Field(..., description="Type of scientific file")
    analysis_type: str = Field(..., description="Type of analysis performed")
    content: Dict[str, Any] = Field(default_factory=dict, description="Analysis content")
    insights: List[str] = Field(default_factory=list, description="Key insights")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    quality_score: Optional[float] = Field(None, description="Data quality score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChangeLog(BaseModel):
    """Log of changes in a project."""
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Change timestamp")
    change_type: ChangeType = Field(..., description="Type of change")
    filename: str = Field(..., description="Changed file")
    old_filename: Optional[str] = Field(None, description="Previous filename for renames")
    size_change: int = Field(0, description="Size change in bytes")
    summary: Optional[str] = Field(None, description="AI-generated summary of changes")
    
    @property
    def is_addition(self) -> bool:
        """Check if this is a file addition."""
        return self.change_type == ChangeType.ADDED
    
    @property
    def is_deletion(self) -> bool:
        """Check if this is a file deletion."""
        return self.change_type == ChangeType.DELETED
    
    @property
    def is_modification(self) -> bool:
        """Check if this is a file modification."""
        return self.change_type == ChangeType.MODIFIED
    
    @property
    def is_rename(self) -> bool:
        """Check if this is a file rename."""
        return self.change_type == ChangeType.RENAMED


class ProvenanceGraph(BaseModel):
    """Graph representation of file dependencies and relationships."""
    
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Graph nodes (files)"
    )
    edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Graph edges (dependencies)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Graph metadata"
    )
    
    def add_node(self, file_metadata: FileMetadata) -> None:
        """Add a file as a node in the graph."""
        node = {
            "id": file_metadata.file_path,
            "label": file_metadata.file_name,
            "type": file_metadata.file_type.value,
            "size": file_metadata.file_size,
            "metadata": {}
        }
        self.nodes.append(node)
    
    def add_edge(self, source: str, target: str, relationship: str = "depends_on") -> None:
        """Add a dependency edge between files."""
        edge = {
            "source": source,
            "target": target,
            "relationship": relationship
        }
        self.edges.append(edge)
    
    def to_networkx(self):
        """Convert to NetworkX graph."""
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes:
            G.add_node(node["id"], **node)
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge["source"], edge["target"], **edge)
        
        return G
    
    def to_json(self) -> str:
        """Export as JSON string."""
        import json
        return json.dumps(self.dict(), indent=2)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save graph to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(self.to_json())


class ProjectMetadata(BaseModel):
    """Metadata for an entire project."""
    
    project_path: str = Field(..., description="Project root path")
    files: List[FileMetadata] = Field(
        default_factory=list,
        description="List of file metadata"
    )
    analysis_results: List[AnalysisResult] = Field(
        default_factory=list,
        description="List of scientific analysis results"
    )
    change_log: ChangeLog = Field(default_factory=ChangeLog, description="Change log")
    provenance_graph: ProvenanceGraph = Field(default_factory=ProvenanceGraph, description="Provenance graph")
    created_at: float = Field(..., description="Project creation time (timestamp)")
    last_modified: float = Field(..., description="Last modification time (timestamp)")
    
    @property
    def name(self) -> str:
        """Get project name from path."""
        return Path(self.project_path).name
    
    @property
    def total_files(self) -> int:
        """Get total number of files."""
        return len(self.files)
    
    @property
    def total_size(self) -> int:
        """Get total size in bytes."""
        return sum(f.file_size for f in self.files)
    
    @property
    def file_types(self) -> Dict[FileType, int]:
        """Get count of files by type."""
        file_types = {}
        for file_meta in self.files:
            file_types[file_meta.file_type] = file_types.get(file_meta.file_type, 0) + 1
        return file_types
    
    def get_files_by_type(self, file_type: FileType) -> List[FileMetadata]:
        """Get all files of a specific type."""
        return [f for f in self.files if f.file_type == file_type]
    
    def get_largest_files(self, limit: int = 10) -> List[FileMetadata]:
        """Get the largest files in the project."""
        return sorted(self.files, key=lambda x: x.file_size, reverse=True)[:limit]
    
    def get_recent_files(self, limit: int = 10) -> List[FileMetadata]:
        """Get the most recently modified files."""
        return sorted(self.files, key=lambda x: x.last_modified, reverse=True)[:limit]
