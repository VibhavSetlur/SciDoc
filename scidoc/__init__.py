"""
SciDoc: A comprehensive science researcher assistant tool.

This package provides tools for tracking GitHub changes, managing documentation,
analyzing data files, and serving as a terminal-based chat assistant.
"""

__version__ = "0.1.0"
__author__ = "SciDoc Team"
__email__ = "scidoc@example.com"

from .core import SciDoc
from .models import (
    FileMetadata,
    ProjectConfig,
    ProjectMetadata,
    ChangeLog,
    ValidationResult,
)
from .config import get_config, load_config

__all__ = [
    "SciDoc",
    "FileMetadata",
    "ProjectConfig",
    "ProjectMetadata",
    "ChangeLog",
    "ValidationResult",
    "get_config",
    "load_config",
]
