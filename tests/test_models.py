"""
Tests for SciDoc models.

This module contains tests for the data models.
"""

import pytest
from datetime import datetime
from pathlib import Path

from scidoc.models import (
    FileMetadata,
    ProjectConfig,
    ProjectMetadata,
    FileType,
    ChangeType,
    SummarizerBackend,
)


def test_file_metadata_creation():
    """Test FileMetadata creation."""
    metadata = FileMetadata(
        filename="test.py",
        file_type=FileType.PYTHON,
        size=1024,
        created=datetime.now(),
        modified=datetime.now(),
        checksum="abc123",
    )
    
    assert metadata.filename == "test.py"
    assert metadata.file_type == FileType.PYTHON
    assert metadata.size == 1024
    assert metadata.name == "test.py"
    assert metadata.extension == ".py"


def test_project_config_defaults():
    """Test ProjectConfig default values."""
    config = ProjectConfig()
    
    assert config.summarizer == SummarizerBackend.HUGGINGFACE
    assert config.max_length == 200
    assert config.github_enabled is True


def test_project_metadata_creation():
    """Test ProjectMetadata creation."""
    metadata = ProjectMetadata(
        project_path="/test/project",
        name="test_project",
        files=[],
    )
    
    assert metadata.project_path == "/test/project"
    assert metadata.name == "test_project"
    assert metadata.total_files == 0
