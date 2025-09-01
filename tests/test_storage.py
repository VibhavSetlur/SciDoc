"""
Tests for SciDoc storage modules.
"""

import pytest
import tempfile
import os
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

from scidoc.storage.metadata_store import MetadataStore
from scidoc.storage.cache_store import CacheStore
from scidoc.storage.file_store import FileStore
from scidoc.models import ProjectMetadata, FileMetadata, ProjectConfig


class TestMetadataStore:
    """Test the metadata store."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.metadata_dir = Path(self.temp_dir) / ".metadata"
        self.metadata_dir.mkdir()
        self.store = MetadataStore(self.metadata_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_metadata(self):
        """Test saving and loading metadata."""
        # Create test metadata
        files = [
            FileMetadata(filename="test1.py", file_type="PYTHON", size=1000),
            FileMetadata(filename="test2.csv", file_type="CSV", size=2000)
        ]
        
        metadata = ProjectMetadata(
            path=self.temp_dir,
            name="test_project",
            files=files
        )
        
        # Save metadata
        self.store.save_metadata(metadata)
        
        # Check that files exist
        json_path = self.metadata_dir / "metadata.json"
        pickle_path = self.metadata_dir / "metadata.pkl"
        
        assert json_path.exists()
        assert pickle_path.exists()
        
        # Load metadata
        loaded_metadata = self.store.load_metadata()
        
        assert loaded_metadata is not None
        assert loaded_metadata.name == metadata.name
        assert len(loaded_metadata.files) == len(metadata.files)
        assert loaded_metadata.files[0].filename == "test1.py"
        assert loaded_metadata.files[1].filename == "test2.csv"
    
    def test_metadata_exists(self):
        """Test checking if metadata exists."""
        # Should not exist initially
        assert not self.store.metadata_exists()
        
        # Create and save metadata
        metadata = ProjectMetadata(
            path=self.temp_dir,
            name="test_project",
            files=[]
        )
        self.store.save_metadata(metadata)
        
        # Should exist now
        assert self.store.metadata_exists()
    
    def test_delete_metadata(self):
        """Test deleting metadata."""
        # Create and save metadata
        metadata = ProjectMetadata(
            path=self.temp_dir,
            name="test_project",
            files=[]
        )
        self.store.save_metadata(metadata)
        
        # Verify it exists
        assert self.store.metadata_exists()
        
        # Delete metadata
        self.store.delete_metadata()
        
        # Verify it's gone
        assert not self.store.metadata_exists()
    
    def test_get_metadata_info(self):
        """Test getting metadata information."""
        # Create and save metadata
        files = [
            FileMetadata(filename="test1.py", file_type="PYTHON", size=1000),
            FileMetadata(filename="test2.csv", file_type="CSV", size=2000)
        ]
        
        metadata = ProjectMetadata(
            path=self.temp_dir,
            name="test_project",
            files=files
        )
        self.store.save_metadata(metadata)
        
        # Get info
        info = self.store.get_metadata_info()
        
        assert isinstance(info, dict)
        assert "json_size" in info
        assert "pickle_size" in info
        assert "last_modified" in info
        assert info["json_size"] > 0
        assert info["pickle_size"] > 0
    
    def test_load_nonexistent_metadata(self):
        """Test loading metadata that doesn't exist."""
        loaded_metadata = self.store.load_metadata()
        assert loaded_metadata is None


class TestCacheStore:
    """Test the cache store."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / ".scidoc_cache"
        self.cache_dir.mkdir()
        self.store = CacheStore(self.cache_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_set_and_get_cache(self):
        """Test setting and getting cache values."""
        # Set cache value
        key = "test_key"
        value = {"data": "test_value", "timestamp": 1234567890}
        
        self.store.set(key, value)
        
        # Get cache value
        retrieved_value = self.store.get(key)
        
        assert retrieved_value is not None
        assert retrieved_value["data"] == value["data"]
        assert retrieved_value["timestamp"] == value["timestamp"]
    
    def test_get_nonexistent_cache(self):
        """Test getting cache value that doesn't exist."""
        value = self.store.get("nonexistent_key")
        assert value is None
    
    def test_delete_cache(self):
        """Test deleting cache value."""
        # Set cache value
        key = "test_key"
        value = {"data": "test_value"}
        
        self.store.set(key, value)
        
        # Verify it exists
        assert self.store.get(key) is not None
        
        # Delete cache value
        self.store.delete(key)
        
        # Verify it's gone
        assert self.store.get(key) is None
    
    def test_clear_cache(self):
        """Test clearing all cache."""
        # Set multiple cache values
        self.store.set("key1", {"data": "value1"})
        self.store.set("key2", {"data": "value2"})
        self.store.set("key3", {"data": "value3"})
        
        # Verify they exist
        assert self.store.get("key1") is not None
        assert self.store.get("key2") is not None
        assert self.store.get("key3") is not None
        
        # Clear cache
        self.store.clear()
        
        # Verify all are gone
        assert self.store.get("key1") is None
        assert self.store.get("key2") is None
        assert self.store.get("key3") is None
    
    def test_get_cache_info(self):
        """Test getting cache information."""
        # Set some cache values
        self.store.set("key1", {"data": "value1"})
        self.store.set("key2", {"data": "value2"})
        
        # Get info
        info = self.store.get_cache_info()
        
        assert isinstance(info, dict)
        assert "total_files" in info
        assert "total_size" in info
        assert "cache_dir" in info
        assert info["total_files"] >= 2
        assert info["total_size"] > 0
        assert str(self.cache_dir) in info["cache_dir"]
    
    def test_cache_persistence(self):
        """Test that cache persists between store instances."""
        # Set cache value
        key = "test_key"
        value = {"data": "persistent_value"}
        
        self.store.set(key, value)
        
        # Create new store instance
        new_store = CacheStore(self.cache_dir)
        
        # Get value from new store
        retrieved_value = new_store.get(key)
        
        assert retrieved_value is not None
        assert retrieved_value["data"] == value["data"]


class TestFileStore:
    """Test the file store."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.metadata_dir = Path(self.temp_dir) / ".metadata"
        self.metadata_dir.mkdir()
        self.store = FileStore(self.metadata_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_file_metadata(self):
        """Test saving and loading file metadata."""
        # Create test file metadata
        file_metadata = FileMetadata(
            filename="test_file.py",
            file_type="PYTHON",
            size=1000,
            content_metadata={"functions": 5, "classes": 2}
        )
        
        # Save file metadata
        self.store.save_file_metadata(file_metadata)
        
        # Check that file exists
        file_path = self.metadata_dir / "files" / "test_file.py.json"
        assert file_path.exists()
        
        # Load file metadata
        loaded_metadata = self.store.load_file_metadata("test_file.py")
        
        assert loaded_metadata is not None
        assert loaded_metadata.filename == file_metadata.filename
        assert loaded_metadata.file_type == file_metadata.file_type
        assert loaded_metadata.size == file_metadata.size
        assert loaded_metadata.content_metadata["functions"] == 5
        assert loaded_metadata.content_metadata["classes"] == 2
    
    def test_load_nonexistent_file_metadata(self):
        """Test loading file metadata that doesn't exist."""
        loaded_metadata = self.store.load_file_metadata("nonexistent_file.py")
        assert loaded_metadata is None
    
    def test_delete_file_metadata(self):
        """Test deleting file metadata."""
        # Create and save file metadata
        file_metadata = FileMetadata(
            filename="test_file.py",
            file_type="PYTHON",
            size=1000
        )
        self.store.save_file_metadata(file_metadata)
        
        # Verify it exists
        assert self.store.load_file_metadata("test_file.py") is not None
        
        # Delete file metadata
        self.store.delete_file_metadata("test_file.py")
        
        # Verify it's gone
        assert self.store.load_file_metadata("test_file.py") is None
    
    def test_list_file_metadata(self):
        """Test listing file metadata."""
        # Create and save multiple file metadata
        files = [
            FileMetadata(filename="file1.py", file_type="PYTHON", size=1000),
            FileMetadata(filename="file2.csv", file_type="CSV", size=2000),
            FileMetadata(filename="file3.md", file_type="MARKDOWN", size=500)
        ]
        
        for file_metadata in files:
            self.store.save_file_metadata(file_metadata)
        
        # List file metadata
        file_list = self.store.list_file_metadata()
        
        assert isinstance(file_list, list)
        assert len(file_list) == 3
        assert "file1.py" in file_list
        assert "file2.csv" in file_list
        assert "file3.md" in file_list
    
    def test_file_metadata_with_special_characters(self):
        """Test file metadata with special characters in filename."""
        # Create file metadata with special characters
        file_metadata = FileMetadata(
            filename="test file (v1.0).py",
            file_type="PYTHON",
            size=1000
        )
        
        # Save file metadata
        self.store.save_file_metadata(file_metadata)
        
        # Load file metadata
        loaded_metadata = self.store.load_file_metadata("test file (v1.0).py")
        
        assert loaded_metadata is not None
        assert loaded_metadata.filename == "test file (v1.0).py"
    
    def test_file_metadata_directory_structure(self):
        """Test that files directory is created automatically."""
        # The files directory should be created when needed
        file_metadata = FileMetadata(
            filename="test_file.py",
            file_type="PYTHON",
            size=1000
        )
        
        self.store.save_file_metadata(file_metadata)
        
        # Check that files directory exists
        files_dir = self.metadata_dir / "files"
        assert files_dir.exists()
        assert files_dir.is_dir()


class TestStorageIntegration:
    """Test storage integration scenarios."""
    
    def test_metadata_store_with_large_files(self):
        """Test metadata store with large number of files."""
        temp_dir = tempfile.mkdtemp()
        metadata_dir = Path(temp_dir) / ".metadata"
        metadata_dir.mkdir()
        store = MetadataStore(metadata_dir)
        
        try:
            # Create many file metadata objects
            files = []
            for i in range(100):
                file_metadata = FileMetadata(
                    filename=f"file_{i}.py",
                    file_type="PYTHON",
                    size=1000 + i,
                    content_metadata={"functions": i % 10, "classes": i % 5}
                )
                files.append(file_metadata)
            
            # Create project metadata
            metadata = ProjectMetadata(
                path=temp_dir,
                name="large_project",
                files=files
            )
            
            # Save metadata
            store.save_metadata(metadata)
            
            # Load metadata
            loaded_metadata = store.load_metadata()
            
            assert loaded_metadata is not None
            assert len(loaded_metadata.files) == 100
            assert loaded_metadata.files[0].filename == "file_0.py"
            assert loaded_metadata.files[99].filename == "file_99.py"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_cache_store_with_complex_objects(self):
        """Test cache store with complex Python objects."""
        temp_dir = tempfile.mkdtemp()
        cache_dir = Path(temp_dir) / ".scidoc_cache"
        cache_dir.mkdir()
        store = CacheStore(cache_dir)
        
        try:
            # Create complex object
            complex_object = {
                "nested_dict": {
                    "list": [1, 2, 3, {"nested": "value"}],
                    "tuple": (1, 2, 3),
                    "set": {1, 2, 3}
                },
                "datetime": "2023-01-01T00:00:00",
                "metadata": {
                    "files": [
                        {"name": "file1.py", "size": 1000},
                        {"name": "file2.csv", "size": 2000}
                    ]
                }
            }
            
            # Set cache
            store.set("complex_key", complex_object)
            
            # Get cache
            retrieved_object = store.get("complex_key")
            
            assert retrieved_object is not None
            assert retrieved_object["nested_dict"]["list"][0] == 1
            assert retrieved_object["nested_dict"]["list"][3]["nested"] == "value"
            assert len(retrieved_object["metadata"]["files"]) == 2
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_file_store_with_different_file_types(self):
        """Test file store with different file types."""
        temp_dir = tempfile.mkdtemp()
        metadata_dir = Path(temp_dir) / ".metadata"
        metadata_dir.mkdir()
        store = FileStore(metadata_dir)
        
        try:
            # Create different file types
            file_types = [
                ("python_file.py", "PYTHON", {"functions": 10, "classes": 3}),
                ("data_file.csv", "CSV", {"columns": 5, "estimated_rows": 1000}),
                ("config_file.yaml", "YAML", {"structure": "object"}),
                ("notebook.ipynb", "JUPYTER", {"total_cells": 15, "code_cells": 10}),
                ("document.md", "MARKDOWN", {"headings": 5, "links": 3})
            ]
            
            for filename, file_type, content_metadata in file_types:
                file_metadata = FileMetadata(
                    filename=filename,
                    file_type=file_type,
                    size=1000,
                    content_metadata=content_metadata
                )
                
                store.save_file_metadata(file_metadata)
            
            # List all files
            file_list = store.list_file_metadata()
            
            assert len(file_list) == 5
            assert "python_file.py" in file_list
            assert "data_file.csv" in file_list
            assert "config_file.yaml" in file_list
            assert "notebook.ipynb" in file_list
            assert "document.md" in file_list
            
            # Load specific file
            python_file = store.load_file_metadata("python_file.py")
            assert python_file.file_type == "PYTHON"
            assert python_file.content_metadata["functions"] == 10
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestStorageErrorHandling:
    """Test storage error handling."""
    
    def test_metadata_store_corrupted_json(self):
        """Test handling corrupted JSON metadata."""
        temp_dir = tempfile.mkdtemp()
        metadata_dir = Path(temp_dir) / ".metadata"
        metadata_dir.mkdir()
        store = MetadataStore(metadata_dir)
        
        try:
            # Create corrupted JSON file
            json_path = metadata_dir / "metadata.json"
            with open(json_path, 'w') as f:
                f.write('{"invalid": json content')
            
            # Should handle gracefully
            loaded_metadata = store.load_metadata()
            assert loaded_metadata is None
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_cache_store_invalid_key(self):
        """Test cache store with invalid keys."""
        temp_dir = tempfile.mkdtemp()
        cache_dir = Path(temp_dir) / ".scidoc_cache"
        cache_dir.mkdir()
        store = CacheStore(cache_dir)
        
        try:
            # Test with None key
            with pytest.raises(ValueError):
                store.set(None, {"data": "value"})
            
            # Test with empty key
            with pytest.raises(ValueError):
                store.set("", {"data": "value"})
            
            # Test with invalid key characters
            with pytest.raises(ValueError):
                store.set("invalid/key", {"data": "value"})
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_file_store_invalid_filename(self):
        """Test file store with invalid filenames."""
        temp_dir = tempfile.mkdtemp()
        metadata_dir = Path(temp_dir) / ".metadata"
        metadata_dir.mkdir()
        store = FileStore(metadata_dir)
        
        try:
            # Test with None filename
            file_metadata = FileMetadata(
                filename=None,
                file_type="PYTHON",
                size=1000
            )
            
            with pytest.raises(ValueError):
                store.save_file_metadata(file_metadata)
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
