"""
Example runs and integration tests for SciDoc.

This module contains comprehensive examples that demonstrate real-world usage
of the SciDoc tool with different project types and scenarios.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from scidoc import SciDoc
from scidoc.models import ProjectConfig, FileMetadata
from scidoc.parsers import create_parser
from scidoc.summarizers import create_summarizer


class TestExampleRuns:
    """Test example runs demonstrating SciDoc functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_dir = Path(self.temp_dir) / "example_project"
        self.project_dir.mkdir()
        
        # Create example project structure
        self._create_example_project()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_example_project(self):
        """Create an example project with various file types."""
        # Create Python files
        python_code = '''
def analyze_data(data):
    """Analyze the given data."""
    return data.mean()

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, data):
        return data * 2
'''
        
        with open(self.project_dir / "analysis.py", "w") as f:
            f.write(python_code)
        
        # Create CSV data file
        csv_data = """sample_id,value,group
sample_1,10.5,control
sample_2,15.2,treatment
sample_3,12.8,control
sample_4,18.1,treatment
"""
        
        with open(self.project_dir / "data.csv", "w") as f:
            f.write(csv_data)
        
        # Create Jupyter notebook
        notebook_content = '''{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Data Analysis Project\\n", "This notebook contains our analysis."]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["import pandas as pd\\n", "df = pd.read_csv('data.csv')\\n", "print(df.head())"],
            "outputs": []
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}'''
        
        with open(self.project_dir / "analysis.ipynb", "w") as f:
            f.write(notebook_content)
        
        # Create README
        readme_content = """# Example Project

This is an example bioinformatics project demonstrating SciDoc functionality.

## Files
- `analysis.py`: Main analysis script
- `data.csv`: Sample data
- `analysis.ipynb`: Jupyter notebook with analysis

## Usage
Run the analysis script to process the data.
"""
        
        with open(self.project_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        # Create configuration file
        config_content = """# Project Configuration
project_name: example_project
description: Example bioinformatics project
version: 1.0.0
"""
        
        with open(self.project_dir / "config.yaml", "w") as f:
            f.write(config_content)
    
    def test_example_bioinformatics_project(self):
        """Test exploring a bioinformatics project."""
        # Initialize SciDoc
        scidoc = SciDoc(self.project_dir)
        
        # Explore the project
        with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "This is a bioinformatics project with data analysis scripts."
            
            result = scidoc.explore()
            
            assert result is not None
            assert len(result.files) == 5  # analysis.py, data.csv, analysis.ipynb, README.md, config.yaml
            assert result.total_files == 5
            
            # Check file types
            file_types = [f.file_type for f in result.files]
            assert "PYTHON" in file_types
            assert "CSV" in file_types
            assert "JUPYTER" in file_types
            assert "MARKDOWN" in file_types
            assert "YAML" in file_types
    
    def test_example_project_with_changes(self):
        """Test detecting and summarizing changes in a project."""
        # Initialize SciDoc and do initial exploration
        scidoc = SciDoc(self.project_dir)
        
        with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Initial project exploration."
            initial_result = scidoc.explore()
        
        # Make some changes to the project
        with open(self.project_dir / "new_script.py", "w") as f:
            f.write("print('New script')")
        
        with open(self.project_dir / "analysis.py", "a") as f:
            f.write("\n# New function\ndef new_function():\n    pass\n")
        
        # Update the project
        with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Changes detected: new script added and analysis updated."
            changes = scidoc.update()
            
            assert changes is not None
            assert len(changes) > 0
    
    def test_example_project_logging(self):
        """Test generating daily logs for a project."""
        scidoc = SciDoc(self.project_dir)
        
        # Generate a log entry
        with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Daily summary: Project contains 5 files with analysis scripts and data."
            
            log_entry = scidoc.generate_log()
            
            assert log_entry is not None
            assert "Daily summary" in log_entry
    
    def test_example_project_graph(self):
        """Test building provenance graph for a project."""
        scidoc = SciDoc(self.project_dir)
        
        # Build the graph
        graph = scidoc.build_graph()
        
        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.edges) >= 0  # May have no edges initially
    
    def test_example_project_validation(self):
        """Test validating a project against rules."""
        scidoc = SciDoc(self.project_dir)
        
        # Define validation rules
        rules = {
            "required_files": ["README.md", "analysis.py"],
            "file_patterns": {
                "*.py": {"max_size": 10000},
                "*.csv": {"min_size": 100}
            }
        }
        
        # Validate the project
        validation_results = scidoc.validate(rules)
        
        assert validation_results is not None
        assert len(validation_results) > 0
    
    def test_example_project_chat(self):
        """Test chat functionality with the project."""
        scidoc = SciDoc(self.project_dir)
        
        # Initialize the project first
        with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Project initialized."
            scidoc.explore()
        
        # Test chat queries
        queries = [
            "How many Python files are in this project?",
            "What is the largest file?",
            "Show me all CSV files"
        ]
        
        for query in queries:
            with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
                mock_summarize.return_value = f"Answer to: {query}"
                response = scidoc.chat(query)
                
                assert response is not None
                assert len(response) > 0


class TestBioinformaticsExample:
    """Test bioinformatics-specific example scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.bio_dir = Path(self.temp_dir) / "bioinformatics_project"
        self.bio_dir.mkdir()
        
        # Create bioinformatics project structure
        self._create_bioinformatics_project()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_bioinformatics_project(self):
        """Create a bioinformatics project with typical file types."""
        # Create FASTQ file
        fastq_content = """@seq1
ACGTACGTACGTACGTACGT
+
IIIIIIIIIIIIIIIIIIII
@seq2
GCTAGCTAGCTAGCTAGCTA
+
IIIIIIIIIIIIIIIIIIII
@seq3
TGCATGCATGCATGCATGCA
+
IIIIIIIIIIIIIIIIIIII
"""
        
        with open(self.bio_dir / "sample.fastq", "w") as f:
            f.write(fastq_content)
        
        # Create VCF file
        vcf_content = """##fileformat=VCFv4.2
##source=example
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	100	rs1	A	T	50	PASS	NS=3
chr1	200	rs2	C	G	60	PASS	NS=3
chr2	150	rs3	T	A	40	PASS	NS=3
"""
        
        with open(self.bio_dir / "variants.vcf", "w") as f:
            f.write(vcf_content)
        
        # Create analysis script
        analysis_script = '''#!/usr/bin/env python3
"""
Bioinformatics analysis script.
"""

import pandas as pd
import numpy as np

def analyze_variants(vcf_file):
    """Analyze variant data from VCF file."""
    # Read VCF file
    variants = pd.read_csv(vcf_file, sep='\\t', comment='#')
    return variants

def process_sequences(fastq_file):
    """Process FASTQ sequences."""
    # Process sequences
    sequences = []
    with open(fastq_file, 'r') as f:
        for line in f:
            if line.startswith('@'):
                continue
            sequences.append(line.strip())
    return sequences

if __name__ == "__main__":
    # Run analysis
    variants = analyze_variants("variants.vcf")
    sequences = process_sequences("sample.fastq")
    print(f"Found {len(variants)} variants and {len(sequences)} sequences")
'''
        
        with open(self.bio_dir / "bio_analysis.py", "w") as f:
            f.write(analysis_script)
        
        # Create results file
        results_content = """Sample_ID,Total_Variants,Quality_Score
sample_1,150,0.95
sample_2,200,0.92
sample_3,175,0.88
"""
        
        with open(self.bio_dir / "results.csv", "w") as f:
            f.write(results_content)
    
    def test_bioinformatics_project_exploration(self):
        """Test exploring a bioinformatics project."""
        scidoc = SciDoc(self.bio_dir)
        
        with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Bioinformatics project with sequencing data and variant analysis."
            
            result = scidoc.explore()
            
            assert result is not None
            assert len(result.files) == 4
            
            # Check for bioinformatics file types
            file_types = [f.file_type for f in result.files]
            assert "FASTQ" in file_types
            assert "VCF" in file_types
            assert "PYTHON" in file_types
            assert "CSV" in file_types
    
    def test_bioinformatics_project_validation(self):
        """Test validating a bioinformatics project."""
        scidoc = SciDoc(self.bio_dir)
        
        # Define bioinformatics-specific validation rules
        rules = {
            "required_files": ["bio_analysis.py", "results.csv"],
            "file_patterns": {
                "*.fastq": {"min_size": 100},
                "*.vcf": {"min_size": 50},
                "*.py": {"max_size": 5000}
            }
        }
        
        validation_results = scidoc.validate(rules)
        
        assert validation_results is not None
        assert len(validation_results) > 0


class TestLargeProjectExample:
    """Test handling large projects with many files."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.large_dir = Path(self.temp_dir) / "large_project"
        self.large_dir.mkdir()
        
        # Create large project structure
        self._create_large_project()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_large_project(self):
        """Create a large project with many files."""
        # Create multiple directories
        for i in range(5):
            subdir = self.large_dir / f"module_{i}"
            subdir.mkdir()
            
            # Create Python files in each subdirectory
            for j in range(10):
                python_file = subdir / f"script_{j}.py"
                with open(python_file, "w") as f:
                    f.write(f"# Module {i}, Script {j}\nprint('Hello from script {j}')\n")
            
            # Create data files
            data_file = subdir / f"data_{i}.csv"
            with open(data_file, "w") as f:
                f.write(f"id,value\n1,{i*10}\n2,{i*20}\n3,{i*30}\n")
        
        # Create configuration files
        config_content = """# Configuration
module_count: 5
scripts_per_module: 10
"""
        
        with open(self.large_dir / "config.yaml", "w") as f:
            f.write(config_content)
    
    def test_large_project_exploration(self):
        """Test exploring a large project."""
        scidoc = SciDoc(self.large_dir)
        
        with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Large project with multiple modules and scripts."
            
            result = scidoc.explore()
            
            assert result is not None
            assert result.total_files >= 55  # 5 modules * (10 scripts + 1 data file) + 1 config
            assert result.total_size > 0
    
    def test_large_project_performance(self):
        """Test performance with large projects."""
        import time
        
        scidoc = SciDoc(self.large_dir)
        
        start_time = time.time()
        
        with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Performance test summary."
            result = scidoc.explore()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert result is not None
        assert processing_time < 10  # Should complete within 10 seconds


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""
    
    def test_full_workflow_example(self):
        """Test a complete workflow from project creation to analysis."""
        # Create a temporary project
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "workflow_project"
        project_dir.mkdir()
        
        try:
            # Create project files
            with open(project_dir / "main.py", "w") as f:
                f.write("print('Main script')\n")
            
            with open(project_dir / "data.csv", "w") as f:
                f.write("id,value\n1,10\n2,20\n")
            
            # Initialize SciDoc
            scidoc = SciDoc(project_dir)
            
            # Step 1: Initial exploration
            with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
                mock_summarize.return_value = "Initial project setup with main script and data."
                initial_result = scidoc.explore()
            
            assert initial_result is not None
            assert len(initial_result.files) == 2
            
            # Step 2: Make changes
            with open(project_dir / "utils.py", "w") as f:
                f.write("def helper():\n    pass\n")
            
            # Step 3: Update project
            with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
                mock_summarize.return_value = "Added utility functions."
                changes = scidoc.update()
            
            assert changes is not None
            
            # Step 4: Generate log
            with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
                mock_summarize.return_value = "Workflow completed successfully."
                log_entry = scidoc.generate_log()
            
            assert log_entry is not None
            
            # Step 5: Build graph
            graph = scidoc.build_graph()
            assert graph is not None
            
            # Step 6: Validate
            rules = {"required_files": ["main.py", "data.csv"]}
            validation = scidoc.validate(rules)
            assert validation is not None
            
            # Step 7: Chat
            with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
                mock_summarize.return_value = "Project contains 3 files: main script, data, and utilities."
                response = scidoc.chat("How many files are in this project?")
            
            assert response is not None
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_error_recovery_example(self):
        """Test error recovery and graceful handling."""
        # Create a project with some problematic files
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "error_project"
        project_dir.mkdir()
        
        try:
            # Create a valid file
            with open(project_dir / "valid.py", "w") as f:
                f.write("print('Valid file')\n")
            
            # Create a corrupted JSON file
            with open(project_dir / "corrupted.json", "w") as f:
                f.write('{"invalid": json content')
            
            # Create a very large file (simulate)
            with open(project_dir / "large.txt", "w") as f:
                f.write("A" * 1000000)  # 1MB file
            
            scidoc = SciDoc(project_dir)
            
            # Should handle errors gracefully
            with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
                mock_summarize.return_value = "Project with some problematic files."
                result = scidoc.explore()
            
            assert result is not None
            assert len(result.files) >= 1  # At least the valid file should be processed
            
        finally:
            shutil.rmtree(temp_dir)


class TestRealWorldExamples:
    """Test real-world usage examples."""
    
    def test_github_repository_example(self):
        """Test simulating a GitHub repository structure."""
        temp_dir = tempfile.mkdtemp()
        repo_dir = Path(temp_dir) / "github_repo"
        repo_dir.mkdir()
        
        try:
            # Create typical GitHub repository structure
            (repo_dir / ".git").mkdir()
            (repo_dir / "src").mkdir()
            (repo_dir / "tests").mkdir()
            (repo_dir / "docs").mkdir()
            
            # Create source files
            with open(repo_dir / "src" / "main.py", "w") as f:
                f.write("def main():\n    pass\n")
            
            with open(repo_dir / "tests" / "test_main.py", "w") as f:
                f.write("def test_main():\n    pass\n")
            
            with open(repo_dir / "README.md", "w") as f:
                f.write("# GitHub Repository\n\nThis is a test repository.")
            
            with open(repo_dir / "requirements.txt", "w") as f:
                f.write("pytest\nrequests\n")
            
            scidoc = SciDoc(repo_dir)
            
            with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
                mock_summarize.return_value = "GitHub repository with source code, tests, and documentation."
                result = scidoc.explore()
            
            assert result is not None
            assert result.total_files >= 4
            
            # Check that .git directory is ignored
            git_files = [f for f in result.files if ".git" in f.filename]
            assert len(git_files) == 0
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_data_science_project_example(self):
        """Test a data science project structure."""
        temp_dir = tempfile.mkdtemp()
        ds_dir = Path(temp_dir) / "data_science_project"
        ds_dir.mkdir()
        
        try:
            # Create data science project structure
            (ds_dir / "data").mkdir()
            (ds_dir / "notebooks").mkdir()
            (ds_dir / "models").mkdir()
            
            # Create data files
            with open(ds_dir / "data" / "dataset.csv", "w") as f:
                f.write("feature1,feature2,target\n1,2,0\n3,4,1\n")
            
            # Create notebook
            notebook_content = '''{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Data Science Project\\n", "Analysis of dataset"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["import pandas as pd\\n", "df = pd.read_csv('data/dataset.csv')\\n", "print(df.head())"],
            "outputs": []
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}'''
            
            with open(ds_dir / "notebooks" / "analysis.ipynb", "w") as f:
                f.write(notebook_content)
            
            # Create model script
            with open(ds_dir / "train_model.py", "w") as f:
                f.write("from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()\n")
            
            scidoc = SciDoc(ds_dir)
            
            with patch.object(scidoc.summarizer, 'summarize') as mock_summarize:
                mock_summarize.return_value = "Data science project with dataset, analysis notebook, and model training."
                result = scidoc.explore()
            
            assert result is not None
            assert result.total_files >= 3
            
            # Check file types
            file_types = [f.file_type for f in result.files]
            assert "CSV" in file_types
            assert "JUPYTER" in file_types
            assert "PYTHON" in file_types
            
        finally:
            shutil.rmtree(temp_dir)
