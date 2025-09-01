"""
SciDoc Summarizer - ChatGPT-like documentation and summary tool.

This module provides intelligent summaries of files and directories,
explaining what they do, what's in them, and answering questions
about the content in a conversational manner.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import ast

from .logger import setup_logging
from .scientific_parser import ScientificParser

logger = setup_logging("scidoc.summarizer")


class DocumentSummarizer:
    """Intelligent document and file summarizer."""
    
    def __init__(self):
        self.scientific_parser = ScientificParser()
        self.file_type_patterns = {
            'python': r'\.py$',
            'jupyter': r'\.ipynb$',
            'markdown': r'\.md$',
            'readme': r'readme\.md$',
            'csv': r'\.csv$',
            'json': r'\.json$',
            'yaml': r'\.ya?ml$',
            'fastq': r'\.fastq$',
            'fasta': r'\.(fasta|fa)$',
            'vcf': r'\.vcf$',
            'bam': r'\.bam$',
            'sam': r'\.sam$',
            'txt': r'\.txt$',
            'log': r'\.log$',
            'config': r'\.(config|conf|ini)$',
            'shell': r'\.(sh|bash|zsh)$',
            'r': r'\.r$',
            'matlab': r'\.m$',
            'cpp': r'\.(cpp|cc|cxx)$',
            'c': r'\.c$',
            'java': r'\.java$',
            'javascript': r'\.js$',
            'typescript': r'\.ts$',
            'html': r'\.html?$',
            'css': r'\.css$',
            'sql': r'\.sql$',
            'latex': r'\.tex$',
            'bibtex': r'\.bib$'
        }
    
    def summarize_directory(self, directory_path: Path) -> str:
        """Provide a ChatGPT-like summary of what's in a directory."""
        logger.info(f"Summarizing directory: {directory_path}")
        
        try:
            # Get all files
            files = list(directory_path.rglob("*"))
            files = [f for f in files if f.is_file() and not any(part.startswith('.') for part in f.parts)]
            
            if not files:
                return f"This directory '{directory_path.name}' appears to be empty or contains only hidden files."
            
            # Analyze file types and structure
            file_types = self._categorize_files(files)
            project_type = self._determine_project_type(files, file_types)
            
            # Generate summary
            summary = self._generate_directory_summary(directory_path, files, file_types, project_type)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing directory: {e}")
            return f"I encountered an error while analyzing the directory '{directory_path.name}'. Please check if the directory exists and is accessible."
    
    def summarize_file(self, file_path: Path) -> str:
        """Provide a ChatGPT-like summary of what a file does."""
        logger.info(f"Summarizing file: {file_path}")
        
        try:
            if not file_path.exists():
                return f"The file '{file_path.name}' does not exist."
            
            # Use scientific parser for scientific files
            if self._is_scientific_file(file_path):
                scientific_analysis = self.scientific_parser.parse_file(file_path)
                return self._format_scientific_summary(file_path, scientific_analysis)
            
            # Fall back to basic summarization for other files
            content = self._read_file_content(file_path)
            if not content:
                return f"The file '{file_path.name}' appears to be empty or unreadable."
            
            file_type = self._get_file_type(file_path)
            
            # Generate summary based on file type
            if file_type == 'python':
                return self._summarize_python_file(file_path, content)
            elif file_type == 'jupyter':
                return self._summarize_jupyter_file(file_path, content)
            elif file_type == 'markdown':
                return self._summarize_markdown_file(file_path, content)
            elif file_type == 'csv':
                return self._summarize_csv_file(file_path, content)
            elif file_type == 'json':
                return self._summarize_json_file(file_path, content)
            elif file_type == 'yaml':
                return self._summarize_yaml_file(file_path, content)
            else:
                return self._summarize_generic_file(file_path, content, file_type)
                
        except Exception as e:
            logger.error(f"Error summarizing file: {e}")
            return f"I encountered an error while analyzing the file '{file_path.name}'. The file might be corrupted or in an unsupported format."
    
    def answer_question(self, question: str, directory_path: Path) -> str:
        """Answer questions about files and content in a ChatGPT-like manner."""
        logger.info(f"Answering question: {question}")
        
        try:
            question_lower = question.lower()
            
            # Handle different types of questions
            if any(word in question_lower for word in ['what', 'how many', 'count', 'files']):
                return self._answer_what_questions(question, directory_path)
            elif any(word in question_lower for word in ['find', 'search', 'where', 'locate']):
                return self._answer_search_questions(question, directory_path)
            elif any(word in question_lower for word in ['recent', 'changed', 'modified', 'updated']):
                return self._answer_change_questions(question, directory_path)
            elif any(word in question_lower for word in ['summary', 'overview', 'describe']):
                return self._answer_summary_questions(question, directory_path)
            else:
                return self._answer_general_questions(question, directory_path)
                
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"I encountered an error while processing your question. Please try rephrasing it or ask about a specific file or directory."
    
    def _categorize_files(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Categorize files by type."""
        categories = {}
        
        for file_path in files:
            file_type = self._get_file_type(file_path)
            if file_type not in categories:
                categories[file_type] = []
            categories[file_type].append(file_path)
        
        return categories
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine the type of a file."""
        filename = file_path.name.lower()
        
        for file_type, pattern in self.file_type_patterns.items():
            if re.search(pattern, filename):
                return file_type
        
        return 'unknown'
    
    def _determine_project_type(self, files: List[Path], file_types: Dict[str, List[Path]]) -> str:
        """Determine the type of project based on files present."""
        if 'python' in file_types and len(file_types['python']) > 0:
            return 'Python Project'
        elif 'jupyter' in file_types and len(file_types['jupyter']) > 0:
            return 'Jupyter Notebook Project'
        elif 'r' in file_types and len(file_types['r']) > 0:
            return 'R Project'
        elif 'matlab' in file_types and len(file_types['matlab']) > 0:
            return 'MATLAB Project'
        elif any(bio_type in file_types for bio_type in ['fastq', 'fasta', 'vcf', 'bam']):
            return 'Bioinformatics Project'
        elif 'csv' in file_types or 'json' in file_types:
            return 'Data Analysis Project'
        else:
            return 'General Project'
    
    def _generate_directory_summary(self, directory_path: Path, files: List[Path], 
                                  file_types: Dict[str, List[Path]], project_type: str) -> str:
        """Generate a conversational summary of a directory."""
        total_files = len(files)
        
        # Count files by type
        type_counts = {file_type: len(file_list) for file_type, file_list in file_types.items()}
        
        # Find key files
        readme_files = [f for f in files if 'readme' in f.name.lower()]
        main_files = [f for f in files if 'main' in f.name.lower()]
        config_files = [f for f in files if any(ext in f.name.lower() for ext in ['.config', '.conf', '.ini', '.yaml', '.yml'])]
        
        # Generate focused summary
        summary = f"This is a {project_type} containing {total_files} files.\n\n"
        
        # Focus on what the project does, not just file counts
        if readme_files:
            readme_content = self._read_file_content(readme_files[0])
            if readme_content:
                # Extract first meaningful paragraph from README
                paragraphs = [p.strip() for p in readme_content.split('\n\n') if p.strip() and not p.strip().startswith('#')]
                if paragraphs:
                    summary += f"Project Description: {paragraphs[0][:300]}"
                    if len(paragraphs[0]) > 300:
                        summary += "..."
                    summary += "\n\n"
        
        # Key files and their purposes
        if main_files:
            summary += f"Main files: {', '.join([f.name for f in main_files])}\n"
        
        if config_files:
            summary += f"Configuration files: {len(config_files)} found\n"
        
        # Project-specific summary
        if project_type == 'Python Project':
            summary += self._add_python_summary(files, file_types)
        elif project_type == 'Bioinformatics Project':
            summary += self._add_bioinformatics_summary(files, file_types)
        elif project_type == 'Data Analysis Project':
            summary += self._add_data_analysis_summary(files, file_types)
        
        return summary
    
    def _add_python_summary(self, files: List[Path], file_types: Dict[str, List[Path]]) -> str:
        """Add Python-specific summary."""
        summary = "\nPython Project Details:\n"
        
        # Check for common Python project patterns
        has_requirements = any('requirements' in f.name.lower() for f in files)
        has_setup = any('setup' in f.name.lower() for f in files)
        has_tests = any('test' in f.name.lower() for f in files)
        
        if has_requirements:
            summary += "- Has dependency management\n"
        if has_setup:
            summary += "- Has setup configuration\n"
        if has_tests:
            summary += "- Includes test files\n"
        
        return summary
    
    def _add_bioinformatics_summary(self, files: List[Path], file_types: Dict[str, List[Path]]) -> str:
        """Add bioinformatics-specific summary."""
        summary = "\nBioinformatics Project Details:\n"
        
        if 'fastq' in file_types:
            summary += f"- Contains {len(file_types['fastq'])} FASTQ files (sequencing data)\n"
        if 'fasta' in file_types:
            summary += f"- Contains {len(file_types['fasta'])} FASTA files (sequence data)\n"
        if 'vcf' in file_types:
            summary += f"- Contains {len(file_types['vcf'])} VCF files (variant data)\n"
        if 'bam' in file_types:
            summary += f"- Contains {len(file_types['bam'])} BAM files (aligned reads)\n"
        
        return summary
    
    def _add_data_analysis_summary(self, files: List[Path], file_types: Dict[str, List[Path]]) -> str:
        """Add data analysis-specific summary."""
        summary = "\nData Analysis Project Details:\n"
        
        if 'csv' in file_types:
            summary += f"- Contains {len(file_types['csv'])} CSV files (tabular data)\n"
        if 'json' in file_types:
            summary += f"- Contains {len(file_types['json'])} JSON files (structured data)\n"
        if 'jupyter' in file_types:
            summary += f"- Contains {len(file_types['jupyter'])} Jupyter notebooks (analysis workflows)\n"
        
        return summary
    
    def _summarize_python_file(self, file_path: Path, content: str) -> str:
        """Summarize a Python file."""
        summary = f"This is a Python script named '{file_path.name}'.\n\n"
        
        # Parse Python code
        try:
            tree = ast.parse(content)
            
            # Extract functions and classes
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.Import)]
            from_imports = [f"{node.module}.{node.names[0].name}" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            
            # Look for docstring first (most important)
            docstring = ast.get_docstring(tree)
            if docstring:
                summary += f"Description: {docstring[:300]}"
                if len(docstring) > 300:
                    summary += "..."
                summary += "\n\n"
            
            # Generate summary
            if functions:
                summary += f"Functions: {', '.join(functions[:5])}"
                if len(functions) > 5:
                    summary += f" and {len(functions) - 5} more"
                summary += "\n"
            
            if classes:
                summary += f"Classes: {', '.join(classes)}\n"
            
            if imports or from_imports:
                all_imports = imports + from_imports
                summary += f"Dependencies: {', '.join(all_imports[:5])}"
                if len(all_imports) > 5:
                    summary += f" and {len(all_imports) - 5} more"
                summary += "\n"
            
        except:
            # Fallback for unparseable Python
            summary += "Content: This appears to be a Python script, but I couldn't parse its structure.\n"
        
        return summary
    
    def _summarize_markdown_file(self, file_path: Path, content: str) -> str:
        """Summarize a Markdown file."""
        summary = f"This is a **Markdown document** named '{file_path.name}'.\n\n"
        
        # Extract headers
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        
        if headers:
            summary += f"📋 **Sections**: {', '.join(headers[:5])}"
            if len(headers) > 5:
                summary += f" and {len(headers) - 5} more"
            summary += "\n"
        
        # Extract first paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and not p.strip().startswith('#')]
        if paragraphs:
            summary += f"\n📝 **Overview**: {paragraphs[0][:300]}"
            if len(paragraphs[0]) > 300:
                summary += "..."
            summary += "\n"
        
        return summary
    
    def _summarize_csv_file(self, file_path: Path, content: str) -> str:
        """Summarize a CSV file."""
        summary = f"This is a **CSV data file** named '{file_path.name}'.\n\n"
        
        lines = content.split('\n')
        if lines:
            # Analyze header
            header = lines[0].split(',')
            summary += f"📊 **Columns**: {len(header)} columns\n"
            summary += f"📋 **Headers**: {', '.join(header[:5])}"
            if len(header) > 5:
                summary += f" and {len(header) - 5} more"
            summary += "\n"
            
            # Count rows
            row_count = len(lines) - 1  # Exclude header
            summary += f"📈 **Rows**: {row_count:,} data rows\n"
        
        return summary
    
    def _summarize_json_file(self, file_path: Path, content: str) -> str:
        """Summarize a JSON file."""
        summary = f"This is a **JSON file** named '{file_path.name}'.\n\n"
        
        try:
            data = json.loads(content)
            
            if isinstance(data, dict):
                summary += f"📋 **Structure**: JSON object with {len(data)} keys\n"
                summary += f"🔑 **Keys**: {', '.join(list(data.keys())[:5])}"
                if len(data) > 5:
                    summary += f" and {len(data) - 5} more"
                summary += "\n"
            elif isinstance(data, list):
                summary += f"📋 **Structure**: JSON array with {len(data)} items\n"
                if data and isinstance(data[0], dict):
                    summary += f"🔑 **Item keys**: {', '.join(list(data[0].keys())[:5])}\n"
            
        except:
            summary += "📋 **Structure**: JSON file (could not parse structure)\n"
        
        return summary
    
    def _summarize_bioinformatics_file(self, file_path: Path, content: str, file_type: str) -> str:
        """Summarize a bioinformatics file."""
        file_type_names = {
            'fastq': 'FASTQ',
            'fasta': 'FASTA',
            'vcf': 'VCF',
            'bam': 'BAM',
            'sam': 'SAM'
        }
        
        file_type_name = file_type_names.get(file_type, file_type.upper())
        summary = f"This is a **{file_type_name} file** named '{file_path.name}'.\n\n"
        
        lines = content.split('\n')
        
        if file_type == 'fastq':
            # FASTQ files have 4 lines per read
            read_count = len(lines) // 4
            summary += f"🧬 **Sequences**: {read_count:,} sequencing reads\n"
            if lines:
                summary += f"📏 **Read length**: ~{len(lines[1])} bases (first read)\n"
        
        elif file_type == 'fasta':
            # FASTA files have 2 lines per sequence
            seq_count = len([line for line in lines if line.startswith('>')])
            summary += f"🧬 **Sequences**: {seq_count:,} sequences\n"
            if lines:
                summary += f"📏 **Sequence length**: ~{len(lines[1])} bases (first sequence)\n"
        
        elif file_type == 'vcf':
            # Count non-header lines
            variant_count = len([line for line in lines if not line.startswith('#')])
            summary += f"🧬 **Variants**: {variant_count:,} genetic variants\n"
        
        return summary
    
    def _summarize_generic_file(self, file_path: Path, content: str, file_type: str) -> str:
        """Summarize a generic file."""
        summary = f"This is a **{file_type} file** named '{file_path.name}'.\n\n"
        
        # Basic file info
        lines = content.split('\n')
        summary += f"📏 **Size**: {len(content):,} characters\n"
        summary += f"📄 **Lines**: {len(lines):,} lines\n"
        
        # Try to extract meaningful content
        if len(content) > 200:
            summary += f"📝 **Preview**: {content[:200]}...\n"
        else:
            summary += f"📝 **Content**: {content}\n"
        
        return summary
    
    def _summarize_jupyter_file(self, file_path: Path, content: str) -> str:
        """Summarize a Jupyter notebook file."""
        summary = f"This is a **Jupyter notebook** named '{file_path.name}'.\n\n"
        
        try:
            notebook_data = json.loads(content)
            cells = notebook_data.get('cells', [])
            
            # Count different cell types
            code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']
            markdown_cells = [cell for cell in cells if cell.get('cell_type') == 'markdown']
            
            summary += f"📓 **Notebook Structure**:\n"
            summary += f"   • **{len(cells)} total cells**\n"
            summary += f"   • **{len(code_cells)} code cells**\n"
            summary += f"   • **{len(markdown_cells)} markdown cells**\n"
            
            # Extract markdown content for overview
            markdown_content = ""
            for cell in markdown_cells:
                source = cell.get('source', [])
                if isinstance(source, list):
                    markdown_content += ''.join(source)
                else:
                    markdown_content += str(source)
            
            if markdown_content:
                summary += f"\n📝 **Overview**: {markdown_content[:300]}"
                if len(markdown_content) > 300:
                    summary += "..."
                summary += "\n"
            
        except:
            summary += "📓 **Content**: This appears to be a Jupyter notebook, but I couldn't parse its structure.\n"
        
        return summary
    
    def _summarize_yaml_file(self, file_path: Path, content: str) -> str:
        """Summarize a YAML file."""
        summary = f"This is a **YAML configuration file** named '{file_path.name}'.\n\n"
        
        try:
            import yaml
            data = yaml.safe_load(content)
            
            if isinstance(data, dict):
                summary += f"⚙️ **Configuration**: YAML object with {len(data)} top-level keys\n"
                summary += f"🔑 **Keys**: {', '.join(list(data.keys())[:5])}"
                if len(data) > 5:
                    summary += f" and {len(data) - 5} more"
                summary += "\n"
            elif isinstance(data, list):
                summary += f"⚙️ **Configuration**: YAML array with {len(data)} items\n"
            
        except:
            summary += "⚙️ **Configuration**: YAML file (could not parse structure)\n"
        
        return summary
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content safely."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    
    def _is_scientific_file(self, file_path: Path) -> bool:
        """Check if a file is a scientific file that should use the scientific parser."""
        scientific_extensions = {'.fastq', '.fasta', '.fa', '.vcf', '.bam', '.sam', '.csv', '.tsv', '.json', '.md', '.txt'}
        return file_path.suffix.lower() in scientific_extensions
    
    def _format_scientific_summary(self, file_path: Path, scientific_analysis: Dict[str, Any]) -> str:
        """Format scientific analysis results into a readable summary."""
        if 'error' in scientific_analysis:
            return f"Error analyzing {file_path.name}: {scientific_analysis['error']}"
        
        file_type = scientific_analysis.get('file_type', 'unknown')
        
        if file_type in ['fastq', 'fasta']:
            return self._format_biological_summary(file_path, scientific_analysis)
        elif file_type == 'vcf':
            return self._format_variant_summary(file_path, scientific_analysis)
        elif file_type == 'csv':
            return self._format_data_summary(file_path, scientific_analysis)
        elif file_type == 'scientific_document':
            return self._format_document_summary(file_path, scientific_analysis)
        else:
            return self._format_generic_scientific_summary(file_path, scientific_analysis)
    
    def _format_biological_summary(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Format biological data summary."""
        bio_data = analysis.get('biological_data', {})
        
        summary = f"This is a {bio_data.get('data_type', 'biological data')} file.\n\n"
        
        if 'sequence_count' in bio_data:
            summary += f"Contains {bio_data['sequence_count']} sequences"
            if 'average_length' in bio_data:
                summary += f" with average length {bio_data['average_length']:.1f} bp"
            summary += ".\n"
        
        if 'gc_content' in bio_data and bio_data['gc_content'] is not None:
            summary += f"GC content: {bio_data['gc_content']:.1f}%\n"
        
        if 'total_bases' in bio_data:
            summary += f"Total bases: {bio_data['total_bases']:,}\n"
        
        if 'total_residues' in bio_data:
            summary += f"Total residues: {bio_data['total_residues']:,}\n"
        
        # Add biological context
        if 'biological_context' in analysis:
            summary += f"\n{analysis['biological_context']}\n"
        
        # Add recommendations if available
        if 'recommendations' in analysis and analysis['recommendations']:
            summary += "\nRecommendations:\n"
            for rec in analysis['recommendations'][:3]:  # Show top 3
                summary += f"• {rec}\n"
        
        return summary
    
    def _format_variant_summary(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Format variant data summary."""
        bio_data = analysis.get('biological_data', {})
        
        summary = f"This is a {bio_data.get('data_type', 'genetic variant')} file.\n\n"
        
        if 'variant_count' in bio_data:
            summary += f"Contains {bio_data['variant_count']} genetic variants"
            if 'chromosomes' in bio_data:
                summary += f" across {len(bio_data['chromosomes'])} chromosomes"
            summary += ".\n"
        
        # Add variant analysis
        if 'variant_analysis' in analysis:
            var_analysis = analysis['variant_analysis']
            
            if 'chromosome_distribution' in var_analysis:
                chrom_dist = var_analysis['chromosome_distribution']
                if chrom_dist:
                    most_affected = max(chrom_dist.items(), key=lambda x: x[1])
                    summary += f"Most variants on chromosome {most_affected[0]} ({most_affected[1]} variants).\n"
            
            if 'variant_types' in bio_data:
                summary += "Variant types:\n"
                for var_type, count in bio_data['variant_types'].items():
                    summary += f"• {var_type}: {count}\n"
        
        # Add biological context
        if 'biological_context' in analysis:
            summary += f"\n{analysis['biological_context']}\n"
        
        # Add recommendations
        if 'recommendations' in analysis and analysis['recommendations']:
            summary += "\nRecommendations:\n"
            for rec in analysis['recommendations'][:3]:
                summary += f"• {rec}\n"
        
        return summary
    
    def _format_data_summary(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Format data file summary."""
        data_analysis = analysis.get('data_analysis', {})
        
        summary = f"This is a {data_analysis.get('data_type', 'data')} file.\n\n"
        
        if 'dimensions' in data_analysis:
            rows, cols = data_analysis['dimensions']
            summary += f"Dataset dimensions: {rows} samples × {cols} variables\n"
        
        if 'columns' in data_analysis:
            summary += f"Columns: {', '.join(data_analysis['columns'][:5])}"
            if len(data_analysis['columns']) > 5:
                summary += f" and {len(data_analysis['columns']) - 5} more"
            summary += "\n"
        
        if 'scientific_relevance' in data_analysis:
            summary += f"Scientific relevance: {data_analysis['scientific_relevance']}\n"
        
        # Add statistical insights
        if 'statistical_insights' in analysis and analysis['statistical_insights']:
            summary += "\nKey insights:\n"
            for insight in analysis['statistical_insights'][:3]:
                summary += f"• {insight}\n"
        
        # Add data quality assessment
        if 'data_quality' in analysis:
            quality = analysis['data_quality']
            if 'quality_score' in quality:
                summary += f"\nData quality score: {quality['quality_score']:.1f}/100\n"
            
            if 'issues' in quality and quality['issues']:
                summary += "Data quality issues:\n"
                for issue in quality['issues'][:3]:
                    summary += f"• {issue}\n"
        
        return summary
    
    def _format_document_summary(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Format scientific document summary."""
        doc_analysis = analysis.get('document_analysis', {})
        
        summary = f"This is a scientific document.\n\n"
        
        if 'title' in doc_analysis and doc_analysis['title']:
            summary += f"Title: {doc_analysis['title']}\n"
        
        if 'authors' in doc_analysis and doc_analysis['authors']:
            summary += f"Authors: {', '.join(doc_analysis['authors'][:3])}"
            if len(doc_analysis['authors']) > 3:
                summary += f" and {len(doc_analysis['authors']) - 3} more"
            summary += "\n"
        
        if 'abstract' in doc_analysis and doc_analysis['abstract']:
            summary += f"\nAbstract: {doc_analysis['abstract'][:200]}"
            if len(doc_analysis['abstract']) > 200:
                summary += "..."
            summary += "\n"
        
        if 'scientific_context' in analysis:
            summary += f"\nScientific domain: {analysis['scientific_context']}\n"
        
        # Add key sections
        for section in ['methods', 'results', 'conclusions']:
            if section in doc_analysis and doc_analysis[section]:
                summary += f"\n{section.title()}: {len(doc_analysis[section])} items documented\n"
        
        # Add recommendations
        if 'recommendations' in analysis and analysis['recommendations']:
            summary += "\nRecommendations:\n"
            for rec in analysis['recommendations'][:3]:
                summary += f"• {rec}\n"
        
        return summary
    
    def _format_generic_scientific_summary(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Format generic scientific file summary."""
        summary = f"This is a scientific file of type: {analysis.get('file_type', 'unknown')}\n\n"
        
        # Add any available insights
        if 'biological_data' in analysis:
            bio_data = analysis['biological_data']
            summary += f"Data type: {bio_data.get('data_type', 'Unknown')}\n"
            if 'biological_context' in analysis:
                summary += f"{analysis['biological_context']}\n"
        
        if 'data_analysis' in analysis:
            data_analysis = analysis['data_analysis']
            summary += f"Data type: {data_analysis.get('data_type', 'Unknown')}\n"
            if 'scientific_relevance' in data_analysis:
                summary += f"Relevance: {data_analysis['scientific_relevance']}\n"
        
        # Add recommendations
        if 'recommendations' in analysis and analysis['recommendations']:
            summary += "\nRecommendations:\n"
            for rec in analysis['recommendations'][:3]:
                summary += f"• {rec}\n"
        
        return summary
    
    # Question answering methods
    def _answer_what_questions(self, question: str, directory_path: Path) -> str:
        """Answer 'what' questions about the directory."""
        files = list(directory_path.rglob("*"))
        files = [f for f in files if f.is_file() and not any(part.startswith('.') for part in f.parts)]
        
        if 'how many files' in question.lower():
            return f"There are **{len(files)} files** in the '{directory_path.name}' directory."
        
        if 'what files' in question.lower():
            file_types = self._categorize_files(files)
            response = f"In the '{directory_path.name}' directory, I found:\n"
            for file_type, file_list in file_types.items():
                if file_list:
                    response += f"• **{len(file_list)} {file_type}** file{'s' if len(file_list) > 1 else ''}\n"
            return response
        
        return f"I'm not sure how to answer that specific question about '{directory_path.name}'. Could you be more specific?"
    
    def _answer_search_questions(self, question: str, directory_path: Path) -> str:
        """Answer search/find questions."""
        files = list(directory_path.rglob("*"))
        files = [f for f in files if f.is_file() and not any(part.startswith('.') for part in f.parts)]
        
        # Extract search terms from question
        search_terms = []
        for word in question.lower().split():
            if word not in ['find', 'search', 'where', 'locate', 'file', 'files', 'in', 'the', 'directory']:
                search_terms.append(word)
        
        if search_terms:
            matching_files = []
            for file_path in files:
                if any(term in file_path.name.lower() for term in search_terms):
                    matching_files.append(file_path)
            
            if matching_files:
                response = f"I found {len(matching_files)} file{'s' if len(matching_files) > 1 else ''} matching your search:\n"
                for file_path in matching_files[:10]:  # Limit to 10 results
                    response += f"• {file_path.name}\n"
                if len(matching_files) > 10:
                    response += f"... and {len(matching_files) - 10} more\n"
                return response
            else:
                return f"I couldn't find any files matching '{' '.join(search_terms)}' in the '{directory_path.name}' directory."
        
        return f"I'm not sure what you're looking for. Could you specify what type of files you want to find?"
    
    def _answer_change_questions(self, question: str, directory_path: Path) -> str:
        """Answer questions about recent changes."""
        files = list(directory_path.rglob("*"))
        files = [f for f in files if f.is_file() and not any(part.startswith('.') for part in f.parts)]
        
        # Get file modification times
        file_times = [(f, f.stat().st_mtime) for f in files]
        file_times.sort(key=lambda x: x[1], reverse=True)
        
        recent_files = file_times[:5]
        
        response = f"Here are the **5 most recently modified files** in '{directory_path.name}':\n"
        for file_path, mtime in recent_files:
            mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            response += f"• {file_path.name} (modified: {mod_time})\n"
        
        return response
    
    def _answer_summary_questions(self, question: str, directory_path: Path) -> str:
        """Answer summary/overview questions."""
        return self.summarize_directory(directory_path)
    
    def _answer_general_questions(self, question: str, directory_path: Path) -> str:
        """Answer general questions."""
        return f"I'm not sure how to answer that specific question about '{directory_path.name}'. Try asking about files, directories, or recent changes."
