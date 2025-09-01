"""
Scientific Parser for SciDoc - Interprets biological and scientific data.

This module provides intelligent parsing of scientific files including:
- Biological sequence data (FASTQ, FASTA, VCF, BAM)
- Scientific documents and papers
- Data files with scientific context
- Laboratory protocols and methods
"""

import re
import json
import csv
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

from .logger import setup_logging

logger = setup_logging("scidoc.scientific_parser")


@dataclass
class BiologicalData:
    """Biological data analysis results."""
    data_type: str  # fastq, fasta, vcf, bam, etc.
    sequence_count: int
    average_length: float
    quality_metrics: Dict[str, Any]
    biological_context: str
    potential_issues: List[str]
    recommendations: List[str]


@dataclass
class ScientificDocument:
    """Scientific document analysis."""
    document_type: str  # paper, protocol, lab_note, etc.
    title: str
    authors: List[str]
    abstract: str
    methods: List[str]
    results: List[str]
    conclusions: List[str]
    references: List[str]
    keywords: List[str]


@dataclass
class DataAnalysis:
    """Data analysis results."""
    data_type: str
    dimensions: Tuple[int, int]
    columns: List[str]
    data_summary: Dict[str, Any]
    statistical_insights: List[str]
    data_quality: Dict[str, Any]
    scientific_relevance: str


class ScientificParser:
    """Intelligent parser for scientific data and documents."""
    
    def __init__(self):
        self.biological_patterns = {
            'fastq': r'^[ACGTN]+$',  # DNA sequences
            'fasta': r'^[ACGTN]+$',
            'protein': r'^[ACDEFGHIKLMNPQRSTVWY]+$',  # Amino acids
            'taxonomy': r'^[A-Z][a-z]+ [a-z]+$',  # Genus species
        }
        
        self.scientific_keywords = {
            'methods': ['method', 'protocol', 'procedure', 'experiment', 'assay', 'technique'],
            'results': ['result', 'finding', 'outcome', 'observation', 'data', 'analysis'],
            'conclusions': ['conclusion', 'summary', 'implication', 'significance', 'interpretation'],
            'biology': ['gene', 'protein', 'sequence', 'mutation', 'expression', 'pathway', 'organism'],
            'chemistry': ['compound', 'reaction', 'molecule', 'concentration', 'pH', 'temperature'],
            'physics': ['force', 'energy', 'velocity', 'acceleration', 'wavelength', 'frequency']
        }
    
    def parse_file(self, file_path) -> Dict[str, Any]:
        """Parse a file and extract scientific information."""
        # Convert string to Path if needed
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        logger.info(f"Parsing scientific file: {file_path}")
        
        try:
            file_type = self._get_file_type(file_path)
            
            if file_type in ['fastq', 'fasta', 'vcf', 'bam', 'sam']:
                return self._parse_biological_file(file_path, file_type)
            elif file_type in ['csv', 'tsv', 'json']:
                return self._parse_data_file(file_path, file_type)
            elif file_type in ['md', 'txt', 'pdf']:
                return self._parse_scientific_document(file_path, file_type)
            else:
                return self._parse_generic_file(file_path, file_type)
                
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return {'error': str(e), 'file_type': 'unknown'}
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine the type of scientific file."""
        filename = file_path.name.lower()
        
        if filename.endswith('.fastq'):
            return 'fastq'
        elif filename.endswith('.fasta') or filename.endswith('.fa'):
            return 'fasta'
        elif filename.endswith('.vcf'):
            return 'vcf'
        elif filename.endswith('.bam'):
            return 'bam'
        elif filename.endswith('.sam'):
            return 'sam'
        elif filename.endswith('.csv'):
            return 'csv'
        elif filename.endswith('.tsv'):
            return 'tsv'
        elif filename.endswith('.json'):
            return 'json'
        elif filename.endswith('.md'):
            return 'md'
        elif filename.endswith('.txt'):
            return 'txt'
        else:
            return 'unknown'
    
    def _parse_biological_file(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Parse biological data files."""
        content = self._read_file_content(file_path)
        if not content:
            return {'error': 'Could not read file content'}
        
        if file_type == 'fastq':
            return self._parse_fastq_file(content, file_path)
        elif file_type == 'fasta':
            return self._parse_fasta_file(content, file_path)
        elif file_type == 'vcf':
            return self._parse_vcf_file(content, file_path)
        elif file_type == 'bam':
            return self._parse_bam_file(content, file_path)
        else:
            return self._parse_generic_bio_file(content, file_path, file_type)
    
    def _parse_fastq_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Parse FASTQ sequencing data."""
        lines = content.split('\n')
        
        # FASTQ format: 4 lines per read
        # Line 1: @sequence_id
        # Line 2: sequence
        # Line 3: + (optional description)
        # Line 4: quality scores
        
        read_count = len(lines) // 4
        sequences = []
        quality_scores = []
        
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                sequence = lines[i + 1]
                quality = lines[i + 3]
                
                if sequence and quality:
                    sequences.append(sequence)
                    quality_scores.append(self._calculate_quality_score(quality))
        
        # Calculate metrics
        avg_length = statistics.mean([len(seq) for seq in sequences]) if sequences else 0
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        # Biological insights
        gc_content = self._calculate_gc_content(sequences)
        potential_issues = self._identify_sequencing_issues(sequences, quality_scores)
        
        return {
            'file_type': 'fastq',
            'biological_data': {
                'data_type': 'DNA Sequencing Data',
                'sequence_count': read_count,
                'average_length': round(avg_length, 2),
                'average_quality': round(avg_quality, 2),
                'gc_content': round(gc_content, 2),
                'total_bases': sum(len(seq) for seq in sequences)
            },
            'quality_metrics': {
                'quality_distribution': self._analyze_quality_distribution(quality_scores),
                'length_distribution': self._analyze_length_distribution(sequences)
            },
            'biological_context': f"FASTQ file containing {read_count} DNA sequencing reads with average length {avg_length:.1f} bp",
            'potential_issues': potential_issues,
            'recommendations': self._generate_sequencing_recommendations(sequences, quality_scores)
        }
    
    def _parse_fasta_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Parse FASTA sequence data."""
        lines = content.split('\n')
        
        sequences = []
        headers = []
        
        current_seq = ""
        current_header = ""
        
        for line in lines:
            if line.startswith('>'):
                if current_seq and current_header:
                    sequences.append(current_seq)
                    headers.append(current_header)
                current_header = line[1:].strip()
                current_seq = ""
            else:
                current_seq += line.strip()
        
        # Add last sequence
        if current_seq and current_header:
            sequences.append(current_seq)
            headers.append(current_header)
        
        # Analyze sequences
        avg_length = statistics.mean([len(seq) for seq in sequences]) if sequences else 0
        gc_content = self._calculate_gc_content(sequences)
        
        # Determine if DNA or protein
        is_protein = self._is_protein_sequence(sequences)
        data_type = "Protein Sequences" if is_protein else "DNA Sequences"
        
        return {
            'file_type': 'fasta',
            'biological_data': {
                'data_type': data_type,
                'sequence_count': len(sequences),
                'average_length': round(avg_length, 2),
                'gc_content': round(gc_content, 2) if not is_protein else None,
                'total_residues': sum(len(seq) for seq in sequences)
            },
            'sequence_analysis': {
                'headers': headers[:5],  # Show first 5 headers
                'length_distribution': self._analyze_length_distribution(sequences),
                'composition': self._analyze_sequence_composition(sequences, is_protein)
            },
            'biological_context': f"FASTA file containing {len(sequences)} {data_type.lower()} with average length {avg_length:.1f} residues",
            'recommendations': self._generate_sequence_recommendations(sequences, is_protein)
        }
    
    def _parse_vcf_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Parse VCF variant data."""
        lines = content.split('\n')
        
        # Skip header lines
        variant_lines = [line for line in lines if not line.startswith('#') and line.strip()]
        
        variants = []
        for line in variant_lines:
            parts = line.split('\t')
            if len(parts) >= 8:
                variant = {
                    'chrom': parts[0],
                    'pos': parts[1],
                    'id': parts[2],
                    'ref': parts[3],
                    'alt': parts[4],
                    'qual': parts[5],
                    'filter': parts[6],
                    'info': parts[7]
                }
                variants.append(variant)
        
        # Analyze variants
        chromosomes = list(set(v['chrom'] for v in variants))
        variant_types = self._classify_variants(variants)
        
        return {
            'file_type': 'vcf',
            'biological_data': {
                'data_type': 'Genetic Variants',
                'variant_count': len(variants),
                'chromosomes': chromosomes,
                'variant_types': variant_types
            },
            'variant_analysis': {
                'chromosome_distribution': self._analyze_chromosome_distribution(variants),
                'quality_distribution': self._analyze_variant_quality(variants),
                'impact_analysis': self._analyze_variant_impact(variants)
            },
            'biological_context': f"VCF file containing {len(variants)} genetic variants across {len(chromosomes)} chromosomes",
            'recommendations': self._generate_variant_recommendations(variants)
        }
    
    def _parse_data_file(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Parse scientific data files."""
        content = self._read_file_content(file_path)
        if not content:
            return {'error': 'Could not read file content'}
        
        if file_type == 'csv':
            return self._parse_csv_file(content, file_path)
        elif file_type == 'json':
            return self._parse_json_file(content, file_path)
        else:
            return self._parse_generic_data_file(content, file_path, file_type)
    
    def _parse_csv_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Parse CSV data with scientific context."""
        lines = content.split('\n')
        if not lines:
            return {'error': 'Empty CSV file'}
        
        # Parse CSV
        reader = csv.reader(lines)
        rows = list(reader)
        
        if not rows:
            return {'error': 'No data in CSV file'}
        
        headers = rows[0]
        data_rows = rows[1:]
        
        # Analyze data
        dimensions = (len(data_rows), len(headers))
        data_summary = self._analyze_csv_data(headers, data_rows)
        scientific_relevance = self._determine_scientific_relevance(headers, data_rows)
        
        return {
            'file_type': 'csv',
            'data_analysis': {
                'data_type': 'Tabular Data',
                'dimensions': dimensions,
                'columns': headers,
                'data_summary': data_summary,
                'scientific_relevance': scientific_relevance
            },
            'statistical_insights': self._generate_statistical_insights(headers, data_rows),
            'data_quality': self._assess_data_quality(headers, data_rows),
            'recommendations': self._generate_data_recommendations(headers, data_rows)
        }
    
    def _parse_scientific_document(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Parse scientific documents and papers."""
        content = self._read_file_content(file_path)
        if not content:
            return {'error': 'Could not read file content'}
        
        # Extract scientific information
        title = self._extract_title(content)
        authors = self._extract_authors(content)
        abstract = self._extract_abstract(content)
        methods = self._extract_methods(content)
        results = self._extract_results(content)
        conclusions = self._extract_conclusions(content)
        keywords = self._extract_keywords(content)
        
        return {
            'file_type': 'scientific_document',
            'document_analysis': {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'methods': methods,
                'results': results,
                'conclusions': conclusions,
                'keywords': keywords
            },
            'scientific_context': self._analyze_scientific_context(content),
            'recommendations': self._generate_document_recommendations(content)
        }
    
    # Helper methods for biological analysis
    def _calculate_gc_content(self, sequences: List[str]) -> float:
        """Calculate GC content of DNA sequences."""
        if not sequences:
            return 0.0
        
        total_gc = 0
        total_bases = 0
        
        for seq in sequences:
            seq_upper = seq.upper()
            gc_count = seq_upper.count('G') + seq_upper.count('C')
            total_gc += gc_count
            total_bases += len(seq)
        
        return (total_gc / total_bases) * 100 if total_bases > 0 else 0.0
    
    def _is_protein_sequence(self, sequences: List[str]) -> bool:
        """Determine if sequences are protein or DNA."""
        if not sequences:
            return False
        
        # Check first few sequences
        sample_size = min(5, len(sequences))
        protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
        
        protein_count = 0
        for seq in sequences[:sample_size]:
            seq_upper = seq.upper()
            if all(char in protein_chars for char in seq_upper):
                protein_count += 1
        
        return protein_count > sample_size / 2
    
    def _calculate_quality_score(self, quality_line: str) -> float:
        """Calculate average quality score from FASTQ quality line."""
        if not quality_line:
            return 0.0
        
        # Convert ASCII quality scores to numerical values
        scores = [ord(char) - 33 for char in quality_line]
        return statistics.mean(scores) if scores else 0.0
    
    def _identify_sequencing_issues(self, sequences: List[str], quality_scores: List[float]) -> List[str]:
        """Identify potential sequencing issues."""
        issues = []
        
        if not sequences or not quality_scores:
            return issues
        
        # Check for low quality
        low_quality_count = sum(1 for score in quality_scores if score < 20)
        if low_quality_count > len(quality_scores) * 0.1:
            issues.append(f"High proportion of low-quality reads ({low_quality_count}/{len(quality_scores)})")
        
        # Check for very short sequences
        short_sequences = sum(1 for seq in sequences if len(seq) < 50)
        if short_sequences > len(sequences) * 0.2:
            issues.append(f"Many short sequences detected ({short_sequences}/{len(sequences)})")
        
        # Check for N bases (ambiguous)
        total_n = sum(seq.count('N') for seq in sequences)
        if total_n > 0:
            issues.append(f"Ambiguous bases (N) detected: {total_n} total")
        
        return issues
    
    def _generate_sequencing_recommendations(self, sequences: List[str], quality_scores: List[float]) -> List[str]:
        """Generate recommendations for sequencing data."""
        recommendations = []
        
        if not sequences or not quality_scores:
            return recommendations
        
        avg_quality = statistics.mean(quality_scores)
        if avg_quality < 25:
            recommendations.append("Consider quality trimming to remove low-quality reads")
        
        if len(sequences) < 1000:
            recommendations.append("Low read count - consider increasing sequencing depth")
        
        # Check for contamination
        gc_content = self._calculate_gc_content(sequences)
        if gc_content < 30 or gc_content > 70:
            recommendations.append("Unusual GC content - check for contamination or mixed samples")
        
        return recommendations
    
    def _classify_variants(self, variants: List[Dict]) -> Dict[str, int]:
        """Classify variants by type."""
        variant_types = {}
        
        for variant in variants:
            ref = variant['ref']
            alt = variant['alt']
            
            if len(ref) == 1 and len(alt) == 1:
                if ref != alt:
                    variant_types['SNP'] = variant_types.get('SNP', 0) + 1
            elif len(ref) > len(alt):
                variant_types['Deletion'] = variant_types.get('Deletion', 0) + 1
            elif len(alt) > len(ref):
                variant_types['Insertion'] = variant_types.get('Insertion', 0) + 1
            else:
                variant_types['Complex'] = variant_types.get('Complex', 0) + 1
        
        return variant_types
    
    # Helper methods for data analysis
    def _analyze_csv_data(self, headers: List[str], data_rows: List[List[str]]) -> Dict[str, Any]:
        """Analyze CSV data for scientific insights."""
        analysis = {}
        
        for i, header in enumerate(headers):
            if i < len(data_rows[0]):
                column_data = [row[i] for row in data_rows if i < len(row)]
                analysis[header] = self._analyze_column(column_data)
        
        return analysis
    
    def _analyze_column(self, column_data: List[str]) -> Dict[str, Any]:
        """Analyze a single column of data."""
        if not column_data:
            return {'type': 'empty', 'count': 0}
        
        # Try to convert to numeric
        numeric_data = []
        for value in column_data:
            try:
                numeric_data.append(float(value))
            except ValueError:
                continue
        
        if numeric_data:
            return {
                'type': 'numeric',
                'count': len(numeric_data),
                'min': min(numeric_data),
                'max': max(numeric_data),
                'mean': statistics.mean(numeric_data),
                'std': statistics.stdev(numeric_data) if len(numeric_data) > 1 else 0
            }
        else:
            # Categorical data
            unique_values = list(set(column_data))
            value_counts = {value: column_data.count(value) for value in unique_values}
            return {
                'type': 'categorical',
                'count': len(column_data),
                'unique_values': len(unique_values),
                'most_common': max(value_counts.items(), key=lambda x: x[1]) if value_counts else None
            }
    
    def _determine_scientific_relevance(self, headers: List[str], data_rows: List[List[str]]) -> str:
        """Determine the scientific relevance of the data."""
        header_text = ' '.join(headers).lower()
        
        if any(word in header_text for word in ['gene', 'protein', 'sequence', 'mutation']):
            return "Biological/Genomic Data"
        elif any(word in header_text for word in ['concentration', 'ph', 'temperature', 'time']):
            return "Laboratory/Experimental Data"
        elif any(word in header_text for word in ['patient', 'sample', 'treatment', 'outcome']):
            return "Clinical/Medical Data"
        elif any(word in header_text for word in ['species', 'taxonomy', 'abundance']):
            return "Ecological/Environmental Data"
        else:
            return "General Scientific Data"
    
    def _generate_statistical_insights(self, headers: List[str], data_rows: List[List[str]]) -> List[str]:
        """Generate statistical insights from the data."""
        insights = []
        
        if not data_rows:
            return insights
        
        # Basic statistics
        insights.append(f"Dataset contains {len(data_rows)} samples with {len(headers)} variables")
        
        # Check for missing data
        missing_data = 0
        for row in data_rows:
            missing_data += sum(1 for cell in row if not cell or cell.strip() == '')
        
        if missing_data > 0:
            missing_percentage = (missing_data / (len(data_rows) * len(headers))) * 100
            insights.append(f"Missing data: {missing_percentage:.1f}% of cells")
        
        # Check for outliers in numeric columns
        for i, header in enumerate(headers):
            if i < len(data_rows[0]):
                column_data = [row[i] for row in data_rows if i < len(row)]
                numeric_data = []
                for value in column_data:
                    try:
                        numeric_data.append(float(value))
                    except ValueError:
                        continue
                
                if numeric_data and len(numeric_data) > 10:
                    mean_val = statistics.mean(numeric_data)
                    std_val = statistics.stdev(numeric_data) if len(numeric_data) > 1 else 0
                    outliers = [x for x in numeric_data if abs(x - mean_val) > 2 * std_val]
                    
                    if outliers:
                        insights.append(f"Potential outliers detected in {header}: {len(outliers)} values")
        
        return insights
    
    # Helper methods for document analysis
    def _extract_title(self, content: str) -> str:
        """Extract document title."""
        lines = content.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                return line.strip()[:100]
        return "Untitled"
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extract author information."""
        # Look for common author patterns
        author_patterns = [
            r'by\s+([A-Z][a-z]+ [A-Z][a-z]+)',
            r'Author[s]?:\s*([A-Z][a-z]+ [A-Z][a-z]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\s+et al\.'
        ]
        
        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            authors.extend(matches)
        
        return list(set(authors))[:5]  # Limit to 5 authors
    
    def _extract_abstract(self, content: str) -> str:
        """Extract abstract or summary."""
        # Look for abstract section
        abstract_patterns = [
            r'Abstract[:\s]*([^#\n]+)',
            r'Summary[:\s]*([^#\n]+)',
            r'Introduction[:\s]*([^#\n]+)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:300]
        
        # Fallback: first paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and not p.strip().startswith('#')]
        return paragraphs[0][:300] if paragraphs else ""
    
    def _extract_methods(self, content: str) -> List[str]:
        """Extract methods and procedures."""
        methods = []
        
        # Look for methods section
        method_patterns = [
            r'Methods?[:\s]*([^#\n]+)',
            r'Procedure[:\s]*([^#\n]+)',
            r'Protocol[:\s]*([^#\n]+)'
        ]
        
        for pattern in method_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                method_text = match.group(1)
                # Split into individual methods
                method_lines = [line.strip() for line in method_text.split('\n') if line.strip()]
                methods.extend(method_lines[:5])  # Limit to 5 methods
        
        return methods
    
    def _extract_results(self, content: str) -> List[str]:
        """Extract results and findings."""
        results = []
        
        # Look for results section
        result_patterns = [
            r'Results?[:\s]*([^#\n]+)',
            r'Findings?[:\s]*([^#\n]+)',
            r'Outcome[:\s]*([^#\n]+)'
        ]
        
        for pattern in result_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                result_text = match.group(1)
                # Split into individual results
                result_lines = [line.strip() for line in result_text.split('\n') if line.strip()]
                results.extend(result_lines[:5])  # Limit to 5 results
        
        return results
    
    def _extract_conclusions(self, content: str) -> List[str]:
        """Extract conclusions and implications."""
        conclusions = []
        
        # Look for conclusions section
        conclusion_patterns = [
            r'Conclusion[:\s]*([^#\n]+)',
            r'Discussion[:\s]*([^#\n]+)',
            r'Implication[:\s]*([^#\n]+)'
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                conclusion_text = match.group(1)
                # Split into individual conclusions
                conclusion_lines = [line.strip() for line in conclusion_text.split('\n') if line.strip()]
                conclusions.extend(conclusion_lines[:5])  # Limit to 5 conclusions
        
        return conclusions
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords and key terms."""
        # Look for keywords section
        keyword_match = re.search(r'Keywords?[:\s]*([^#\n]+)', content, re.IGNORECASE)
        if keyword_match:
            keyword_text = keyword_match.group(1)
            keywords = [kw.strip() for kw in keyword_text.split(',')]
            return [kw for kw in keywords if len(kw) > 2][:10]
        
        # Fallback: extract important terms
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
        word_freq = {}
        for word in words:
            if word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _analyze_scientific_context(self, content: str) -> str:
        """Analyze the scientific context of the document."""
        content_lower = content.lower()
        
        # Determine scientific field
        if any(word in content_lower for word in ['gene', 'protein', 'dna', 'rna', 'sequence']):
            return "Molecular Biology/Genomics"
        elif any(word in content_lower for word in ['cell', 'tissue', 'organism', 'species']):
            return "Cell Biology/Ecology"
        elif any(word in content_lower for word in ['chemical', 'reaction', 'molecule', 'compound']):
            return "Chemistry/Biochemistry"
        elif any(word in content_lower for word in ['patient', 'clinical', 'medical', 'disease']):
            return "Medical/Clinical Research"
        elif any(word in content_lower for word in ['experiment', 'protocol', 'method', 'assay']):
            return "Laboratory Research"
        else:
            return "General Scientific Research"
    
    def _generate_document_recommendations(self, content: str) -> List[str]:
        """Generate recommendations for scientific documents."""
        recommendations = []
        
        # Check for missing sections
        content_lower = content.lower()
        
        if 'abstract' not in content_lower and 'summary' not in content_lower:
            recommendations.append("Consider adding an abstract or summary section")
        
        if 'method' not in content_lower and 'procedure' not in content_lower:
            recommendations.append("Include detailed methods/procedures for reproducibility")
        
        if 'result' not in content_lower and 'finding' not in content_lower:
            recommendations.append("Clearly present results and findings")
        
        if 'conclusion' not in content_lower and 'discussion' not in content_lower:
            recommendations.append("Add conclusions or discussion section")
        
        if 'reference' not in content_lower and 'citation' not in content_lower:
            recommendations.append("Include proper references and citations")
        
        return recommendations
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content safely."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    
    # Additional helper methods for data analysis
    def _analyze_quality_distribution(self, quality_scores: List[float]) -> Dict[str, Any]:
        """Analyze quality score distribution."""
        if not quality_scores:
            return {}
        
        return {
            'min': min(quality_scores),
            'max': max(quality_scores),
            'mean': statistics.mean(quality_scores),
            'std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            'quartiles': [
                statistics.quantiles(quality_scores, n=4)[0] if len(quality_scores) > 1 else 0,
                statistics.quantiles(quality_scores, n=4)[1] if len(quality_scores) > 1 else 0,
                statistics.quantiles(quality_scores, n=4)[2] if len(quality_scores) > 1 else 0
            ]
        }
    
    def _analyze_length_distribution(self, sequences: List[str]) -> Dict[str, Any]:
        """Analyze sequence length distribution."""
        if not sequences:
            return {}
        
        lengths = [len(seq) for seq in sequences]
        
        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': statistics.mean(lengths),
            'std': statistics.stdev(lengths) if len(lengths) > 1 else 0,
            'median': statistics.median(lengths)
        }
    
    def _analyze_sequence_composition(self, sequences: List[str], is_protein: bool) -> Dict[str, Any]:
        """Analyze sequence composition."""
        if not sequences:
            return {}
        
        if is_protein:
            # Protein composition
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            composition = {}
            for aa in amino_acids:
                total_count = sum(seq.upper().count(aa) for seq in sequences)
                composition[aa] = total_count
        else:
            # DNA composition
            bases = 'ACGTN'
            composition = {}
            for base in bases:
                total_count = sum(seq.upper().count(base) for seq in sequences)
                composition[base] = total_count
        
        return composition
    
    def _analyze_chromosome_distribution(self, variants: List[Dict]) -> Dict[str, int]:
        """Analyze variant distribution across chromosomes."""
        chrom_counts = {}
        for variant in variants:
            chrom = variant['chrom']
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
        return chrom_counts
    
    def _analyze_variant_quality(self, variants: List[Dict]) -> Dict[str, Any]:
        """Analyze variant quality scores."""
        if not variants:
            return {}
        
        qualities = []
        for variant in variants:
            try:
                qual = float(variant['qual'])
                if qual > 0:
                    qualities.append(qual)
            except ValueError:
                continue
        
        if not qualities:
            return {}
        
        return {
            'min': min(qualities),
            'max': max(qualities),
            'mean': statistics.mean(qualities),
            'median': statistics.median(qualities)
        }
    
    def _analyze_variant_impact(self, variants: List[Dict]) -> List[str]:
        """Analyze potential impact of variants."""
        insights = []
        
        # Count variants by chromosome
        chrom_counts = self._analyze_chromosome_distribution(variants)
        if chrom_counts:
            most_affected = max(chrom_counts.items(), key=lambda x: x[1])
            insights.append(f"Most variants on chromosome {most_affected[0]} ({most_affected[1]} variants)")
        
        # Check for high-impact variants
        high_quality_variants = [v for v in variants if float(v.get('qual', 0)) > 50]
        if high_quality_variants:
            insights.append(f"{len(high_quality_variants)} high-quality variants (QUAL > 50)")
        
        return insights
    
    def _generate_variant_recommendations(self, variants: List[Dict]) -> List[str]:
        """Generate recommendations for variant analysis."""
        recommendations = []
        
        if not variants:
            return recommendations
        
        # Check variant quality
        low_quality = [v for v in variants if float(v.get('qual', 0)) < 30]
        if low_quality:
            recommendations.append(f"Filter out {len(low_quality)} low-quality variants (QUAL < 30)")
        
        # Check for rare variants
        if len(variants) > 1000:
            recommendations.append("Large variant set - consider filtering by frequency or impact")
        
        # Check chromosome coverage
        chrom_counts = self._analyze_chromosome_distribution(variants)
        if len(chrom_counts) < 5:
            recommendations.append("Limited chromosome coverage - check for technical issues")
        
        return recommendations
    
    def _generate_data_recommendations(self, headers: List[str], data_rows: List[List[str]]) -> List[str]:
        """Generate recommendations for data analysis."""
        recommendations = []
        
        if not data_rows:
            return recommendations
        
        # Check data size
        if len(data_rows) < 10:
            recommendations.append("Small dataset - consider collecting more samples")
        
        # Check for missing data
        missing_data = 0
        for row in data_rows:
            missing_data += sum(1 for cell in row if not cell or cell.strip() == '')
        
        if missing_data > 0:
            missing_percentage = (missing_data / (len(data_rows) * len(headers))) * 100
            if missing_percentage > 20:
                recommendations.append("High missing data rate - consider imputation or removal")
        
        # Check for data types
        numeric_columns = 0
        for i, header in enumerate(headers):
            if i < len(data_rows[0]):
                column_data = [row[i] for row in data_rows if i < len(row)]
                try:
                    [float(cell) for cell in column_data if cell.strip()]
                    numeric_columns += 1
                except ValueError:
                    continue
        
        if numeric_columns > 0:
            recommendations.append(f"{numeric_columns} numeric columns available for statistical analysis")
        
        return recommendations
    
    def _assess_data_quality(self, headers: List[str], data_rows: List[List[str]]) -> Dict[str, Any]:
        """Assess the quality of the data."""
        if not data_rows:
            return {'quality_score': 0, 'issues': ['No data']}
        
        issues = []
        score = 100
        
        # Check for missing data
        missing_data = 0
        for row in data_rows:
            missing_data += sum(1 for cell in row if not cell or cell.strip() == '')
        
        if missing_data > 0:
            missing_percentage = (missing_data / (len(data_rows) * len(headers))) * 100
            issues.append(f"Missing data: {missing_percentage:.1f}%")
            score -= min(30, missing_percentage * 0.5)
        
        # Check for consistent row lengths
        row_lengths = [len(row) for row in data_rows]
        if len(set(row_lengths)) > 1:
            issues.append("Inconsistent row lengths")
            score -= 20
        
        # Check for empty rows
        empty_rows = sum(1 for row in data_rows if all(not cell or not cell.strip() for cell in row))
        if empty_rows > 0:
            issues.append(f"{empty_rows} empty rows")
            score -= 10
        
        # Check for duplicate rows
        unique_rows = len(set(tuple(row) for row in data_rows))
        if unique_rows < len(data_rows):
            duplicate_count = len(data_rows) - unique_rows
            issues.append(f"{duplicate_count} duplicate rows")
            score -= 15
        
        return {
            'quality_score': max(0, score),
            'issues': issues,
            'missing_data_percentage': missing_data / (len(data_rows) * len(headers)) * 100 if data_rows else 0
        }
    
    def _generate_sequence_recommendations(self, sequences: List[str], is_protein: bool) -> List[str]:
        """Generate recommendations for sequence data."""
        recommendations = []
        
        if not sequences:
            return recommendations
        
        # Check sequence count
        if len(sequences) < 10:
            recommendations.append("Low sequence count - consider adding more sequences for robust analysis")
        
        # Check sequence lengths
        lengths = [len(seq) for seq in sequences]
        avg_length = statistics.mean(lengths)
        
        if avg_length < 50:
            recommendations.append("Short sequences detected - consider longer reads for better coverage")
        
        if avg_length > 1000:
            recommendations.append("Long sequences detected - consider fragmentation for better analysis")
        
        # Check for ambiguous characters
        total_ambiguous = sum(seq.count('N') for seq in sequences)
        if total_ambiguous > 0:
            recommendations.append(f"Ambiguous characters (N) detected: {total_ambiguous} total")
        
        # Protein-specific recommendations
        if is_protein:
            recommendations.append("Protein sequences detected - consider domain analysis and structure prediction")
        else:
            # DNA-specific recommendations
            gc_content = self._calculate_gc_content(sequences)
            if gc_content < 30 or gc_content > 70:
                recommendations.append("Unusual GC content - check for contamination or mixed samples")
        
        return recommendations
    
    def _parse_generic_file(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Parse generic files."""
        content = self._read_file_content(file_path)
        if not content:
            return {'error': 'Could not read file content'}
        
        return {
            'file_type': file_type,
            'generic_analysis': {
                'data_type': 'Generic File',
                'file_size': len(content),
                'line_count': len(content.split('\n')),
                'content_preview': content[:200] + '...' if len(content) > 200 else content
            },
            'recommendations': ['Consider using a more specific parser for better analysis']
        }
    
    def _parse_json_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Parse JSON files."""
        try:
            data = json.loads(content)
            
            # Analyze JSON structure
            if isinstance(data, dict):
                keys = list(data.keys())
                structure_type = "Object"
            elif isinstance(data, list):
                keys = [f"item_{i}" for i in range(min(5, len(data)))]
                structure_type = "Array"
            else:
                keys = []
                structure_type = "Primitive"
            
            return {
                'file_type': 'json',
                'data_analysis': {
                    'data_type': 'JSON Data',
                    'structure_type': structure_type,
                    'top_level_keys': keys[:10],  # Show first 10 keys
                    'data_size': len(str(data))
                },
                'recommendations': ['JSON data detected - consider schema validation']
            }
        except json.JSONDecodeError as e:
            return {
                'file_type': 'json',
                'error': f'Invalid JSON: {str(e)}',
                'recommendations': ['Fix JSON syntax errors']
            }
