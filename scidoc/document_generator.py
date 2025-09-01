"""
Scientific Document Generator for SciDoc.

This module generates .scidoc files that contain:
- Scientific analysis results
- Biological data insights
- Statistical summaries
- Recommendations and next steps
- Metadata and provenance information
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .logger import setup_logging
from .scientific_parser import ScientificParser

logger = setup_logging("scidoc.document_generator")


@dataclass
class SciDocMetadata:
    """Metadata for .scidoc files."""
    generated_at: str
    scidoc_version: str
    source_files: List[str]
    analysis_type: str
    scientific_domain: str
    data_quality_score: Optional[float]
    recommendations_count: int
    insights_count: int


@dataclass
class SciDocContent:
    """Content structure for .scidoc files."""
    summary: str
    scientific_analysis: Dict[str, Any]
    biological_insights: Optional[Dict[str, Any]]
    statistical_summary: Optional[Dict[str, Any]]
    data_quality_assessment: Optional[Dict[str, Any]]
    recommendations: List[str]
    next_steps: List[str]
    technical_details: Dict[str, Any]


class SciDocGenerator:
    """Generates comprehensive .scidoc files for scientific projects."""
    
    def __init__(self):
        self.parser = ScientificParser()
        self.scidoc_version = "1.0.0"
    
    def generate_scidoc(self, directory_path: Path, output_path: Optional[Path] = None) -> Path:
        """Generate a comprehensive .scidoc file for a directory."""
        logger.info(f"Generating .scidoc file for: {directory_path}")
        
        if output_path is None:
            output_path = directory_path / f"{directory_path.name}.scidoc"
        
        # Analyze the directory
        analysis_results = self._analyze_directory(directory_path)
        
        # Generate content
        scidoc_content = self._create_scidoc_content(directory_path, analysis_results)
        
        # Write the file
        self._write_scidoc_file(output_path, scidoc_content)
        
        logger.info(f"Generated .scidoc file: {output_path}")
        return output_path
    
    def _analyze_directory(self, directory_path: Path) -> Dict[str, Any]:
        """Analyze a directory and its contents."""
        analysis = {
            'directory_info': {
                'name': directory_path.name,
                'path': str(directory_path),
                'total_files': 0,
                'file_types': {},
                'scientific_files': [],
                'biological_files': [],
                'data_files': [],
                'document_files': []
            },
            'file_analyses': {},
            'overall_assessment': {},
            'recommendations': []
        }
        
        # Get all files
        all_files = list(directory_path.rglob('*'))
        files = [f for f in all_files if f.is_file() and not f.name.startswith('.')]
        
        analysis['directory_info']['total_files'] = len(files)
        
        # Analyze each file
        for file_path in files:
            try:
                file_analysis = self.parser.parse_file(file_path)
                analysis['file_analyses'][file_path.name] = file_analysis
                
                # Categorize files
                self._categorize_file(file_path, file_analysis, analysis)
                
            except Exception as e:
                logger.warning(f"Could not analyze file {file_path}: {e}")
                analysis['file_analyses'][file_path.name] = {'error': str(e)}
        
        # Generate overall assessment
        analysis['overall_assessment'] = self._generate_overall_assessment(analysis)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_overall_recommendations(analysis)
        
        return analysis
    
    def _categorize_file(self, file_path: Path, file_analysis: Dict[str, Any], analysis: Dict[str, Any]):
        """Categorize a file based on its analysis."""
        file_type = file_analysis.get('file_type', 'unknown')
        
        if file_type in ['fastq', 'fasta', 'vcf', 'bam', 'sam']:
            analysis['directory_info']['biological_files'].append(file_path.name)
        elif file_type in ['csv', 'tsv', 'json']:
            analysis['directory_info']['data_files'].append(file_path.name)
        elif file_type in ['md', 'txt', 'pdf']:
            analysis['directory_info']['document_files'].append(file_path.name)
        
        # Count file types
        if file_type not in analysis['directory_info']['file_types']:
            analysis['directory_info']['file_types'][file_type] = 0
        analysis['directory_info']['file_types'][file_type] += 1
        
        # Check if it's a scientific file
        if 'biological_data' in file_analysis or 'data_analysis' in file_analysis or 'document_analysis' in file_analysis:
            analysis['directory_info']['scientific_files'].append(file_path.name)
    
    def _generate_overall_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of the project."""
        assessment = {
            'project_type': 'Unknown',
            'scientific_domain': 'Unknown',
            'data_quality_score': 0.0,
            'complexity_level': 'Low',
            'research_maturity': 'Unknown'
        }
        
        # Determine project type
        bio_count = len(analysis['directory_info']['biological_files'])
        data_count = len(analysis['directory_info']['data_files'])
        doc_count = len(analysis['directory_info']['document_files'])
        
        if bio_count > 0:
            assessment['project_type'] = 'Biological Research'
            assessment['scientific_domain'] = 'Biology/Genomics'
        elif data_count > 0:
            assessment['project_type'] = 'Data Analysis'
            assessment['scientific_domain'] = 'Data Science'
        elif doc_count > 0:
            assessment['project_type'] = 'Documentation/Research'
            assessment['scientific_domain'] = 'General Research'
        
        # Calculate data quality score
        quality_scores = []
        for file_analysis in analysis['file_analyses'].values():
            if 'data_quality' in file_analysis and 'quality_score' in file_analysis['data_quality']:
                quality_scores.append(file_analysis['data_quality']['quality_score'])
        
        if quality_scores:
            assessment['data_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # Determine complexity level
        total_files = analysis['directory_info']['total_files']
        if total_files > 100:
            assessment['complexity_level'] = 'High'
        elif total_files > 20:
            assessment['complexity_level'] = 'Medium'
        else:
            assessment['complexity_level'] = 'Low'
        
        # Determine research maturity
        if 'README' in [f.lower() for f in analysis['directory_info']['scientific_files']]:
            assessment['research_maturity'] = 'Established'
        elif total_files > 10:
            assessment['research_maturity'] = 'Developing'
        else:
            assessment['research_maturity'] = 'Initial'
        
        return assessment
    
    def _generate_overall_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations for the project."""
        recommendations = []
        
        # Check for missing documentation
        if not any('readme' in f.lower() for f in analysis['directory_info']['scientific_files']):
            recommendations.append("Add a README file to document the project purpose and structure")
        
        # Check for data quality
        if analysis['overall_assessment']['data_quality_score'] < 70:
            recommendations.append("Address data quality issues - consider data cleaning and validation")
        
        # Check for biological data
        if analysis['directory_info']['biological_files']:
            recommendations.append("Consider adding metadata files for biological samples and experimental conditions")
        
        # Check for data files
        if analysis['directory_info']['data_files']:
            recommendations.append("Document data sources, collection methods, and processing steps")
        
        # Check for code files
        if 'python' in analysis['directory_info']['file_types']:
            recommendations.append("Add code documentation and requirements.txt for reproducibility")
        
        # Check project size
        if analysis['directory_info']['total_files'] > 50:
            recommendations.append("Consider organizing files into logical subdirectories")
        
        return recommendations
    
    def _create_scidoc_content(self, directory_path: Path, analysis: Dict[str, Any]) -> SciDocContent:
        """Create the content for the .scidoc file."""
        # Generate summary
        summary = self._generate_project_summary(directory_path, analysis)
        
        # Extract scientific analysis
        scientific_analysis = self._extract_scientific_analysis(analysis)
        
        # Extract biological insights
        biological_insights = self._extract_biological_insights(analysis)
        
        # Extract statistical summary
        statistical_summary = self._extract_statistical_summary(analysis)
        
        # Extract data quality assessment
        data_quality_assessment = self._extract_data_quality_assessment(analysis)
        
        # Get recommendations and next steps
        recommendations = analysis.get('recommendations', [])
        next_steps = self._generate_next_steps(analysis)
        
        # Technical details
        technical_details = self._extract_technical_details(analysis)
        
        return SciDocContent(
            summary=summary,
            scientific_analysis=scientific_analysis,
            biological_insights=biological_insights,
            statistical_summary=statistical_summary,
            data_quality_assessment=data_quality_assessment,
            recommendations=recommendations,
            next_steps=next_steps,
            technical_details=technical_details
        )
    
    def _generate_project_summary(self, directory_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive project summary."""
        assessment = analysis['overall_assessment']
        info = analysis['directory_info']
        
        summary = f"This is a {assessment['project_type']} project in the {assessment['scientific_domain']} domain. "
        summary += f"The project contains {info['total_files']} files with a complexity level of {assessment['complexity_level']}. "
        
        if info['biological_files']:
            summary += f"It includes {len(info['biological_files'])} biological data files for genomic or sequence analysis. "
        
        if info['data_files']:
            summary += f"There are {len(info['data_files'])} data files for statistical analysis and research. "
        
        if info['document_files']:
            summary += f"The project includes {len(info['document_files'])} documentation and research files. "
        
        summary += f"Overall data quality score: {assessment['data_quality_score']:.1f}/100. "
        summary += f"Research maturity level: {assessment['research_maturity']}."
        
        return summary
    
    def _extract_scientific_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scientific analysis from file analyses."""
        scientific_analysis = {
            'file_type_distribution': analysis['directory_info']['file_types'],
            'scientific_files_count': len(analysis['directory_info']['scientific_files']),
            'biological_files_count': len(analysis['directory_info']['biological_files']),
            'data_files_count': len(analysis['directory_info']['data_files']),
            'document_files_count': len(analysis['directory_info']['document_files'])
        }
        
        # Add specific scientific insights
        scientific_insights = []
        for filename, file_analysis in analysis['file_analyses'].items():
            if 'biological_data' in file_analysis:
                bio_data = file_analysis['biological_data']
                scientific_insights.append({
                    'file': filename,
                    'type': bio_data.get('data_type', 'Unknown'),
                    'count': bio_data.get('sequence_count', bio_data.get('variant_count', 0)),
                    'context': bio_data.get('biological_context', '')
                })
            elif 'data_analysis' in file_analysis:
                data_analysis = file_analysis['data_analysis']
                scientific_insights.append({
                    'file': filename,
                    'type': data_analysis.get('data_type', 'Unknown'),
                    'dimensions': data_analysis.get('dimensions', (0, 0)),
                    'relevance': data_analysis.get('scientific_relevance', 'Unknown')
                })
        
        scientific_analysis['scientific_insights'] = scientific_insights
        
        return scientific_analysis
    
    def _extract_biological_insights(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract biological insights from the analysis."""
        if not analysis['directory_info']['biological_files']:
            return None
        
        biological_insights = {
            'total_biological_files': len(analysis['directory_info']['biological_files']),
            'file_types': {},
            'total_sequences': 0,
            'total_variants': 0,
            'quality_metrics': {}
        }
        
        for filename, file_analysis in analysis['file_analyses'].items():
            if 'biological_data' in file_analysis:
                bio_data = file_analysis['biological_data']
                file_type = bio_data.get('data_type', 'Unknown')
                
                if file_type not in biological_insights['file_types']:
                    biological_insights['file_types'][file_type] = 0
                biological_insights['file_types'][file_type] += 1
                
                if 'sequence_count' in bio_data:
                    biological_insights['total_sequences'] += bio_data['sequence_count']
                if 'variant_count' in bio_data:
                    biological_insights['total_variants'] += bio_data['variant_count']
                
                # Collect quality metrics
                if 'quality_metrics' in file_analysis:
                    biological_insights['quality_metrics'][filename] = file_analysis['quality_metrics']
        
        return biological_insights
    
    def _extract_statistical_summary(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract statistical summary from the analysis."""
        if not analysis['directory_info']['data_files']:
            return None
        
        statistical_summary = {
            'total_data_files': len(analysis['directory_info']['data_files']),
            'data_types': {},
            'total_samples': 0,
            'total_variables': 0,
            'statistical_insights': []
        }
        
        for filename, file_analysis in analysis['file_analyses'].items():
            if 'data_analysis' in file_analysis:
                data_analysis = file_analysis['data_analysis']
                data_type = data_analysis.get('data_type', 'Unknown')
                
                if data_type not in statistical_summary['data_types']:
                    statistical_summary['data_types'][data_type] = 0
                statistical_summary['data_types'][data_type] += 1
                
                dimensions = data_analysis.get('dimensions', (0, 0))
                statistical_summary['total_samples'] += dimensions[0]
                statistical_summary['total_variables'] += dimensions[1]
                
                # Collect statistical insights
                if 'statistical_insights' in file_analysis:
                    statistical_summary['statistical_insights'].extend(file_analysis['statistical_insights'])
        
        return statistical_summary
    
    def _extract_data_quality_assessment(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract data quality assessment from the analysis."""
        quality_assessments = []
        
        for filename, file_analysis in analysis['file_analyses'].items():
            if 'data_quality' in file_analysis:
                quality_assessments.append({
                    'file': filename,
                    'quality_score': file_analysis['data_quality'].get('quality_score', 0),
                    'issues': file_analysis['data_quality'].get('issues', [])
                })
        
        if not quality_assessments:
            return None
        
        overall_score = sum(qa['quality_score'] for qa in quality_assessments) / len(quality_assessments)
        all_issues = []
        for qa in quality_assessments:
            all_issues.extend(qa['issues'])
        
        return {
            'overall_quality_score': overall_score,
            'file_quality_scores': quality_assessments,
            'common_issues': list(set(all_issues)),
            'files_with_issues': len([qa for qa in quality_assessments if qa['issues']])
        }
    
    def _generate_next_steps(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate next steps for the project."""
        next_steps = []
        
        # Based on project type
        if analysis['overall_assessment']['project_type'] == 'Biological Research':
            next_steps.extend([
                "Validate biological data quality and completeness",
                "Document experimental protocols and sample information",
                "Consider downstream analysis pipelines",
                "Implement version control for data files"
            ])
        elif analysis['overall_assessment']['project_type'] == 'Data Analysis':
            next_steps.extend([
                "Perform exploratory data analysis",
                "Implement statistical testing procedures",
                "Create data visualization summaries",
                "Document analysis workflows and code"
            ])
        
        # General next steps
        next_steps.extend([
            "Review and address data quality issues",
            "Create comprehensive project documentation",
            "Implement reproducible research practices",
            "Consider collaboration and sharing protocols"
        ])
        
        return next_steps
    
    def _extract_technical_details(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical details from the analysis."""
        return {
            'total_files': analysis['directory_info']['total_files'],
            'file_type_breakdown': analysis['directory_info']['file_types'],
            'analysis_timestamp': datetime.now().isoformat(),
            'scidoc_version': self.scidoc_version,
            'analysis_scope': 'comprehensive'
        }
    
    def _write_scidoc_file(self, output_path: Path, content: SciDocContent):
        """Write the .scidoc file to disk."""
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        scidoc_data = asdict(content)
        
        # Add metadata
        metadata = SciDocMetadata(
            generated_at=datetime.now().isoformat(),
            scidoc_version=self.scidoc_version,
            source_files=[],  # Will be populated by caller
            analysis_type="comprehensive",
            scientific_domain="mixed",
            data_quality_score=scidoc_data.get('data_quality_assessment', {}).get('overall_quality_score'),
            recommendations_count=len(scidoc_data.get('recommendations', [])),
            insights_count=len(scidoc_data.get('scientific_analysis', {}).get('scientific_insights', []))
        )
        
        scidoc_data['metadata'] = asdict(metadata)
        
        # Write as JSON (more readable than YAML for complex data)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scidoc_data, f, indent=2, ensure_ascii=False)
    
    def generate_summary_scidoc(self, directory_path: Path, output_path: Optional[Path] = None) -> Path:
        """Generate a simplified summary .scidoc file."""
        if output_path is None:
            output_path = directory_path / f"{directory_path.name}_summary.scidoc"
        
        # Analyze the directory
        analysis_results = self._analyze_directory(directory_path)
        
        # Create simplified content
        summary_content = SciDocContent(
            summary=self._generate_project_summary(directory_path, analysis_results),
            scientific_analysis={
                'project_type': analysis_results['overall_assessment']['project_type'],
                'scientific_domain': analysis_results['overall_assessment']['scientific_domain'],
                'total_files': analysis_results['directory_info']['total_files'],
                'key_findings': self._extract_key_findings(analysis_results)
            },
            biological_insights=None,
            statistical_summary=None,
            data_quality_assessment=None,
            recommendations=analysis_results.get('recommendations', [])[:5],  # Top 5
            next_steps=self._generate_next_steps(analysis_results)[:3],  # Top 3
            technical_details={
                'analysis_timestamp': datetime.now().isoformat(),
                'scidoc_version': self.scidoc_version,
                'analysis_scope': 'summary'
            }
        )
        
        # Write the file
        self._write_scidoc_file(output_path, summary_content)
        
        return output_path
    
    def _extract_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []
        
        # Add project type finding
        findings.append(f"Project type: {analysis['overall_assessment']['project_type']}")
        
        # Add file count finding
        findings.append(f"Total files: {analysis['directory_info']['total_files']}")
        
        # Add biological data finding if present
        if analysis['directory_info']['biological_files']:
            findings.append(f"Biological data files: {len(analysis['directory_info']['biological_files'])}")
        
        # Add data quality finding
        quality_score = analysis['overall_assessment']['data_quality_score']
        findings.append(f"Data quality score: {quality_score:.1f}/100")
        
        return findings
