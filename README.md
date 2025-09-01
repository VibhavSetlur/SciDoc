# SciDoc: Intelligent Scientific Documentation & Analysis Assistant

SciDoc is a scientific co-assistant that provides intelligent analysis, summarization, and documentation for research projects. It interprets biological data, scientific documents, and generates meaningful insights for researchers.

## Scientific Capabilities

- **Biological Data Analysis**: Intelligent parsing of FASTQ, FASTA, VCF, BAM, and SAM files
- **Sequence Analysis**: GC content, quality metrics, sequence composition, and biological context
- **Variant Analysis**: SNP classification, chromosome distribution, and quality assessment
- **Data Quality Assessment**: Statistical analysis, outlier detection, and data validation
- **Scientific Document Parsing**: Extract abstracts, methods, results, and conclusions from papers
- **Intelligent Recommendations**: Context-aware suggestions for data quality and research workflow

## Commands

SciDoc provides 4 essential commands:

- **`explore`** - Analyze project structure with user-friendly summaries and cache difference review
- **`summarize`** - Get intelligent, context-aware summaries of files and directories
- **`chat`** - Interactive Q&A about project content and scientific data
- **`generate`** - Create comprehensive .scidoc files with scientific analysis and insights

## Quick Setup

```bash
# One-time setup
./setup.sh

# Usage
scidoc --help
scidoc explore example_project/
scidoc summarize example_project/
scidoc chat example_project/
scidoc generate example_project/
```

## .scidoc Files

SciDoc generates comprehensive `.scidoc` files containing:

- **Project Summary**: Overview and scientific assessment
- **Scientific Analysis**: File type distribution and insights
- **Biological Insights**: Sequence and variant analysis for genomic data
- **Statistical Summary**: Data analysis results and quality metrics
- **Data Quality Assessment**: Issues, scores, and recommendations
- **Next Steps**: Research workflow recommendations

### Example .scidoc Generation

```bash
# Generate comprehensive .scidoc file
scidoc generate my_research_project/

# Generate summary .scidoc only
scidoc generate my_research_project/ --summary
```

## Scientific File Support

### Biological Data
- **FASTQ**: Sequencing reads with quality scores and biological context
- **FASTA**: DNA/protein sequences with composition analysis
- **VCF**: Genetic variants with chromosome distribution and impact analysis
- **BAM/SAM**: Aligned sequencing data

### Data Files
- **CSV/TSV**: Statistical analysis with scientific relevance detection
- **JSON**: Structured data interpretation
- **YAML**: Configuration and metadata files

### Documents
- **Markdown**: Scientific paper and protocol parsing
- **Text**: Research notes and laboratory protocols
- **PDF**: Document structure extraction (when available)

## Analysis Features

- **Context-Aware Summaries**: Focus on scientific relevance, not just file counts
- **Biological Insights**: Automatic detection of sequencing issues and quality problems
- **Statistical Validation**: Outlier detection, missing data assessment, and data quality scoring
- **Research Recommendations**: Domain-specific suggestions for improving research workflows
- **Scientific Relevance**: Automatic classification of data types and research domains

## Installation

### Prerequisites
- Python 3.10+
- Conda or Miniconda

### Quick Setup
```bash
git clone <repository>
cd SciDoc
./setup.sh
```

The setup script will:
- Create a conda environment with all dependencies
- Install SciDoc in development mode
- Download the AI model for summarization
- Attempt to create a global `scidoc` command (user-friendly, no sudo required)
- Set up an example project

### Manual Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Install SciDoc
conda run -n scidoc pip install -e .

# Download model
conda run -n scidoc python setup_model.py

# Create global command (optional, no sudo required)
# The setup script will attempt this automatically
```

## Usage Examples

### Explore a Project
```bash
scidoc explore my_research_project/
```

### Get Intelligent Summaries
```bash
# Directory summary
scidoc summarize my_research_project/

# File summary
scidoc summarize my_research_project/data.fastq
```

### Interactive Chat
```bash
scidoc chat my_research_project/
```

### Generate Scientific Documentation
```bash
# Comprehensive analysis
scidoc generate my_research_project/

# Summary only
scidoc generate my_research_project/ --summary
```

## Configuration

SciDoc uses intelligent defaults but can be configured through environment variables:

- `SCIDOC_CACHE_DIR`: Cache directory for metadata
- `SCIDOC_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `SCIDOC_MODEL_PATH`: Path to AI model for summarization

## Example Projects

SciDoc comes with example projects to demonstrate its capabilities:

- **Python Project**: Code analysis and documentation
- **Data Analysis**: CSV processing and statistical insights
- **Bioinformatics**: Sequence data analysis and quality assessment

## Contributing

SciDoc is designed for researchers and scientists. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or feature requests:

1. Check the existing issues
2. Create a new issue with detailed information
3. Include example files and expected behavior

---

**SciDoc**: Making scientific research more accessible, understandable, and reproducible through intelligent documentation and analysis.
