# SciDoc Installation Guide

## Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd SciDoc

# Run the setup script
./setup.sh
```

That's it! The setup script will:
- Create a conda environment with all dependencies
- Install SciDoc in development mode
- Download the AI model for summarization
- Create a global `scidoc` command (no sudo required)
- Set up an example project

## Usage

After setup, you can use SciDoc with:

```bash
# Get help
scidoc --help

# Explore a project
scidoc explore my_research_project/

# Get summaries
scidoc summarize my_research_project/

# Interactive chat
scidoc chat my_research_project/

# Generate .scidoc files
scidoc generate my_research_project/
```

## Troubleshooting

### Global Command Not Working

If the `scidoc` command isn't found, the setup script will create it in `~/.local/bin`. Make sure this directory is in your PATH:

```bash
# Add to your shell configuration (~/.bashrc, ~/.zshrc, etc.)
export PATH="$HOME/.local/bin:$PATH"

# Then restart your terminal or run:
source ~/.bashrc
```

### Fallback Option

If the global command doesn't work, you can still use:

```bash
./scidoc.sh --help
./scidoc.sh explore my_research_project/
```

### Permission Issues

The setup script tries to create the global command without requiring sudo. If you encounter issues:

1. Check if `~/.local/bin` exists and is in your PATH
2. Try running the setup script again
3. Use the fallback `./scidoc.sh` option

## Requirements

- Python 3.10+
- Conda or Miniconda
- Git

## Manual Setup (Advanced)

If you prefer manual setup:

```bash
# Create conda environment
conda env create -f environment.yml

# Install SciDoc
conda run -n scidoc pip install -e .

# Download model
conda run -n scidoc python setup_model.py

# Create global command manually
ln -sf $(pwd)/scidoc.sh ~/.local/bin/scidoc
```

## Support

For issues or questions:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include example files and expected behavior
