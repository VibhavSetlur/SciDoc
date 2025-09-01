#!/bin/bash
# SciDoc Setup - Simple one-time setup

set -e

echo "SciDoc Setup - Documentation & Summary Assistant"
echo "================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$python_version >= 3.10" | bc -l) -eq 0 ]]; then
    echo "Error: Python 3.10+ is required, found Python $python_version"
    exit 1
fi

echo "Python $python_version detected"

# Create conda environment (skip if exists)
if conda env list | grep -q "^scidoc "; then
    echo "Using existing conda environment 'scidoc'"
else
    echo "Creating conda environment 'scidoc'..."
    conda env create -f environment.yml
fi

# Install SciDoc
echo "Installing SciDoc..."
conda run -n scidoc pip install -e .

# Download model (skip if exists)
if [ ! -d "models/flan-t5-base" ]; then
    echo "Downloading AI model..."
    conda run -n scidoc python setup_model.py
else
    echo "AI model already exists"
fi

# Create global scidoc command
echo "Creating global scidoc command..."

# Create the runner script
cat > scidoc.sh << 'EOF'
#!/bin/bash
# SciDoc Runner

# Suppress warnings
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""
export TF_ENABLE_ONEDNN_OPTS=0
export TF_ENABLE_DEPRECATION_WARNINGS=0
export PYTHONWARNINGS="ignore"

# Run SciDoc
conda run -n scidoc scidoc "$@" 2>/dev/null
exit $?
EOF

chmod +x scidoc.sh

# Create global command (user-friendly approach)
echo "Setting up global scidoc command..."

# Ensure ~/.local/bin exists
mkdir -p ~/.local/bin

# Try to create global command without sudo first
if ln -sf $(pwd)/scidoc.sh ~/.local/bin/scidoc 2>/dev/null; then
    echo "Global command created in ~/.local/bin/scidoc"
    
    # Check if ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo ""
        echo "IMPORTANT: ~/.local/bin is not in your PATH"
        echo "Add this line to your ~/.bashrc or ~/.zshrc:"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
        echo "Then restart your terminal or run: source ~/.bashrc"
    else
        echo "Global command is ready to use!"
    fi
elif command -v sudo >/dev/null 2>&1; then
    echo "Attempting to create global command with sudo..."
    if sudo ln -sf $(pwd)/scidoc.sh /usr/local/bin/scidoc 2>/dev/null; then
        echo "Global command created in /usr/local/bin/scidoc"
    else
        echo "Could not create global command. You can still use ./scidoc.sh"
    fi
else
    echo "Could not create global command. You can still use ./scidoc.sh"
fi

# Create example project
echo "Creating example project..."
mkdir -p example_project
cat > example_project/main.py << 'EOF'
#!/usr/bin/env python3
"""
Example Python script for SciDoc testing.
"""

import pandas as pd
import numpy as np

def process_data(data):
    """Process the input data."""
    return data * 2

def main():
    """Main function."""
    print("Hello from SciDoc example!")
    data = np.array([1, 2, 3, 4, 5])
    result = process_data(data)
    print(f"Processed data: {result}")

if __name__ == "__main__":
    main()
EOF

cat > example_project/data.csv << 'EOF'
name,age,city
Alice,25,New York
Bob,30,San Francisco
Charlie,35,Chicago
EOF

cat > example_project/README.md << 'EOF'
# Example Project

This is an example project for testing SciDoc functionality.

## Files

- `main.py`: Main Python script
- `data.csv`: Sample data file
- `README.md`: This documentation file

## Usage

Run the main script:
```bash
python main.py
```
EOF

echo ""
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  scidoc --help"
echo "  scidoc explore example_project/"
echo "  scidoc summarize example_project/"
echo "  scidoc chat example_project/"
echo "  scidoc generate example_project/"
echo ""
echo "Try: scidoc summarize example_project/"
echo ""
echo ""
echo "Setup Summary:"
echo "✓ SciDoc environment created"
echo "✓ Dependencies installed"
echo "✓ AI model downloaded"
echo "✓ Global command created (if possible)"
echo ""
echo "You can now use:"
echo "  scidoc --help                    (if global command works)"
echo "  ./scidoc.sh --help               (fallback option)"
echo ""
echo "If the global command doesn't work, you can still use ./scidoc.sh"
