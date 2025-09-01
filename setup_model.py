#!/usr/bin/env python3
"""
Setup script for SciDoc - Downloads the HuggingFace model.

This script downloads the flan-t5-base model from HuggingFace and sets up
the local model directory for offline use with SciDoc.
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(model_name="google/flan-t5-base", model_dir="./models"):
    """
    Download the specified model from HuggingFace.
    
    Args:
        model_name (str): Name of the model to download
        model_dir (str): Directory to save the model
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Model will be saved to: {model_dir}")
        
        # Create model directory
        model_path = Path(model_dir) / model_name.split('/')[-1]
        model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        
        logger.info("Downloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(model_path)
        
        logger.info(f"Model successfully downloaded to: {model_path}")
        
        # Create a README file for the models directory
        readme_path = Path(model_dir) / "README.md"
        readme_content = """# SciDoc Models

This directory contains the AI models used by SciDoc for text summarization and analysis.

## Model Information

- **flan-t5-base**: Google's Flan-T5 model for text generation and summarization
- **Size**: Approximately 1GB
- **License**: Apache 2.0
- **Source**: HuggingFace Hub

## Git LFS

Large model files are tracked using Git LFS (Large File Storage). When cloning the repository:

1. Install Git LFS: `git lfs install`
2. Pull LFS files: `git lfs pull`

## Manual Download

If you need to download the model manually:

```bash
python setup_model.py
```

## Model Usage

The model is automatically loaded by SciDoc when needed. No manual configuration required.
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info("Created models/README.md")
        return str(model_path)
        
    except ImportError:
        logger.error("transformers library not found. Please install it first:")
        logger.error("pip install transformers torch")
        return None
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def main():
    """Main function to download the model."""
    parser = argparse.ArgumentParser(description="Download HuggingFace model for SciDoc")
    parser.add_argument("--model", default="google/flan-t5-base", 
                       help="Model name to download (default: google/flan-t5-base)")
    parser.add_argument("--dir", default="./models", 
                       help="Directory to save the model (default: ./models)")
    
    args = parser.parse_args()
    
    logger.info("Starting model download for SciDoc...")
    
    # Download the model
    model_path = download_model(args.model, args.dir)
    
    if model_path:
        logger.info("Model setup completed successfully!")
        logger.info(f"Model location: {model_path}")
        logger.info("You can now use SciDoc with the downloaded model.")
    else:
        logger.error("Model setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
