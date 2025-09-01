"""
Professional Logging System for SciDoc.

This module provides structured logging and complete warning suppression
for a production-level documentation assistant.
"""

import logging
import warnings
import os
import sys
from typing import Optional
from pathlib import Path

# Suppress ALL warnings globally
warnings.filterwarnings('ignore')

# Set environment variables for TensorFlow/CUDA suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_LOGGING_LEVEL'] = 'ERROR'

# Suppress specific TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

# Suppress CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Suppress NLTK warnings
os.environ['NLTK_DATA'] = str(Path.home() / '.nltk_data')


class SciDocLogger:
    """Professional logging system for SciDoc."""
    
    def __init__(self, name: str = "scidoc", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers with professional formatting."""
        # Console handler with rich formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Professional formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # File handler for detailed logs
        log_dir = Path.home() / '.scidoc' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'scidoc.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


def setup_logging(name: str = "scidoc", level: str = "INFO") -> SciDocLogger:
    """Setup professional logging for SciDoc."""
    return SciDocLogger(name, level)


def suppress_warnings():
    """Comprehensive warning suppression for production use."""
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Suppress specific library warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=ImportWarning)
    
    # Suppress TensorFlow warnings
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        pass
    
    # Suppress NLTK warnings
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except ImportError:
        pass


# Initialize global logger
logger = setup_logging()
suppress_warnings()
