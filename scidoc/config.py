"""
Configuration management for SciDoc.

This module handles loading and managing configuration from files,
environment variables, and command-line arguments.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from .models import ProjectConfig, SummarizerBackend


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        "summarizer": SummarizerBackend.HUGGINGFACE,
        "model_path": "./models/flan-t5-base",
        "max_length": 200,
        "temperature": 0.7,
        "github_enabled": False,
        "github_token": None,
        "github_webhook_secret": None,
        "metadata_dir": ".metadata",
        "cache_dir": ".scidoc_cache",
        "enabled_parsers": [
            "fastq", "bam", "vcf", "csv", "json", "yaml",
            "python", "jupyter", "markdown", "image", "pdf"
        ],
        "validation_rules_file": None,
        "auto_fix": False,
        "log_level": "INFO",
        "log_format": "json",
        "log_file": None,
    }


def load_env_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}
    
    # Summarizer configuration
    if os.getenv("SCIDOC_SUMMARIZER"):
        config["summarizer"] = SummarizerBackend(os.getenv("SCIDOC_SUMMARIZER"))
    
    if os.getenv("SCIDOC_MODEL_PATH"):
        config["model_path"] = os.getenv("SCIDOC_MODEL_PATH")
    
    if os.getenv("SCIDOC_MAX_LENGTH"):
        config["max_length"] = int(os.getenv("SCIDOC_MAX_LENGTH"))
    
    if os.getenv("SCIDOC_TEMPERATURE"):
        config["temperature"] = float(os.getenv("SCIDOC_TEMPERATURE"))
    
    # GitHub configuration
    if os.getenv("SCIDOC_GITHUB_ENABLED"):
        config["github_enabled"] = os.getenv("SCIDOC_GITHUB_ENABLED").lower() == "true"
    
    if os.getenv("GITHUB_TOKEN"):
        config["github_token"] = os.getenv("GITHUB_TOKEN")
    
    if os.getenv("SCIDOC_GITHUB_WEBHOOK_SECRET"):
        config["github_webhook_secret"] = os.getenv("SCIDOC_GITHUB_WEBHOOK_SECRET")
    
    # Storage configuration
    if os.getenv("SCIDOC_METADATA_DIR"):
        config["metadata_dir"] = os.getenv("SCIDOC_METADATA_DIR")
    
    if os.getenv("SCIDOC_CACHE_DIR"):
        config["cache_dir"] = os.getenv("SCIDOC_CACHE_DIR")
    
    # Parser configuration
    if os.getenv("SCIDOC_ENABLED_PARSERS"):
        config["enabled_parsers"] = os.getenv("SCIDOC_ENABLED_PARSERS").split(",")
    
    # Validation configuration
    if os.getenv("SCIDOC_VALIDATION_RULES_FILE"):
        config["validation_rules_file"] = os.getenv("SCIDOC_VALIDATION_RULES_FILE")
    
    if os.getenv("SCIDOC_AUTO_FIX"):
        config["auto_fix"] = os.getenv("SCIDOC_AUTO_FIX").lower() == "true"
    
    # Logging configuration
    if os.getenv("SCIDOC_LOG_LEVEL"):
        config["log_level"] = os.getenv("SCIDOC_LOG_LEVEL")
    
    if os.getenv("SCIDOC_LOG_FORMAT"):
        config["log_format"] = os.getenv("SCIDOC_LOG_FORMAT")
    
    if os.getenv("SCIDOC_LOG_FILE"):
        config["log_file"] = os.getenv("SCIDOC_LOG_FILE")
    
    return config


def find_config_file(project_path: Path) -> Optional[Path]:
    """Find configuration file in project directory."""
    config_names = ["scidoc.yaml", "scidoc.yml", ".scidoc.yaml", ".scidoc.yml"]
    
    for config_name in config_names:
        config_path = project_path / config_name
        if config_path.exists():
            return config_path
    
    # Check parent directories
    for parent in project_path.parents:
        for config_name in config_names:
            config_path = parent / config_name
            if config_path.exists():
                return config_path
    
    return None


def load_config(project_path: Optional[Path] = None) -> ProjectConfig:
    """Load configuration from file and environment variables."""
    if project_path is None:
        project_path = Path.cwd()
    
    # Start with default configuration
    config_data = get_default_config()
    
    # Load from file if exists
    config_file = find_config_file(project_path)
    if config_file:
        try:
            file_config = ProjectConfig.from_file(config_file)
            config_data.update(file_config.dict())
        except Exception as e:
            print(f"Warning: Failed to load config from {config_file}: {e}")
    
    # Override with environment variables
    env_config = load_env_config()
    config_data.update(env_config)
    
    return ProjectConfig(**config_data)


def get_config(project_path: Optional[Path] = None) -> ProjectConfig:
    """Get configuration for the current project."""
    return load_config(project_path)


def create_default_config(project_path: Path, config_name: str = "scidoc.yaml") -> Path:
    """Create a default configuration file in the project directory."""
    config_path = project_path / config_name
    
    if config_path.exists():
        raise FileExistsError(f"Configuration file {config_path} already exists")
    
    # Create default configuration
    config = ProjectConfig()
    config.save(config_path)
    
    return config_path


def validate_config(config: ProjectConfig) -> bool:
    """Validate configuration values."""
    errors = []
    
    # Check model path
    if config.model_path and not Path(config.model_path).exists():
        if not config.model_path.startswith(("http://", "https://")):
            errors.append(f"Model path does not exist: {config.model_path}")
    
    # Check directories
    if config.metadata_dir:
        metadata_path = Path(config.metadata_dir)
        if metadata_path.exists() and not metadata_path.is_dir():
            errors.append(f"Metadata directory is not a directory: {config.metadata_dir}")
    
    if config.cache_dir:
        cache_path = Path(config.cache_dir)
        if cache_path.exists() and not cache_path.is_dir():
            errors.append(f"Cache directory is not a directory: {config.cache_dir}")
    
    # Check GitHub configuration
    if config.github_enabled and not config.github_token:
        errors.append("GitHub integration enabled but no token provided")
    
    # Check validation rules file
    if config.validation_rules_file and not Path(config.validation_rules_file).exists():
        errors.append(f"Validation rules file does not exist: {config.validation_rules_file}")
    
    # Check log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.log_level.upper() not in valid_log_levels:
        errors.append(f"Invalid log level: {config.log_level}")
    
    # Check log format
    valid_log_formats = ["json", "text"]
    if config.log_format.lower() not in valid_log_formats:
        errors.append(f"Invalid log format: {config.log_format}")
    
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        return False
    
    return True


def setup_logging(config: ProjectConfig) -> None:
    """Setup logging based on configuration."""
    import logging
    import structlog
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if config.log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=open(config.log_file, "w") if config.log_file else None,
        level=getattr(logging, config.log_level.upper()),
    )


def get_cache_dir(config: ProjectConfig, project_path: Path) -> Path:
    """Get the cache directory for the project."""
    cache_dir = Path(config.cache_dir)
    
    if cache_dir.is_absolute():
        return cache_dir
    else:
        return project_path / cache_dir


def get_metadata_dir(config: ProjectConfig, project_path: Path) -> Path:
    """Get the metadata directory for the project."""
    metadata_dir = Path(config.metadata_dir)
    
    if metadata_dir.is_absolute():
        return metadata_dir
    else:
        return project_path / metadata_dir


def ensure_directories(config: ProjectConfig, project_path: Path) -> None:
    """Ensure all necessary directories exist."""
    # Create metadata directory
    metadata_dir = get_metadata_dir(config, project_path)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache directory
    cache_dir = get_cache_dir(config, project_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model directory if using local model
    if config.model_path and not config.model_path.startswith(("http://", "https://")):
        model_path = Path(config.model_path)
        if not model_path.is_absolute():
            model_path = project_path / model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
