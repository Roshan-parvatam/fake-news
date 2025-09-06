# config/settings.py
"""
General System Settings for Fake News Detection

This module contains system-wide settings, defaults, and configuration
management utilities that apply across all agents.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging


@dataclass
class SystemSettings:
    """
    🔧 SYSTEM-WIDE SETTINGS
    
    Core configuration settings that apply across the entire fake news detection system.
    These settings control logging, API keys, file paths, and system behavior.
    """
    
    # API Configuration
    gemini_api_key: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    
    # File Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "cache")
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "fake_news_detection.log"
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    
    # Performance Settings
    max_workers: int = 4
    default_timeout: int = 300  # seconds
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    
    # API Rate Limiting
    gemini_rate_limit: float = 4.0  # seconds between requests
    openai_rate_limit: float = 1.0  # seconds between requests
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Text Processing Limits
    max_article_length: int = 50000  # characters
    max_batch_size: int = 32
    default_max_tokens: int = 2048
    
    # Quality Thresholds
    min_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    evidence_quality_threshold: float = 6.0
    bias_detection_threshold: float = 5.0
    
    # Feature Flags
    enable_detailed_analysis: bool = True
    enable_cross_verification: bool = True
    enable_metrics_collection: bool = True
    enable_debug_mode: bool = False
    
    # Database Settings (if using databases)
    db_connection_string: str = field(default_factory=lambda: os.getenv('DB_CONNECTION_STRING', ''))
    db_pool_size: int = 10
    db_timeout: int = 30
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate API keys
        if not self.gemini_api_key:
            logging.warning("GEMINI_API_KEY not found in environment variables")
        
        # Setup logging
        self._setup_logging()
    
    def validate_api_keys(self) -> bool:
        """
        🔐 VALIDATE API KEYS
        
        Check if required API keys are configured and non-empty.
        This is the method that was missing and causing your startup error.
        
        Returns:
            bool: True if GEMINI_API_KEY is configured, False otherwise
        """
        # Check if Gemini API key exists and is not empty
        gemini_valid = bool(self.gemini_api_key and self.gemini_api_key.strip())
        
        # You can add additional API key validations here if needed
        # openai_valid = bool(self.openai_api_key and self.openai_api_key.strip())
        
        return gemini_valid
    
    def _setup_logging(self):
        """Configure system-wide logging"""
        # Create formatters
        formatter = logging.Formatter(self.log_format)
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            log_file_path = self.logs_dir / self.log_file
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary format"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def save_to_file(self, file_path: Optional[Path] = None):
        """Save current settings to JSON file"""
        if file_path is None:
            file_path = self.project_root / "config" / "current_settings.json"
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'SystemSettings':
        """Load settings from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert path strings back to Path objects
        path_fields = ['project_root', 'models_dir', 'data_dir', 'logs_dir', 'cache_dir']
        for field in path_fields:
            if field in data:
                data[field] = Path(data[field])
        
        return cls(**data)


# Global settings instance
_settings_instance: Optional[SystemSettings] = None


def get_settings() -> SystemSettings:
    """
    Get global settings instance (singleton pattern)
    
    Returns:
        SystemSettings instance
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SystemSettings()
    return _settings_instance


def update_settings(**kwargs) -> SystemSettings:
    """
    Update global settings with new values
    
    Args:
        **kwargs: Settings to update
        
    Returns:
        Updated SystemSettings instance
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SystemSettings()
    
    for key, value in kwargs.items():
        if hasattr(_settings_instance, key):
            setattr(_settings_instance, key, value)
        else:
            raise ValueError(f"Unknown setting: {key}")
    
    return _settings_instance


def reset_settings():
    """Reset settings to default values"""
    global _settings_instance
    _settings_instance = SystemSettings()


# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "log_level": "DEBUG",
    "enable_debug_mode": True,
    "enable_detailed_analysis": True,
    "gemini_rate_limit": 2.0,  # Faster for dev
    "max_retries": 1
}


PRODUCTION_CONFIG = {
    "log_level": "INFO", 
    "enable_debug_mode": False,
    "enable_detailed_analysis": False,  # For performance
    "gemini_rate_limit": 4.0,
    "max_retries": 3,
    "enable_metrics_collection": True
}


TESTING_CONFIG = {
    "log_level": "ERROR",
    "enable_file_logging": False,
    "enable_caching": False,
    "max_retries": 0,
    "default_timeout": 10
}


def apply_environment_config(environment: str = "development"):
    """
    Apply environment-specific configuration
    
    Args:
        environment: "development", "production", or "testing"
    """
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "testing": TESTING_CONFIG
    }
    
    if environment not in configs:
        raise ValueError(f"Unknown environment: {environment}")
    
    update_settings(**configs[environment])
