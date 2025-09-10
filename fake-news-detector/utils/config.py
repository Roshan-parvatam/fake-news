"""
Enhanced Configuration Management for Fake News Detection System

Production-ready configuration with environment awareness,
validation, and security best practices.

Features:
- Environment-based configuration switching
- Secure API key management
- Path validation and setup
- Configuration validation and health checks
- Performance optimization settings
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

logger = logging.getLogger(__name__)


@dataclass
class ConfigPaths:
    """Centralized path configuration"""
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    def __post_init__(self):
        # Data directories
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.CACHE_DIR = self.DATA_DIR / "cache"
        
        # Model directories
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.SAVED_MODELS_DIR = self.MODELS_DIR / "saved_models"
        
        # Logs directory
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        
        # Config directory
        self.CONFIG_DIR = self.PROJECT_ROOT / "config"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all required directories"""
        directories = [
            self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR,
            self.MODELS_DIR, self.SAVED_MODELS_DIR, self.LOGS_DIR,
            self.CONFIG_DIR, self.CACHE_DIR
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o755)
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")


class Config:
    """
    Enhanced configuration class with environment awareness and validation
    """
    
    def __init__(self):
        # Initialize paths
        self.paths = ConfigPaths()
        
        # Environment settings
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
        self.DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # API Configuration
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
        
        # Performance Settings
        self.MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
        self.DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '300'))
        self.ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
        self.CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))
        
        # Content Processing Limits
        self.MAX_ARTICLE_LENGTH = int(os.getenv('MAX_ARTICLE_LENGTH', '50000'))
        self.MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '32'))
        self.DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', '2048'))
        
        # Rate Limiting
        self.GEMINI_RATE_LIMIT = float(os.getenv('GEMINI_RATE_LIMIT', '4.0'))
        self.OPENAI_RATE_LIMIT = float(os.getenv('OPENAI_RATE_LIMIT', '1.0'))
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
        self.RETRY_DELAY = float(os.getenv('RETRY_DELAY', '1.0'))
        
        # Quality Thresholds
        self.MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
        self.HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.8'))
        self.EVIDENCE_QUALITY_THRESHOLD = float(os.getenv('EVIDENCE_QUALITY_THRESHOLD', '6.0'))
        
        # Feature Flags
        self.ENABLE_DETAILED_ANALYSIS = os.getenv('ENABLE_DETAILED_ANALYSIS', 'true').lower() == 'true'
        self.ENABLE_CROSS_VERIFICATION = os.getenv('ENABLE_CROSS_VERIFICATION', 'true').lower() == 'true'
        self.ENABLE_METRICS_COLLECTION = os.getenv('ENABLE_METRICS_COLLECTION', 'true').lower() == 'true'
        
        # Database Settings (optional)
        self.DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING', '')
        self.DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
        
        # Apply environment-specific settings
        self._apply_environment_config()
    
    def _apply_environment_config(self):
        """Apply environment-specific configuration overrides"""
        if self.ENVIRONMENT == 'development':
            self.LOG_LEVEL = 'DEBUG'
            self.GEMINI_RATE_LIMIT = 2.0
            self.MAX_RETRIES = 1
            self.ENABLE_DETAILED_ANALYSIS = True
        elif self.ENVIRONMENT == 'testing':
            self.LOG_LEVEL = 'ERROR'
            self.ENABLE_CACHING = False
            self.MAX_RETRIES = 0
            self.DEFAULT_TIMEOUT = 10
        elif self.ENVIRONMENT == 'production':
            self.LOG_LEVEL = 'INFO'
            self.ENABLE_DETAILED_ANALYSIS = False
            self.ENABLE_METRICS_COLLECTION = True
            self.LOG_TO_FILE = True
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration and return status report
        
        Returns:
            Dictionary with validation results
        """
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'environment': self.ENVIRONMENT,
            'timestamp': datetime.now().isoformat()
        }
        
        # API Key validation
        if not self.GEMINI_API_KEY:
            validation_report['errors'].append('GEMINI_API_KEY is not configured')
            validation_report['valid'] = False
        elif len(self.GEMINI_API_KEY) < 10:
            validation_report['warnings'].append('GEMINI_API_KEY appears to be invalid')
        
        # Directory validation
        required_dirs = [
            self.paths.DATA_DIR,
            self.paths.MODELS_DIR,
            self.paths.LOGS_DIR
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                validation_report['errors'].append(f'Required directory does not exist: {directory}')
                validation_report['valid'] = False
        
        # Performance settings validation
        if self.MAX_WORKERS < 1 or self.MAX_WORKERS > 32:
            validation_report['warnings'].append('MAX_WORKERS should be between 1 and 32')
        
        if self.DEFAULT_TIMEOUT < 10:
            validation_report['warnings'].append('DEFAULT_TIMEOUT is very low (< 10 seconds)')
        
        # Environment-specific validation
        if self.ENVIRONMENT == 'production':
            if self.DEBUG:
                validation_report['warnings'].append('DEBUG mode enabled in production')
            if not self.LOG_TO_FILE:
                validation_report['warnings'].append('File logging disabled in production')
        
        return validation_report
    
    def get_scraper_config(self) -> Dict[str, Any]:
        """Get web scraper configuration"""
        return {
            'request_delay': 2.0,
            'timeout': self.DEFAULT_TIMEOUT,
            'max_retries': self.MAX_RETRIES,
            'user_agent': 'Mozilla/5.0 (compatible; FakeNewsBot/1.0)',
            'min_article_length': 200,
            'max_article_length': self.MAX_ARTICLE_LENGTH,
            'requests_per_minute': 30,
            'concurrent_requests': min(self.MAX_WORKERS, 5)
        }
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get text preprocessing configuration"""
        return {
            'min_text_length': 50,
            'max_text_length': self.MAX_ARTICLE_LENGTH,
            'target_language': 'en',
            'remove_html': True,
            'remove_urls': True,
            'normalize_whitespace': True,
            'min_word_count': 10
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'max_tokens': self.DEFAULT_MAX_TOKENS,
            'temperature': 0.3,
            'top_p': 0.9,
            'rate_limit': self.GEMINI_RATE_LIMIT,
            'timeout': self.DEFAULT_TIMEOUT,
            'max_retries': self.MAX_RETRIES
        }
    
    def export_config(self, file_path: Optional[Path] = None) -> Path:
        """
        Export current configuration to JSON file
        
        Args:
            file_path: Optional path for export file
            
        Returns:
            Path to exported configuration file
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.paths.CONFIG_DIR / f"config_export_{timestamp}.json"
        
        config_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'environment': self.ENVIRONMENT,
                'version': '3.2.0'
            },
            'settings': {
                'environment': self.ENVIRONMENT,
                'debug': self.DEBUG,
                'log_level': self.LOG_LEVEL,
                'max_workers': self.MAX_WORKERS,
                'default_timeout': self.DEFAULT_TIMEOUT,
                'enable_caching': self.ENABLE_CACHING,
                'max_article_length': self.MAX_ARTICLE_LENGTH,
                'gemini_rate_limit': self.GEMINI_RATE_LIMIT,
                'max_retries': self.MAX_RETRIES,
                'min_confidence_threshold': self.MIN_CONFIDENCE_THRESHOLD,
                'enable_detailed_analysis': self.ENABLE_DETAILED_ANALYSIS,
                'enable_metrics_collection': self.ENABLE_METRICS_COLLECTION
            },
            'paths': {
                'project_root': str(self.paths.PROJECT_ROOT),
                'data_dir': str(self.paths.DATA_DIR),
                'models_dir': str(self.paths.MODELS_DIR),
                'logs_dir': str(self.paths.LOGS_DIR)
            }
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            logger.info(f"Configuration exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise
        
        return file_path
    
    def print_status_report(self):
        """Print comprehensive configuration status report"""
        print("🔍 Configuration Status Report")
        print("=" * 60)
        
        # Validation
        validation = self.validate_configuration()
        status_icon = "✅" if validation['valid'] else "❌"
        print(f"{status_icon} Overall Status: {'VALID' if validation['valid'] else 'INVALID'}")
        
        if validation['errors']:
            print("\n❌ Errors:")
            for error in validation['errors']:
                print(f"   - {error}")
        
        if validation['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   - {warning}")
        
        # Environment Info
        print(f"\n🌍 Environment: {self.ENVIRONMENT}")
        print(f"🐛 Debug Mode: {self.DEBUG}")
        print(f"📝 Log Level: {self.LOG_LEVEL}")
        
        # API Configuration
        print(f"\n🔑 API Keys:")
        print(f"   - Gemini: {'✅ Configured' if self.GEMINI_API_KEY else '❌ Missing'}")
        print(f"   - OpenAI: {'✅ Configured' if self.OPENAI_API_KEY else '❌ Missing'}")
        
        # Performance Settings
        print(f"\n⚡ Performance:")
        print(f"   - Max Workers: {self.MAX_WORKERS}")
        print(f"   - Default Timeout: {self.DEFAULT_TIMEOUT}s")
        print(f"   - Caching: {'Enabled' if self.ENABLE_CACHING else 'Disabled'}")
        print(f"   - Rate Limit: {self.GEMINI_RATE_LIMIT}s")
        
        # Directories
        print(f"\n📁 Directories:")
        dirs_status = [
            ("Data", self.paths.DATA_DIR),
            ("Models", self.paths.MODELS_DIR),
            ("Logs", self.paths.LOGS_DIR)
        ]
        
        for name, path in dirs_status:
            status = "✅" if path.exists() else "❌"
            print(f"   - {name}: {status} {path}")
        
        print("=" * 60)


# Create global configuration instance
config = Config()

# Backward compatibility - expose commonly used paths and settings
PROJECT_ROOT = config.paths.PROJECT_ROOT
DATA_DIR = config.paths.DATA_DIR
RAW_DATA_DIR = config.paths.RAW_DATA_DIR
PROCESSED_DATA_DIR = config.paths.PROCESSED_DATA_DIR
MODELS_DIR = config.paths.MODELS_DIR
SAVED_MODELS_DIR = config.paths.SAVED_MODELS_DIR
LOGS_DIR = config.paths.LOGS_DIR

# Environment and API settings
ENVIRONMENT = config.ENVIRONMENT
GEMINI_API_KEY = config.GEMINI_API_KEY
OPENAI_API_KEY = config.OPENAI_API_KEY
LOG_LEVEL = config.LOG_LEVEL

# Processing settings
MAX_ARTICLE_LENGTH = config.MAX_ARTICLE_LENGTH
MAX_WORKERS = config.MAX_WORKERS
DEFAULT_TIMEOUT = config.DEFAULT_TIMEOUT

# Convenience functions
def get_dataset_paths() -> Dict[str, Path]:
    """Get paths for dataset files"""
    return {
        'true_file': RAW_DATA_DIR / "True.csv",
        'fake_file': RAW_DATA_DIR / "Fake.csv",
        'scraped_data': RAW_DATA_DIR / "scraped_articles.csv"
    }

def validate_setup() -> Dict[str, bool]:
    """Validate system setup"""
    return config.validate_configuration()

def print_setup_status():
    """Print setup status (backward compatibility)"""
    config.print_status_report()

# Export configuration on import for debugging
if __name__ == "__main__":
    print_setup_status()
