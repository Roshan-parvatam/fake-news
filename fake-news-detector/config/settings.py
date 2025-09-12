# config/settings.py

"""
Enhanced System Settings for Production Fake News Detection

Production-ready configuration management with modern Python patterns,
comprehensive validation, and enhanced security practices.

Features:
- Clean environment-based configuration
- Robust validation with clear error messages
- Secure API key management
- Modular configuration components
- Enhanced logging and monitoring
- Performance optimization settings

Version: 3.2.0 - Modern Production Edition
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib
from enum import Enum

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from fake-news-detector directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.getLogger(__name__).debug(f"✅ Loaded environment variables from {env_path}")
    else:
        # Fallback to default .env loading
        load_dotenv()
        logging.getLogger(__name__).debug("✅ Loaded environment variables from default .env")
except ImportError:
    logging.getLogger(__name__).warning("⚠️ python-dotenv not available - using system environment variables only")


class Environment(Enum):
    """Supported environments with validation."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DirectoryConfig:
    """Directory configuration with automatic creation."""
    project_root: Path
    models_dir: Path
    data_dir: Path
    logs_dir: Path
    cache_dir: Path
    config_dir: Path

    def __post_init__(self):
        """Create directories with proper permissions."""
        for directory in [self.models_dir, self.data_dir, self.logs_dir, 
                         self.cache_dir, self.config_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o755)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not create directory {directory}: {e}")

    @classmethod
    def from_project_root(cls, project_root: Optional[Path] = None) -> 'DirectoryConfig':
        """Create directory config from project root."""
        if project_root is None:
            project_root = Path(__file__).parent.parent
            
        return cls(
            project_root=project_root,
            models_dir=project_root / "models",
            data_dir=project_root / "data",
            logs_dir=project_root / "logs",
            cache_dir=project_root / "cache",
            config_dir=project_root / "config"
        )


@dataclass
class APIConfig:
    """API configuration with validation."""
    gemini_api_key: str
    openai_api_key: str
    api_key_rotation_enabled: bool
    gemini_rate_limit: float
    openai_rate_limit: float
    max_retries: int
    retry_delay: float
    exponential_backoff: bool
    circuit_breaker_threshold: int

    def __post_init__(self):
        """Validate API configuration."""
        if not self.gemini_api_key or len(self.gemini_api_key.strip()) < 10:
            raise ValueError("GEMINI_API_KEY is missing or too short (minimum 10 characters)")
        
        if self.gemini_api_key.startswith('test_') or self.gemini_api_key == 'your_api_key_here':
            raise ValueError("GEMINI_API_KEY appears to be a placeholder or test key")

        if self.gemini_rate_limit <= 0 or self.openai_rate_limit <= 0:
            raise ValueError("Rate limits must be positive numbers")

        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError("max_retries must be between 0 and 10")

    def validate_api_keys(self) -> bool:
        """Enhanced API key validation."""
        try:
            # Check Gemini API key
            gemini_valid = (
                bool(self.gemini_api_key) and
                len(self.gemini_api_key.strip()) >= 10 and
                not self.gemini_api_key.startswith('test_') and
                not self.gemini_api_key == 'your_api_key_here'
            )

            # Check OpenAI API key if provided
            openai_valid = True
            if self.openai_api_key:
                openai_valid = (
                    len(self.openai_api_key.strip()) >= 10 and
                    not self.openai_api_key.startswith('test_')
                )

            return gemini_valid and openai_valid
        except Exception:
            return False

    @classmethod
    def from_environment(cls) -> 'APIConfig':
        """Create API config from environment variables."""
        return cls(
            gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            api_key_rotation_enabled=_parse_bool(os.getenv('API_KEY_ROTATION', 'false')),
            gemini_rate_limit=float(os.getenv('GEMINI_RATE_LIMIT', '4.0')),
            openai_rate_limit=float(os.getenv('OPENAI_RATE_LIMIT', '1.0')),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            retry_delay=float(os.getenv('RETRY_DELAY', '1.0')),
            exponential_backoff=True,
            circuit_breaker_threshold=int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', '5'))
        )


@dataclass
class PerformanceConfig:
    """Performance and resource configuration."""
    max_workers: int
    default_timeout: int
    memory_limit_mb: int
    cpu_limit_percent: float
    max_article_length: int
    max_batch_size: int
    default_max_tokens: int

    def __post_init__(self):
        """Validate performance configuration."""
        if self.memory_limit_mb < 512:
            raise ValueError("Memory limit too low (minimum 512MB)")
        
        if not 1 <= self.max_workers <= 32:
            raise ValueError("max_workers must be between 1 and 32")
        
        if self.default_timeout < 10:
            raise ValueError("default_timeout too low (minimum 10 seconds)")

    @classmethod
    def from_environment(cls) -> 'PerformanceConfig':
        """Create performance config from environment variables."""
        return cls(
            max_workers=int(os.getenv('MAX_WORKERS', '4')),
            default_timeout=int(os.getenv('DEFAULT_TIMEOUT', '300')),
            memory_limit_mb=int(os.getenv('MEMORY_LIMIT_MB', '2048')),
            cpu_limit_percent=float(os.getenv('CPU_LIMIT_PERCENT', '80.0')),
            max_article_length=int(os.getenv('MAX_ARTICLE_LENGTH', '50000')),
            max_batch_size=int(os.getenv('MAX_BATCH_SIZE', '32')),
            default_max_tokens=int(os.getenv('DEFAULT_MAX_TOKENS', '2048'))
        )


@dataclass
class LoggingConfig:
    """Logging configuration with enhanced features."""
    log_level: LogLevel
    log_format: str
    log_file: str
    enable_console_logging: bool
    enable_file_logging: bool
    enable_structured_logging: bool
    log_rotation_enabled: bool
    max_log_files: int
    max_log_size_mb: int

    @classmethod
    def from_environment(cls, environment: Environment) -> 'LoggingConfig':
        """Create logging config based on environment."""
        # Environment-specific defaults
        env_defaults = {
            Environment.DEVELOPMENT: {
                'log_level': LogLevel.DEBUG,
                'enable_file_logging': False,
                'enable_structured_logging': True
            },
            Environment.TESTING: {
                'log_level': LogLevel.ERROR,
                'enable_file_logging': False,
                'enable_structured_logging': False
            },
            Environment.STAGING: {
                'log_level': LogLevel.INFO,
                'enable_file_logging': True,
                'enable_structured_logging': True
            },
            Environment.PRODUCTION: {
                'log_level': LogLevel.WARNING,
                'enable_file_logging': True,
                'enable_structured_logging': True
            }
        }

        defaults = env_defaults.get(environment, env_defaults[Environment.DEVELOPMENT])
        
        return cls(
            log_level=LogLevel(os.getenv('LOG_LEVEL', defaults['log_level'].value)),
            log_format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            log_file="fake_news_detection.log",
            enable_console_logging=True,
            enable_file_logging=_parse_bool(os.getenv('ENABLE_FILE_LOGGING', str(defaults['enable_file_logging']))),
            enable_structured_logging=_parse_bool(os.getenv('STRUCTURED_LOGGING', str(defaults['enable_structured_logging']))),
            log_rotation_enabled=True,
            max_log_files=10,
            max_log_size_mb=100
        )


class SystemSettings:
    """
    Production-Ready System Settings
    
    Modern, clean configuration management with proper separation of concerns,
    robust validation, and environment-aware defaults.
    """

    def __init__(self, environment: Optional[str] = None):
        """
        Initialize system settings with environment awareness.
        
        Args:
            environment: Target environment (development, testing, staging, production)
        """
        # Determine environment
        env_str = environment or os.getenv('ENVIRONMENT', 'development')
        try:
            self.environment = Environment(env_str)
        except ValueError:
            logging.getLogger(__name__).warning(f"Invalid environment '{env_str}', using development")
            self.environment = Environment.DEVELOPMENT

        self.debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Environment validation
        if self.environment == Environment.PRODUCTION and self.debug_mode:
            raise ValueError("Debug mode should be disabled in production")

        # Initialize configuration components
        self.directories = DirectoryConfig.from_project_root()
        self.api = APIConfig.from_environment()
        self.performance = PerformanceConfig.from_environment()
        self.logging = LoggingConfig.from_environment(self.environment)

        # Feature flags and additional settings
        self._init_feature_flags()
        self._init_caching_config()
        self._init_security_config()
        self._init_monitoring_config()

        # Setup logging
        self.setup_logging()

        # Log initialization
        self._log_initialization()

    def _init_feature_flags(self):
        """Initialize feature flags based on environment."""
        env_features = {
            Environment.DEVELOPMENT: {
                'enable_detailed_analysis': True,
                'enable_cross_verification': True,
                'enable_metrics_collection': True,
                'enable_safety_fallbacks': True,
                'enable_async_processing': True
            },
            Environment.TESTING: {
                'enable_detailed_analysis': False,
                'enable_cross_verification': False,
                'enable_metrics_collection': False,
                'enable_safety_fallbacks': True,
                'enable_async_processing': False
            },
            Environment.STAGING: {
                'enable_detailed_analysis': True,
                'enable_cross_verification': True,
                'enable_metrics_collection': True,
                'enable_safety_fallbacks': True,
                'enable_async_processing': True
            },
            Environment.PRODUCTION: {
                'enable_detailed_analysis': True,
                'enable_cross_verification': True,
                'enable_metrics_collection': True,
                'enable_safety_fallbacks': True,
                'enable_async_processing': True
            }
        }

        defaults = env_features[self.environment]
        
        self.enable_detailed_analysis = _parse_bool(os.getenv('ENABLE_DETAILED_ANALYSIS', str(defaults['enable_detailed_analysis'])))
        self.enable_cross_verification = _parse_bool(os.getenv('ENABLE_CROSS_VERIFICATION', str(defaults['enable_cross_verification'])))
        self.enable_metrics_collection = _parse_bool(os.getenv('ENABLE_METRICS', str(defaults['enable_metrics_collection'])))
        self.enable_safety_fallbacks = _parse_bool(os.getenv('ENABLE_SAFETY_FALLBACKS', str(defaults['enable_safety_fallbacks'])))
        self.enable_async_processing = _parse_bool(os.getenv('ENABLE_ASYNC', str(defaults['enable_async_processing'])))

    def _init_caching_config(self):
        """Initialize caching configuration."""
        self.enable_caching = _parse_bool(os.getenv('ENABLE_CACHING', 'true'))
        self.cache_ttl = int(os.getenv('CACHE_TTL', '3600'))
        self.cache_size_limit_mb = int(os.getenv('CACHE_SIZE_LIMIT_MB', '512'))
        self.cache_backend = os.getenv('CACHE_BACKEND', 'memory')

    def _init_security_config(self):
        """Initialize security configuration."""
        self.input_sanitization_enabled = True
        self.content_filter_enabled = _parse_bool(os.getenv('CONTENT_FILTER', 'true'))
        self.enable_request_validation = True
        self.rate_limiting_enabled = _parse_bool(os.getenv('RATE_LIMITING', 'true'))
        self.cors_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')

        # Quality thresholds
        self.min_confidence_threshold = 0.6
        self.high_confidence_threshold = 0.8
        self.evidence_quality_threshold = 6.0
        self.bias_detection_threshold = 5.0

    def _init_monitoring_config(self):
        """Initialize monitoring and alerting configuration."""
        self.enable_health_checks = True
        self.health_check_interval = 300
        self.enable_prometheus_metrics = _parse_bool(os.getenv('PROMETHEUS_METRICS', 'false'))
        self.alert_email_enabled = _parse_bool(os.getenv('ALERT_EMAIL', 'false'))
        self.alert_webhook_url = os.getenv('ALERT_WEBHOOK_URL', '')

    def setup_logging(self):
        """Setup enhanced logging with environment-appropriate configuration."""
        # Get root logger and clear existing handlers
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(getattr(logging, self.logging.log_level.value))

        # Create formatter
        if self.logging.enable_structured_logging:
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(self.logging.log_format)

        # Console handler
        if self.logging.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler with rotation
        if self.logging.enable_file_logging:
            try:
                if self.logging.log_rotation_enabled:
                    from logging.handlers import RotatingFileHandler
                    log_file_path = self.directories.logs_dir / self.logging.log_file
                    file_handler = RotatingFileHandler(
                        log_file_path,
                        maxBytes=self.logging.max_log_size_mb * 1024 * 1024,
                        backupCount=self.logging.max_log_files
                    )
                else:
                    log_file_path = self.directories.logs_dir / self.logging.log_file
                    file_handler = logging.FileHandler(log_file_path)
                
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")

    def _log_initialization(self):
        """Log system initialization information."""
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("FAKE NEWS DETECTION SYSTEM - INITIALIZATION")
        logger.info("=" * 60)
        logger.info(f"Environment: {self.environment.value}")
        logger.info(f"Debug Mode: {self.debug_mode}")
        logger.info(f"Log Level: {self.logging.log_level.value}")
        logger.info(f"API Keys Valid: {self.api.validate_api_keys()}")
        logger.info(f"Caching Enabled: {self.enable_caching}")
        logger.info(f"Async Processing: {self.enable_async_processing}")
        logger.info(f"Max Workers: {self.performance.max_workers}")
        logger.info(f"Memory Limit: {self.performance.memory_limit_mb}MB")
        logger.info("=" * 60)

    def validate_api_keys(self) -> bool:
        """Validate API keys."""
        return self.api.validate_api_keys()

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration if available."""
        db_connection_string = os.getenv('DB_CONNECTION_STRING', '')
        if not db_connection_string:
            return {}

        return {
            "connection_string": db_connection_string,
            "pool_size": int(os.getenv('DB_POOL_SIZE', '10')),
            "timeout": int(os.getenv('DB_TIMEOUT', '30')),
            "ssl_enabled": self.environment == Environment.PRODUCTION
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration."""
        return {
            "enabled": self.enable_caching,
            "backend": self.cache_backend,
            "ttl": self.cache_ttl,
            "size_limit_mb": self.cache_size_limit_mb,
            "directory": str(self.directories.cache_dir) if self.cache_backend == "file" else None
        }

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "input_sanitization": self.input_sanitization_enabled,
            "content_filtering": self.content_filter_enabled,
            "rate_limiting": self.rate_limiting_enabled,
            "cors_origins": self.cors_origins,
            "max_article_length": self.performance.max_article_length,
            "request_validation": self.enable_request_validation
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration."""
        return {
            "max_workers": self.performance.max_workers,
            "memory_limit_mb": self.performance.memory_limit_mb,
            "cpu_limit_percent": self.performance.cpu_limit_percent,
            "default_timeout": self.performance.default_timeout,
            "async_enabled": self.enable_async_processing,
            "batch_size": self.performance.max_batch_size
        }

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert settings to dictionary format."""
        result = {
            "environment": self.environment.value,
            "debug_mode": self.debug_mode,
            "directories": {
                "project_root": str(self.directories.project_root),
                "models_dir": str(self.directories.models_dir),
                "data_dir": str(self.directories.data_dir),
                "logs_dir": str(self.directories.logs_dir),
                "cache_dir": str(self.directories.cache_dir),
                "config_dir": str(self.directories.config_dir)
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "memory_limit_mb": self.performance.memory_limit_mb,
                "default_timeout": self.performance.default_timeout
            },
            "features": {
                "enable_detailed_analysis": self.enable_detailed_analysis,
                "enable_async_processing": self.enable_async_processing,
                "enable_caching": self.enable_caching
            },
            "logging": {
                "log_level": self.logging.log_level.value,
                "enable_file_logging": self.logging.enable_file_logging
            }
        }

        if include_sensitive:
            result["api"] = {
                "gemini_api_key": self.api.gemini_api_key,
                "openai_api_key": self.api.openai_api_key
            }
        else:
            result["api"] = {
                "gemini_api_key": f"***{self.api.gemini_api_key[-4:]}" if self.api.gemini_api_key else None,
                "openai_api_key": f"***{self.api.openai_api_key[-4:]}" if self.api.openai_api_key else None
            }

        return result

    def export_config(self, file_path: Optional[Path] = None, include_sensitive: bool = False):
        """Export configuration to JSON file."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.directories.config_dir / f"exported_config_{timestamp}.json"

        config_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment.value,
                "version": "3.2.0",
                "include_sensitive": include_sensitive
            },
            "settings": self.to_dict(include_sensitive=include_sensitive)
        }

        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        return file_path

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            import psutil
            
            # System resources
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage(str(self.directories.project_root))
            disk_percent = (disk_usage.used / disk_usage.total) * 100

            # Determine health status
            health_issues = []
            if memory_percent > 85:
                health_issues.append("High memory usage")
            if cpu_percent > 90:
                health_issues.append("High CPU usage")
            if disk_percent > 90:
                health_issues.append("Low disk space")

            status = "healthy" if not health_issues else "degraded"

            return {
                "status": status,
                "issues": health_issues,
                "resources": {
                    "memory_percent": memory_percent,
                    "cpu_percent": cpu_percent,
                    "disk_percent": disk_percent
                },
                "configuration": {
                    "environment": self.environment.value,
                    "api_keys_valid": self.validate_api_keys(),
                    "caching_enabled": self.enable_caching,
                    "logging_enabled": self.logging.enable_file_logging
                }
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e),
                "configuration": {
                    "environment": self.environment.value,
                    "api_keys_valid": self.validate_api_keys()
                }
            }


# Utility functions
def _parse_bool(value: str) -> bool:
    """Parse boolean values from environment variables."""
    return str(value).lower() in ('true', '1', 'yes', 'on', 'enabled')


# Global settings instance (singleton pattern with thread safety)
_settings_instance: Optional[SystemSettings] = None
_settings_lock = None

try:
    import threading
    _settings_lock = threading.Lock()
except ImportError:
    pass


def get_settings() -> SystemSettings:
    """Get global settings instance with thread-safe initialization."""
    global _settings_instance
    
    if _settings_instance is None:
        if _settings_lock:
            with _settings_lock:
                if _settings_instance is None:
                    _settings_instance = SystemSettings()
        else:
            _settings_instance = SystemSettings()
    
    return _settings_instance


def update_settings(**kwargs) -> SystemSettings:
    """Update global settings with new values."""
    global _settings_instance
    
    if _settings_instance is None:
        _settings_instance = SystemSettings()

    # Apply updates by recreating with new environment if needed
    if 'environment' in kwargs:
        _settings_instance = SystemSettings(environment=kwargs['environment'])
    else:
        # For other updates, would need to implement field-by-field updates
        # This is a simplified version
        for key, value in kwargs.items():
            if hasattr(_settings_instance, key):
                setattr(_settings_instance, key, value)

    return _settings_instance


def reset_settings():
    """Reset settings to default values."""
    global _settings_instance
    _settings_instance = SystemSettings()


# Environment-specific configuration presets
ENVIRONMENT_CONFIGS = {
    "development": {
        "enable_detailed_analysis": True,
        "enable_file_logging": False,
        "memory_limit_mb": 1024
    },
    "testing": {
        "enable_detailed_analysis": False,
        "enable_file_logging": False,
        "enable_caching": False,
        "memory_limit_mb": 512
    },
    "staging": {
        "enable_detailed_analysis": True,
        "memory_limit_mb": 1536
    },
    "production": {
        "enable_detailed_analysis": True,
        "enable_file_logging": True,
        "enable_prometheus_metrics": True,
        "memory_limit_mb": 2048
    }
}


def apply_environment_config(environment: str = None):
    """Apply environment-specific configuration."""
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    if environment not in ENVIRONMENT_CONFIGS:
        raise ValueError(f"Unknown environment: {environment}. Available: {list(ENVIRONMENT_CONFIGS.keys())}")

    config_updates = ENVIRONMENT_CONFIGS[environment]
    update_settings(**config_updates)

    logger = logging.getLogger(__name__)
    logger.info(f"Applied {environment} configuration with {len(config_updates)} settings")


# Initialize with environment config on import
try:
    apply_environment_config()
except Exception as e:
    print(f"Warning: Could not apply environment configuration: {e}")
