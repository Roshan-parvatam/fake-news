"""
Enhanced System Settings for Production Fake News Detection

Production-ready configuration management with environment awareness,
security best practices, and comprehensive monitoring capabilities.

Features:
- Environment-based configuration switching
- Secure API key management with validation
- Resource monitoring and limits
- Comprehensive logging configuration
- Performance optimization settings
- Security hardening options
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
from datetime import datetime
import hashlib


@dataclass
class SystemSettings:
    """
    Production-Ready System Settings
    
    Comprehensive configuration management with environment awareness,
    security best practices, and performance optimization.
    """
    
    # Environment Configuration
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    debug_mode: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    
    # API Configuration with Security
    gemini_api_key: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    api_key_rotation_enabled: bool = field(default_factory=lambda: os.getenv('API_KEY_ROTATION', 'false').lower() == 'true')
    
    # Directory Structure
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "cache")
    config_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "config")
    
    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    log_file: str = "fake_news_detection.log"
    enable_console_logging: bool = True
    enable_file_logging: bool = field(default_factory=lambda: os.getenv('ENABLE_FILE_LOGGING', 'true').lower() == 'true')
    enable_structured_logging: bool = field(default_factory=lambda: os.getenv('STRUCTURED_LOGGING', 'true').lower() == 'true')
    log_rotation_enabled: bool = True
    max_log_files: int = 10
    max_log_size_mb: int = 100
    
    # Performance and Resource Management
    max_workers: int = field(default_factory=lambda: int(os.getenv('MAX_WORKERS', '4')))
    default_timeout: int = field(default_factory=lambda: int(os.getenv('DEFAULT_TIMEOUT', '300')))
    memory_limit_mb: int = field(default_factory=lambda: int(os.getenv('MEMORY_LIMIT_MB', '2048')))
    cpu_limit_percent: float = field(default_factory=lambda: float(os.getenv('CPU_LIMIT_PERCENT', '80.0')))
    
    # Caching Configuration
    enable_caching: bool = field(default_factory=lambda: os.getenv('ENABLE_CACHING', 'true').lower() == 'true')
    cache_ttl: int = field(default_factory=lambda: int(os.getenv('CACHE_TTL', '3600')))
    cache_size_limit_mb: int = field(default_factory=lambda: int(os.getenv('CACHE_SIZE_LIMIT_MB', '512')))
    cache_backend: str = field(default_factory=lambda: os.getenv('CACHE_BACKEND', 'memory'))  # memory, redis, file
    
    # API Rate Limiting and Retry Logic
    gemini_rate_limit: float = field(default_factory=lambda: float(os.getenv('GEMINI_RATE_LIMIT', '4.0')))
    openai_rate_limit: float = field(default_factory=lambda: float(os.getenv('OPENAI_RATE_LIMIT', '1.0')))
    max_retries: int = field(default_factory=lambda: int(os.getenv('MAX_RETRIES', '3')))
    retry_delay: float = field(default_factory=lambda: float(os.getenv('RETRY_DELAY', '1.0')))
    exponential_backoff: bool = True
    circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', '5')))
    
    # Input Validation and Security
    max_article_length: int = field(default_factory=lambda: int(os.getenv('MAX_ARTICLE_LENGTH', '50000')))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv('MAX_BATCH_SIZE', '32')))
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv('DEFAULT_MAX_TOKENS', '2048')))
    input_sanitization_enabled: bool = True
    content_filter_enabled: bool = field(default_factory=lambda: os.getenv('CONTENT_FILTER', 'true').lower() == 'true')
    
    # Quality and Confidence Thresholds
    min_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    evidence_quality_threshold: float = 6.0
    bias_detection_threshold: float = 5.0
    
    # Feature Flags
    enable_detailed_analysis: bool = field(default_factory=lambda: os.getenv('ENABLE_DETAILED_ANALYSIS', 'true').lower() == 'true')
    enable_cross_verification: bool = field(default_factory=lambda: os.getenv('ENABLE_CROSS_VERIFICATION', 'true').lower() == 'true')
    enable_metrics_collection: bool = field(default_factory=lambda: os.getenv('ENABLE_METRICS', 'true').lower() == 'true')
    enable_safety_fallbacks: bool = field(default_factory=lambda: os.getenv('ENABLE_SAFETY_FALLBACKS', 'true').lower() == 'true')
    enable_async_processing: bool = field(default_factory=lambda: os.getenv('ENABLE_ASYNC', 'true').lower() == 'true')
    
    # Database Configuration (Optional)
    db_connection_string: str = field(default_factory=lambda: os.getenv('DB_CONNECTION_STRING', ''))
    db_pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '10')))
    db_timeout: int = field(default_factory=lambda: int(os.getenv('DB_TIMEOUT', '30')))
    
    # Monitoring and Alerting
    enable_health_checks: bool = True
    health_check_interval: int = 300  # seconds
    enable_prometheus_metrics: bool = field(default_factory=lambda: os.getenv('PROMETHEUS_METRICS', 'false').lower() == 'true')
    alert_email_enabled: bool = field(default_factory=lambda: os.getenv('ALERT_EMAIL', 'false').lower() == 'true')
    alert_webhook_url: str = field(default_factory=lambda: os.getenv('ALERT_WEBHOOK_URL', ''))
    
    # Security Configuration
    enable_request_validation: bool = True
    rate_limiting_enabled: bool = field(default_factory=lambda: os.getenv('RATE_LIMITING', 'true').lower() == 'true')
    cors_origins: List[str] = field(default_factory=lambda: os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(','))
    
    def __post_init__(self):
        """Post-initialization setup and validation"""
        # Ensure directories exist
        self._create_directories()
        
        # Validate configuration
        self._validate_configuration()
        
        # Setup logging
        self._setup_enhanced_logging()
        
        # Log initialization
        self._log_initialization()

    def _create_directories(self):
        """Create necessary directories with proper permissions"""
        directories = [
            self.models_dir,
            self.data_dir, 
            self.logs_dir,
            self.cache_dir,
            self.config_dir
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o755)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {e}")

    def _validate_configuration(self):
        """Comprehensive configuration validation"""
        validation_errors = []
        
        # API Key validation
        if not self.gemini_api_key or len(self.gemini_api_key.strip()) < 10:
            validation_errors.append("GEMINI_API_KEY is missing or invalid")
        
        # Resource limits validation
        if self.memory_limit_mb < 512:
            validation_errors.append("Memory limit too low (minimum 512MB)")
        
        if self.max_workers < 1 or self.max_workers > 32:
            validation_errors.append("Invalid max_workers value (1-32)")
        
        # Performance validation
        if self.default_timeout < 10:
            validation_errors.append("Default timeout too low (minimum 10 seconds)")
        
        # Environment-specific validation
        if self.environment == "production":
            if self.debug_mode:
                validation_errors.append("Debug mode should be disabled in production")
            
            if not self.enable_file_logging:
                validation_errors.append("File logging should be enabled in production")
        
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
            raise ValueError(error_msg)

    def _setup_enhanced_logging(self):
        """Setup production-ready logging with rotation and formatting"""
        # Create formatters
        if self.enable_structured_logging:
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(self.log_format)
        
        # Get root logger and clear existing handlers
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Console handler with colors (if available)
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.enable_file_logging:
            try:
                if self.log_rotation_enabled:
                    from logging.handlers import RotatingFileHandler
                    log_file_path = self.logs_dir / self.log_file
                    file_handler = RotatingFileHandler(
                        log_file_path,
                        maxBytes=self.max_log_size_mb * 1024 * 1024,
                        backupCount=self.max_log_files
                    )
                else:
                    log_file_path = self.logs_dir / self.log_file
                    file_handler = logging.FileHandler(log_file_path)
                
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")

    def _log_initialization(self):
        """Log system initialization information"""
        logger = logging.getLogger(__name__)
        
        logger.info("="*60)
        logger.info("FAKE NEWS DETECTION SYSTEM - INITIALIZATION")
        logger.info("="*60)
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Debug Mode: {self.debug_mode}")
        logger.info(f"Log Level: {self.log_level}")
        logger.info(f"API Keys Configured: {self.validate_api_keys()}")
        logger.info(f"Caching Enabled: {self.enable_caching}")
        logger.info(f"Async Processing: {self.enable_async_processing}")
        logger.info(f"Safety Fallbacks: {self.enable_safety_fallbacks}")
        logger.info(f"Max Workers: {self.max_workers}")
        logger.info(f"Memory Limit: {self.memory_limit_mb}MB")
        logger.info("="*60)

    def validate_api_keys(self) -> bool:
        """
        Enhanced API key validation with security checks
        
        Returns:
            True if API keys are valid and secure
        """
        # Check Gemini API key
        gemini_valid = (
            bool(self.gemini_api_key) and 
            len(self.gemini_api_key.strip()) >= 10 and
            not self.gemini_api_key.startswith('test_') and
            not self.gemini_api_key == 'your_api_key_here'
        )
        
        # Optional: Check OpenAI API key if provided
        openai_valid = True
        if self.openai_api_key:
            openai_valid = (
                len(self.openai_api_key.strip()) >= 10 and
                not self.openai_api_key.startswith('test_')
            )
        
        return gemini_valid and openai_valid

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration if available"""
        if not self.db_connection_string:
            return {}
        
        return {
            "connection_string": self.db_connection_string,
            "pool_size": self.db_pool_size,
            "timeout": self.db_timeout,
            "ssl_enabled": self.environment == "production"
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration"""
        return {
            "enabled": self.enable_caching,
            "backend": self.cache_backend,
            "ttl": self.cache_ttl,
            "size_limit_mb": self.cache_size_limit_mb,
            "directory": str(self.cache_dir) if self.cache_backend == "file" else None
        }

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "input_sanitization": self.input_sanitization_enabled,
            "content_filtering": self.content_filter_enabled,
            "rate_limiting": self.rate_limiting_enabled,
            "cors_origins": self.cors_origins,
            "max_article_length": self.max_article_length,
            "request_validation": self.enable_request_validation
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration"""
        return {
            "max_workers": self.max_workers,
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_limit_percent": self.cpu_limit_percent,
            "default_timeout": self.default_timeout,
            "async_enabled": self.enable_async_processing,
            "batch_size": self.max_batch_size
        }

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert settings to dictionary format
        
        Args:
            include_sensitive: Whether to include API keys and sensitive data
        """
        result = {}
        
        for key, value in self.__dict__.items():
            # Handle sensitive data
            if not include_sensitive and 'api_key' in key.lower():
                if value:
                    result[key] = f"***{value[-4:]}" if len(value) > 4 else "***"
                else:
                    result[key] = None
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        
        return result

    def export_config(self, file_path: Optional[Path] = None, include_sensitive: bool = False):
        """Export configuration to JSON file"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.config_dir / f"exported_config_{timestamp}.json"
        
        config_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment,
                "version": "3.2.0",
                "include_sensitive": include_sensitive
            },
            "settings": self.to_dict(include_sensitive=include_sensitive)
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        return file_path

    @classmethod
    def load_from_file(cls, file_path: Path) -> 'SystemSettings':
        """Load settings from exported JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract settings from export format
        settings_data = data.get("settings", data)
        
        # Convert path strings back to Path objects
        path_fields = ['project_root', 'models_dir', 'data_dir', 'logs_dir', 'cache_dir', 'config_dir']
        for field in path_fields:
            if field in settings_data:
                settings_data[field] = Path(settings_data[field])
        
        return cls(**settings_data)

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        import psutil
        
        try:
            # System resources
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage(str(self.project_root))
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
                    "environment": self.environment,
                    "api_keys_valid": self.validate_api_keys(),
                    "caching_enabled": self.enable_caching,
                    "logging_enabled": self.enable_file_logging
                }
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e),
                "configuration": {
                    "environment": self.environment,
                    "api_keys_valid": self.validate_api_keys()
                }
            }


# Global settings instance (singleton pattern)
_settings_instance: Optional[SystemSettings] = None

def get_settings() -> SystemSettings:
    """Get global settings instance with thread-safe initialization"""
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
    
    # Validate updates
    for key, value in kwargs.items():
        if not hasattr(_settings_instance, key):
            raise ValueError(f"Unknown setting: {key}")
        setattr(_settings_instance, key, value)
    
    # Re-run post-init validation
    _settings_instance._validate_configuration()
    
    return _settings_instance

def reset_settings():
    """Reset settings to default values"""
    global _settings_instance
    _settings_instance = SystemSettings()

# Environment-specific configuration presets
ENVIRONMENT_CONFIGS = {
    "development": {
        "log_level": "DEBUG",
        "debug_mode": True,
        "enable_detailed_analysis": True,
        "gemini_rate_limit": 2.0,
        "max_retries": 1,
        "enable_file_logging": False,
        "memory_limit_mb": 1024
    },
    
    "testing": {
        "log_level": "ERROR",
        "debug_mode": False,
        "enable_file_logging": False,
        "enable_caching": False,
        "max_retries": 0,
        "default_timeout": 10,
        "memory_limit_mb": 512
    },
    
    "staging": {
        "log_level": "INFO",
        "debug_mode": False,
        "enable_detailed_analysis": True,
        "gemini_rate_limit": 3.0,
        "max_retries": 2,
        "memory_limit_mb": 1536
    },
    
    "production": {
        "log_level": "INFO",
        "debug_mode": False,
        "enable_detailed_analysis": False,
        "gemini_rate_limit": 4.0,
        "max_retries": 3,
        "enable_metrics_collection": True,
        "enable_file_logging": True,
        "log_rotation_enabled": True,
        "memory_limit_mb": 2048,
        "enable_prometheus_metrics": True
    }
}

def apply_environment_config(environment: str = None):
    """
    Apply environment-specific configuration
    
    Args:
        environment: Target environment ("development", "testing", "staging", "production")
                    If None, uses ENVIRONMENT variable or defaults to "development"
    """
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
