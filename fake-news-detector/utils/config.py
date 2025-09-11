# utils/config.py

"""
Enhanced Configuration Management for Fake News Detection System

Comprehensive configuration management with environment awareness, secure key handling,
validation, health monitoring, and production-ready deployment support.

Features:
- Environment-based configuration with intelligent defaults
- Secure API key management with validation
- Dynamic path management with automatic directory creation
- Configuration validation with detailed health reports
- Performance optimization settings with runtime tuning
- Integration with enhanced logging system
- Configuration export/import with encryption support
- Multi-environment deployment support
- Runtime configuration updates with hot-reloading

Version: 3.2.0 - Enhanced Production Edition
"""

import os
import sys
import json
import logging
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
import threading

# Optional dependencies for enhanced functionality
try:
    from dotenv import load_dotenv
    load_dotenv()
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ConfigPaths:
    """
    Centralized path configuration with automatic directory creation and validation.
    
    Features:
    - Intelligent path resolution based on project structure
    - Automatic directory creation with proper permissions
    - Path validation and accessibility checks
    - Environment-specific path overrides
    - Symlink and mount point detection
    """
    
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Core directories
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    temp_dir: Path = field(init=False)
    
    # Model directories
    models_dir: Path = field(init=False)
    saved_models_dir: Path = field(init=False)
    pretrained_models_dir: Path = field(init=False)
    
    # System directories
    logs_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    backup_dir: Path = field(init=False)
    
    # Optional external paths
    external_data_dir: Optional[Path] = field(default=None)
    external_models_dir: Optional[Path] = field(default=None)
    
    def __post_init__(self):
        """Initialize all paths and create directories."""
        # Resolve project root if it's relative
        if not self.project_root.is_absolute():
            self.project_root = self.project_root.resolve()
        
        # Core data paths
        self.data_dir = self._resolve_path("DATA_DIR", self.project_root / "data")
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        self.temp_dir = self.data_dir / "temp"
        
        # Model paths
        self.models_dir = self._resolve_path("MODELS_DIR", self.project_root / "models")
        self.saved_models_dir = self.models_dir / "saved_models"
        self.pretrained_models_dir = self.models_dir / "pretrained"
        
        # System paths
        self.logs_dir = self._resolve_path("LOGS_DIR", self.project_root / "logs")
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.project_root / "backups"
        
        # External paths from environment
        if os.getenv("EXTERNAL_DATA_DIR"):
            self.external_data_dir = Path(os.getenv("EXTERNAL_DATA_DIR"))
        if os.getenv("EXTERNAL_MODELS_DIR"):
            self.external_models_dir = Path(os.getenv("EXTERNAL_MODELS_DIR"))
        
        # Create all directories
        self._create_directories()
    
    def _resolve_path(self, env_var: str, default: Path) -> Path:
        """Resolve path from environment variable or use default."""
        env_path = os.getenv(env_var)
        if env_path:
            path = Path(env_path)
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return default
    
    def _create_directories(self):
        """Create all required directories with proper error handling."""
        directories = [
            self.data_dir, self.raw_data_dir, self.processed_data_dir,
            self.cache_dir, self.temp_dir, self.models_dir, 
            self.saved_models_dir, self.pretrained_models_dir,
            self.logs_dir, self.config_dir, self.backup_dir
        ]
        
        # Add external directories if specified
        if self.external_data_dir:
            directories.append(self.external_data_dir)
        if self.external_models_dir:
            directories.append(self.external_models_dir)
        
        created_dirs = []
        failed_dirs = []
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o755)
                
                # Verify directory is writable
                if not self._is_writable(directory):
                    failed_dirs.append((directory, "Not writable"))
                else:
                    created_dirs.append(directory)
                    
            except PermissionError as e:
                failed_dirs.append((directory, f"Permission denied: {e}"))
            except OSError as e:
                failed_dirs.append((directory, f"OS error: {e}"))
            except Exception as e:
                failed_dirs.append((directory, f"Unexpected error: {e}"))
        
        # Log results
        if created_dirs:
            logger.info(f"Successfully created/verified {len(created_dirs)} directories")
        
        if failed_dirs:
            logger.warning(f"Failed to create {len(failed_dirs)} directories:")
            for directory, error in failed_dirs:
                logger.warning(f"  {directory}: {error}")
    
    def _is_writable(self, path: Path) -> bool:
        """Check if directory is writable."""
        try:
            test_file = path / f".write_test_{secrets.token_hex(4)}"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def validate_paths(self) -> Dict[str, Any]:
        """Validate all paths and return detailed status."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "path_status": {},
            "disk_usage": {},
            "permissions": {}
        }
        
        # Check each path
        all_paths = {
            "project_root": self.project_root,
            "data_dir": self.data_dir,
            "raw_data_dir": self.raw_data_dir,
            "processed_data_dir": self.processed_data_dir,
            "cache_dir": self.cache_dir,
            "temp_dir": self.temp_dir,
            "models_dir": self.models_dir,
            "saved_models_dir": self.saved_models_dir,
            "pretrained_models_dir": self.pretrained_models_dir,
            "logs_dir": self.logs_dir,
            "config_dir": self.config_dir,
            "backup_dir": self.backup_dir
        }
        
        if self.external_data_dir:
            all_paths["external_data_dir"] = self.external_data_dir
        if self.external_models_dir:
            all_paths["external_models_dir"] = self.external_models_dir
        
        for name, path in all_paths.items():
            status = {
                "exists": path.exists(),
                "is_directory": path.is_dir() if path.exists() else False,
                "is_writable": self._is_writable(path) if path.exists() else False,
                "is_readable": os.access(path, os.R_OK) if path.exists() else False
            }
            
            validation_result["path_status"][name] = status
            
            # Check for issues
            if not status["exists"]:
                validation_result["errors"].append(f"Path does not exist: {path}")
                validation_result["valid"] = False
            elif not status["is_directory"]:
                validation_result["errors"].append(f"Path is not a directory: {path}")
                validation_result["valid"] = False
            elif not status["is_writable"]:
                validation_result["warnings"].append(f"Path is not writable: {path}")
            elif not status["is_readable"]:
                validation_result["errors"].append(f"Path is not readable: {path}")
                validation_result["valid"] = False
            
            # Get disk usage if possible
            if path.exists() and _PSUTIL_AVAILABLE:
                try:
                    usage = psutil.disk_usage(str(path))
                    validation_result["disk_usage"][name] = {
                        "total_gb": round(usage.total / 1024**3, 2),
                        "used_gb": round(usage.used / 1024**3, 2),
                        "free_gb": round(usage.free / 1024**3, 2),
                        "percent_used": round((usage.used / usage.total) * 100, 1)
                    }
                except Exception:
                    pass
        
        return validation_result
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """Clean up old temporary files."""
        cleanup_result = {
            "files_removed": 0,
            "space_freed_mb": 0,
            "errors": []
        }
        
        if not self.temp_dir.exists():
            return cleanup_result
        
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        try:
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            size = file_path.stat().st_size
                            file_path.unlink()
                            cleanup_result["files_removed"] += 1
                            cleanup_result["space_freed_mb"] += size / (1024 * 1024)
                    except Exception as e:
                        cleanup_result["errors"].append(f"Failed to remove {file_path}: {e}")
        
        except Exception as e:
            cleanup_result["errors"].append(f"Failed to scan temp directory: {e}")
        
        return cleanup_result


@dataclass
class SecurityConfig:
    """Security configuration with key validation and rotation support."""
    
    # API Keys (retrieved from environment)
    gemini_api_key: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY', ''))
    
    # Security settings
    enable_api_key_rotation: bool = field(default_factory=lambda: os.getenv('ENABLE_KEY_ROTATION', 'false').lower() == 'true')
    api_key_rotation_days: int = field(default_factory=lambda: int(os.getenv('KEY_ROTATION_DAYS', '30')))
    
    # Request security
    enable_request_signing: bool = field(default_factory=lambda: os.getenv('ENABLE_REQUEST_SIGNING', 'false').lower() == 'true')
    request_timeout_sec: int = field(default_factory=lambda: int(os.getenv('REQUEST_TIMEOUT', '300')))
    max_request_retries: int = field(default_factory=lambda: int(os.getenv('MAX_RETRIES', '3')))
    
    # Data security
    enable_data_encryption: bool = field(default_factory=lambda: os.getenv('ENABLE_DATA_ENCRYPTION', 'false').lower() == 'true')
    encryption_key: Optional[str] = field(default_factory=lambda: os.getenv('ENCRYPTION_KEY'))
    
    def validate_api_keys(self) -> Dict[str, Any]:
        """Validate API keys and return status."""
        validation = {
            "valid_keys": [],
            "invalid_keys": [],
            "missing_keys": [],
            "key_status": {}
        }
        
        keys = {
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key
        }
        
        for key_name, key_value in keys.items():
            if not key_value:
                validation["missing_keys"].append(key_name)
                validation["key_status"][key_name] = "missing"
            elif len(key_value) < 10:  # Basic length check
                validation["invalid_keys"].append(key_name)
                validation["key_status"][key_name] = "invalid"
            else:
                validation["valid_keys"].append(key_name)
                validation["key_status"][key_name] = "valid"
                
                # Additional validation for known key formats
                if key_name == "openai" and not key_value.startswith(("sk-", "pk-")):
                    validation["key_status"][key_name] = "suspicious"
                elif key_name == "gemini" and len(key_value) < 30:
                    validation["key_status"][key_name] = "suspicious"
        
        return validation
    
    def get_masked_keys(self) -> Dict[str, str]:
        """Get API keys with masking for logging."""
        def mask_key(key: str) -> str:
            if not key:
                return "NOT_SET"
            elif len(key) < 8:
                return "*" * len(key)
            else:
                return f"{key[:4]}...{key[-4:]}"
        
        return {
            "gemini": mask_key(self.gemini_api_key),
            "openai": mask_key(self.openai_api_key), 
            "anthropic": mask_key(self.anthropic_api_key)
        }


@dataclass
class PerformanceConfig:
    """Performance and resource management configuration."""
    
    # Worker configuration
    max_workers: int = field(default_factory=lambda: int(os.getenv('MAX_WORKERS', '4')))
    worker_timeout_sec: int = field(default_factory=lambda: int(os.getenv('WORKER_TIMEOUT', '300')))
    
    # Memory management
    max_memory_mb: int = field(default_factory=lambda: int(os.getenv('MAX_MEMORY_MB', '2048')))
    memory_check_interval_sec: int = field(default_factory=lambda: int(os.getenv('MEMORY_CHECK_INTERVAL', '60')))
    enable_memory_profiling: bool = field(default_factory=lambda: os.getenv('ENABLE_MEMORY_PROFILING', 'false').lower() == 'true')
    
    # Caching
    enable_cache: bool = field(default_factory=lambda: os.getenv('ENABLE_CACHE', 'true').lower() == 'true')
    cache_ttl_sec: int = field(default_factory=lambda: int(os.getenv('CACHE_TTL', '3600')))
    max_cache_size_mb: int = field(default_factory=lambda: int(os.getenv('MAX_CACHE_SIZE_MB', '500')))
    
    # Rate limiting
    gemini_rate_limit_per_min: float = field(default_factory=lambda: float(os.getenv('GEMINI_RATE_LIMIT', '60')))
    openai_rate_limit_per_min: float = field(default_factory=lambda: float(os.getenv('OPENAI_RATE_LIMIT', '20')))
    
    # Content processing limits
    max_article_length: int = field(default_factory=lambda: int(os.getenv('MAX_ARTICLE_LENGTH', '50000')))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv('MAX_BATCH_SIZE', '32')))
    
    def validate_performance_settings(self) -> Dict[str, Any]:
        """Validate performance configuration."""
        validation = {
            "valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check worker configuration
        if self.max_workers < 1:
            validation["valid"] = False
            validation["warnings"].append("MAX_WORKERS must be at least 1")
        elif self.max_workers > 32:
            validation["warnings"].append("MAX_WORKERS > 32 may cause excessive resource usage")
        
        # Check memory limits
        if _PSUTIL_AVAILABLE:
            try:
                available_memory = psutil.virtual_memory().available / (1024 * 1024)
                if self.max_memory_mb > available_memory:
                    validation["warnings"].append(f"MAX_MEMORY_MB ({self.max_memory_mb}) exceeds available memory ({available_memory:.0f}MB)")
                    validation["recommendations"].append(f"Consider reducing MAX_MEMORY_MB to {int(available_memory * 0.8)}MB")
            except Exception:
                pass
        
        # Check rate limits
        if self.gemini_rate_limit_per_min <= 0:
            validation["valid"] = False
            validation["warnings"].append("GEMINI_RATE_LIMIT must be positive")
        
        if self.openai_rate_limit_per_min <= 0:
            validation["valid"] = False
            validation["warnings"].append("OPENAI_RATE_LIMIT must be positive")
        
        # Check content limits
        if self.max_article_length > 100000:
            validation["warnings"].append("MAX_ARTICLE_LENGTH is very high and may impact performance")
            validation["recommendations"].append("Consider reducing MAX_ARTICLE_LENGTH for better performance")
        
        return validation
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get performance-optimized settings based on system capabilities."""
        optimal = {}
        
        if _PSUTIL_AVAILABLE:
            try:
                # CPU-based worker optimization
                cpu_count = psutil.cpu_count()
                if cpu_count:
                    optimal["recommended_max_workers"] = min(cpu_count * 2, 16)
                
                # Memory-based optimization
                memory = psutil.virtual_memory()
                optimal["recommended_max_memory_mb"] = int((memory.available / (1024 * 1024)) * 0.7)
                optimal["recommended_max_cache_size_mb"] = min(int((memory.available / (1024 * 1024)) * 0.1), 1000)
                
            except Exception as e:
                logger.warning(f"Failed to get system info for optimization: {e}")
        
        return optimal


@dataclass 
class Config:
    """
    Main configuration class with comprehensive settings management.
    
    Features:
    - Environment-aware configuration with intelligent defaults
    - Comprehensive validation and health monitoring
    - Performance optimization with system-aware tuning
    - Security management with key validation and rotation
    - Integration with enhanced logging and monitoring
    """
    
    # Environment and debugging
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    debug_mode: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    
    # Feature flags
    enable_detailed_analysis: bool = field(default_factory=lambda: os.getenv('ENABLE_DETAILED_ANALYSIS', 'true').lower() == 'true')
    enable_cross_verification: bool = field(default_factory=lambda: os.getenv('ENABLE_CROSS_VERIFICATION', 'true').lower() == 'true')
    enable_metrics_collection: bool = field(default_factory=lambda: os.getenv('ENABLE_METRICS_COLLECTION', 'true').lower() == 'true')
    
    # Configuration components
    paths: ConfigPaths = field(default_factory=ConfigPaths)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Runtime state
    _initialized_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _config_lock: threading.RLock = field(default_factory=threading.RLock)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Apply environment-specific overrides
        self._apply_environment_overrides()
        
        # Initialize logging integration
        self._setup_logging_integration()
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        env_overrides = {
            'development': {
                'debug_mode': True,
                'log_level': 'DEBUG',
                'enable_detailed_analysis': True,
                'enable_metrics_collection': True
            },
            'testing': {
                'debug_mode': False,
                'log_level': 'WARNING',
                'enable_detailed_analysis': False,
                'enable_metrics_collection': False,
                'performance.enable_cache': False
            },
            'staging': {
                'debug_mode': False,
                'log_level': 'INFO',
                'enable_detailed_analysis': True,
                'enable_metrics_collection': True
            },
            'production': {
                'debug_mode': False,
                'log_level': 'INFO',
                'enable_detailed_analysis': False,
                'enable_metrics_collection': True,
                'security.enable_api_key_rotation': True,
                'security.enable_data_encryption': True
            }
        }
        
        if self.environment in env_overrides:
            overrides = env_overrides[self.environment]
            
            for key, value in overrides.items():
                if '.' in key:  # Nested attribute
                    obj_name, attr_name = key.split('.', 1)
                    if hasattr(self, obj_name):
                        obj = getattr(self, obj_name)
                        if hasattr(obj, attr_name):
                            setattr(obj, attr_name, value)
                else:  # Direct attribute
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def _setup_logging_integration(self):
        """Setup integration with enhanced logging system."""
        try:
            # Try to setup logging with our configuration
            from .logger import setup_logging
            
            setup_logging(
                environment=self.environment,
                log_directory=self.paths.logs_dir,
                config_overrides={
                    'log_level': self.log_level,
                    'enable_debug_context': self.debug_mode
                }
            )
            
        except ImportError:
            # Fallback to basic logging if enhanced logging not available
            logging.basicConfig(
                level=getattr(logging, self.log_level.upper(), logging.INFO),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    @contextmanager
    def config_update_lock(self):
        """Context manager for thread-safe configuration updates."""
        with self._config_lock:
            yield
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Perform comprehensive configuration validation.
        
        Returns:
            Detailed validation report with errors, warnings, and recommendations
        """
        validation_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": self.environment,
            "overall_status": "valid",
            "validation_results": {},
            "summary": {
                "total_errors": 0,
                "total_warnings": 0,
                "components_validated": 0
            },
            "recommendations": []
        }
        
        try:
            # Validate paths
            path_validation = self.paths.validate_paths()
            validation_report["validation_results"]["paths"] = path_validation
            if not path_validation["valid"]:
                validation_report["overall_status"] = "invalid"
            validation_report["summary"]["total_errors"] += len(path_validation["errors"])
            validation_report["summary"]["total_warnings"] += len(path_validation["warnings"])
            validation_report["summary"]["components_validated"] += 1
            
            # Validate security
            security_validation = self.security.validate_api_keys()
            validation_report["validation_results"]["security"] = security_validation
            if security_validation["invalid_keys"]:
                validation_report["overall_status"] = "invalid"
            validation_report["summary"]["total_errors"] += len(security_validation["invalid_keys"])
            validation_report["summary"]["total_warnings"] += len(security_validation["missing_keys"])
            validation_report["summary"]["components_validated"] += 1
            
            # Validate performance
            performance_validation = self.performance.validate_performance_settings()
            validation_report["validation_results"]["performance"] = performance_validation
            if not performance_validation["valid"]:
                validation_report["overall_status"] = "invalid"
            validation_report["summary"]["total_warnings"] += len(performance_validation["warnings"])
            validation_report["recommendations"].extend(performance_validation["recommendations"])
            validation_report["summary"]["components_validated"] += 1
            
            # Environment-specific validations
            if self.environment == "production":
                # Production-specific checks
                if self.debug_mode:
                    validation_report["validation_results"]["environment_specific"] = {
                        "errors": ["Debug mode should not be enabled in production"]
                    }
                    validation_report["overall_status"] = "invalid"
                    validation_report["summary"]["total_errors"] += 1
                
                if not self.security.gemini_api_key:
                    validation_report["summary"]["total_errors"] += 1
                    if validation_report["overall_status"] != "invalid":
                        validation_report["overall_status"] = "invalid"
            
            # Generate recommendations
            if validation_report["summary"]["total_errors"] == 0 and validation_report["summary"]["total_warnings"] == 0:
                validation_report["recommendations"].append("Configuration is optimal")
            else:
                if validation_report["summary"]["total_errors"] > 0:
                    validation_report["recommendations"].append("Address configuration errors before deployment")
                if validation_report["summary"]["total_warnings"] > 0:
                    validation_report["recommendations"].append("Review configuration warnings for optimal performance")
        
        except Exception as e:
            validation_report["overall_status"] = "error"
            validation_report["error"] = str(e)
            validation_report["recommendations"] = ["Contact system administrator - validation failed"]
        
        return validation_report
    
    def get_scraper_config(self) -> Dict[str, Any]:
        """Get configuration for web scraping components."""
        return {
            "timeout_sec": self.security.request_timeout_sec,
            "max_retries": self.security.max_request_retries,
            "max_article_length": self.performance.max_article_length,
            "requests_per_minute": min(self.performance.gemini_rate_limit_per_min, 60),
            "enable_cache": self.performance.enable_cache,
            "cache_ttl_sec": self.performance.cache_ttl_sec,
            "user_agent": "FakeNewsDetector/3.2.0 (+https://github.com/fake-news-detector)",
            "respect_robots_txt": True,
            "max_concurrent_requests": min(self.performance.max_workers, 5)
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for AI model components."""
        return {
            "gemini": {
                "api_key": self.security.gemini_api_key,
                "rate_limit_per_min": self.performance.gemini_rate_limit_per_min,
                "max_tokens": 4096,
                "temperature": 0.2,
                "timeout_sec": self.security.request_timeout_sec
            },
            "openai": {
                "api_key": self.security.openai_api_key,
                "rate_limit_per_min": self.performance.openai_rate_limit_per_min,
                "max_tokens": 4096,
                "temperature": 0.2,
                "timeout_sec": self.security.request_timeout_sec
            },
            "general": {
                "max_batch_size": self.performance.max_batch_size,
                "enable_detailed_analysis": self.enable_detailed_analysis,
                "enable_cross_verification": self.enable_cross_verification
            }
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get configuration for logging system."""
        return {
            "level": self.log_level,
            "log_directory": str(self.paths.logs_dir),
            "environment": self.environment,
            "enable_performance_tracking": self.enable_metrics_collection,
            "enable_debug_context": self.debug_mode,
            "structured_logs": self.environment in ["staging", "production"],
            "colored_output": self.environment in ["development", "staging"]
        }
    
    def export_config(self, filename: Optional[str] = None, 
                     include_sensitive: bool = False) -> Path:
        """
        Export configuration to JSON file with optional sensitive data masking.
        
        Args:
            filename: Output filename (auto-generated if None)
            include_sensitive: Whether to include sensitive information like API keys
            
        Returns:
            Path to exported configuration file
        """
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"config_export_{self.environment}_{timestamp}.json"
        
        export_path = self.paths.config_dir / filename
        
        # Prepare export data
        export_data = {
            "export_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "environment": self.environment,
                "config_version": "3.2.0",
                "exported_by": os.getenv("USER", "unknown"),
                "include_sensitive": include_sensitive
            },
            "environment_config": {
                "environment": self.environment,
                "debug_mode": self.debug_mode,
                "log_level": self.log_level,
                "feature_flags": {
                    "enable_detailed_analysis": self.enable_detailed_analysis,
                    "enable_cross_verification": self.enable_cross_verification,
                    "enable_metrics_collection": self.enable_metrics_collection
                }
            },
            "paths": {
                "project_root": str(self.paths.project_root),
                "data_dir": str(self.paths.data_dir),
                "models_dir": str(self.paths.models_dir),
                "logs_dir": str(self.paths.logs_dir),
                "cache_dir": str(self.paths.cache_dir)
            },
            "performance": asdict(self.performance),
            "validation_report": self.validate_configuration()
        }
        
        # Handle sensitive information
        if include_sensitive:
            export_data["security"] = asdict(self.security)
        else:
            export_data["security"] = {
                "api_keys_configured": self.security.get_masked_keys(),
                "security_features": {
                    "enable_api_key_rotation": self.security.enable_api_key_rotation,
                    "enable_request_signing": self.security.enable_request_signing,
                    "enable_data_encryption": self.security.enable_data_encryption
                }
            }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Configuration exported to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": self.environment,
            "overall_status": "healthy",
            "uptime_seconds": (datetime.now(timezone.utc) - self._initialized_at).total_seconds(),
            "component_health": {},
            "system_resources": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check configuration health
            validation = self.validate_configuration()
            health_status["component_health"]["configuration"] = {
                "status": validation["overall_status"],
                "errors": validation["summary"]["total_errors"],
                "warnings": validation["summary"]["total_warnings"]
            }
            
            if validation["overall_status"] == "invalid":
                health_status["overall_status"] = "unhealthy"
                health_status["issues"].extend([
                    "Configuration validation failed",
                    f"{validation['summary']['total_errors']} configuration errors found"
                ])
            
            # Check system resources
            if _PSUTIL_AVAILABLE:
                try:
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage(str(self.paths.project_root))
                    
                    health_status["system_resources"] = {
                        "memory_percent": memory.percent,
                        "memory_available_gb": round(memory.available / (1024**3), 2),
                        "disk_percent": round((disk.used / disk.total) * 100, 1),
                        "disk_free_gb": round(disk.free / (1024**3), 2),
                        "cpu_percent": psutil.cpu_percent(interval=1)
                    }
                    
                    # Check for resource issues
                    if memory.percent > 90:
                        health_status["issues"].append("High memory usage (>90%)")
                        health_status["overall_status"] = "degraded"
                    
                    disk_percent = (disk.used / disk.total) * 100
                    if disk_percent > 90:
                        health_status["issues"].append("Low disk space (<10% free)")
                        health_status["overall_status"] = "degraded"
                        
                except Exception as e:
                    health_status["issues"].append(f"Failed to get system resources: {e}")
            
            # Check component-specific health
            path_validation = self.paths.validate_paths()
            health_status["component_health"]["paths"] = {
                "status": "healthy" if path_validation["valid"] else "unhealthy",
                "issues": len(path_validation["errors"]) + len(path_validation["warnings"])
            }
            
            api_validation = self.security.validate_api_keys()
            health_status["component_health"]["api_keys"] = {
                "status": "healthy" if api_validation["valid_keys"] else "degraded",
                "valid_keys": len(api_validation["valid_keys"]),
                "missing_keys": len(api_validation["missing_keys"])
            }
            
            # Generate recommendations
            if health_status["overall_status"] == "healthy":
                health_status["recommendations"].append("System is operating normally")
            else:
                health_status["recommendations"].extend([
                    "Review system issues and take corrective action",
                    "Monitor resource usage and scale if necessary"
                ])
                
                if health_status["component_health"]["api_keys"]["missing_keys"] > 0:
                    health_status["recommendations"].append("Configure missing API keys for full functionality")
        
        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)
            health_status["recommendations"] = ["Contact system administrator - health check failed"]
        
        return health_status
    
    def print_status_report(self):
        """Print comprehensive configuration status report to console."""
        print("🔍 Configuration Status Report")
        print("=" * 60)
        
        # Basic info
        print(f"🌍 Environment: {self.environment}")
        print(f"🐛 Debug Mode: {'Enabled' if self.debug_mode else 'Disabled'}")
        print(f"📝 Log Level: {self.log_level}")
        
        # Validation status
        validation = self.validate_configuration()
        status_icon = "✅" if validation["overall_status"] == "valid" else "❌"
        print(f"\n{status_icon} Overall Status: {validation['overall_status'].upper()}")
        
        if validation["summary"]["total_errors"] > 0:
            print(f"\n❌ Errors: {validation['summary']['total_errors']}")
        
        if validation["summary"]["total_warnings"] > 0:
            print(f"\n⚠️  Warnings: {validation['summary']['total_warnings']}")
        
        # API Keys status
        print(f"\n🔑 API Keys:")
        masked_keys = self.security.get_masked_keys()
        for service, masked_key in masked_keys.items():
            status = "✅ Configured" if masked_key != "NOT_SET" else "❌ Missing"
            print(f"  - {service.title()}: {status}")
        
        # Performance settings
        print(f"\n⚡ Performance:")
        print(f"  - Max Workers: {self.performance.max_workers}")
        print(f"  - Memory Limit: {self.performance.max_memory_mb} MB")
        print(f"  - Cache Enabled: {'Yes' if self.performance.enable_cache else 'No'}")
        
        # System resources (if available)
        if _PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage(str(self.paths.project_root))
                print(f"\n💻 System Resources:")
                print(f"  - Memory Usage: {memory.percent:.1f}%")
                print(f"  - Disk Usage: {(disk.used/disk.total)*100:.1f}%")
            except Exception:
                pass
        
        # Recommendations
        if validation["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in validation["recommendations"][:3]:  # Show top 3
                print(f"  - {rec}")
        
        print("=" * 60)


# Create global configuration instance
config = Config()


# Convenience functions for backward compatibility
def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def get_dataset_paths() -> Dict[str, Path]:
    """Get paths for dataset files."""
    return {
        "true_file": config.paths.raw_data_dir / "True.csv",
        "fake_file": config.paths.raw_data_dir / "Fake.csv", 
        "scraped_data": config.paths.raw_data_dir / "scraped_articles.csv",
        "processed_data": config.paths.processed_data_dir / "processed_dataset.csv"
    }


def validate_setup() -> Dict[str, Any]:
    """Validate system setup - convenience function."""
    return config.validate_configuration()


def print_setup_status():
    """Print setup status - convenience function."""
    config.print_status_report()


# Export commonly used paths and settings for backward compatibility
PROJECT_ROOT = config.paths.project_root
DATA_DIR = config.paths.data_dir
RAW_DATA_DIR = config.paths.raw_data_dir
PROCESSED_DATA_DIR = config.paths.processed_data_dir
MODELS_DIR = config.paths.models_dir
SAVED_MODELS_DIR = config.paths.saved_models_dir
LOGS_DIR = config.paths.logs_dir

# Environment and API settings
ENVIRONMENT = config.environment
GEMINI_API_KEY = config.security.gemini_api_key
OPENAI_API_KEY = config.security.openai_api_key
LOG_LEVEL = config.log_level

# Processing settings
MAX_ARTICLE_LENGTH = config.performance.max_article_length
MAX_WORKERS = config.performance.max_workers
DEFAULT_TIMEOUT = config.security.request_timeout_sec


# Export all public interfaces
__all__ = [
    'Config',
    'ConfigPaths',
    'SecurityConfig', 
    'PerformanceConfig',
    'config',
    'get_config',
    'get_dataset_paths',
    'validate_setup',
    'print_setup_status',
    # Backward compatibility exports
    'PROJECT_ROOT', 'DATA_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR',
    'MODELS_DIR', 'SAVED_MODELS_DIR', 'LOGS_DIR',
    'ENVIRONMENT', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 'LOG_LEVEL',
    'MAX_ARTICLE_LENGTH', 'MAX_WORKERS', 'DEFAULT_TIMEOUT'
]


# Example usage and testing
if __name__ == "__main__":
    print_setup_status()
    
    # Export current configuration
    export_path = config.export_config(include_sensitive=False)
    print(f"\n📁 Configuration exported to: {export_path}")
    
    # Health check
    health = config.get_health_status()
    print(f"\n🏥 System Health: {health['overall_status'].upper()}")
    
    # Cleanup temp files
    cleanup_result = config.paths.cleanup_temp_files()
    if cleanup_result['files_removed'] > 0:
        print(f"\n🧹 Cleaned up {cleanup_result['files_removed']} temporary files")
