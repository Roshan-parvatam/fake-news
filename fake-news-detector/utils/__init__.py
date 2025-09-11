# utils/__init__.py

"""
Enhanced Utils Package for Production Fake News Detection

Comprehensive utility package providing essential functionality for text processing,
web scraping, logging, configuration management, and system monitoring.

Components:
- helpers: Advanced text sanitization, validation, and metadata extraction
- url_scraper: Production-ready multi-strategy web scraping with rate limiting
- logger: Enhanced logging with colors, rotation, and structured output
- config: Environment-aware configuration with validation and health monitoring

Features:
- Graceful import handling with detailed error reporting
- Component availability tracking and health monitoring
- Factory functions for easy component instantiation
- Comprehensive status reporting and diagnostics
- Version tracking and compatibility checking
- Production-ready error handling and recovery

Version: 3.2.0 - Enhanced Production Edition
"""

import logging
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

# Package metadata
__version__ = "3.2.0"
__author__ = "Enhanced Fake News Detection Team"
__description__ = "Production-ready utilities for comprehensive fake news detection"
__license__ = "MIT"
__status__ = "Production"

# Component availability tracking
_component_status = {
    'helpers': {'available': False, 'error': None, 'features': []},
    'url_scraper': {'available': False, 'error': None, 'features': []},
    'logger': {'available': False, 'error': None, 'features': []},
    'config': {'available': False, 'error': None, 'features': []}
}

# Imported components storage
_imported_components = {}

# Configure package logger - will be enhanced after logger module is loaded
logger = logging.getLogger(__name__)

# ============================================================================
# HELPERS MODULE - Text Processing and Validation
# ============================================================================

try:
    from .helpers import (
        sanitize_text,
        validate_input_data,
        extract_metadata,
        generate_content_hash,
        clean_filename,
        get_system_info,
        format_bytes,
        safe_json_parse,
        truncate_text,
        is_valid_email,
        is_valid_url,
        extract_domain,
        create_error_response
    )
    
    _component_status['helpers'] = {
        'available': True,
        'error': None,
        'features': [
            'advanced_text_sanitization', 'xss_prevention', 'input_validation',
            'metadata_extraction', 'content_hashing', 'system_monitoring'
        ]
    }
    
    _imported_components['helpers'] = [
        'sanitize_text', 'validate_input_data', 'extract_metadata',
        'generate_content_hash', 'clean_filename', 'get_system_info',
        'format_bytes', 'safe_json_parse', 'truncate_text',
        'is_valid_email', 'is_valid_url', 'extract_domain', 'create_error_response'
    ]
    
    logger.info("✅ Helpers module loaded successfully")
    
except ImportError as e:
    _component_status['helpers'] = {
        'available': False,
        'error': str(e),
        'features': []
    }
    logger.warning(f"⚠️ Helpers module failed to load: {e}")
    
    # Provide fallback functions
    def sanitize_text(text: str, **kwargs) -> str:
        """Fallback text sanitization."""
        return str(text).strip() if text else ""
    
    def create_error_response(message: str, code: str = "ERROR", details=None) -> Dict[str, Any]:
        """Fallback error response."""
        return {"success": False, "error": {"message": message, "code": code}}

# ============================================================================
# URL SCRAPER MODULE - Web Content Extraction
# ============================================================================

try:
    from .url_scraper import (
        ProductionNewsScraper,
        ScrapingResult,
        RateLimitConfig,
        create_scraper,
        validate_url,
        is_news_url,
        extract_domain as scraper_extract_domain
    )
    
    _component_status['url_scraper'] = {
        'available': True,
        'error': None,
        'features': [
            'multi_strategy_scraping', 'adaptive_rate_limiting', 'content_validation',
            'async_support', 'robots_txt_compliance', 'quality_scoring'
        ]
    }
    
    _imported_components['url_scraper'] = [
        'ProductionNewsScraper', 'ScrapingResult', 'RateLimitConfig',
        'create_scraper', 'validate_url', 'is_news_url'
    ]
    
    logger.info("✅ URL Scraper module loaded successfully")
    
except ImportError as e:
    _component_status['url_scraper'] = {
        'available': False,
        'error': str(e),
        'features': []
    }
    logger.warning(f"⚠️ URL Scraper module failed to load: {e}")
    
    # Provide fallback scraper
    class ScrapingResult:
        def __init__(self, success=False, text="", error="Module not available", **kwargs):
            self.success = success
            self.text = text
            self.error = error
            self.__dict__.update(kwargs)
    
    def create_scraper(config=None):
        """Fallback scraper that returns error results."""
        class FallbackScraper:
            def scrape_article(self, url, **kwargs):
                return ScrapingResult(
                    success=False,
                    url=url,
                    error="URL Scraper module not available"
                )
        return FallbackScraper()

# ============================================================================
# LOGGER MODULE - Enhanced Logging System
# ============================================================================

try:
    from .logger import (
        LoggingManager,
        EnhancedLogger,
        LogConfig,
        setup_logging,
        get_logger
    )
    
    _component_status['logger'] = {
        'available': True,
        'error': None,
        'features': [
            'colored_console_output', 'rotating_file_logs', 'structured_json_logs',
            'performance_tracking', 'environment_awareness', 'health_monitoring'
        ]
    }
    
    _imported_components['logger'] = [
        'LoggingManager', 'EnhancedLogger', 'LogConfig', 
        'setup_logging', 'get_logger'
    ]
    
    # Update logger to use enhanced logger
    logger = get_logger(__name__)
    logger.debug("✅ Logger module loaded successfully")
    
except ImportError as e:
    _component_status['logger'] = {
        'available': False,
        'error': str(e),
        'features': []
    }
    logger.warning(f"⚠️ Logger module failed to load: {e}")
    
    # Provide fallback logging
    def setup_logging(environment="development", **kwargs):
        """Fallback logging setup."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()
    
    def get_logger(name: str):
        """Fallback logger getter."""
        return logging.getLogger(name)

# ============================================================================
# CONFIG MODULE - Configuration Management
# ============================================================================

try:
    from .config import (
        Config,
        ConfigPaths,
        SecurityConfig,
        PerformanceConfig,
        config,
        get_config,
        validate_setup,
        print_setup_status,
        get_dataset_paths
    )
    
    _component_status['config'] = {
        'available': True,
        'error': None,
        'features': [
            'environment_awareness', 'secure_key_management', 'path_validation',
            'performance_optimization', 'health_monitoring', 'configuration_export'
        ]
    }
    
    _imported_components['config'] = [
        'Config', 'ConfigPaths', 'SecurityConfig', 'PerformanceConfig',
        'config', 'get_config', 'validate_setup', 'print_setup_status',
        'get_dataset_paths'
    ]
    
    logger.info("✅ Config module loaded successfully")
    
except ImportError as e:
    _component_status['config'] = {
        'available': False,
        'error': str(e),
        'features': []
    }
    logger.warning(f"⚠️ Config module failed to load: {e}")
    
    # Provide fallback config
    class Config:
        def __init__(self):
            self.environment = "development"
            self.debug_mode = True
        
        def validate_configuration(self):
            return {"valid": False, "error": "Config module not available"}
    
    config = Config()
    get_config = lambda: config
    validate_setup = lambda: {"valid": False, "error": "Config module not available"}
    print_setup_status = lambda: print("Config module not available")

# ============================================================================
# PACKAGE INITIALIZATION AND UTILITIES
# ============================================================================

def get_package_status() -> Dict[str, Any]:
    """
    Get comprehensive package status and component availability.
    
    Returns:
        Dictionary with detailed package and component status
    """
    available_components = [name for name, status in _component_status.items() if status['available']]
    total_components = len(_component_status)
    
    return {
        'package_info': {
            'name': 'utils',
            'version': __version__,
            'author': __author__,
            'description': __description__,
            'status': __status__,
            'timestamp': datetime.now().isoformat()
        },
        'component_status': _component_status.copy(),
        'imported_components': _imported_components.copy(),
        'summary': {
            'available_components': len(available_components),
            'total_components': total_components,
            'availability_percentage': (len(available_components) / total_components) * 100,
            'all_components_available': len(available_components) == total_components,
            'available_component_list': available_components
        },
        'features': {
            component: details['features'] 
            for component, details in _component_status.items() 
            if details['available']
        }
    }


def get_package_health() -> Dict[str, Any]:
    """
    Get comprehensive package health status with recommendations.
    
    Returns:
        Dictionary with health status and actionable recommendations
    """
    status = get_package_status()
    health_info = {
        'overall_status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'package_version': __version__,
        'component_health': {},
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Analyze component health
    for component, details in _component_status.items():
        if details['available']:
            health_info['component_health'][component] = {
                'status': 'healthy',
                'features': len(details['features']),
                'feature_list': details['features']
            }
        else:
            health_info['component_health'][component] = {
                'status': 'unavailable',
                'error': details['error']
            }
            health_info['issues'].append(f"{component}: {details['error']}")
    
    # Determine overall health
    available_count = len([c for c in _component_status.values() if c['available']])
    total_count = len(_component_status)
    
    if available_count == total_count:
        health_info['overall_status'] = 'healthy'
        health_info['recommendations'].append("All components available and functioning")
    elif available_count >= total_count * 0.75:  # 75% threshold
        health_info['overall_status'] = 'degraded'
        health_info['warnings'].append("Some components unavailable")
        health_info['recommendations'].extend([
            "Check missing component dependencies",
            "System will operate with reduced functionality"
        ])
    else:
        health_info['overall_status'] = 'critical'
        health_info['issues'].append("Multiple components unavailable")
        health_info['recommendations'].extend([
            "Install missing dependencies immediately",
            "Check system configuration and environment setup"
        ])
    
    # Component-specific recommendations
    if not _component_status['helpers']['available']:
        health_info['recommendations'].append("Text processing functionality limited without helpers module")
    
    if not _component_status['url_scraper']['available']:
        health_info['recommendations'].append("Web scraping functionality unavailable - install requests, beautifulsoup4")
    
    if not _component_status['logger']['available']:
        health_info['recommendations'].append("Enhanced logging unavailable - using basic logging")
    
    if not _component_status['config']['available']:
        health_info['recommendations'].append("Configuration management limited - using fallback settings")
    
    return health_info


def initialize_package(environment: str = "development", 
                      config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize the utils package with environment-specific settings.
    
    Args:
        environment: Target environment (development, testing, staging, production)
        config_overrides: Optional configuration overrides
        
    Returns:
        Initialization results and status
    """
    initialization_result = {
        'success': False,
        'environment': environment,
        'timestamp': datetime.now().isoformat(),
        'initialization_steps': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        # Step 1: Setup logging
        if _component_status['logger']['available']:
            try:
                log_config_overrides = config_overrides.get('logging', {}) if config_overrides else {}
                logging_manager = setup_logging(environment, **log_config_overrides)
                initialization_result['initialization_steps']['logging'] = 'success'
            except Exception as e:
                initialization_result['initialization_steps']['logging'] = f'failed: {e}'
                initialization_result['warnings'].append(f"Logging setup failed: {e}")
        else:
            initialization_result['initialization_steps']['logging'] = 'skipped: module unavailable'
        
        # Step 2: Validate configuration
        if _component_status['config']['available']:
            try:
                validation_result = validate_setup()
                if validation_result.get('valid', False):
                    initialization_result['initialization_steps']['config'] = 'success'
                else:
                    initialization_result['initialization_steps']['config'] = f"validation_failed"
                    initialization_result['warnings'].extend(validation_result.get('errors', []))
            except Exception as e:
                initialization_result['initialization_steps']['config'] = f'failed: {e}'
                initialization_result['warnings'].append(f"Configuration validation failed: {e}")
        else:
            initialization_result['initialization_steps']['config'] = 'skipped: module unavailable'
        
        # Step 3: Test component functionality
        if _component_status['url_scraper']['available']:
            try:
                test_scraper = create_scraper({'requests_per_minute': 1})
                initialization_result['initialization_steps']['scraper'] = 'success'
            except Exception as e:
                initialization_result['initialization_steps']['scraper'] = f'test_failed: {e}'
                initialization_result['warnings'].append(f"Scraper test failed: {e}")
        else:
            initialization_result['initialization_steps']['scraper'] = 'skipped: module unavailable'
        
        # Step 4: Health check
        health_status = get_package_health()
        initialization_result['health_status'] = health_status['overall_status']
        
        if health_status['overall_status'] in ['healthy', 'degraded']:
            initialization_result['success'] = True
        
        # Add recommendations
        initialization_result['recommendations'] = health_status.get('recommendations', [])
        
    except Exception as e:
        initialization_result['errors'].append(f"Initialization failed: {str(e)}")
        initialization_result['success'] = False
    
    return initialization_result


def print_package_info():
    """Print comprehensive package information to console."""
    status = get_package_status()
    
    print("🎯 Utils Package Information")
    print("=" * 50)
    print(f"📦 Package: {status['package_info']['name']} v{status['package_info']['version']}")
    print(f"👥 Author: {status['package_info']['author']}")
    print(f"📝 Description: {status['package_info']['description']}")
    print(f"🏷️  Status: {status['package_info']['status']}")
    
    print(f"\n📊 Component Availability: {status['summary']['availability_percentage']:.1f}%")
    print(f"✅ Available: {status['summary']['available_components']}/{status['summary']['total_components']}")
    
    print("\n🔧 Component Status:")
    for component, details in _component_status.items():
        if details['available']:
            features = f" ({len(details['features'])} features)" if details['features'] else ""
            print(f"  ✅ {component}{features}")
        else:
            print(f"  ❌ {component}: {details['error']}")
    
    print(f"\n🚀 Available Features:")
    for component, features in status['features'].items():
        if features:
            print(f"  {component}: {', '.join(features)}")
    
    print("=" * 50)


def get_available_functions() -> List[str]:
    """Get list of all available functions from loaded components."""
    available_functions = []
    for component, functions in _imported_components.items():
        if _component_status[component]['available']:
            available_functions.extend(functions)
    return available_functions


# ============================================================================
# DYNAMIC EXPORTS BASED ON COMPONENT AVAILABILITY
# ============================================================================

def _build_exports() -> List[str]:
    """Build exports list dynamically based on available components."""
    exports = [
        # Package utilities
        '__version__', '__author__', '__description__',
        'get_package_status', 'get_package_health', 'initialize_package',
        'print_package_info', 'get_available_functions'
    ]
    
    # Add available component exports
    for component, functions in _imported_components.items():
        if _component_status[component]['available']:
            exports.extend(functions)
    
    return exports

__all__ = _build_exports()

# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

# Auto-initialize package with basic setup
try:
    _auto_init_result = initialize_package(
        environment=os.getenv('ENVIRONMENT', 'development')
    )
    
    if not _auto_init_result['success']:
        logger.warning("Package auto-initialization completed with warnings")
        
except Exception as e:
    logger.error(f"Package auto-initialization failed: {e}")

# Log package status
available_count = len([c for c in _component_status.values() if c['available']])
total_count = len(_component_status)

if available_count == total_count:
    logger.info(f"🎉 Utils package v{__version__} fully loaded - all {total_count} components available")
elif available_count > 0:
    logger.info(f"⚠️ Utils package v{__version__} partially loaded - {available_count}/{total_count} components available")
else:
    logger.error(f"❌ Utils package v{__version__} failed to load - no components available")

# Export initialization result for monitoring
__all__.extend(['_auto_init_result', '_component_status'])
