"""
Enhanced Utils Package for Production Fake News Detection

Comprehensive utility package with text processing, logging, scraping,
configuration management, and system utilities.

Features:
- Advanced text sanitization and metadata extraction
- Production-ready logging with color and file rotation
- Multi-strategy news article scraping
- Environment-aware configuration management
- System monitoring and health checks

Version: 3.2.0 - Production Enhanced Edition
"""

# Import main utility functions
from .helpers import (
    sanitize_text,
    extract_metadata,
    validate_input_data,
    generate_content_hash,
    clean_filename,
    get_system_info,
    format_bytes,
    safe_json_parse,
    truncate_text,
    is_valid_email,
    create_error_response
)

from .logger import (
    setup_logging,
    get_logger,
    setup_logger,  # Backward compatibility
    LogConfig,
    ColorFormatter
)

from .url_scraper import (
    ProductionNewsScraper,
    ScrapingResult,
    create_scraper,
    validate_url,
    is_news_url,
    extract_domain
)

from .config import (
    Config,
    config,
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    GEMINI_API_KEY,
    get_dataset_paths,
    validate_setup,
    print_setup_status
)

# Package metadata
__version__ = "3.2.0"
__author__ = "Fake News Detection Team"
__description__ = "Production-ready utilities for fake news detection system"
__all__ = [
    # Helper functions
    'sanitize_text',
    'extract_metadata',
    'validate_input_data',
    'generate_content_hash',
    'clean_filename',
    'get_system_info',
    'format_bytes',
    'safe_json_parse',
    'truncate_text',
    'is_valid_email',
    'create_error_response',
    
    # Logging
    'setup_logging',
    'get_logger',
    'setup_logger',
    'LogConfig',
    'ColorFormatter',
    
    # Scraping
    'ProductionNewsScraper',
    'ScrapingResult',
    'create_scraper',
    'validate_url',
    'is_news_url',
    'extract_domain',
    
    # Configuration
    'Config',
    'config',
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR', 
    'LOGS_DIR',
    'GEMINI_API_KEY',
    'get_dataset_paths',
    'validate_setup',
    'print_setup_status'
]

def initialize_utils(environment: str = "development") -> dict:
    """
    Initialize the utils package with environment-specific settings
    
    Args:
        environment: Target environment ("development", "testing", "production")
        
    Returns:
        Initialization status dictionary
    """
    try:
        # Setup logging
        logging_manager = setup_logging(environment)
        
        # Validate configuration
        validation = validate_setup()
        
        # Get system info
        system_info = get_system_info()
        
        return {
            'success': True,
            'environment': environment,
            'logging_initialized': True,
            'config_valid': validation.get('valid', False),
            'system_info': system_info,
            'version': __version__
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'version': __version__
        }

def get_package_status() -> dict:
    """
    Get comprehensive package status and health information
    
    Returns:
        Package status dictionary
    """
    try:
        config_validation = validate_setup()
        system_info = get_system_info()
        
        return {
            'package_version': __version__,
            'configuration': {
                'valid': config_validation.get('valid', False),
                'errors': config_validation.get('errors', []),
                'warnings': config_validation.get('warnings', [])
            },
            'system': system_info,
            'components': {
                'helpers': True,
                'logger': True,
                'scraper': True,
                'config': True
            },
            'status': 'healthy' if config_validation.get('valid', False) else 'degraded'
        }
        
    except Exception as e:
        return {
            'package_version': __version__,
            'status': 'error',
            'error': str(e)
        }

# Auto-initialize package
_initialization_result = initialize_utils()

if not _initialization_result['success']:
    import warnings
    warnings.warn(f"Utils package initialization failed: {_initialization_result.get('error', 'Unknown error')}")

# Export initialization result
__all__.extend(['initialize_utils', 'get_package_status', '_initialization_result'])
