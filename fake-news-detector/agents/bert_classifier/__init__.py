# agents/bert_classifier/__init__.py

"""
BERT Classifier Package - Enhanced Production Ready

Comprehensive BERT classification package with modular architecture,
enhanced error handling, and production-grade reliability.

Components:
- BERTClassifier: Main classification agent with async support and comprehensive monitoring
- FakeNewsDataset: Efficient dataset class for batch processing with preprocessing integration
- TextPreprocessor: Advanced text preprocessing with security validation and performance tracking
- DeviceManager: Intelligent device management with optimization and health monitoring
- ModelManager: Comprehensive model loading, validation, and resource management

Key Features:
- Dynamic configuration management with environment awareness
- Enhanced error handling with recovery strategies and detailed reporting
- Performance monitoring and health checking for production environments
- Async processing capabilities with session tracking and audit trails
- Memory management and resource optimization
- Security validation and content filtering
- Comprehensive logging with structured format and session tracking
- Integration with enhanced exception system for consistent error handling

Version: 3.2.0 - Enhanced Production Edition
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

# Core component imports with enhanced error handling
try:
    from .classifier import BERTClassifier, FakeNewsDataset
    _classifier_available = True
except ImportError as e:
    _classifier_available = False
    logging.getLogger(__name__).error(f"Failed to import classifier components: {e}")

try:
    from .preprocessing import TextPreprocessor
    _preprocessing_available = True
except ImportError as e:
    _preprocessing_available = False
    logging.getLogger(__name__).error(f"Failed to import preprocessing components: {e}")

try:
    from .model_utils import DeviceManager, ModelManager
    _model_utils_available = True
except ImportError as e:
    _model_utils_available = False
    logging.getLogger(__name__).error(f"Failed to import model utilities: {e}")

# Enhanced exception integration
try:
    from agents.llm_explanation.exceptions import (
        handle_llm_explanation_exception,
        ErrorContext,
        log_exception_with_context
    )
    _enhanced_exceptions_available = True
except ImportError:
    _enhanced_exceptions_available = False

# Configuration system integration with enhanced error handling
try:
    from config import get_model_config, get_settings
    _config_available = True
except ImportError:
    _config_available = False
    logging.getLogger(__name__).warning("Config system not available - using fallback configuration")

# Package metadata with comprehensive information
__version__ = "3.2.0"
__author__ = "Enhanced Fake News Detection Team"
__description__ = "Production-ready BERT classifier with comprehensive error handling, performance monitoring, and enhanced integrations"
__license__ = "MIT"
__status__ = "Production"
__compatibility__ = "Python 3.8+"

# Component availability status
__component_status__ = {
    "classifier_available": _classifier_available,
    "preprocessing_available": _preprocessing_available,
    "model_utils_available": _model_utils_available,
    "config_available": _config_available,
    "enhanced_exceptions_available": _enhanced_exceptions_available
}

# Enhanced configuration functions with comprehensive error handling
def get_bert_config(fallback: bool = True) -> Dict[str, Any]:
    """
    Get BERT classifier configuration with enhanced error handling and fallbacks.

    Args:
        fallback: Whether to use fallback configuration if config system unavailable

    Returns:
        Configuration dictionary with BERT classifier settings

    Example:
        >>> config = get_bert_config()
        >>> classifier = BERTClassifier(config)
    """
    try:
        if _config_available:
            config = get_model_config('bert_classifier')
            if config:
                return config
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load BERT config: {e}")

    if fallback:
        return get_fallback_bert_config()
    else:
        raise RuntimeError("BERT configuration not available and fallback disabled")


def get_system_config(fallback: bool = True) -> Dict[str, Any]:
    """
    Get system configuration with enhanced error handling and fallbacks.

    Args:
        fallback: Whether to use fallback configuration if config system unavailable

    Returns:
        System configuration dictionary

    Example:
        >>> config = get_system_config()
        >>> models_dir = Path(config.get('models_dir', './models'))
    """
    try:
        if _config_available:
            settings = get_settings()
            if settings:
                return settings.__dict__ if hasattr(settings, '__dict__') else {}
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load system config: {e}")

    if fallback:
        return get_fallback_system_config()
    else:
        raise RuntimeError("System configuration not available and fallback disabled")


def get_fallback_bert_config() -> Dict[str, Any]:
    """
    Get fallback BERT configuration for use when config system is unavailable.

    Returns:
        Fallback BERT configuration dictionary with safe defaults
    """
    return {
        'model_name': 'bert-base-uncased',
        'max_length': 512,
        'batch_size': 16,
        'device': 'auto',
        'enable_preprocessing': True,
        'enable_metrics': True,
        'model_path': 'bert_fake_news_classifier',
        'preprocessing_config': {
            'max_length': 2000,
            'remove_urls': True,
            'remove_emails': True,
            'enable_security_checks': True
        },
        'confidence_threshold': 0.8,
        'min_text_length': 10,
        'max_retries': 3,
        'timeout': 30,
        'config_source': 'fallback'
    }


def get_fallback_system_config() -> Dict[str, Any]:
    """
    Get fallback system configuration for use when config system is unavailable.

    Returns:
        Fallback system configuration dictionary with safe defaults
    """
    return {
        'models_dir': './models',
        'logs_dir': './logs',
        'cache_dir': './cache',
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'debug_mode': os.getenv('DEBUG', 'false').lower() == 'true',
        'enable_logging': True,
        'log_level': 'INFO',
        'config_source': 'fallback'
    }


def create_bert_classifier(config: Optional[Dict[str, Any]] = None, 
                          session_id: Optional[str] = None) -> 'BERTClassifier':
    """
    Create BERT classifier instance with enhanced configuration and error handling.

    Args:
        config: Optional configuration dictionary to override defaults
        session_id: Optional session ID for tracking

    Returns:
        Configured BERTClassifier instance

    Raises:
        ImportError: If classifier components are not available
        RuntimeError: If configuration fails

    Example:
        >>> classifier = create_bert_classifier()
        >>> classifier = create_bert_classifier({'batch_size': 32, 'device': 'cuda'})
    """
    if not _classifier_available:
        raise ImportError("BERTClassifier not available - check component imports")

    try:
        # Get base configuration
        base_config = get_bert_config()
        
        # Merge with user overrides
        if config:
            base_config.update(config)

        # Create classifier instance
        classifier = BERTClassifier(base_config)
        
        logging.getLogger(__name__).info(
            f"BERT classifier created successfully",
            extra={'session_id': session_id, 'config_source': base_config.get('config_source')}
        )
        
        return classifier

    except Exception as e:
        error_msg = f"Failed to create BERT classifier: {str(e)}"
        logging.getLogger(__name__).error(error_msg, extra={'session_id': session_id})
        
        if _enhanced_exceptions_available:
            context = ErrorContext(
                session_id=session_id,
                operation="classifier_creation"
            )
            standardized_error = handle_llm_explanation_exception(e, context)
            log_exception_with_context(standardized_error, session_id, {'package': 'bert_classifier'})
        
        raise RuntimeError(error_msg)


def create_text_preprocessor(config: Optional[Dict[str, Any]] = None,
                           session_id: Optional[str] = None) -> 'TextPreprocessor':
    """
    Create text preprocessor instance with enhanced configuration.

    Args:
        config: Optional configuration dictionary to override defaults
        session_id: Optional session ID for tracking

    Returns:
        Configured TextPreprocessor instance

    Raises:
        ImportError: If preprocessing components are not available

    Example:
        >>> preprocessor = create_text_preprocessor()
        >>> preprocessor = create_text_preprocessor({'max_length': 1000, 'remove_urls': True})
    """
    if not _preprocessing_available:
        raise ImportError("TextPreprocessor not available - check component imports")

    try:
        # Get preprocessing config from BERT config
        bert_config = get_bert_config()
        base_config = bert_config.get('preprocessing_config', {})
        
        # Merge with user overrides
        if config:
            base_config.update(config)

        preprocessor = TextPreprocessor(base_config)
        
        logging.getLogger(__name__).info(
            f"Text preprocessor created successfully",
            extra={'session_id': session_id}
        )
        
        return preprocessor

    except Exception as e:
        error_msg = f"Failed to create text preprocessor: {str(e)}"
        logging.getLogger(__name__).error(error_msg, extra={'session_id': session_id})
        raise RuntimeError(error_msg)


def create_device_manager(device_preference: str = 'auto',
                         config: Optional[Dict[str, Any]] = None,
                         session_id: Optional[str] = None) -> 'DeviceManager':
    """
    Create device manager instance with enhanced configuration.

    Args:
        device_preference: Device preference ('auto', 'cpu', 'cuda', 'mps')
        config: Optional configuration dictionary
        session_id: Optional session ID for tracking

    Returns:
        Configured DeviceManager instance

    Raises:
        ImportError: If model utilities are not available

    Example:
        >>> device_manager = create_device_manager('cuda')
        >>> device_manager = create_device_manager('auto', {'memory_threshold_gb': 4.0})
    """
    if not _model_utils_available:
        raise ImportError("DeviceManager not available - check component imports")

    try:
        device_manager = DeviceManager(device_preference, config)
        
        logging.getLogger(__name__).info(
            f"Device manager created: {device_manager.get_device()}",
            extra={'session_id': session_id}
        )
        
        return device_manager

    except Exception as e:
        error_msg = f"Failed to create device manager: {str(e)}"
        logging.getLogger(__name__).error(error_msg, extra={'session_id': session_id})
        raise RuntimeError(error_msg)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information and component status.

    Returns:
        Dictionary with package metadata and component availability

    Example:
        >>> info = get_package_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Components available: {info['components_available']}")
    """
    return {
        'package_metadata': {
            'name': 'bert_classifier',
            'version': __version__,
            'author': __author__,
            'description': __description__,
            'license': __license__,
            'status': __status__,
            'compatibility': __compatibility__
        },
        'component_status': __component_status__,
        'components_available': sum(__component_status__.values()),
        'total_components': len(__component_status__),
        'all_components_available': all(__component_status__.values()),
        'capabilities': {
            'async_processing': _classifier_available,
            'text_preprocessing': _preprocessing_available,
            'device_management': _model_utils_available,
            'enhanced_error_handling': _enhanced_exceptions_available,
            'configuration_system': _config_available
        },
        'features': [
            'Enhanced BERT classification with async support',
            'Advanced text preprocessing with security validation',
            'Intelligent device management and optimization',
            'Comprehensive performance monitoring and health checking',
            'Production-grade error handling and recovery',
            'Memory management and resource optimization',
            'Session tracking and audit trails',
            'Integration with enhanced exception system'
        ]
    }


def check_package_health() -> Dict[str, Any]:
    """
    Perform comprehensive package health check for production monitoring.

    Returns:
        Dictionary with detailed health status and recommendations

    Example:
        >>> health = check_package_health()
        >>> if health['overall_status'] == 'healthy':
        ...     print("Package ready for use")
    """
    health_info = {
        'overall_status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'package_version': __version__,
        'issues': [],
        'warnings': [],
        'recommendations': []
    }

    try:
        # Check component availability
        missing_components = [comp for comp, available in __component_status__.items() if not available]
        
        if missing_components:
            health_info['warnings'].extend([f"Component not available: {comp}" for comp in missing_components])
            if len(missing_components) > 2:
                health_info['overall_status'] = 'degraded'

        # Check critical components
        if not _classifier_available:
            health_info['issues'].append("BERTClassifier not available - core functionality disabled")
            health_info['overall_status'] = 'critical'

        if not _model_utils_available:
            health_info['issues'].append("Model utilities not available - device management disabled")
            health_info['overall_status'] = 'degraded' if health_info['overall_status'] == 'healthy' else health_info['overall_status']

        # Configuration checks
        try:
            config = get_bert_config()
            if config.get('config_source') == 'fallback':
                health_info['warnings'].append("Using fallback configuration - config system not available")
        except Exception as e:
            health_info['issues'].append(f"Configuration check failed: {str(e)}")

        # Generate recommendations
        if health_info['overall_status'] == 'healthy':
            health_info['recommendations'].append("Package is ready for production use")
        else:
            health_info['recommendations'].append("Address component issues before production use")
            health_info['recommendations'].append("Check import dependencies and installation")

        if not _enhanced_exceptions_available:
            health_info['recommendations'].append("Install enhanced exception system for better error handling")

        if not _config_available:
            health_info['recommendations'].append("Configure config system for dynamic configuration management")

        return health_info

    except Exception as e:
        return {
            'overall_status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'recommendations': ['Contact system administrator for package health issues']
        }


def get_version_info() -> Dict[str, Any]:
    """
    Get detailed version information for debugging and support.

    Returns:
        Dictionary with version details and dependency information
    """
    version_info = {
        'package_version': __version__,
        'package_status': __status__,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'component_versions': {},
        'dependencies': {}
    }

    # Check component versions
    if _classifier_available:
        try:
            version_info['component_versions']['classifier'] = '3.2.0_enhanced'
        except:
            pass

    if _preprocessing_available:
        try:
            version_info['component_versions']['preprocessing'] = '3.2.0_enhanced'
        except:
            pass

    if _model_utils_available:
        try:
            version_info['component_versions']['model_utils'] = '3.2.0_enhanced'
        except:
            pass

    # Check key dependencies
    try:
        import torch
        version_info['dependencies']['torch'] = torch.__version__
    except ImportError:
        version_info['dependencies']['torch'] = 'not_available'

    try:
        import transformers
        version_info['dependencies']['transformers'] = transformers.__version__
    except ImportError:
        version_info['dependencies']['transformers'] = 'not_available'

    return version_info


# Enhanced export management with dynamic exports based on availability
def _build_exports() -> List[str]:
    """Build dynamic exports list based on component availability."""
    exports = ['get_package_info', 'check_package_health', 'get_version_info']

    if _classifier_available:
        exports.extend(['BERTClassifier', 'FakeNewsDataset', 'create_bert_classifier'])

    if _preprocessing_available:
        exports.extend(['TextPreprocessor', 'create_text_preprocessor'])

    if _model_utils_available:
        exports.extend(['DeviceManager', 'ModelManager', 'create_device_manager'])

    if _config_available:
        exports.extend(['get_bert_config', 'get_system_config'])
    else:
        exports.extend(['get_fallback_bert_config', 'get_fallback_system_config'])

    return exports


# Dynamic exports based on component availability
__all__ = _build_exports()

# Package initialization with enhanced error handling
def _initialize_package() -> bool:
    """Initialize package with comprehensive validation and setup."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Initializing BERT Classifier Package v{__version__}")
        
        # Check component status
        available_components = sum(__component_status__.values())
        total_components = len(__component_status__)
        
        logger.info(f"Component availability: {available_components}/{total_components}")
        
        # Log component status
        for component, available in __component_status__.items():
            status = "✅" if available else "❌"
            logger.info(f"{status} {component}: {'Available' if available else 'Not Available'}")

        # Check critical components
        if not _classifier_available:
            logger.error("❌ Critical: BERTClassifier not available")
            return False

        # Log configuration status
        config_status = "✅ Available" if _config_available else "⚠️ Using fallback"
        logger.info(f"Configuration system: {config_status}")

        # Log exception system status
        exception_status = "✅ Available" if _enhanced_exceptions_available else "⚠️ Basic exceptions"
        logger.info(f"Enhanced exceptions: {exception_status}")

        logger.info(f"✅ BERT Classifier Package v{__version__} initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Package initialization failed: {str(e)}")
        return False


# Initialize package on import
_package_initialized = _initialize_package()

if _package_initialized:
    logging.getLogger(__name__).info(f"🎯 BERT Classifier Package v{__version__} ready for production")
else:
    logging.getLogger(__name__).error(f"⚠️ BERT Classifier Package v{__version__} initialization completed with errors")

# Add package status to exports
__all__.extend(['__version__', '__status__', '__component_status__'])
