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
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from pathlib import Path

# Core component imports with enhanced error handling
logger = logging.getLogger(__name__)

# Component availability tracking
_component_status = {
    "classifier_available": False,
    "preprocessing_available": False, 
    "model_utils_available": False,
    "config_integration_available": False,
    "enhanced_exceptions_available": False
}

# Import core components with graceful error handling
try:
    from agents.bert_classifier.classifier import BERTClassifier, FakeNewsDataset
    _component_status["classifier_available"] = True
    logger.debug("✅ Classifier components imported successfully")
except ImportError as e:
    _component_status["classifier_available"] = False
    logger.error(f"❌ Failed to import classifier components: {e}")

try:
    from agents.bert_classifier.preprocessing import TextPreprocessor
    _component_status["preprocessing_available"] = True
    logger.debug("✅ Preprocessing components imported successfully") 
except ImportError as e:
    _component_status["preprocessing_available"] = False
    logger.error(f"❌ Failed to import preprocessing components: {e}")

try:
    from agents.bert_classifier.model_utils import DeviceManager, ModelManager
    _component_status["model_utils_available"] = True
    logger.debug("✅ Model utilities imported successfully")
except ImportError as e:
    _component_status["model_utils_available"] = False
    logger.error(f"❌ Failed to import model utilities: {e}")

# Enhanced exception integration
try:
    from agents.llm_explanation.exceptions import (
        handle_llm_explanation_exception,
        ErrorContext,
        log_exception_with_context
    )
    _component_status["enhanced_exceptions_available"] = True
    logger.debug("✅ Enhanced exception system available")
except ImportError:
    _component_status["enhanced_exceptions_available"] = False
    logger.debug("ℹ️ Enhanced exception system not available - using basic error handling")

# Configuration system integration with fallback
try:
    from config import get_model_config, get_settings
    _component_status["config_integration_available"] = True
    logger.debug("✅ Configuration system integration available")
    
    def get_bert_config(fallback: bool = True) -> Dict[str, Any]:
        """
        Get BERT classifier configuration with enhanced error handling.
        
        Args:
            fallback: Whether to use fallback configuration if main config unavailable
            
        Returns:
            Configuration dictionary for BERT classifier
        """
        try:
            config = get_model_config('bert_classifier')
            if config:
                return config
        except Exception as e:
            logger.warning(f"Failed to load BERT config: {e}")
            
        if fallback:
            return get_fallback_bert_config()
        else:
            raise RuntimeError("BERT configuration not available and fallback disabled")

    def get_system_config(fallback: bool = True) -> Any:
        """
        Get system configuration with enhanced error handling.
        
        Args:
            fallback: Whether to use fallback if main config unavailable
            
        Returns:
            System configuration object or dictionary
        """
        try:
            return get_settings()
        except Exception as e:
            logger.warning(f"Failed to load system config: {e}")
            
        if fallback:
            return get_fallback_system_config()
        else:
            raise RuntimeError("System configuration not available and fallback disabled")
            
except ImportError as e:
    _component_status["config_integration_available"] = False
    logger.warning(f"Configuration system not available: {e}")
    
    def get_bert_config(fallback: bool = True) -> Dict[str, Any]:
        """Fallback BERT configuration function."""
        return get_fallback_bert_config()
    
    def get_system_config(fallback: bool = True) -> Dict[str, Any]:
        """Fallback system configuration function.""" 
        return get_fallback_system_config()

# Fallback configuration functions
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
        'model_path': 'models/bert_fake_news_classifier',
        'preprocessing_config': {
            'max_length': 2000,
            'remove_urls': True,
            'remove_emails': True,
            'enable_security_checks': True,
            'normalize_quotes': True,
            'remove_excessive_punctuation': True,
            'handle_special_characters': True
        },
        'prediction_threshold': 0.5,
        'high_confidence_threshold': 0.8,
        'timeout_seconds': 30,
        'max_retries': 3,
        'config_source': 'fallback'
    }

def get_fallback_system_config() -> Dict[str, Any]:
    """
    Get fallback system configuration.
    
    Returns:
        Fallback system configuration dictionary
    """
    return {
        'models_dir': './models',
        'logs_dir': './logs', 
        'cache_dir': './cache',
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'debug_mode': os.getenv('DEBUG', 'false').lower() == 'true',
        'enable_logging': True,
        'log_level': 'INFO',
        'max_workers': int(os.getenv('MAX_WORKERS', '4')),
        'memory_limit_mb': int(os.getenv('MEMORY_LIMIT_MB', '2048')),
        'config_source': 'fallback'
    }

# Factory functions for easy component creation
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
    """
    if not _component_status["classifier_available"]:
        raise ImportError("BERTClassifier not available - check component imports")
    
    try:
        # Get base configuration
        base_config = get_bert_config()
        
        # Merge with user overrides
        if config:
            base_config.update(config)
        
        # Create classifier instance
        classifier = BERTClassifier(base_config)
        
        logger.info(
            f"BERT classifier created successfully",
            extra={'session_id': session_id, 'config_source': base_config.get('config_source')}
        )
        
        return classifier
        
    except Exception as e:
        error_msg = f"Failed to create BERT classifier: {str(e)}"
        logger.error(error_msg, extra={'session_id': session_id})
        
        if _component_status["enhanced_exceptions_available"]:
            context = ErrorContext(
                session_id=session_id,
                operation="classifier_creation",
                component="bert_classifier_package"
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
    """
    if not _component_status["preprocessing_available"]:
        raise ImportError("TextPreprocessor not available - check component imports")
    
    try:
        # Get preprocessing config from BERT config
        bert_config = get_bert_config()
        base_config = bert_config.get('preprocessing_config', {})
        
        # Merge with user overrides
        if config:
            base_config.update(config)
        
        preprocessor = TextPreprocessor(base_config)
        
        logger.info(
            f"Text preprocessor created successfully", 
            extra={'session_id': session_id}
        )
        
        return preprocessor
        
    except Exception as e:
        error_msg = f"Failed to create text preprocessor: {str(e)}"
        logger.error(error_msg, extra={'session_id': session_id})
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
    """
    if not _component_status["model_utils_available"]:
        raise ImportError("DeviceManager not available - check component imports")
    
    try:
        device_manager = DeviceManager(device_preference, config)
        
        logger.info(
            f"Device manager created: {device_manager.get_device()}",
            extra={'session_id': session_id}
        )
        
        return device_manager
        
    except Exception as e:
        error_msg = f"Failed to create device manager: {str(e)}"
        logger.error(error_msg, extra={'session_id': session_id})
        raise RuntimeError(error_msg)

def create_model_manager(device_manager: 'DeviceManager',
                        config: Optional[Dict[str, Any]] = None,
                        session_id: Optional[str] = None) -> 'ModelManager':
    """
    Create model manager instance with enhanced configuration.
    
    Args:
        device_manager: DeviceManager instance
        config: Optional configuration dictionary
        session_id: Optional session ID for tracking
        
    Returns:
        Configured ModelManager instance
        
    Raises:
        ImportError: If model utilities are not available
    """
    if not _component_status["model_utils_available"]:
        raise ImportError("ModelManager not available - check component imports")
    
    try:
        model_manager = ModelManager(device_manager, config)
        
        logger.info(
            f"Model manager created successfully",
            extra={'session_id': session_id}
        )
        
        return model_manager
        
    except Exception as e:
        error_msg = f"Failed to create model manager: {str(e)}"
        logger.error(error_msg, extra={'session_id': session_id})
        raise RuntimeError(error_msg)

# Package information and health functions
def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information and component status.
    
    Returns:
        Dictionary with package metadata and component availability
    """
    return {
        'package_metadata': {
            'name': 'bert_classifier',
            'version': __version__,
            'author': __author__,
            'description': __description__,
            'license': 'MIT',
            'status': 'Production',
            'compatibility': 'Python 3.8+'
        },
        'component_status': _component_status,
        'components_available': sum(_component_status.values()),
        'total_components': len(_component_status),
        'all_components_available': all(_component_status.values()),
        'capabilities': {
            'async_processing': _component_status["classifier_available"],
            'text_preprocessing': _component_status["preprocessing_available"],
            'device_management': _component_status["model_utils_available"],
            'enhanced_error_handling': _component_status["enhanced_exceptions_available"],
            'configuration_system': _component_status["config_integration_available"]
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
        missing_components = [
            comp for comp, available in _component_status.items() 
            if not available
        ]
        
        if missing_components:
            health_info['warnings'].extend([
                f"Component not available: {comp}" 
                for comp in missing_components
            ])
            if len(missing_components) > 2:
                health_info['overall_status'] = 'degraded'
        
        # Check critical components
        if not _component_status["classifier_available"]:
            health_info['issues'].append("BERTClassifier not available - core functionality disabled")
            health_info['overall_status'] = 'critical'
            
        if not _component_status["model_utils_available"]:
            health_info['issues'].append("Model utilities not available - device management disabled")
            if health_info['overall_status'] == 'healthy':
                health_info['overall_status'] = 'degraded'
        
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
        
        if not _component_status["enhanced_exceptions_available"]:
            health_info['recommendations'].append("Install enhanced exception system for better error handling")
            
        if not _component_status["config_integration_available"]:
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
        'package_status': 'Production',
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'component_versions': {},
        'dependencies': {}
    }
    
    # Check component versions
    if _component_status["classifier_available"]:
        version_info['component_versions']['classifier'] = '3.2.0_enhanced'
        
    if _component_status["preprocessing_available"]:
        version_info['component_versions']['preprocessing'] = '3.2.0_enhanced'
        
    if _component_status["model_utils_available"]:
        version_info['component_versions']['model_utils'] = '3.2.0_enhanced'
    
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

# Build dynamic exports based on component availability
def _build_exports() -> List[str]:
    """Build dynamic exports list based on component availability."""
    exports = [
        'get_package_info', 'check_package_health', 'get_version_info',
        'get_bert_config', 'get_system_config'
    ]
    
    if _component_status["classifier_available"]:
        exports.extend(['BERTClassifier', 'FakeNewsDataset', 'create_bert_classifier'])
        
    if _component_status["preprocessing_available"]:
        exports.extend(['TextPreprocessor', 'create_text_preprocessor'])
        
    if _component_status["model_utils_available"]:
        exports.extend(['DeviceManager', 'ModelManager', 'create_device_manager', 'create_model_manager'])
    
    return exports

# Package metadata
__version__ = "3.2.0"
__author__ = "Enhanced Fake News Detection Team"
__description__ = "Production-ready BERT classifier with comprehensive error handling, performance monitoring, and enhanced integrations"
__license__ = "MIT"
__status__ = "Production"

# Dynamic exports based on component availability
__all__ = _build_exports()

# Package initialization
def _initialize_package() -> bool:
    """Initialize package with comprehensive validation and setup."""
    try:
        logger.info(f"Initializing BERT Classifier Package v{__version__}")
        
        # Check component status
        available_components = sum(_component_status.values())
        total_components = len(_component_status)
        
        logger.info(f"Component availability: {available_components}/{total_components}")
        
        # Log component status
        for component, available in _component_status.items():
            status = "✅" if available else "❌"
            logger.info(f"{status} {component}: {'Available' if available else 'Not Available'}")
        
        # Check critical components
        if not _component_status["classifier_available"]:
            logger.error("❌ Critical: BERTClassifier not available")
            return False
            
        # Log configuration status  
        config_status = "✅ Available" if _component_status["config_integration_available"] else "⚠️ Using fallback"
        logger.info(f"Configuration system: {config_status}")
        
        # Log exception system status
        exception_status = "✅ Available" if _component_status["enhanced_exceptions_available"] else "ℹ️ Basic exceptions"
        logger.info(f"Enhanced exceptions: {exception_status}")
        
        logger.info(f"✅ BERT Classifier Package v{__version__} initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Package initialization failed: {str(e)}")
        return False

# Initialize package on import
_package_initialized = _initialize_package()

# Export package status
__all__.extend(['__version__', '__status__', '_component_status', '_package_initialized'])

# Final status message
if _package_initialized:
    logger.info(f"🎯 BERT Classifier Package v{__version__} ready for production")
else:
    logger.error(f"⚠️ BERT Classifier Package v{__version__} initialization completed with errors")
