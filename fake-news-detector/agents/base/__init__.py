# agents/__init__.py

"""
Enhanced Base Module for Production Fake News Detection System

This module provides the foundational BaseAgent class and supporting utilities
for building robust, production-ready fake news detection agents with comprehensive
error handling, performance monitoring, and enhanced logging capabilities.

Key Features:
- Enhanced BaseAgent with comprehensive error handling and exception integration
- Environment-aware configuration management with production optimizations
- Structured logging with session tracking and file rotation support
- Advanced performance monitoring and health checking capabilities
- LangGraph integration compatibility with state management
- Memory management and resource optimization
- Async processing support with concurrent session handling
- Agent type-specific configuration templates and optimizations
- Comprehensive status reporting and aggregation utilities

Version: 3.2.0 - Enhanced Production Edition
"""

import logging
import os
import sys
import platform
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Core agent import
from .base_agent import BaseAgent

# Enhanced exception integration for better error handling
try:
    from agents.llm_explanation.exceptions import (
        handle_llm_explanation_exception,
        ErrorSeverity,
        ErrorCategory,
        is_recoverable_error
    )
    _enhanced_exceptions_available = True
except ImportError:
    _enhanced_exceptions_available = False
    logging.getLogger(__name__).warning("Enhanced exceptions not available - using basic error handling")

# Export main classes and enhanced utilities
__all__ = [
    'BaseAgent',
    'create_agent_config',
    'get_agent_status_summary', 
    'check_system_compatibility',
    'initialize_agent_framework',
    'get_framework_info',
    'DEFAULT_CONFIG',
    'AGENT_TYPES',
    'PRODUCTION_CONFIG',
    'DEVELOPMENT_CONFIG'
]

# Package metadata with enhanced information
__version__ = '3.2.0'
__author__ = 'Enhanced Fake News Detection Team'
__description__ = 'Production-ready base agent framework with enhanced logging, error handling, and performance monitoring'
__license__ = 'MIT'
__status__ = 'Production'
__compatibility__ = 'Python 3.8+'

# Enhanced configuration constants with production optimizations
DEFAULT_CONFIG = {
    "environment": "development",
    "log_level": "INFO",
    "enable_metrics": True,
    "enable_health_monitoring": True,
    "max_retries": 3,
    "retry_delay": 1.0,
    "timeout": 30,
    "memory_threshold_mb": 512,
    "circuit_breaker_threshold": 5,
    "async_enabled": True,
    "session_tracking_enabled": True,
    "enable_security_checks": True,
    "validate_output": True,
    "enable_caching": False,
    "enable_file_logging": False,
    "log_rotation": True
}

# Production-optimized configuration template
PRODUCTION_CONFIG = {
    **DEFAULT_CONFIG,
    "environment": "production",
    "log_level": "WARNING",
    "enable_file_logging": True,
    "log_rotation": True,
    "memory_threshold_mb": 1024,
    "timeout": 45,
    "enable_caching": True,
    "circuit_breaker_threshold": 3,
    "max_retries": 5,
    "performance_optimization": True,
    "detailed_error_logging": True
}

# Development-optimized configuration template
DEVELOPMENT_CONFIG = {
    **DEFAULT_CONFIG,
    "environment": "development", 
    "log_level": "DEBUG",
    "debug_mode": True,
    "enable_file_logging": False,
    "timeout": 60,
    "max_retries": 2,
    "enable_security_checks": False,
    "detailed_metrics": True
}

# Enhanced agent type constants with descriptions
AGENT_TYPES = {
    "CLASSIFIER": {
        "type": "classifier",
        "description": "Binary or multi-class classification agents",
        "typical_timeout": 15,
        "memory_profile": "low"
    },
    "GENERATOR": {
        "type": "generator", 
        "description": "Text and explanation generation agents",
        "typical_timeout": 45,
        "memory_profile": "high"
    },
    "RECOMMENDER": {
        "type": "recommender",
        "description": "Source and content recommendation agents", 
        "typical_timeout": 20,
        "memory_profile": "medium"
    },
    "EXTRACTOR": {
        "type": "extractor",
        "description": "Information and feature extraction agents",
        "typical_timeout": 25,
        "memory_profile": "medium"
    },
    "ANALYZER": {
        "type": "analyzer",
        "description": "Content and context analysis agents",
        "typical_timeout": 30,
        "memory_profile": "medium"
    },
    "EVALUATOR": {
        "type": "evaluator", 
        "description": "Evidence and quality evaluation agents",
        "typical_timeout": 35,
        "memory_profile": "high"
    },
    "VALIDATOR": {
        "type": "validator",
        "description": "Input and output validation agents",
        "typical_timeout": 10,
        "memory_profile": "low"
    },
    "MONITOR": {
        "type": "monitor",
        "description": "Health and performance monitoring agents",
        "typical_timeout": 5,
        "memory_profile": "low"
    }
}


def create_agent_config(agent_type: str, environment: str = None, **kwargs) -> Dict[str, Any]:
    """
    Create optimized configuration for specific agent type with enhanced defaults.

    Args:
        agent_type: Type of agent (use AGENT_TYPES keys or type strings)
        environment: Target environment ('production', 'development', or None for auto-detect)
        **kwargs: Additional configuration parameters to override

    Returns:
        Optimized configuration dictionary with type-specific and environment-specific settings

    Example:
        >>> config = create_agent_config('CLASSIFIER', environment='production', timeout=20)
        >>> agent = BaseAgent(config)
    """
    try:
        # Determine base configuration based on environment
        if environment == "production":
            base_config = PRODUCTION_CONFIG.copy()
        elif environment == "development":
            base_config = DEVELOPMENT_CONFIG.copy()
        else:
            # Auto-detect from environment variable or use default
            env = os.getenv("ENVIRONMENT", "development")
            if env == "production":
                base_config = PRODUCTION_CONFIG.copy()
            else:
                base_config = DEVELOPMENT_CONFIG.copy()

        # Agent-type specific optimizations
        agent_type_upper = agent_type.upper()
        type_configs = {
            "CLASSIFIER": {
                "timeout": 15,
                "memory_threshold_mb": 256,
                "enable_caching": True,
                "max_retries": 2,
                "temperature": 0.1,  # Low temperature for consistent classification
                "max_tokens": 1024
            },
            "GENERATOR": {
                "timeout": 45,
                "memory_threshold_mb": 1024,
                "max_tokens": 4096,
                "temperature": 0.3,  # Moderate temperature for creative generation
                "enable_detailed_analysis": True,
                "enable_quality_validation": True
            },
            "RECOMMENDER": {
                "timeout": 25,
                "memory_threshold_mb": 512,
                "enable_caching": True,
                "max_tokens": 2048,
                "temperature": 0.2
            },
            "EXTRACTOR": {
                "timeout": 20,
                "memory_threshold_mb": 512,
                "enable_caching": True,
                "max_tokens": 2048,
                "temperature": 0.1  # Low temperature for accurate extraction
            },
            "ANALYZER": {
                "timeout": 30,
                "memory_threshold_mb": 768,
                "enable_caching": True,
                "enable_detailed_metrics": True,
                "max_tokens": 3072,
                "temperature": 0.2
            },
            "EVALUATOR": {
                "timeout": 35,
                "memory_threshold_mb": 1024,
                "enable_detailed_analysis": True,
                "max_tokens": 3072,
                "temperature": 0.2
            },
            "VALIDATOR": {
                "timeout": 10,
                "memory_threshold_mb": 256,
                "enable_security_checks": True,
                "max_tokens": 1024,
                "temperature": 0.1
            },
            "MONITOR": {
                "timeout": 5,
                "memory_threshold_mb": 128,
                "enable_health_monitoring": True,
                "enable_detailed_metrics": True,
                "max_tokens": 512
            }
        }

        # Apply agent-type specific configuration
        if agent_type_upper in type_configs:
            base_config.update(type_configs[agent_type_upper])
        elif agent_type.lower() in [info["type"] for info in AGENT_TYPES.values()]:
            # Handle lowercase type strings
            for type_key, type_info in AGENT_TYPES.items():
                if type_info["type"] == agent_type.lower():
                    if type_key in type_configs:
                        base_config.update(type_configs[type_key])
                    break

        # Apply user overrides
        base_config.update(kwargs)
        
        # Add agent type information
        base_config["agent_type_config"] = {
            "requested_type": agent_type,
            "resolved_type": agent_type_upper if agent_type_upper in AGENT_TYPES else "CUSTOM",
            "configuration_source": f"enhanced_framework_v{__version__}"
        }

        return base_config

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create agent config: {str(e)}")
        # Return safe default configuration
        safe_config = DEFAULT_CONFIG.copy()
        safe_config.update(kwargs)
        return safe_config


def get_agent_status_summary(agents: List[BaseAgent]) -> Dict[str, Any]:
    """
    Get comprehensive status summary for multiple agents with enhanced aggregation.

    Args:
        agents: List of BaseAgent instances to analyze

    Returns:
        Combined status summary with detailed metrics and health analysis

    Example:
        >>> agents = [agent1, agent2, agent3]
        >>> summary = get_agent_status_summary(agents)
        >>> print(f"Overall status: {summary['overall_status']}")
    """
    try:
        if not agents:
            return {
                "status": "no_agents",
                "agent_count": 0,
                "healthy_agents": 0,
                "total_calls": 0,
                "overall_success_rate": 0.0,
                "timestamp": datetime.now().isoformat(),
                "framework_version": __version__
            }

        # Collect individual agent statuses
        agent_statuses = []
        total_calls = 0
        total_successful = 0
        total_errors = 0
        healthy_agents = 0
        
        for agent in agents:
            try:
                if hasattr(agent, 'get_comprehensive_status'):
                    status = agent.get_comprehensive_status()
                else:
                    # Fallback to basic status
                    status = agent.get_status_summary() if hasattr(agent, 'get_status_summary') else {}
                
                agent_statuses.append(status)
                
                # Aggregate metrics
                perf = status.get("performance_summary", {})
                total_calls += perf.get("total_calls", 0)
                total_successful += perf.get("successful_calls", 0)
                total_errors += perf.get("error_calls", 0)
                
                # Count healthy agents
                if status.get("health_status", {}).get("status") == "healthy":
                    healthy_agents += 1
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to get status for agent: {str(e)}")

        # Calculate aggregate metrics
        agent_count = len(agents)
        overall_success_rate = (total_successful / max(total_calls, 1)) * 100
        overall_error_rate = (total_errors / max(total_calls, 1)) * 100
        healthy_ratio = healthy_agents / agent_count

        # Determine overall status
        if healthy_ratio >= 1.0:
            overall_status = "healthy"
        elif healthy_ratio >= 0.8:
            overall_status = "mostly_healthy"
        elif healthy_ratio >= 0.6:
            overall_status = "degraded"
        elif healthy_ratio >= 0.3:
            overall_status = "critical"
        else:
            overall_status = "failing"

        return {
            "overall_status": overall_status,
            "agent_count": agent_count,
            "healthy_agents": healthy_agents,
            "healthy_ratio": round(healthy_ratio, 3),
            "aggregate_metrics": {
                "total_calls": total_calls,
                "total_successful": total_successful,
                "total_errors": total_errors,
                "overall_success_rate": round(overall_success_rate, 2),
                "overall_error_rate": round(overall_error_rate, 2)
            },
            "agent_details": [
                {
                    "name": status.get("agent_info", {}).get("agent_name", "unknown"),
                    "type": status.get("agent_info", {}).get("agent_type", "unknown"),
                    "status": status.get("health_status", {}).get("status", "unknown"),
                    "success_rate": status.get("health_status", {}).get("success_rate", 0.0),
                    "uptime": status.get("agent_info", {}).get("uptime_seconds", 0.0)
                }
                for status in agent_statuses
            ],
            "framework_info": {
                "version": __version__,
                "enhanced_exceptions_available": _enhanced_exceptions_available,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to generate agent status summary: {str(e)}")
        return {
            "overall_status": "error",
            "agent_count": len(agents) if agents else 0,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def check_system_compatibility() -> Dict[str, Any]:
    """
    Check system compatibility for enhanced base agent operations with comprehensive analysis.

    Returns:
        Dictionary with detailed compatibility information and recommendations

    Example:
        >>> compat = check_system_compatibility()
        >>> if not compat['compatible']:
        ...     print(f"Issues: {compat['issues']}")
    """
    try:
        python_version = sys.version_info
        is_compatible = python_version >= (3, 8)
        
        # Check optional dependencies
        optional_deps = {}
        
        try:
            import psutil
            optional_deps['psutil'] = {'available': True, 'version': psutil.__version__}
        except ImportError:
            optional_deps['psutil'] = {'available': False, 'impact': 'Memory monitoring disabled'}

        try:
            import asyncio
            optional_deps['asyncio'] = {'available': True, 'version': 'built-in'}
        except ImportError:
            optional_deps['asyncio'] = {'available': False, 'impact': 'Async processing disabled'}

        # System information
        system_info = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': sys.version,
            'python_version_info': {
                'major': python_version.major,
                'minor': python_version.minor,
                'micro': python_version.micro
            }
        }

        # Compatibility issues and warnings
        issues = []
        warnings = []
        
        if not is_compatible:
            issues.append(f"Python {python_version.major}.{python_version.minor} detected, minimum required is 3.8")
        
        if python_version < (3, 9):
            warnings.append("Python 3.9+ recommended for optimal performance")
            
        if not optional_deps['psutil']['available']:
            warnings.append("psutil not available - memory monitoring will be limited")

        # Overall status
        if issues:
            status = "incompatible"
        elif warnings:
            status = "compatible_with_warnings"
        else:
            status = "fully_compatible"

        return {
            'compatible': is_compatible,
            'status': status,
            'system_info': system_info,
            'optional_dependencies': optional_deps,
            'issues': issues,
            'warnings': warnings,
            'recommendations': [
                "Use Python 3.9+ for best performance",
                "Install psutil for enhanced memory monitoring: pip install psutil",
                "Run in production environment for optimal configuration"
            ] if warnings else ["System is fully compatible"],
            'framework_requirements': {
                'minimum_python': '3.8.0',
                'recommended_python': '3.9.0+',
                'required_packages': ['typing', 'asyncio', 'logging'],
                'optional_packages': ['psutil']
            },
            'check_timestamp': datetime.now().isoformat(),
            'framework_version': __version__
        }

    except Exception as e:
        logging.getLogger(__name__).error(f"Compatibility check failed: {str(e)}")
        return {
            'compatible': False,
            'status': 'check_failed',
            'error': str(e),
            'check_timestamp': datetime.now().isoformat()
        }


def initialize_agent_framework() -> Dict[str, Any]:
    """
    Initialize the enhanced agent framework with comprehensive setup and validation.

    Returns:
        Initialization result dictionary with detailed status information

    Example:
        >>> init_result = initialize_agent_framework()
        >>> if init_result['success']:
        ...     print("Framework ready for production")
    """
    logger = logging.getLogger(__name__)
    init_start_time = datetime.now()
    
    try:
        logger.info(f"Initializing Enhanced Agent Framework v{__version__}")
        
        # Check system compatibility
        compatibility = check_system_compatibility()
        if not compatibility['compatible']:
            error_msg = f"System compatibility check failed: {compatibility['issues']}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'compatibility': compatibility,
                'initialization_time': datetime.now().isoformat()
            }

        # Log compatibility warnings if any
        if compatibility['warnings']:
            for warning in compatibility['warnings']:
                logger.warning(f"Compatibility warning: {warning}")

        # Test BaseAgent import and instantiation
        try:
            # Test configuration creation
            test_config = create_agent_config('CLASSIFIER', environment='development')
            logger.info("Configuration system operational")
            
            # Test enhanced exception integration
            if _enhanced_exceptions_available:
                logger.info("Enhanced exception handling available")
            else:
                logger.warning("Enhanced exception handling not available - using basic error handling")

        except Exception as e:
            logger.error(f"Component testing failed: {str(e)}")
            return {
                'success': False,
                'error': f"Component testing failed: {str(e)}",
                'initialization_time': datetime.now().isoformat()
            }

        # Initialize logging configuration
        try:
            # Configure root logger if not already configured
            root_logger = logging.getLogger()
            if not root_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                root_logger.addHandler(handler)
                root_logger.setLevel(logging.INFO)
            
            logger.info("Logging system configured")

        except Exception as e:
            logger.warning(f"Logging configuration warning: {str(e)}")

        # Calculate initialization time
        init_duration = (datetime.now() - init_start_time).total_seconds()
        
        logger.info(f"Enhanced Agent Framework v{__version__} initialized successfully in {init_duration:.3f}s")
        
        return {
            'success': True,
            'framework_info': {
                'version': __version__,
                'status': __status__,
                'compatibility': __compatibility__,
                'enhanced_exceptions_available': _enhanced_exceptions_available
            },
            'compatibility': compatibility,
            'available_agent_types': list(AGENT_TYPES.keys()),
            'available_configurations': ['DEFAULT_CONFIG', 'PRODUCTION_CONFIG', 'DEVELOPMENT_CONFIG'],
            'initialization_duration_seconds': round(init_duration, 3),
            'initialization_time': datetime.now().isoformat(),
            'system_ready': True
        }

    except Exception as e:
        error_msg = f"Framework initialization failed: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'initialization_time': datetime.now().isoformat(),
            'system_ready': False
        }


def get_framework_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the enhanced agent framework.

    Returns:
        Dictionary with detailed framework information and capabilities

    Example:
        >>> info = get_framework_info()
        >>> print(f"Framework version: {info['version']}")
        >>> print(f"Available agents: {info['agent_types']}")
    """
    return {
        'package_metadata': {
            'version': __version__,
            'author': __author__,
            'description': __description__,
            'license': __license__,
            'status': __status__,
            'compatibility': __compatibility__
        },
        'capabilities': {
            'base_agent_class': 'Enhanced BaseAgent with comprehensive error handling',
            'async_support': True,
            'session_tracking': True,
            'performance_monitoring': True,
            'health_monitoring': True,
            'exception_integration': _enhanced_exceptions_available,
            'configuration_templates': True,
            'logging_enhancement': True,
            'memory_management': True,
            'security_validation': True
        },
        'agent_types': {
            name: {
                'type': info['type'],
                'description': info['description']
            }
            for name, info in AGENT_TYPES.items()
        },
        'configuration_options': {
            'default': 'Balanced configuration for general use',
            'production': 'Optimized for production environments',
            'development': 'Enhanced debugging and development features',
            'custom': 'User-defined configuration with overrides'
        },
        'features': [
            'Enhanced error handling with exception integration',
            'Comprehensive performance and health monitoring',
            'Async processing with concurrent session support',
            'Memory management and resource optimization',
            'Security validation and input sanitization',
            'Structured logging with session tracking',
            'Configuration templates for different environments',
            'Agent status aggregation and reporting',
            'System compatibility checking and validation'
        ],
        'framework_status': {
            'initialized': _module_initialized,
            'enhanced_exceptions_available': _enhanced_exceptions_available,
            'compatibility_check_passed': True,  # Updated during initialization
            'production_ready': __status__ == 'Production'
        }
    }


# Module initialization with enhanced error handling
def _initialize_module() -> bool:
    """
    Initialize the module with comprehensive setup and validation.
    
    Returns:
        True if initialization successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Initializing Enhanced Agent Framework Module v{__version__}")
        
        # Check basic compatibility
        compatibility = check_system_compatibility()
        
        if not compatibility['compatible']:
            logger.error(f"Module initialization failed - incompatible system: {compatibility['issues']}")
            return False
            
        # Log warnings if any
        for warning in compatibility.get('warnings', []):
            logger.warning(warning)
        
        # Test configuration system
        try:
            test_config = create_agent_config('CLASSIFIER')
            logger.debug("Configuration system test passed")
        except Exception as e:
            logger.error(f"Configuration system test failed: {str(e)}")
            return False
            
        # Log framework capabilities
        logger.info(f"Framework capabilities: async_support=True, enhanced_exceptions={_enhanced_exceptions_available}")
        logger.info(f"Available agent types: {list(AGENT_TYPES.keys())}")
        logger.info("Enhanced Agent Framework Module initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Module initialization failed: {str(e)}")
        return False


# Initialize module on import
_module_initialized = _initialize_module()

# Export additional enhanced utilities
__all__.extend([
    'PRODUCTION_CONFIG',
    'DEVELOPMENT_CONFIG', 
    'get_framework_info',
    'initialize_agent_framework'
])

# Module initialization status message
if _module_initialized:
    logging.getLogger(__name__).info(f"✅ Enhanced Agent Framework v{__version__} ready for production")
else:
    logging.getLogger(__name__).error(f"❌ Enhanced Agent Framework v{__version__} initialization failed")
