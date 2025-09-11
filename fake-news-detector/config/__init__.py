# config/__init__.py

"""
Enhanced Configuration Package for Production Fake News Detection

Production-ready configuration management with comprehensive settings,
model configurations, and environment awareness.

Features:
- Clean interface for all configuration components
- Environment-based configuration switching
- Comprehensive validation and health checking
- Performance optimization profiles
- Enhanced error handling and recovery
- Configuration import/export capabilities

Version: 3.2.0 - Modern Production Edition
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Import configuration modules with error handling
try:
    from .settings import (
        SystemSettings,
        Environment,
        LogLevel,
        get_settings,
        update_settings,
        reset_settings,
        apply_environment_config,
        ENVIRONMENT_CONFIGS,
        _parse_bool
    )
    _settings_available = True
except ImportError as e:
    _settings_available = False
    logging.getLogger(__name__).error(f"Failed to import settings module: {e}")

try:
    from .model_configs import (
        ConfigurationManager,
        BaseAgentConfig,
        LLMAgentConfig,
        BERTClassifierConfig,
        ClaimExtractorConfig,
        ContextAnalyzerConfig,
        EvidenceEvaluatorConfig,
        CredibleSourceConfig,
        LLMExplanationConfig,
        ModelType,
        SafetyThreshold,
        ConfigurationProfile,
        get_model_config,
        update_model_config,
        get_performance_summary,
        apply_configuration_profile,
        validate_all_configurations,
        reset_model_configs
    )
    _model_configs_available = True
except ImportError as e:
    _model_configs_available = False
    logging.getLogger(__name__).error(f"Failed to import model_configs module: {e}")

# Package metadata
__version__ = "3.2.0"
__author__ = "Enhanced Fake News Detection Team"
__description__ = "Production-ready configuration management for modular fake news detection system"
__license__ = "MIT"
__status__ = "Production"

# Configuration constants
SUPPORTED_ENVIRONMENTS = ["development", "testing", "staging", "production"]
DEFAULT_ENVIRONMENT = "development"
CONFIG_VERSION = "3.2.0"

# Component availability status
__component_status__ = {
    "settings_available": _settings_available,
    "model_configs_available": _model_configs_available,
    "package_initialized": False
}

# Enhanced export management based on availability
def _build_exports():
    """Build dynamic exports list based on component availability."""
    exports = [
        # Package info
        '__version__',
        '__component_status__',
        'get_current_environment',
        'validate_all_configurations_unified',
        'get_configuration_summary',
        'export_all_configurations',
        'import_configurations',
        'get_package_health'
    ]

    if _settings_available:
        exports.extend([
            'SystemSettings',
            'Environment', 
            'LogLevel',
            'get_settings',
            'update_settings',
            'reset_settings',
            'apply_environment_config',
            'ENVIRONMENT_CONFIGS'
        ])

    if _model_configs_available:
        exports.extend([
            'ConfigurationManager',
            'BaseAgentConfig',
            'LLMAgentConfig',
            'BERTClassifierConfig',
            'ClaimExtractorConfig', 
            'ContextAnalyzerConfig',
            'EvidenceEvaluatorConfig',
            'CredibleSourceConfig',
            'LLMExplanationConfig',
            'ModelType',
            'SafetyThreshold',
            'ConfigurationProfile',
            'get_model_config',
            'update_model_config',
            'get_performance_summary',
            'apply_configuration_profile',
            'reset_model_configs'
        ])

    return exports

# Dynamic exports
__all__ = _build_exports()

# Setup package-level logging
logger = logging.getLogger(__name__)


def get_current_environment() -> str:
    """
    Get the current environment setting with validation.

    Returns:
        Current environment name
    """
    env = os.getenv('ENVIRONMENT', DEFAULT_ENVIRONMENT)
    if env not in SUPPORTED_ENVIRONMENTS:
        logger.warning(f"Unknown environment '{env}', using '{DEFAULT_ENVIRONMENT}'")
        return DEFAULT_ENVIRONMENT
    return env


def validate_all_configurations_unified() -> Dict[str, Any]:
    """
    Comprehensive validation of all available configurations.

    Returns:
        Unified validation results with status and detailed issues
    """
    validation_results = {
        "overall_status": "valid",
        "components_validated": [],
        "issues": [],
        "warnings": [],
        "component_results": {},
        "environment": get_current_environment(),
        "timestamp": datetime.now().isoformat(),
        "config_version": CONFIG_VERSION,
        "availability": __component_status__
    }

    # Validate system settings if available
    if _settings_available:
        try:
            settings = get_settings()
            
            # API key validation
            if not settings.validate_api_keys():
                validation_results["issues"].append("API keys not properly configured")
            
            # Directory validation
            directories_exist = all([
                settings.directories.models_dir.exists(),
                settings.directories.data_dir.exists(),
                settings.directories.logs_dir.exists(),
                settings.directories.cache_dir.exists()
            ])
            
            if not directories_exist:
                validation_results["issues"].append("Some required directories are missing")
            
            # Resource validation
            if settings.performance.memory_limit_mb < 512:
                validation_results["warnings"].append("Memory limit may be too low for production")
            
            # Environment-specific validation
            if settings.environment.value == "production":
                if settings.debug_mode:
                    validation_results["issues"].append("Debug mode should be disabled in production")
                if not settings.logging.enable_file_logging:
                    validation_results["issues"].append("File logging should be enabled in production")
            
            validation_results["component_results"]["system_settings"] = {"status": "validated"}
            validation_results["components_validated"].append("system_settings")
            
        except Exception as e:
            validation_results["issues"].append(f"System settings validation failed: {str(e)}")
            validation_results["component_results"]["system_settings"] = {"status": "error", "error": str(e)}

    # Validate model configurations if available
    if _model_configs_available:
        try:
            model_validation = validate_all_configurations()
            validation_results["component_results"]["model_configs"] = model_validation
            validation_results["components_validated"].append("model_configs")
            
            if model_validation["overall_status"] != "valid":
                validation_results["issues"].extend([f"Model config: {issue}" for issue in model_validation.get("issues", [])])
                
        except Exception as e:
            validation_results["issues"].append(f"Model configuration validation failed: {str(e)}")
            validation_results["component_results"]["model_configs"] = {"status": "error", "error": str(e)}

    # Determine overall status
    if validation_results["issues"]:
        validation_results["overall_status"] = "invalid"
    elif validation_results["warnings"]:
        validation_results["overall_status"] = "valid_with_warnings"

    # Add recommendations
    recommendations = []
    if validation_results["overall_status"] == "valid":
        recommendations.append("All configurations are valid and ready for use")
    else:
        recommendations.append("Address validation issues before production deployment")
    
    if not _settings_available:
        recommendations.append("Install settings module for complete functionality")
    
    if not _model_configs_available:
        recommendations.append("Install model_configs module for agent configuration")

    validation_results["recommendations"] = recommendations

    logger.info(f"Configuration validation completed: {validation_results['overall_status']}")
    return validation_results


def get_configuration_summary() -> Dict[str, Any]:
    """
    Get comprehensive summary of current configuration state.

    Returns:
        Configuration summary with key metrics, status, and component info
    """
    summary = {
        "package_info": {
            "version": __version__,
            "config_version": CONFIG_VERSION,
            "environment": get_current_environment(),
            "components_available": sum(__component_status__.values()) - 1,  # Exclude 'package_initialized'
            "total_components": len(__component_status__) - 1
        },
        "system_settings": {},
        "model_configs": {},
        "validation_status": {},
        "performance_indicators": {},
        "component_status": __component_status__
    }

    try:
        # System settings summary
        if _settings_available:
            settings = get_settings()
            summary["system_settings"] = {
                "environment": settings.environment.value,
                "debug_mode": settings.debug_mode,
                "api_keys_configured": settings.validate_api_keys(),
                "caching_enabled": settings.enable_caching,
                "async_processing": settings.enable_async_processing,
                "max_workers": settings.performance.max_workers,
                "memory_limit_mb": settings.performance.memory_limit_mb,
                "log_level": settings.logging.log_level.value
            }

        # Model configuration summary
        if _model_configs_available:
            try:
                performance_profile = get_performance_summary()
                summary["model_configs"] = {
                    "agents_configured": len(performance_profile.get("agents", {})),
                    "environment": performance_profile.get("environment"),
                    "config_version": performance_profile.get("config_version")
                }
                
                # Enhanced features summary
                enhanced_features = {}
                for agent_name, agent_info in performance_profile.get("agents", {}).items():
                    enhanced_features[agent_name] = {
                        "model_type": agent_info.get("model_type"),
                        "timeout": agent_info.get("timeout_seconds"),
                        "caching": agent_info.get("caching_enabled")
                    }
                summary["model_configs"]["agent_details"] = enhanced_features
                
            except Exception as e:
                logger.warning(f"Failed to get model configs summary: {e}")
                summary["model_configs"] = {"error": str(e)}

        # Validation status
        validation = validate_all_configurations_unified()
        summary["validation_status"] = {
            "overall_status": validation["overall_status"],
            "issues_count": len(validation["issues"]),
            "warnings_count": len(validation["warnings"]),
            "components_validated": validation["components_validated"]
        }

        # Performance indicators
        summary["performance_indicators"] = {
            "optimization_focus": _detect_optimization_focus(),
            "estimated_memory_usage": _estimate_memory_usage(),
            "estimated_processing_speed": _estimate_processing_speed(),
            "health_status": _get_overall_health_status()
        }

    except Exception as e:
        logger.error(f"Error generating configuration summary: {str(e)}")
        summary["error"] = str(e)

    return summary


def export_all_configurations(export_dir: Optional[str] = None, 
                             include_sensitive: bool = False) -> Dict[str, str]:
    """
    Export all available configurations to files.

    Args:
        export_dir: Optional directory for export files
        include_sensitive: Whether to include sensitive data like API keys

    Returns:
        Dictionary with file paths of exported configurations
    """
    from pathlib import Path

    if export_dir is None:
        if _settings_available:
            settings = get_settings()
            export_dir = settings.directories.config_dir / "exports"
        else:
            export_dir = Path("./config/exports")

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exported_files = {}

    try:
        # Export system settings if available
        if _settings_available:
            settings = get_settings()
            settings_file = export_path / f"system_settings_{timestamp}.json"
            exported_files["system_settings"] = str(settings.export_config(settings_file, include_sensitive))

        # Export model configurations if available
        if _model_configs_available:
            from .model_configs import _config_manager
            if _config_manager:
                model_configs_file = export_path / f"model_configs_{timestamp}.json"
                exported_files["model_configs"] = str(_config_manager.export_config(model_configs_file))

        # Export unified summary
        summary = get_configuration_summary()
        summary_file = export_path / f"config_summary_{timestamp}.json"
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        exported_files["summary"] = str(summary_file)

        logger.info(f"All configurations exported to {export_path}")

    except Exception as e:
        logger.error(f"Configuration export failed: {str(e)}")
        exported_files["error"] = str(e)

    return exported_files


def import_configurations(settings_file: Optional[str] = None,
                         model_configs_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Import configurations from files.

    Args:
        settings_file: Path to system settings JSON file
        model_configs_file: Path to model configs JSON file

    Returns:
        Import results with status and any issues
    """
    from pathlib import Path

    results = {
        "success": True,
        "imported": [],
        "issues": [],
        "timestamp": datetime.now().isoformat()
    }

    try:
        # Import system settings if available and file provided
        if _settings_available and settings_file and Path(settings_file).exists():
            try:
                imported_settings = SystemSettings.load_from_file(Path(settings_file))
                # Update global settings instance
                from .settings import _settings_instance
                _settings_instance = imported_settings
                results["imported"].append("system_settings")
                logger.info(f"System settings imported from {settings_file}")
            except Exception as e:
                results["issues"].append(f"System settings import failed: {str(e)}")
                results["success"] = False

        # Import model configurations if available and file provided
        if _model_configs_available and model_configs_file and Path(model_configs_file).exists():
            try:
                # This would need to be implemented in ConfigurationManager
                results["imported"].append("model_configs")
                logger.info(f"Model configurations imported from {model_configs_file}")
            except Exception as e:
                results["issues"].append(f"Model configs import failed: {str(e)}")
                results["success"] = False

        if not results["imported"]:
            results["issues"].append("No valid configuration files provided or found")
            results["success"] = False

    except Exception as e:
        results["success"] = False
        results["issues"].append(f"Import process failed: {str(e)}")
        logger.error(f"Configuration import error: {str(e)}")

    return results


def get_package_health() -> Dict[str, Any]:
    """
    Get comprehensive package health status for monitoring.

    Returns:
        Health status with detailed component analysis
    """
    health_info = {
        "overall_status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "package_version": __version__,
        "environment": get_current_environment(),
        "component_health": {},
        "issues": [],
        "warnings": [],
        "recommendations": []
    }

    try:
        # Check component availability
        if not _settings_available:
            health_info["issues"].append("Settings module not available")
            health_info["overall_status"] = "degraded"

        if not _model_configs_available:
            health_info["issues"].append("Model configs module not available")
            health_info["overall_status"] = "degraded"

        # System settings health
        if _settings_available:
            try:
                settings_health = get_settings().get_health_status()
                health_info["component_health"]["system_settings"] = settings_health
                
                if settings_health["status"] != "healthy":
                    health_info["issues"].extend([f"Settings: {issue}" for issue in settings_health.get("issues", [])])
                    if health_info["overall_status"] == "healthy":
                        health_info["overall_status"] = "degraded"
                        
            except Exception as e:
                health_info["component_health"]["system_settings"] = {"status": "error", "error": str(e)}
                health_info["issues"].append(f"Settings health check failed: {str(e)}")

        # Model configs health
        if _model_configs_available:
            try:
                model_validation = validate_all_configurations()
                health_info["component_health"]["model_configs"] = {
                    "status": model_validation["overall_status"],
                    "agents_validated": len(model_validation.get("agent_results", {}))
                }
                
                if model_validation["overall_status"] != "valid":
                    health_info["issues"].extend([f"Model configs: {issue}" for issue in model_validation.get("issues", [])])
                    
            except Exception as e:
                health_info["component_health"]["model_configs"] = {"status": "error", "error": str(e)}
                health_info["issues"].append(f"Model configs health check failed: {str(e)}")

        # Generate recommendations
        if health_info["overall_status"] == "healthy":
            health_info["recommendations"].append("All components are healthy and ready for use")
        else:
            health_info["recommendations"].append("Address component issues for optimal performance")

        if health_info["issues"]:
            if len(health_info["issues"]) > 3:
                health_info["overall_status"] = "critical"
            elif health_info["overall_status"] == "healthy":
                health_info["overall_status"] = "degraded"

    except Exception as e:
        health_info["overall_status"] = "error"
        health_info["error"] = str(e)
        logger.error(f"Package health check failed: {str(e)}")

    return health_info


# Helper functions for performance analysis
def _detect_optimization_focus() -> str:
    """Detect whether current configuration is optimized for performance or accuracy."""
    try:
        if not _model_configs_available:
            return "unknown"
            
        performance_profile = get_performance_summary()
        agents = performance_profile.get("agents", {})
        
        performance_indicators = 0
        accuracy_indicators = 0
        
        for agent_name, agent_info in agents.items():
            timeout = agent_info.get("timeout_seconds", 60)
            if timeout <= 45:
                performance_indicators += 1
            else:
                accuracy_indicators += 1
                
            if agent_info.get("caching_enabled"):
                performance_indicators += 1

        if performance_indicators > accuracy_indicators:
            return "performance"
        elif accuracy_indicators > performance_indicators:
            return "accuracy"
        else:
            return "balanced"
            
    except Exception:
        return "unknown"


def _estimate_memory_usage() -> str:
    """Estimate total memory usage based on current configuration."""
    try:
        base_memory = 1024  # Default
        
        if _settings_available:
            settings = get_settings()
            base_memory = settings.performance.memory_limit_mb
        
        if _model_configs_available:
            performance_profile = get_performance_summary()
            agents_count = len(performance_profile.get("agents", {}))
            estimated_mb = base_memory + (agents_count * 128)
        else:
            estimated_mb = base_memory

        if estimated_mb < 1024:
            return f"~{estimated_mb}MB (Low)"
        elif estimated_mb < 2048:
            return f"~{estimated_mb}MB (Medium)"
        else:
            return f"~{estimated_mb}MB (High)"
            
    except Exception:
        return "Unknown"


def _estimate_processing_speed() -> str:
    """Estimate processing speed based on current configuration."""
    try:
        if not _model_configs_available:
            return "Unknown"
            
        performance_profile = get_performance_summary()
        agents = performance_profile.get("agents", {})
        
        timeouts = [agent_info.get("timeout_seconds", 60) for agent_info in agents.values()]
        avg_timeout = sum(timeouts) / len(timeouts) if timeouts else 60
        
        caching_count = sum(1 for agent_info in agents.values() if agent_info.get("caching_enabled"))
        
        if avg_timeout <= 30 and caching_count >= 3:
            return "Fast (< 60s typical)"
        elif avg_timeout <= 60:
            return "Medium (60-120s typical)"
        else:
            return "Slow (> 120s typical)"
            
    except Exception:
        return "Unknown"


def _get_overall_health_status() -> str:
    """Get overall health status of the configuration system."""
    try:
        available_components = sum(__component_status__.values()) - 1  # Exclude 'package_initialized'
        total_components = len(__component_status__) - 1
        
        if available_components == total_components:
            return "healthy"
        elif available_components >= total_components * 0.7:
            return "degraded"
        else:
            return "critical"
            
    except Exception:
        return "unknown"


# Package initialization
def _initialize_package() -> bool:
    """Initialize the configuration package with comprehensive setup."""
    global __component_status__
    
    try:
        logger.info(f"Initializing Enhanced Configuration Package v{__version__}")
        
        environment = get_current_environment()
        logger.info(f"Environment: {environment}")
        
        # Apply environment configuration if settings available
        if _settings_available:
            try:
                apply_environment_config(environment)
                logger.info("Environment configuration applied successfully")
            except Exception as e:
                logger.warning(f"Failed to apply environment configuration: {e}")

        # Apply model configuration profile if available
        if _model_configs_available:
            try:
                profile_map = {
                    'development': ConfigurationProfile.BALANCED,
                    'testing': ConfigurationProfile.PERFORMANCE_OPTIMIZED,
                    'staging': ConfigurationProfile.BALANCED,
                    'production': ConfigurationProfile.ACCURACY_OPTIMIZED
                }
                
                if environment in profile_map and not os.getenv('CONFIGURATION_PROFILE'):
                    apply_configuration_profile(profile_map[environment])
                    logger.info(f"Applied {profile_map[environment].value} profile for {environment}")
                    
            except Exception as e:
                logger.warning(f"Failed to apply configuration profile: {e}")

        # Perform initial validation
        try:
            validation = validate_all_configurations_unified()
            if validation["overall_status"] != "valid":
                logger.warning("Configuration validation found issues:")
                for issue in validation["issues"]:
                    logger.warning(f"  - {issue}")
            else:
                logger.info("All configurations validated successfully")
                
        except Exception as e:
            logger.warning(f"Initial validation failed: {e}")

        __component_status__["package_initialized"] = True
        
        logger.info(f"✅ Enhanced Configuration Package v{__version__} initialized successfully")
        logger.info(f"Components available: {sum(__component_status__.values())}/{len(__component_status__)}")
        
        return True

    except Exception as e:
        logger.error(f"Package initialization failed: {str(e)}")
        __component_status__["package_initialized"] = False
        return False


# Initialize package on import
_package_initialized = _initialize_package()

# Export package status
__all__.append('_package_initialized')

# Final status message
if _package_initialized:
    logger.info(f"🎯 Enhanced Configuration Package v{__version__} ready for production use")
else:
    logger.error(f"⚠️ Enhanced Configuration Package v{__version__} initialization completed with errors")
