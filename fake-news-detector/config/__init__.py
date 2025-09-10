"""
Enhanced Configuration Package for Production Fake News Detection

Production-ready configuration management with comprehensive settings,
model configurations, and environment awareness.

Features:
- Environment-based configuration switching
- Comprehensive model configuration management  
- Performance and accuracy optimization profiles
- Enhanced security and validation
- Monitoring and health checking capabilities

Version: 3.2.0 - Production Enhanced Edition
"""

import os
import logging
from typing import Dict, Any, Optional

# Import main configuration classes
from .settings import (
    SystemSettings,
    get_settings,
    update_settings,
    reset_settings,
    apply_environment_config,
    ENVIRONMENT_CONFIGS
)

from .model_configs import (
    ModelConfigs,
    get_model_config,
    update_model_config,
    get_performance_summary,
    apply_configuration_profile,
    get_available_profiles,
    reset_model_configs,
    CONFIGURATION_PROFILES
)

# Export all public interfaces
__all__ = [
    # Core classes
    'SystemSettings',
    'ModelConfigs',
    
    # System settings functions
    'get_settings',
    'update_settings',
    'reset_settings',
    'apply_environment_config',
    
    # Model configuration functions
    'get_model_config',
    'update_model_config', 
    'get_performance_summary',
    'apply_configuration_profile',
    'get_available_profiles',
    'reset_model_configs',
    
    # Utility functions
    'get_current_environment',
    'validate_configuration',
    'get_configuration_summary',
    'export_all_configurations',
    'import_configurations',
    
    # Constants
    'ENVIRONMENT_CONFIGS',
    'CONFIGURATION_PROFILES'
]

# Package metadata
__version__ = "3.2.0"
__author__ = "Fake News Detection Team"
__description__ = "Production-ready configuration management for modular fake news detection system"
__license__ = "MIT"

# Configuration constants
SUPPORTED_ENVIRONMENTS = ["development", "testing", "staging", "production"]
DEFAULT_ENVIRONMENT = "development"
CONFIG_VERSION = "3.2.0"

# Setup package-level logging
logger = logging.getLogger(__name__)


def get_current_environment() -> str:
    """
    Get the current environment setting
    
    Returns:
        Current environment name
    """
    return os.getenv('ENVIRONMENT', DEFAULT_ENVIRONMENT)


def validate_configuration() -> Dict[str, Any]:
    """
    Comprehensive validation of all configurations
    
    Returns:
        Validation results with status and any issues found
    """
    validation_results = {
        "overall_status": "valid",
        "issues": [],
        "warnings": [],
        "system_settings": {},
        "model_configs": {},
        "environment": get_current_environment(),
        "timestamp": "",
        "config_version": CONFIG_VERSION
    }
    
    try:
        from datetime import datetime
        validation_results["timestamp"] = datetime.now().isoformat()
        
        # Validate system settings
        try:
            settings = get_settings()
            
            # API key validation
            if not settings.validate_api_keys():
                validation_results["issues"].append("API keys not properly configured")
            else:
                validation_results["system_settings"]["api_keys"] = "valid"
            
            # Directory validation
            directories_exist = all([
                settings.models_dir.exists(),
                settings.data_dir.exists(),
                settings.logs_dir.exists(),
                settings.cache_dir.exists()
            ])
            
            if directories_exist:
                validation_results["system_settings"]["directories"] = "valid"
            else:
                validation_results["issues"].append("Some required directories are missing")
            
            # Resource limits validation
            if settings.memory_limit_mb < 512:
                validation_results["warnings"].append("Memory limit may be too low for production use")
            
            if settings.max_workers > 16:
                validation_results["warnings"].append("High worker count may cause resource contention")
                
        except Exception as e:
            validation_results["issues"].append(f"System settings validation failed: {str(e)}")
        
        # Validate model configurations
        try:
            performance_summary = get_performance_summary()
            
            if performance_summary:
                validation_results["model_configs"]["performance_profile"] = "available"
                
                # Check for missing model configurations
                required_agents = ["bert_classifier", "claim_extractor", "context_analyzer", 
                                 "evidence_evaluator", "credible_source", "llm_explanation"]
                
                missing_agents = []
                for agent_name in required_agents:
                    try:
                        config = get_model_config(agent_name)
                        if not config or "model_name" not in config:
                            missing_agents.append(agent_name)
                    except Exception:
                        missing_agents.append(agent_name)
                
                if missing_agents:
                    validation_results["issues"].append(f"Missing configurations for agents: {missing_agents}")
                else:
                    validation_results["model_configs"]["all_agents"] = "configured"
                    
        except Exception as e:
            validation_results["issues"].append(f"Model configuration validation failed: {str(e)}")
        
        # Determine overall status
        if validation_results["issues"]:
            validation_results["overall_status"] = "invalid"
        elif validation_results["warnings"]:
            validation_results["overall_status"] = "valid_with_warnings"
        
        logger.info(f"Configuration validation completed: {validation_results['overall_status']}")
        
    except Exception as e:
        validation_results["overall_status"] = "error"
        validation_results["issues"].append(f"Validation process failed: {str(e)}")
        logger.error(f"Configuration validation error: {str(e)}")
    
    return validation_results


def get_configuration_summary() -> Dict[str, Any]:
    """
    Get comprehensive summary of current configuration state
    
    Returns:
        Configuration summary with key metrics and status
    """
    summary = {
        "package_info": {
            "version": __version__,
            "config_version": CONFIG_VERSION,
            "environment": get_current_environment()
        },
        "system_settings": {},
        "model_configs": {},
        "validation_status": {},
        "performance_indicators": {}
    }
    
    try:
        # System settings summary
        settings = get_settings()
        summary["system_settings"] = {
            "api_keys_configured": settings.validate_api_keys(),
            "caching_enabled": settings.enable_caching,
            "debug_mode": settings.debug_mode,
            "max_workers": settings.max_workers,
            "memory_limit_mb": settings.memory_limit_mb,
            "log_level": settings.log_level,
            "environment": settings.environment
        }
        
        # Model configuration summary
        performance_profile = get_performance_summary()
        if performance_profile:
            summary["model_configs"] = {
                "agents_configured": len(performance_profile.get("agents", {})),
                "environment": performance_profile.get("environment"),
                "config_version": performance_profile.get("configuration_version")
            }
            
            # Extract enhanced features summary
            enhanced_features = {}
            for agent_name, agent_info in performance_profile.get("agents", {}).items():
                enhanced_features[agent_name] = len(agent_info.get("enhanced_features", []))
            
            summary["model_configs"]["enhanced_features_count"] = enhanced_features
        
        # Validation status
        validation = validate_configuration()
        summary["validation_status"] = {
            "overall_status": validation["overall_status"],
            "issues_count": len(validation["issues"]),
            "warnings_count": len(validation["warnings"])
        }
        
        # Performance indicators
        summary["performance_indicators"] = {
            "optimized_for": _detect_optimization_focus(),
            "estimated_memory_usage": _estimate_memory_usage(),
            "estimated_processing_speed": _estimate_processing_speed()
        }
        
    except Exception as e:
        logger.error(f"Error generating configuration summary: {str(e)}")
        summary["error"] = str(e)
    
    return summary


def export_all_configurations(export_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Export all configurations to files
    
    Args:
        export_dir: Optional directory for export files
        
    Returns:
        Dictionary with file paths of exported configurations
    """
    from pathlib import Path
    from datetime import datetime
    
    if export_dir is None:
        settings = get_settings()
        export_dir = settings.config_dir / "exports"
    
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exported_files = {}
    
    try:
        # Export system settings
        settings = get_settings()
        settings_file = export_path / f"system_settings_{timestamp}.json"
        exported_files["system_settings"] = str(settings.export_config(settings_file))
        
        # Export model configurations
        model_configs = ModelConfigs()
        model_configs_file = export_path / f"model_configs_{timestamp}.json"
        exported_files["model_configs"] = str(model_configs.export_configuration(model_configs_file))
        
        # Export configuration summary
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
    Import configurations from files
    
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
        "timestamp": ""
    }
    
    try:
        from datetime import datetime
        results["timestamp"] = datetime.now().isoformat()
        
        # Import system settings
        if settings_file and Path(settings_file).exists():
            try:
                imported_settings = SystemSettings.load_from_file(Path(settings_file))
                # Update global settings instance
                global _settings_instance
                from .settings import _settings_instance
                _settings_instance = imported_settings
                results["imported"].append("system_settings")
                logger.info(f"System settings imported from {settings_file}")
            except Exception as e:
                results["issues"].append(f"System settings import failed: {str(e)}")
                results["success"] = False
        
        # Import model configurations
        if model_configs_file and Path(model_configs_file).exists():
            try:
                imported_model_configs = ModelConfigs.load_from_export(Path(model_configs_file))
                # Update global model configs instance
                global _model_configs_instance
                from .model_configs import _model_configs_instance
                _model_configs_instance = imported_model_configs
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


def _detect_optimization_focus() -> str:
    """Detect whether current configuration is optimized for performance or accuracy"""
    try:
        performance_profile = get_performance_summary()
        agents = performance_profile.get("agents", {})
        
        # Check various indicators
        performance_indicators = 0
        accuracy_indicators = 0
        
        for agent_name, agent_info in agents.items():
            # Check timeout values (lower = performance oriented)
            timeout = agent_info.get("timeout", 30)
            if timeout <= 45:
                performance_indicators += 1
            else:
                accuracy_indicators += 1
            
            # Check caching (enabled = performance oriented)
            if agent_info.get("caching_enabled"):
                performance_indicators += 1
            
            # Check enhanced features (more = accuracy oriented)
            enhanced_count = len(agent_info.get("enhanced_features", []))
            if enhanced_count >= 3:
                accuracy_indicators += 1
            elif enhanced_count <= 1:
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
    """Estimate total memory usage based on current configuration"""
    try:
        settings = get_settings()
        base_memory = settings.memory_limit_mb
        
        # Add estimates based on model configurations
        performance_profile = get_performance_summary()
        agents_count = len(performance_profile.get("agents", {}))
        
        # Rough estimation
        estimated_mb = base_memory + (agents_count * 128)  # ~128MB per agent
        
        if estimated_mb < 1024:
            return f"~{estimated_mb}MB (Low)"
        elif estimated_mb < 2048:
            return f"~{estimated_mb}MB (Medium)"
        else:
            return f"~{estimated_mb}MB (High)"
            
    except Exception:
        return "Unknown"


def _estimate_processing_speed() -> str:
    """Estimate processing speed based on current configuration"""
    try:
        performance_profile = get_performance_summary()
        agents = performance_profile.get("agents", {})
        
        # Calculate average timeout across agents
        timeouts = [agent_info.get("timeout", 30) for agent_info in agents.values()]
        avg_timeout = sum(timeouts) / len(timeouts) if timeouts else 30
        
        # Check for performance optimizations
        caching_count = sum(1 for agent_info in agents.values() 
                           if agent_info.get("caching_enabled"))
        
        # Estimate based on timeouts and optimizations
        if avg_timeout <= 30 and caching_count >= 3:
            return "Fast (< 60s typical)"
        elif avg_timeout <= 60:
            return "Medium (60-120s typical)"
        else:
            return "Slow (> 120s typical)"
            
    except Exception:
        return "Unknown"


# Package initialization
def _initialize_package():
    """Initialize the configuration package"""
    try:
        environment = get_current_environment()
        
        logger.info(f"Configuration package v{__version__} initializing...")
        logger.info(f"Environment: {environment}")
        
        # Validate environment
        if environment not in SUPPORTED_ENVIRONMENTS:
            logger.warning(f"Unsupported environment '{environment}'. Supported: {SUPPORTED_ENVIRONMENTS}")
        
        # Perform basic validation
        validation = validate_configuration()
        if validation["overall_status"] == "invalid":
            logger.warning("Configuration validation found issues:")
            for issue in validation["issues"]:
                logger.warning(f"  - {issue}")
        
        logger.info("Configuration package initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration package initialization failed: {str(e)}")
        return False


# Initialize on import
_package_initialized = _initialize_package()

# Export initialization status
__all__.append('_package_initialized')
