"""
Enhanced Base Module for Production Fake News Detection System

This module provides the foundational BaseAgent class and supporting utilities
for building robust, production-ready fake news detection agents.

Key Features:
- Production-ready BaseAgent with comprehensive error handling
- Environment-aware configuration management
- Structured logging and performance monitoring
- LangGraph integration compatibility
- Memory management and resource optimization
- Async processing support

Version: 3.2.0 - Production Enhanced Edition
"""

from .base_agent import BaseAgent

# Export main classes
__all__ = [
    'BaseAgent'
]

# Package metadata
__version__ = '3.2.0'
__author__ = 'Fake News Detection Team'
__description__ = 'Production-ready base agent framework for fake news detection with LangGraph integration'
__license__ = 'MIT'

# Configuration constants
DEFAULT_CONFIG = {
    "environment": "development",
    "log_level": "INFO",
    "enable_metrics": True,
    "max_retries": 3,
    "timeout": 30,
    "memory_threshold_mb": 512
}

# Agent type constants
AGENT_TYPES = {
    "CLASSIFIER": "classifier",
    "GENERATOR": "generator", 
    "RECOMMENDER": "recommender",
    "EXTRACTOR": "extractor",
    "ANALYZER": "analyzer",
    "EVALUATOR": "evaluator"
}

def create_agent_config(agent_type: str, **kwargs) -> dict:
    """
    Create optimized configuration for specific agent type.
    
    Args:
        agent_type: Type of agent (use AGENT_TYPES constants)
        **kwargs: Additional configuration parameters
        
    Returns:
        Optimized configuration dictionary
    """
    base_config = DEFAULT_CONFIG.copy()
    
    # Agent-type specific optimizations
    type_configs = {
        AGENT_TYPES["CLASSIFIER"]: {
            "timeout": 15,
            "memory_threshold_mb": 256,
            "enable_caching": True
        },
        AGENT_TYPES["GENERATOR"]: {
            "timeout": 45,
            "max_tokens": 4096,
            "temperature": 0.3
        },
        AGENT_TYPES["ANALYZER"]: {
            "timeout": 30,
            "memory_threshold_mb": 512,
            "enable_caching": True
        }
    }
    
    if agent_type in type_configs:
        base_config.update(type_configs[agent_type])
    
    # Apply user overrides
    base_config.update(kwargs)
    
    return base_config

def get_agent_status_summary(agents: list) -> dict:
    """
    Get status summary for multiple agents.
    
    Args:
        agents: List of BaseAgent instances
        
    Returns:
        Combined status summary
    """
    if not agents:
        return {"status": "no_agents", "count": 0}
    
    statuses = [agent.get_comprehensive_status() for agent in agents]
    
    healthy_count = sum(1 for s in statuses if s["status"] == "healthy")
    total_calls = sum(s["performance_summary"]["total_calls"] for s in statuses)
    avg_success_rate = sum(s["performance_summary"]["success_rate"] for s in statuses) / len(statuses)
    
    overall_status = "healthy" if healthy_count == len(agents) else "degraded"
    
    return {
        "overall_status": overall_status,
        "agent_count": len(agents),
        "healthy_agents": healthy_count,
        "total_processing_calls": total_calls,
        "average_success_rate": avg_success_rate,
        "agents": [{"name": s["agent_name"], "status": s["status"]} for s in statuses]
    }

# Version compatibility check
def check_compatibility() -> dict:
    """
    Check system compatibility and requirements.
    
    Returns:
        Compatibility status dictionary
    """
    import sys
    import platform
    
    compatibility = {
        "python_version": sys.version,
        "python_compatible": sys.version_info >= (3, 8),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "status": "compatible"
    }
    
    if not compatibility["python_compatible"]:
        compatibility["status"] = "incompatible"
        compatibility["issues"] = ["Python 3.8+ required"]
    
    return compatibility

# Module initialization
def _initialize_module():
    """Initialize the base module with environment checks."""
    import logging
    import os
    
    logger = logging.getLogger(__name__)
    
    # Check environment
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Check compatibility
    compat = check_compatibility()
    if compat["status"] != "compatible":
        logger.warning(f"Compatibility issues detected: {compat.get('issues', [])}")
    
    # Log initialization
    logger.info(f"Base agents module v{__version__} initialized")
    logger.info(f"Environment: {environment}")
    logger.info(f"Python: {compat['python_version']}")
    
    return True

# Initialize on import
_module_initialized = _initialize_module()

# Export helper functions
__all__.extend([
    'DEFAULT_CONFIG',
    'AGENT_TYPES', 
    'create_agent_config',
    'get_agent_status_summary',
    'check_compatibility'
])
