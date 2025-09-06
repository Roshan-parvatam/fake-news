# config/__init__.py
"""
Configuration Package for Fake News Detection System

This package contains all configuration management utilities and settings
for the modular fake news detection agents.

Modules:
- settings: General system settings and defaults
- model_configs: Model-specific configurations for each agent
- prompts_config: Centralized prompt template management
"""

from .settings import SystemSettings, get_settings
from .model_configs import ModelConfigs, get_model_config
from .prompts_config import PromptsConfig, get_prompt_template

__all__ = [
    'SystemSettings',
    'ModelConfigs', 
    'PromptsConfig',
    'get_settings',
    'get_model_config',
    'get_prompt_template'
]

# Version info
__version__ = "2.0.0"
__description__ = "Configuration management for modular fake news detection system"
