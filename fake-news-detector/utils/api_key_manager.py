# utils/api_key_manager.py

"""
Centralized API Key Management for Fake News Detection System

Provides consistent API key loading, validation, and configuration across all agents.
Ensures proper environment variable handling and fallback mechanisms.

Version: 1.0.0
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class APIKeyManager:
    """
    Centralized API key management with validation and fallback support.
    
    Features:
    - Consistent environment variable loading
    - API key format validation
    - Fallback mechanisms for different environments
    - Secure key masking for logging
    """
    
    # Standardized environment variable names
    GEMINI_API_KEY_VAR = 'GEMINI_API_KEY'
    OPENAI_API_KEY_VAR = 'OPENAI_API_KEY'
    
    @staticmethod
    def load_gemini_api_key(config: Optional[Dict[str, Any]] = None, 
                           system_settings: Optional[Any] = None) -> str:
        """
        Load Gemini API key with standardized fallback mechanisms.
        
        Args:
            config: Agent configuration dictionary
            system_settings: System settings object
            
        Returns:
            str: Validated API key
            
        Raises:
            ValueError: If no valid API key is found
        """
        # Primary: Environment variable (standardized)
        api_key = os.getenv(APIKeyManager.GEMINI_API_KEY_VAR)
        
        # Fallback 1: Legacy environment variable names
        if not api_key:
            api_key = os.getenv('GOOGLE_API_KEY')
            
        # Fallback 2: System settings
        if not api_key and system_settings:
            api_key = getattr(system_settings, 'gemini_api_key', None)
            
        # Fallback 3: Agent config
        if not api_key and config:
            api_key = config.get('api_key')
            
        # Validate API key
        if not api_key:
            raise ValueError(
                f"Gemini API key not found. Please set {APIKeyManager.GEMINI_API_KEY_VAR} "
                "in your .env file"
            )
            
        # Validate format
        if not APIKeyManager._validate_gemini_key_format(api_key):
            raise ValueError(
                "Invalid Gemini API key format. Please check your API key"
            )
            
        logger.debug(f"✅ Gemini API key loaded successfully")
        return api_key
    
    @staticmethod
    def load_openai_api_key(config: Optional[Dict[str, Any]] = None,
                           system_settings: Optional[Any] = None) -> Optional[str]:
        """
        Load OpenAI API key with standardized fallback mechanisms.
        
        Args:
            config: Agent configuration dictionary
            system_settings: System settings object
            
        Returns:
            Optional[str]: Validated API key or None if not found
        """
        # Primary: Environment variable
        api_key = os.getenv(APIKeyManager.OPENAI_API_KEY_VAR)
        
        # Fallback: System settings
        if not api_key and system_settings:
            api_key = getattr(system_settings, 'openai_api_key', None)
            
        # Fallback: Agent config
        if not api_key and config:
            api_key = config.get('openai_api_key')
            
        # Validate format if found
        if api_key and not APIKeyManager._validate_openai_key_format(api_key):
            logger.warning("Invalid OpenAI API key format detected")
            return None
            
        return api_key
    
    @staticmethod
    def _validate_gemini_key_format(api_key: str) -> bool:
        """
        Validate Gemini API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            bool: True if valid format
        """
        if not api_key or not isinstance(api_key, str):
            return False
            
        # Gemini API keys should start with 'AI' and be at least 20 characters
        return (api_key.startswith('AI') and 
                len(api_key.strip()) >= 20 and
                not api_key.startswith('test_') and
                api_key != 'your_api_key_here')
    
    @staticmethod
    def _validate_openai_key_format(api_key: str) -> bool:
        """
        Validate OpenAI API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            bool: True if valid format
        """
        if not api_key or not isinstance(api_key, str):
            return False
            
        # OpenAI API keys should start with 'sk-' and be at least 20 characters
        return (api_key.startswith('sk-') and 
                len(api_key.strip()) >= 20 and
                not api_key.startswith('test_') and
                api_key != 'your_api_key_here')
    
    @staticmethod
    def mask_api_key(api_key: str, show_chars: int = 4) -> str:
        """
        Mask API key for safe logging.
        
        Args:
            api_key: API key to mask
            show_chars: Number of characters to show at the end
            
        Returns:
            str: Masked API key
        """
        if not api_key or len(api_key) <= show_chars:
            return "***"
            
        return f"***{api_key[-show_chars:]}"
    
    @staticmethod
    def get_api_key_status() -> Dict[str, Any]:
        """
        Get status of all API keys.
        
        Returns:
            Dict[str, Any]: Status information for all API keys
        """
        gemini_key = os.getenv(APIKeyManager.GEMINI_API_KEY_VAR)
        openai_key = os.getenv(APIKeyManager.OPENAI_API_KEY_VAR)
        
        return {
            'gemini': {
                'loaded': bool(gemini_key),
                'valid': APIKeyManager._validate_gemini_key_format(gemini_key) if gemini_key else False,
                'masked': APIKeyManager.mask_api_key(gemini_key) if gemini_key else None
            },
            'openai': {
                'loaded': bool(openai_key),
                'valid': APIKeyManager._validate_openai_key_format(openai_key) if openai_key else False,
                'masked': APIKeyManager.mask_api_key(openai_key) if openai_key else None
            }
        }
