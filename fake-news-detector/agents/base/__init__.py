# agents/base/__init__.py
"""
Base module for the fake news detection system.

This module provides the foundational BaseAgent class that all other agents inherit from.
It includes common functionality for configuration management, logging, and standardized
interfaces without hardcoding models or parameters.

Designed for LangGraph integration with flexible, non-hardcoded architecture.
"""

from .base_agent import BaseAgent

__all__ = ['BaseAgent']

# Version info
__version__ = '2.0.0'
__author__ = 'Your Name'
__description__ = 'Modular base agent class for fake news detection system - LangGraph ready'
