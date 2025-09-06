# agents/bert_classifier/__init__.py
"""
BERT Classifier Package with Config Integration

Enhanced BERT classification agent with modular architecture and
configuration integration for fake news detection.

Components:
- BERTClassifier: Main classification agent with config support
- TextPreprocessor: Configurable text preprocessing
- DeviceManager: Enhanced device management
- ModelManager: Model loading and validation
"""

from .classifier import BERTClassifier, FakeNewsDataset
from .preprocessing import TextPreprocessor
from .model_utils import DeviceManager, ModelManager

# ✅ OPTIONAL: EXPOSE CONFIG FUNCTIONS FOR CONVENIENCE
try:
    from config import get_model_config, get_settings
    
    def get_bert_config():
        """Convenience function to get BERT classifier configuration"""
        return get_model_config('bert_classifier')
    
    def get_system_settings():
        """Convenience function to get system settings"""
        return get_settings()
    
    # Add to exports
    __all__ = [
        'BERTClassifier',
        'FakeNewsDataset', 
        'TextPreprocessor',
        'DeviceManager',
        'ModelManager',
        'get_bert_config',
        'get_system_settings'
    ]
    
except ImportError:
    # Config not available, use basic exports
    __all__ = [
        'BERTClassifier',
        'FakeNewsDataset',
        'TextPreprocessor', 
        'DeviceManager',
        'ModelManager'
    ]

# Version and metadata
__version__ = "2.0.0"
__description__ = "BERT classifier with modular architecture and config integration"
__config_integrated__ = True
