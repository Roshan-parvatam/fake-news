# agents/__init__.py

"""
Agents Package - Enhanced Production Ready

Comprehensive interface for all fake news detection agents with robust
error handling, graceful degradation, and production monitoring capabilities.

Components:
- BERTClassifier: Advanced BERT-based classification with preprocessing
- ClaimExtractorAgent: Intelligent claim extraction with LLM integration  
- ContextAnalyzerAgent: Enhanced bias and manipulation analysis (v3.2)
- EvidenceEvaluatorAgent: Advanced evidence evaluation with verification links (v3.2)
- CredibleSourceAgent: Contextual source recommendations with safety handling (v3.2)
- LLMExplanationAgent: Comprehensive explanation generation

Features:
- Graceful component loading with detailed error reporting
- Component availability tracking and health monitoring
- Factory functions for easy agent instantiation
- Comprehensive logging and debugging capabilities
- Version tracking and compatibility checking
- Production-ready error handling and recovery

Version: 3.2.0 - Enhanced Production Edition
"""

import logging
import os
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime

# Configure package-level logging
logger = logging.getLogger(__name__)

# Component availability tracking
_component_availability = {
    'bert_classifier': False,
    'claim_extractor': False,
    'context_analyzer': False,
    'evidence_evaluator': False,
    'credible_source': False,
    'llm_explanation': False
}

# Component imports with comprehensive error handling
_imported_components = {}

# BERT Classifier Components
try:
    from agents.bert_classifier import (
        BERTClassifier,
        FakeNewsDataset, 
        create_bert_classifier,
        TextPreprocessor,
        DeviceManager,
        ModelManager,
        get_bert_config,
        get_package_info as bert_package_info,
        check_package_health as bert_health_check
    )
    _component_availability['bert_classifier'] = True
    _imported_components['bert_classifier'] = {
        'classes': ['BERTClassifier', 'FakeNewsDataset', 'TextPreprocessor', 'DeviceManager', 'ModelManager'],
        'functions': ['create_bert_classifier', 'get_bert_config'],
        'status': 'loaded',
        'version': '3.2.0'
    }
    logger.debug("✅ BERT Classifier components loaded successfully")
    
except ImportError as e:
    _component_availability['bert_classifier'] = False
    _imported_components['bert_classifier'] = {
        'status': 'failed',
        'error': str(e),
        'fallback': 'mock_implementations_available'
    }
    logger.warning(f"⚠️ BERT Classifier components unavailable: {e}")

# Claim Extractor Agent
try:
    from agents.claim_extractor import ClaimExtractorAgent
    _component_availability['claim_extractor'] = True
    _imported_components['claim_extractor'] = {
        'classes': ['ClaimExtractorAgent'],
        'status': 'loaded',
        'version': '3.2.0'
    }
    logger.debug("✅ Claim Extractor Agent loaded successfully")
    
except ImportError as e:
    _component_availability['claim_extractor'] = False
    _imported_components['claim_extractor'] = {
        'status': 'failed',
        'error': str(e)
    }
    logger.warning(f"⚠️ Claim Extractor Agent unavailable: {e}")

# Context Analyzer Agent (Enhanced v3.2)
try:
    from agents.context_analyzer import ContextAnalyzerAgent
    _component_availability['context_analyzer'] = True
    _imported_components['context_analyzer'] = {
        'classes': ['ContextAnalyzerAgent'],
        'status': 'loaded',
        'version': '3.2.0_enhanced',
        'features': ['llm_scoring', 'safety_handling', 'enhanced_bias_detection']
    }
    logger.debug("✅ Context Analyzer Agent loaded successfully (Enhanced v3.2)")
    
except ImportError as e:
    _component_availability['context_analyzer'] = False
    _imported_components['context_analyzer'] = {
        'status': 'failed',
        'error': str(e)
    }
    logger.warning(f"⚠️ Context Analyzer Agent unavailable: {e}")

# Evidence Evaluator Agent (Enhanced v3.2)
try:
    from agents.evidence_evaluator import EvidenceEvaluatorAgent
    _component_availability['evidence_evaluator'] = True
    _imported_components['evidence_evaluator'] = {
        'classes': ['EvidenceEvaluatorAgent'],
        'status': 'loaded',
        'version': '3.2.0_enhanced',
        'features': ['specific_verification_links', 'institutional_fallbacks', 'enhanced_scoring']
    }
    logger.debug("✅ Evidence Evaluator Agent loaded successfully (Enhanced v3.2)")
    
except ImportError as e:
    _component_availability['evidence_evaluator'] = False
    _imported_components['evidence_evaluator'] = {
        'status': 'failed',
        'error': str(e)
    }
    logger.warning(f"⚠️ Evidence Evaluator Agent unavailable: {e}")

# Credible Source Agent (Enhanced v3.2)
try:
    from agents.credible_source import CredibleSourceAgent
    _component_availability['credible_source'] = True
    _imported_components['credible_source'] = {
        'classes': ['CredibleSourceAgent'],
        'status': 'loaded',
        'version': '3.2.0_enhanced',
        'features': ['contextual_recommendations', 'expert_identification', 'safety_awareness']
    }
    logger.debug("✅ Credible Source Agent loaded successfully (Enhanced v3.2)")
    
except ImportError as e:
    _component_availability['credible_source'] = False
    _imported_components['credible_source'] = {
        'status': 'failed',
        'error': str(e)
    }
    logger.warning(f"⚠️ Credible Source Agent unavailable: {e}")

# LLM Explanation Agent
try:
    from agents.llm_explanation import LLMExplanationAgent
    _component_availability['llm_explanation'] = True
    _imported_components['llm_explanation'] = {
        'classes': ['LLMExplanationAgent'],
        'status': 'loaded',
        'version': '3.2.0'
    }
    logger.debug("✅ LLM Explanation Agent loaded successfully")
    
except ImportError as e:
    _component_availability['llm_explanation'] = False
    _imported_components['llm_explanation'] = {
        'status': 'failed',
        'error': str(e)
    }
    logger.warning(f"⚠️ LLM Explanation Agent unavailable: {e}")

# Package metadata
__version__ = "3.2.0"
__author__ = "Enhanced Fake News Detection Team"
__description__ = "Production-ready agents for comprehensive fake news detection and analysis"
__license__ = "MIT"
__status__ = "Production"

# Dynamic exports based on component availability
def _build_exports() -> List[str]:
    """Build exports list based on successfully loaded components."""
    exports = [
        '__version__', '__author__', '__description__',
        'get_agents_status', 'get_agents_health', 'initialize_all_agents',
        'create_agent_factory', 'get_available_agents'
    ]
    
    # Add available components
    for component, available in _component_availability.items():
        if available and component in _imported_components:
            component_info = _imported_components[component]
            if 'classes' in component_info:
                exports.extend(component_info['classes'])
            if 'functions' in component_info:
                exports.extend(component_info['functions'])
    
    return exports

__all__ = _build_exports()

# Package utility functions
def get_agents_status() -> Dict[str, Any]:
    """
    Get comprehensive status of all agent components.
    
    Returns:
        Dictionary with detailed component availability and status
    """
    available_count = sum(_component_availability.values())
    total_count = len(_component_availability)
    
    return {
        'package_info': {
            'version': __version__,
            'author': __author__,
            'description': __description__,
            'status': __status__
        },
        'component_availability': _component_availability.copy(),
        'component_details': _imported_components.copy(),
        'summary': {
            'available_components': available_count,
            'total_components': total_count,
            'availability_percentage': (available_count / total_count) * 100,
            'all_components_available': available_count == total_count
        },
        'timestamp': datetime.now().isoformat()
    }


def get_agents_health() -> Dict[str, Any]:
    """
    Get comprehensive health check of all agent components.
    
    Returns:
        Dictionary with health status and recommendations
    """
    health_info = {
        'overall_status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'package_version': __version__,
        'component_health': {},
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    try:
        # Check each component's health
        for component, available in _component_availability.items():
            if available:
                component_info = _imported_components[component]
                health_info['component_health'][component] = {
                    'status': 'healthy',
                    'version': component_info.get('version', 'unknown'),
                    'features': component_info.get('features', [])
                }
            else:
                error = _imported_components.get(component, {}).get('error', 'Component unavailable')
                health_info['component_health'][component] = {
                    'status': 'unavailable',
                    'error': error
                }
                health_info['issues'].append(f"{component}: {error}")
        
        # Check for BERT classifier specific health
        if _component_availability['bert_classifier']:
            try:
                bert_health = bert_health_check()
                if bert_health['overall_status'] != 'healthy':
                    health_info['warnings'].append(f"BERT Classifier health issues: {bert_health.get('issues', [])}")
            except Exception as e:
                health_info['warnings'].append(f"Could not check BERT health: {str(e)}")
        
        # Determine overall status
        available_count = sum(_component_availability.values())
        total_count = len(_component_availability)
        
        if available_count == total_count:
            health_info['overall_status'] = 'healthy'
            health_info['recommendations'].append("All components available and healthy")
        elif available_count >= total_count * 0.8:  # 80% threshold
            health_info['overall_status'] = 'degraded'
            health_info['recommendations'].append("Most components available - check failed components")
        else:
            health_info['overall_status'] = 'critical'
            health_info['recommendations'].extend([
                "Multiple components unavailable - check dependencies",
                "Verify installation and configuration"
            ])
        
        # Add specific recommendations
        if not _component_availability['bert_classifier']:
            health_info['recommendations'].append("Install BERT Classifier dependencies (torch, transformers)")
        
        missing_enhanced = [comp for comp, avail in _component_availability.items() 
                          if not avail and comp in ['context_analyzer', 'evidence_evaluator', 'credible_source']]
        if missing_enhanced:
            health_info['recommendations'].append(f"Install enhanced v3.2 components: {missing_enhanced}")
            
    except Exception as e:
        health_info['overall_status'] = 'error'
        health_info['error'] = str(e)
        health_info['recommendations'] = ['Contact system administrator - health check failed']
    
    return health_info


def initialize_all_agents() -> Dict[str, Any]:
    """
    Initialize all available agents and return initialization results.
    
    Returns:
        Dictionary with initialization results for each component
    """
    initialization_results = {
        'timestamp': datetime.now().isoformat(),
        'package_version': __version__,
        'results': {},
        'summary': {
            'successful': 0,
            'failed': 0,
            'total': 0
        }
    }
    
    logger.info("🚀 Initializing all available agents...")
    
    # Initialize BERT Classifier
    if _component_availability['bert_classifier']:
        try:
            # Test BERT classifier creation
            classifier = create_bert_classifier()
            initialization_results['results']['bert_classifier'] = {
                'status': 'success',
                'instance_created': True,
                'model_loaded': hasattr(classifier, 'model') and classifier.model is not None
            }
            initialization_results['summary']['successful'] += 1
        except Exception as e:
            initialization_results['results']['bert_classifier'] = {
                'status': 'failed',
                'error': str(e)
            }
            initialization_results['summary']['failed'] += 1
        initialization_results['summary']['total'] += 1
    
    # Initialize other agents
    agent_classes = {
        'claim_extractor': 'ClaimExtractorAgent',
        'context_analyzer': 'ContextAnalyzerAgent', 
        'evidence_evaluator': 'EvidenceEvaluatorAgent',
        'credible_source': 'CredibleSourceAgent',
        'llm_explanation': 'LLMExplanationAgent'
    }
    
    for component, class_name in agent_classes.items():
        if _component_availability[component]:
            try:
                # Get the class from globals and test instantiation
                agent_class = globals().get(class_name)
                if agent_class:
                    agent_instance = agent_class()
                    initialization_results['results'][component] = {
                        'status': 'success',
                        'instance_created': True,
                        'class_name': class_name
                    }
                    initialization_results['summary']['successful'] += 1
                else:
                    initialization_results['results'][component] = {
                        'status': 'failed',
                        'error': f"Class {class_name} not found in globals"
                    }
                    initialization_results['summary']['failed'] += 1
            except Exception as e:
                initialization_results['results'][component] = {
                    'status': 'failed',
                    'error': str(e)
                }
                initialization_results['summary']['failed'] += 1
            initialization_results['summary']['total'] += 1
    
    # Log summary
    successful = initialization_results['summary']['successful']
    total = initialization_results['summary']['total']
    
    if successful == total:
        logger.info(f"✅ All {total} agents initialized successfully")
    else:
        failed = initialization_results['summary']['failed']
        logger.warning(f"⚠️ Initialized {successful}/{total} agents ({failed} failed)")
    
    return initialization_results


def create_agent_factory(agent_type: str, **kwargs) -> Optional[Any]:
    """
    Factory function to create agent instances with error handling.
    
    Args:
        agent_type: Type of agent to create
        **kwargs: Additional arguments for agent creation
        
    Returns:
        Agent instance or None if creation fails
    """
    try:
        if agent_type == 'bert_classifier' and _component_availability['bert_classifier']:
            return create_bert_classifier(kwargs.get('config'), kwargs.get('session_id'))
            
        elif agent_type == 'claim_extractor' and _component_availability['claim_extractor']:
            return ClaimExtractorAgent(**kwargs)
            
        elif agent_type == 'context_analyzer' and _component_availability['context_analyzer']:
            return ContextAnalyzerAgent(**kwargs)
            
        elif agent_type == 'evidence_evaluator' and _component_availability['evidence_evaluator']:
            return EvidenceEvaluatorAgent(**kwargs)
            
        elif agent_type == 'credible_source' and _component_availability['credible_source']:
            return CredibleSourceAgent(**kwargs)
            
        elif agent_type == 'llm_explanation' and _component_availability['llm_explanation']:
            return LLMExplanationAgent(**kwargs)
            
        else:
            logger.error(f"Agent type '{agent_type}' not available or unknown")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create {agent_type} agent: {str(e)}")
        return None


def get_available_agents() -> List[str]:
    """
    Get list of available agent types.
    
    Returns:
        List of available agent type names
    """
    return [agent for agent, available in _component_availability.items() if available]


def get_enhanced_agents() -> List[str]:
    """
    Get list of enhanced v3.2 agents with advanced features.
    
    Returns:
        List of enhanced agent names
    """
    enhanced_agents = []
    for component, details in _imported_components.items():
        if (details.get('status') == 'loaded' and 
            '3.2.0_enhanced' in details.get('version', '') and
            details.get('features')):
            enhanced_agents.append(component)
    return enhanced_agents


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information.
    
    Returns:
        Dictionary with package metadata and component information
    """
    enhanced_agents = get_enhanced_agents()
    available_agents = get_available_agents()
    
    return {
        'package_metadata': {
            'name': 'agents',
            'version': __version__,
            'author': __author__, 
            'description': __description__,
            'license': __license__,
            'status': __status__
        },
        'components': {
            'total_agents': len(_component_availability),
            'available_agents': len(available_agents),
            'enhanced_agents': len(enhanced_agents),
            'availability_rate': len(available_agents) / len(_component_availability) * 100
        },
        'available_agent_list': available_agents,
        'enhanced_agent_list': enhanced_agents,
        'features': [
            'Production-ready agent implementations',
            'Enhanced error handling and recovery',
            'Comprehensive logging and monitoring',
            'Factory functions for easy instantiation',
            'Health monitoring and diagnostics',
            'Version tracking and compatibility checking'
        ],
        'component_details': _imported_components.copy()
    }


# Package initialization logging
def _initialize_package():
    """Initialize package with comprehensive status reporting."""
    try:
        available_count = sum(_component_availability.values())
        total_count = len(_component_availability)
        
        logger.info(f"🎯 Agents Package v{__version__} initialized")
        logger.info(f"📊 Component availability: {available_count}/{total_count}")
        
        # Log available components
        for component, available in _component_availability.items():
            status = "✅ Available" if available else "❌ Unavailable"
            version_info = ""
            
            if available and component in _imported_components:
                details = _imported_components[component]
                version = details.get('version', '')
                features = details.get('features', [])
                if version:
                    version_info = f" (v{version})"
                if features:
                    version_info += f" - Features: {', '.join(features)}"
            
            logger.info(f"  {status} {component}{version_info}")
        
        # Enhanced agents summary
        enhanced_agents = get_enhanced_agents()
        if enhanced_agents:
            logger.info(f"🚀 Enhanced v3.2 agents: {', '.join(enhanced_agents)}")
        
        if available_count == total_count:
            logger.info("🎉 All agents loaded successfully - full functionality available")
        elif available_count >= total_count * 0.8:
            logger.warning("⚠️ Most agents available - some functionality may be limited")
        else:
            logger.error("❌ Multiple agents unavailable - system functionality significantly limited")
            
        return True
        
    except Exception as e:
        logger.error(f"Package initialization error: {str(e)}")
        return False

# Initialize package
_package_initialized = _initialize_package()

# Export initialization status
__all__.extend(['_package_initialized', '_component_availability'])

# Final status message
if _package_initialized:
    available_count = sum(_component_availability.values())
    total_count = len(_component_availability)
    logger.info(f"🎯 Agents Package v{__version__} ready - {available_count}/{total_count} components operational")
else:
    logger.error(f"⚠️ Agents Package v{__version__} initialization completed with errors")
