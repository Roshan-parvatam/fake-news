# config/model_configs.py
"""
Model Configurations for Fake News Detection Agents

This module contains model-specific configurations for each agent,
including AI model settings, processing parameters, and performance tuning.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class ModelConfigs:
    """
    ðŸ¤– MODEL CONFIGURATIONS FOR ALL AGENTS
    
    Centralized configuration for all AI models and processing parameters
    used by the fake news detection agents.
    """
    
    # BERT Classifier Configuration
    bert_classifier: Dict[str, Any] = None
    
    # LLM Explanation Configuration  
    llm_explanation: Dict[str, Any] = None
    
    # Credible Source Configuration
    credible_source: Dict[str, Any] = None
    
    # Claim Extractor Configuration
    claim_extractor: Dict[str, Any] = None
    
    # Context Analyzer Configuration
    context_analyzer: Dict[str, Any] = None
    
    # Evidence Evaluator Configuration
    evidence_evaluator: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default configurations for all agents"""
        if self.bert_classifier is None:
            self.bert_classifier = self._get_bert_classifier_config()
        
        if self.llm_explanation is None:
            self.llm_explanation = self._get_llm_explanation_config()
        
        if self.credible_source is None:
            self.credible_source = self._get_credible_source_config()
        
        if self.claim_extractor is None:
            self.claim_extractor = self._get_claim_extractor_config()
        
        if self.context_analyzer is None:
            self.context_analyzer = self._get_context_analyzer_config()
        
        if self.evidence_evaluator is None:
            self.evidence_evaluator = self._get_evidence_evaluator_config()
    
    def _get_bert_classifier_config(self) -> Dict[str, Any]:
        """Configuration for BERT Classifier Agent"""
        return {
            # Model Settings
            "model_name": "bert-base-uncased",
            "model_path": "models/bert_fake_news",
            "max_length": 512,
            "num_labels": 2,
            
            # Processing Settings
            "batch_size": 16,
            "enable_preprocessing": True,
            "device": "auto",  # auto, cpu, cuda, mps
            
            # Preprocessing Configuration
            "preprocessing_config": {
                "max_length": 2000,
                "remove_urls": True,
                "remove_emails": True,
                "normalize_quotes": True,
                "remove_excessive_punctuation": True,
                "handle_special_characters": True
            },
            
            # Performance Settings
            "enable_metrics": True,
            "cache_predictions": True,
            "prediction_threshold": 0.5,
            
            # LangGraph Integration
            "state_key": "bert_classification",
            "next_agents": ["claim_extractor", "context_analyzer"],
            
            # Quality Thresholds
            "high_confidence_threshold": 0.8,
            "low_confidence_threshold": 0.6,
            "uncertain_prediction_threshold": 0.7
        }
    
    def _get_llm_explanation_config(self) -> Dict[str, Any]:
        """Configuration for LLM Explanation Agent"""
        return {
            # Model Settings
            "model_name": "gemini-2.5-flash",
            "temperature": 0.3,
            "max_tokens": 3072,
            "top_p": 0.9,
            "top_k": 40,
            
            # Analysis Settings
            "enable_detailed_analysis": True,
            "confidence_threshold": 0.75,  # Trigger detailed analysis below this
            "enable_source_analysis": True,
            "enable_confidence_analysis": True,
            
            # Response Settings
            "response_format": "structured",
            "include_evidence_summary": True,
            "include_verification_suggestions": True,
            "max_explanation_length": 2000,
            
            # API Settings
            "rate_limit_seconds": 4.0,
            "max_retries": 3,
            "timeout_seconds": 60,
            
            # LangGraph Integration
            "state_key": "llm_explanation",
            "next_agents": [],  # Final agent
            
            # Safety Settings (relaxed to reduce false blocks)
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }
    
    def _get_credible_source_config(self) -> Dict[str, Any]:
        """Configuration for Credible Source Agent"""
        return {
            # Model Settings
            "model_name": "gemini-2.5-flash",
            "temperature": 0.2,  # Lower for consistent recommendations
            "max_tokens": 2048,
            
            # Source Analysis Settings
            "enable_cross_verification": True,
            "max_sources_per_recommendation": 10,
            "min_source_reliability_score": 6.0,
            "enable_domain_classification": True,
            
            # Database Settings
            "source_database_size": 850,  # Approximate number of sources
            "enable_source_caching": True,
            "cache_ttl_hours": 24,
            
            # Domain Classification
            "domain_confidence_threshold": 0.3,
            "max_domains_to_consider": 3,
            
            # Verification Protocol
            "protocol_depth": "comprehensive",  # basic, standard, comprehensive
            "include_verification_timeline": True,
            "include_expert_contacts": True,
            
            # API Settings
            "rate_limit_seconds": 4.0,
            "max_retries": 3,
            
            # LangGraph Integration
            "state_key": "source_recommendations",
            "next_agents": ["llm_explanation"],
            
            # Safety Settings (relaxed to handle sensitive real-world text)
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }
    
    def _get_claim_extractor_config(self) -> Dict[str, Any]:
        """Configuration for Claim Extractor Agent"""
        return {
            # Model Settings
            "model_name": "gemini-2.5-flash",
            "temperature": 0.3,
            "max_tokens": 2048,
            
            # Extraction Settings
            "max_claims_per_article": 8,
            "min_claim_length": 10,
            "enable_verification_analysis": True,
            "enable_claim_prioritization": True,
            
            # Pattern Analysis
            "enable_pattern_preprocessing": True,
            "pattern_confidence_threshold": 0.5,
            "claim_richness_threshold": 5.0,
            
            # Claim Categories
            "supported_claim_types": [
                "Statistical", "Event", "Attribution", 
                "Research", "Policy", "Causal", "Other"
            ],
            "priority_levels": [1, 2, 3],  # 1=Critical, 2=Important, 3=Minor
            
            # Parsing Settings
            "enable_fallback_parsing": True,
            "max_parsing_attempts": 3,
            "parsing_quality_threshold": 60,  # percentage
            
            # API Settings
            "rate_limit_seconds": 4.0,
            "max_retries": 3,
            
            # LangGraph Integration
            "state_key": "extracted_claims",
            "next_agents": ["evidence_evaluator", "context_analyzer"],
            
            # Safety Settings (relaxed to reduce safety blocks during extraction)
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }
    
    def _get_context_analyzer_config(self) -> Dict[str, Any]:
        """Configuration for Context Analyzer Agent"""
        return {
            # Model Settings
            "model_name": "gemini-2.5-flash", 
            "temperature": 0.4,  # Higher for nuanced analysis
            "max_tokens": 3072,
            
            # Analysis Settings
            "enable_detailed_analysis": True,
            "bias_threshold": 5.0,
            "manipulation_threshold": 6.0,
            "enable_propaganda_analysis": True,
            
            # Bias Detection
            "bias_detection_modes": [
                "political_bias", "emotional_bias", "selection_bias", 
                "linguistic_bias", "cultural_bias"
            ],
            "emotional_analysis_depth": "comprehensive",
            
            # Manipulation Detection
            "propaganda_techniques_count": 10,
            "fallacy_detection_enabled": True,
            "manipulation_scoring_algorithm": "weighted_average",
            
            # Pattern Analysis
            "pattern_database_size": {
                "bias_indicators": 200,
                "emotional_keywords": 150,
                "framing_patterns": 80,
                "linguistic_patterns": 60
            },
            
            # API Settings
            "rate_limit_seconds": 4.0,
            "max_retries": 3,
            
            # LangGraph Integration
            "state_key": "context_analysis",
            "next_agents": ["evidence_evaluator"],
            
            # Safety Settings (relaxed to allow nuanced context analysis)
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }
    
    def _get_evidence_evaluator_config(self) -> Dict[str, Any]:
        """Configuration for Evidence Evaluator Agent"""
        return {
            # Model Settings
            "model_name": "gemini-2.5-flash",
            "temperature": 0.3,  # Lower for consistent analysis
            "max_tokens": 3072,
            
            # Evaluation Settings
            "enable_detailed_analysis": True,
            "evidence_threshold": 6.0,
            "enable_fallacy_detection": True,
            "enable_gap_analysis": True,
            
            # Evidence Criteria
            "evidence_types": [
                "statistical_evidence", "documentary_evidence",
                "testimonial_evidence", "circumstantial_evidence"
            ],
            "source_quality_tiers": ["primary", "expert", "institutional", "journalistic"],
            
            # Scoring Weights
            "scoring_weights": {
                "source_quality": 0.4,
                "logical_consistency": 0.3,
                "evidence_completeness": 0.3
            },
            
            # Fallacy Detection
            "fallacy_types_count": 10,
            "reasoning_quality_threshold": 5.0,
            "logical_health_threshold": 6.0,
            
            # Quality Thresholds
            "high_quality_threshold": 7.0,
            "medium_quality_threshold": 5.0,
            "poor_quality_threshold": 3.0,
            
            # API Settings
            "rate_limit_seconds": 4.0,
            "max_retries": 3,
            
            # LangGraph Integration
            "state_key": "evidence_evaluation",
            "next_agents": ["credible_source"],
            
            # Safety Settings (relaxed for evidence summarization)
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Configuration dictionary for the agent
        """
        agent_configs = {
            "bert_classifier": self.bert_classifier,
            "llm_explanation": self.llm_explanation,
            "credible_source": self.credible_source,
            "claim_extractor": self.claim_extractor,
            "context_analyzer": self.context_analyzer,
            "evidence_evaluator": self.evidence_evaluator
        }
        
        if agent_name not in agent_configs:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        return agent_configs[agent_name].copy()
    
    def update_agent_config(self, agent_name: str, **kwargs):
        """
        Update configuration for specific agent
        
        Args:
            agent_name: Name of the agent
            **kwargs: Configuration updates
        """
        if agent_name not in ["bert_classifier", "llm_explanation", "credible_source", 
                             "claim_extractor", "context_analyzer", "evidence_evaluator"]:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent_config = getattr(self, agent_name)
        agent_config.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all configurations to dictionary format"""
        return {
            "bert_classifier": self.bert_classifier,
            "llm_explanation": self.llm_explanation,
            "credible_source": self.credible_source,
            "claim_extractor": self.claim_extractor,
            "context_analyzer": self.context_analyzer,
            "evidence_evaluator": self.evidence_evaluator
        }
    
    def save_to_file(self, file_path: Optional[Path] = None):
        """Save all model configurations to JSON file"""
        if file_path is None:
            from .settings import get_settings
            settings = get_settings()
            file_path = settings.project_root / "config" / "model_configs.json"
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'ModelConfigs':
        """Load model configurations from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)

# Global model configs instance
_model_configs_instance: Optional[ModelConfigs] = None

def get_model_config(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get model configuration for specific agent or all agents
    
    Args:
        agent_name: Name of specific agent, or None for all configs
        
    Returns:
        Configuration dictionary
    """
    global _model_configs_instance
    if _model_configs_instance is None:
        _model_configs_instance = ModelConfigs()
    
    if agent_name is None:
        return _model_configs_instance.to_dict()
    else:
        return _model_configs_instance.get_agent_config(agent_name)

def update_model_config(agent_name: str, **kwargs):
    """
    Update model configuration for specific agent
    
    Args:
        agent_name: Name of the agent
        **kwargs: Configuration updates
    """
    global _model_configs_instance
    if _model_configs_instance is None:
        _model_configs_instance = ModelConfigs()
    
    _model_configs_instance.update_agent_config(agent_name, **kwargs)

# Predefined configuration profiles
PERFORMANCE_OPTIMIZED_CONFIG = {
    "bert_classifier": {
        "batch_size": 32,
        "enable_preprocessing": False,  # Skip for speed
        "cache_predictions": True
    },
    "llm_explanation": {
        "enable_detailed_analysis": False,
        "max_tokens": 1024,  # Shorter responses
        "rate_limit_seconds": 2.0
    },
    "context_analyzer": {
        "enable_detailed_analysis": False,
        "propaganda_analysis": False
    }
}

ACCURACY_OPTIMIZED_CONFIG = {
    "bert_classifier": {
        "enable_preprocessing": True,
        "batch_size": 8  # Smaller batches for stability
    },
    "llm_explanation": {
        "enable_detailed_analysis": True,
        "max_tokens": 4096,
        "temperature": 0.2  # More deterministic
    },
    "context_analyzer": {
        "enable_detailed_analysis": True,
        "enable_propaganda_analysis": True
    },
    "evidence_evaluator": {
        "enable_detailed_analysis": True,
        "enable_fallacy_detection": True
    }
}

def apply_config_profile(profile_name: str):
    """
    Apply predefined configuration profile
    
    Args:
        profile_name: "performance" or "accuracy"
    """
    profiles = {
        "performance": PERFORMANCE_OPTIMIZED_CONFIG,
        "accuracy": ACCURACY_OPTIMIZED_CONFIG
    }
    
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile: {profile_name}")
    
    profile = profiles[profile_name]
    for agent_name, agent_config in profile.items():
        update_model_config(agent_name, **agent_config)