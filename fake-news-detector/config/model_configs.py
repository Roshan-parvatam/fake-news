# config/model_configs.py

"""
Enhanced Model Configurations for Production Fake News Detection

Modern, modular model configuration management with clean architecture,
comprehensive validation, and environment-aware optimization.

Features:
- Modular configuration classes with inheritance
- Environment-specific optimization profiles
- Comprehensive validation with clear error messages
- Dynamic configuration updates with validation
- Performance profiling and monitoring
- Clean separation of concerns

Version: 3.2.0 - Modern Production Edition
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    BERT = "bert"
    GEMINI = "gemini"
    OPENAI = "openai"


class SafetyThreshold(Enum):
    """Safety threshold levels."""
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"


@dataclass
class SafetySettings:
    """Safety configuration for LLM models."""
    harassment_threshold: SafetyThreshold = SafetyThreshold.BLOCK_ONLY_HIGH
    hate_speech_threshold: SafetyThreshold = SafetyThreshold.BLOCK_ONLY_HIGH
    sexually_explicit_threshold: SafetyThreshold = SafetyThreshold.BLOCK_ONLY_HIGH
    dangerous_content_threshold: SafetyThreshold = SafetyThreshold.BLOCK_ONLY_HIGH

    def to_dict(self) -> List[Dict[str, str]]:
        """Convert to API-compatible format."""
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": self.harassment_threshold.value},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": self.hate_speech_threshold.value},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": self.sexually_explicit_threshold.value},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": self.dangerous_content_threshold.value}
        ]


@dataclass
class BaseAgentConfig(ABC):
    """Base configuration class for all agents."""
    
    # Core settings
    model_name: str
    model_type: ModelType
    timeout_seconds: int = 30
    max_retries: int = 3
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 1800
    rate_limit_seconds: float = 1.0
    
    # LangGraph integration
    state_key: str = ""
    next_agents: List[str] = field(default_factory=list)
    
    # Monitoring
    enable_metrics: bool = True
    enable_performance_tracking: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    @abstractmethod
    def validate(self):
        """Validate configuration parameters."""
        # Common validations
        if self.timeout_seconds < 5 or self.timeout_seconds > 300:
            raise ValueError(f"timeout_seconds must be between 5 and 300, got {self.timeout_seconds}")
        
        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError(f"max_retries must be between 0 and 10, got {self.max_retries}")
        
        if self.rate_limit_seconds < 0:
            raise ValueError(f"rate_limit_seconds must be positive, got {self.rate_limit_seconds}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and hasattr(value[0], 'to_dict'):
                result[key] = [item.to_dict() for item in value]
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**data)


@dataclass
class LLMAgentConfig(BaseAgentConfig):
    """Base configuration for LLM-based agents."""
    
    # LLM-specific parameters
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    
    # Safety settings
    safety_settings: SafetySettings = field(default_factory=SafetySettings)
    enable_safety_fallbacks: bool = True

    def validate(self):
        """Validate LLM-specific parameters."""
        super().validate()
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        
        if self.max_tokens < 100 or self.max_tokens > 8192:
            raise ValueError(f"max_tokens must be between 100 and 8192, got {self.max_tokens}")
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")


@dataclass
class BERTClassifierConfig(BaseAgentConfig):
    """Configuration for BERT Classifier agent."""
    
    model_type: ModelType = ModelType.BERT
    model_path: str = "models/bert_fake_news"
    max_length: int = 512
    batch_size: int = 16
    device: str = "auto"
    
    # Preprocessing settings
    enable_preprocessing: bool = True
    preprocessing_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_length": 2000,
        "remove_urls": True,
        "remove_emails": True,
        "normalize_quotes": True,
        "remove_excessive_punctuation": True,
        "handle_special_characters": True
    })
    
    # Thresholds
    prediction_threshold: float = 0.5
    high_confidence_threshold: float = 0.8
    
    # Performance optimization
    enable_gradient_checkpointing: bool = False
    use_fast_tokenizer: bool = True
    
    def __post_init__(self):
        """Set default state key and next agents."""
        if not self.state_key:
            self.state_key = "bert_classification"
        if not self.next_agents:
            self.next_agents = ["claim_extractor", "context_analyzer"]
        super().__post_init__()

    def validate(self):
        """Validate BERT-specific parameters."""
        super().validate()
        
        if self.max_length < 50 or self.max_length > 2048:
            raise ValueError(f"max_length must be between 50 and 2048, got {self.max_length}")
        
        if self.batch_size < 1 or self.batch_size > 128:
            raise ValueError(f"batch_size must be between 1 and 128, got {self.batch_size}")
        
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError(f"prediction_threshold must be between 0.0 and 1.0, got {self.prediction_threshold}")


@dataclass
class ClaimExtractorConfig(LLMAgentConfig):
    """Configuration for Claim Extractor agent."""
    
    model_type: ModelType = ModelType.GEMINI
    
    # Extraction settings
    max_claims_per_article: int = 8
    min_claim_length: int = 10
    max_claim_length: int = 500
    enable_verification_analysis: bool = True
    enable_claim_prioritization: bool = True
    
    # Claim types and priorities
    supported_claim_types: List[str] = field(default_factory=lambda: [
        "Statistical", "Event", "Attribution", "Research",
        "Policy", "Causal", "Medical", "Financial", "Other"
    ])
    
    # Quality control
    parsing_quality_threshold: int = 60
    max_parsing_attempts: int = 3
    
    def __post_init__(self):
        """Set default configuration."""
        if not self.state_key:
            self.state_key = "extracted_claims"
        if not self.next_agents:
            self.next_agents = ["context_analyzer", "evidence_evaluator"]
        super().__post_init__()

    def validate(self):
        """Validate claim extractor parameters."""
        super().validate()
        
        if self.max_claims_per_article < 1 or self.max_claims_per_article > 20:
            raise ValueError(f"max_claims_per_article must be between 1 and 20, got {self.max_claims_per_article}")


@dataclass
class ContextAnalyzerConfig(LLMAgentConfig):
    """Configuration for Context Analyzer agent."""
    
    model_type: ModelType = ModelType.GEMINI
    
    # Analysis settings
    enable_detailed_analysis: bool = True
    enable_llm_scoring: bool = True
    enable_propaganda_analysis: bool = True
    enable_sentiment_analysis: bool = True
    
    # Thresholds
    bias_threshold: float = 5.0
    manipulation_threshold: float = 6.0
    scoring_consistency_threshold: float = 0.15
    
    # Analysis modes
    bias_detection_modes: List[str] = field(default_factory=lambda: [
        "political_bias", "emotional_bias", "selection_bias",
        "linguistic_bias", "cultural_bias", "confirmation_bias"
    ])
    
    # Scoring ranges
    scoring_ranges: Dict[str, List[int]] = field(default_factory=lambda: {
        "bias_score": [0, 100],
        "credibility_score": [0, 100],
        "risk_score": [0, 100],
        "manipulation_score": [0, 100]
    })
    
    def __post_init__(self):
        """Set default configuration."""
        if not self.state_key:
            self.state_key = "context_analysis"
        if not self.next_agents:
            self.next_agents = ["evidence_evaluator"]
        super().__post_init__()

    def validate(self):
        """Validate context analyzer parameters."""
        super().validate()
        
        if not 0.0 <= self.bias_threshold <= 10.0:
            raise ValueError(f"bias_threshold must be between 0.0 and 10.0, got {self.bias_threshold}")


@dataclass
class EvidenceEvaluatorConfig(LLMAgentConfig):
    """Configuration for Evidence Evaluator agent."""
    
    model_type: ModelType = ModelType.GEMINI
    
    # Evaluation settings
    enable_detailed_analysis: bool = True
    enable_specific_verification_links: bool = True
    enable_institutional_fallbacks: bool = True
    enable_fallacy_detection: bool = True
    
    # Quality thresholds
    evidence_threshold: float = 6.0
    fallacy_confidence_threshold: float = 0.7
    
    # Source quality tiers
    source_quality_tiers: Dict[str, List[str]] = field(default_factory=lambda: {
        "tier_1": ["peer_reviewed", "government_official", "academic"],
        "tier_2": ["institutional", "expert_verified", "established_media"],
        "tier_3": ["journalistic", "verified_blogger", "citizen_journalism"],
        "tier_4": ["social_media", "unverified", "anonymous"]
    })
    
    # Fallacy types
    fallacy_types: List[str] = field(default_factory=lambda: [
        "ad_hominem", "straw_man", "false_dilemma", "appeal_to_authority",
        "circular_reasoning", "correlation_causation", "cherry_picking",
        "hasty_generalization", "appeal_to_emotion", "red_herring"
    ])
    
    def __post_init__(self):
        """Set default configuration."""
        if not self.state_key:
            self.state_key = "evidence_evaluation"
        if not self.next_agents:
            self.next_agents = ["credible_source"]
        super().__post_init__()


@dataclass
class CredibleSourceConfig(LLMAgentConfig):
    """Configuration for Credible Source agent."""
    
    model_type: ModelType = ModelType.GEMINI
    
    # Recommendation settings
    enable_contextual_recommendations: bool = True
    enable_domain_classification: bool = True
    enable_expert_identification: bool = True
    max_sources_per_recommendation: int = 10
    min_source_reliability_score: float = 6.0
    
    # Domain categories
    domain_categories: List[str] = field(default_factory=lambda: [
        "medical_health", "politics_government", "science_technology",
        "business_finance", "education", "environment", "sports",
        "entertainment", "social_issues", "international_affairs"
    ])
    
    # Expert contact types
    expert_contact_types: List[str] = field(default_factory=lambda: [
        "academic_researchers", "industry_professionals",
        "government_officials", "certified_experts",
        "fact_checkers", "investigative_journalists"
    ])
    
    def __post_init__(self):
        """Set default configuration."""
        if not self.state_key:
            self.state_key = "source_recommendations"
        if not self.next_agents:
            self.next_agents = ["llm_explanation"]
        super().__post_init__()


@dataclass
class LLMExplanationConfig(LLMAgentConfig):
    """Configuration for LLM Explanation agent."""
    
    model_type: ModelType = ModelType.GEMINI
    
    # Generation settings
    enable_detailed_analysis: bool = True
    enable_confidence_analysis: bool = True
    enable_source_analysis: bool = True
    enable_methodology_explanation: bool = True
    
    # Content formatting
    response_format: str = "structured_markdown"
    include_evidence_summary: bool = True
    include_verification_suggestions: bool = True
    include_confidence_intervals: bool = True
    max_explanation_length: int = 2500
    
    # Quality assurance
    explanation_quality_check: bool = True
    factual_consistency_validation: bool = True
    readability_optimization: bool = True
    bias_neutrality_check: bool = True
    
    def __post_init__(self):
        """Set default configuration."""
        if not self.state_key:
            self.state_key = "llm_explanation"
        if not self.next_agents:
            self.next_agents = []  # Final agent
        super().__post_init__()


class ConfigurationManager:
    """
    Central configuration manager for all agents.
    
    Provides environment-aware configuration management, validation,
    and dynamic updates with proper error handling.
    """

    # Agent configuration classes mapping
    AGENT_CONFIGS = {
        "bert_classifier": BERTClassifierConfig,
        "claim_extractor": ClaimExtractorConfig,
        "context_analyzer": ContextAnalyzerConfig,
        "evidence_evaluator": EvidenceEvaluatorConfig,
        "credible_source": CredibleSourceConfig,
        "llm_explanation": LLMExplanationConfig
    }

    def __init__(self, environment: str = None):
        """
        Initialize configuration manager.
        
        Args:
            environment: Target environment (development, testing, staging, production)
        """
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config_version = "3.2.0"
        self.created_at = datetime.now().isoformat()
        
        # Initialize agent configurations
        self._configs: Dict[str, BaseAgentConfig] = {}
        self._load_default_configurations()
        self._apply_environment_optimizations()

    def _load_default_configurations(self):
        """Load default configurations for all agents."""
        for agent_name, config_class in self.AGENT_CONFIGS.items():
            try:
                # Get environment-specific model name
                model_name = self._get_default_model_name(agent_name)
                
                # Create configuration with defaults
                if config_class == BERTClassifierConfig:
                    config = config_class(
                        model_name=model_name,
                        model_type=ModelType.BERT
                    )
                else:
                    config = config_class(
                        model_name=model_name,
                        model_type=ModelType.GEMINI
                    )
                
                self._configs[agent_name] = config
                logger.debug(f"Loaded default configuration for {agent_name}")
                
            except Exception as e:
                logger.error(f"Failed to load configuration for {agent_name}: {e}")
                raise

    def _get_default_model_name(self, agent_name: str) -> str:
        """Get default model name for agent based on environment."""
        defaults = {
            "bert_classifier": "bert-base-uncased",
            "claim_extractor": os.getenv('CLAIM_MODEL', 'gemini-1.5-pro'),
            "context_analyzer": os.getenv('CONTEXT_MODEL', 'gemini-1.5-pro'),
            "evidence_evaluator": os.getenv('EVIDENCE_MODEL', 'gemini-1.5-pro'),
            "credible_source": os.getenv('SOURCE_MODEL', 'gemini-1.5-pro'),
            "llm_explanation": os.getenv('EXPLANATION_MODEL', 'gemini-1.5-pro')
        }
        return defaults.get(agent_name, 'gemini-1.5-pro')

    def _apply_environment_optimizations(self):
        """Apply environment-specific optimizations."""
        optimizations = {
            "development": {
                "timeout_multiplier": 1.5,
                "enable_detailed_logging": True,
                "cache_ttl_multiplier": 0.5
            },
            "testing": {
                "timeout_multiplier": 0.5,
                "enable_caching": False,
                "max_retries": 0
            },
            "staging": {
                "timeout_multiplier": 1.0,
                "enable_caching": True,
                "cache_ttl_multiplier": 1.0
            },
            "production": {
                "timeout_multiplier": 1.0,
                "enable_caching": True,
                "cache_ttl_multiplier": 2.0
            }
        }

        env_opts = optimizations.get(self.environment, optimizations["development"])
        
        for agent_name, config in self._configs.items():
            # Apply timeout adjustments
            if "timeout_multiplier" in env_opts:
                config.timeout_seconds = int(config.timeout_seconds * env_opts["timeout_multiplier"])
            
            # Apply caching settings
            if "enable_caching" in env_opts:
                config.enable_caching = env_opts["enable_caching"]
            
            # Apply cache TTL adjustments
            if "cache_ttl_multiplier" in env_opts:
                config.cache_ttl = int(config.cache_ttl * env_opts["cache_ttl_multiplier"])
            
            # Apply retry settings
            if "max_retries" in env_opts:
                config.max_retries = env_opts["max_retries"]

        logger.info(f"Applied {self.environment} environment optimizations")

    def get_config(self, agent_name: str) -> BaseAgentConfig:
        """
        Get configuration for specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent configuration
            
        Raises:
            KeyError: If agent name is not found
        """
        if agent_name not in self._configs:
            raise KeyError(f"Unknown agent: {agent_name}. Available: {list(self._configs.keys())}")
        
        return self._configs[agent_name]

    def update_config(self, agent_name: str, **kwargs):
        """
        Update configuration for specific agent.
        
        Args:
            agent_name: Name of the agent
            **kwargs: Configuration parameters to update
            
        Raises:
            KeyError: If agent name is not found
            ValueError: If validation fails
        """
        if agent_name not in self._configs:
            raise KeyError(f"Unknown agent: {agent_name}")
        
        config = self._configs[agent_name]
        
        # Apply updates
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter '{key}' for {agent_name}")
        
        # Re-validate configuration
        try:
            config.validate()
            logger.info(f"Updated {agent_name} configuration: {list(kwargs.keys())}")
        except Exception as e:
            logger.error(f"Configuration validation failed for {agent_name}: {e}")
            raise

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all agent configurations as dictionaries."""
        return {
            agent_name: config.to_dict()
            for agent_name, config in self._configs.items()
        }

    def validate_all_configs(self) -> Dict[str, Any]:
        """
        Validate all agent configurations.
        
        Returns:
            Validation results with status and issues
        """
        results = {
            "overall_status": "valid",
            "agent_results": {},
            "issues": [],
            "validated_at": datetime.now().isoformat()
        }

        for agent_name, config in self._configs.items():
            try:
                config.validate()
                results["agent_results"][agent_name] = {"status": "valid"}
            except Exception as e:
                results["agent_results"][agent_name] = {
                    "status": "invalid",
                    "error": str(e)
                }
                results["issues"].append(f"{agent_name}: {str(e)}")

        if results["issues"]:
            results["overall_status"] = "invalid"

        return results

    def get_performance_profile(self) -> Dict[str, Any]:
        """Get performance profile of current configuration."""
        profile = {
            "environment": self.environment,
            "config_version": self.config_version,
            "agents": {}
        }

        for agent_name, config in self._configs.items():
            agent_profile = {
                "model_name": config.model_name,
                "model_type": config.model_type.value,
                "timeout_seconds": config.timeout_seconds,
                "caching_enabled": config.enable_caching,
                "rate_limit": config.rate_limit_seconds
            }

            # Add LLM-specific info
            if isinstance(config, LLMAgentConfig):
                agent_profile.update({
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "safety_fallbacks": config.enable_safety_fallbacks
                })

            profile["agents"][agent_name] = agent_profile

        return profile

    def export_config(self, file_path: Optional[Path] = None) -> Path:
        """Export all configurations to JSON file."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = Path(f"model_configs_{timestamp}.json")

        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment,
                "config_version": self.config_version
            },
            "configurations": self.get_all_configs()
        }

        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Configurations exported to {file_path}")
        return file_path


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_model_config(agent_name: Optional[str] = None) -> Union[Dict[str, Any], BaseAgentConfig]:
    """
    Get model configuration for specific agent or all agents.
    
    Args:
        agent_name: Name of specific agent, or None for all configs
        
    Returns:
        Configuration dictionary or BaseAgentConfig instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()

    if agent_name is None:
        return _config_manager.get_all_configs()
    else:
        return _config_manager.get_config(agent_name).to_dict()


def update_model_config(agent_name: str, **kwargs):
    """Update model configuration for specific agent."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()

    _config_manager.update_config(agent_name, **kwargs)


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary of current configuration."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()

    return _config_manager.get_performance_profile()


def validate_all_configurations() -> Dict[str, Any]:
    """Validate all model configurations."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()

    return _config_manager.validate_all_configs()


def reset_model_configs():
    """Reset model configurations to defaults."""
    global _config_manager
    _config_manager = ConfigurationManager()


# Configuration profiles for different use cases
class ConfigurationProfile(Enum):
    """Available configuration profiles."""
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    ACCURACY_OPTIMIZED = "accuracy_optimized"
    BALANCED = "balanced"


def apply_configuration_profile(profile: ConfigurationProfile):
    """
    Apply a predefined configuration profile.
    
    Args:
        profile: Configuration profile to apply
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()

    profile_settings = {
        ConfigurationProfile.PERFORMANCE_OPTIMIZED: {
            "bert_classifier": {"batch_size": 32, "enable_preprocessing": False},
            "claim_extractor": {"max_claims_per_article": 5, "timeout_seconds": 30},
            "context_analyzer": {"enable_detailed_analysis": False, "timeout_seconds": 45},
            "evidence_evaluator": {"enable_detailed_analysis": False, "timeout_seconds": 45},
            "credible_source": {"max_sources_per_recommendation": 5, "timeout_seconds": 30},
            "llm_explanation": {"enable_detailed_analysis": False, "max_tokens": 1024}
        },
        ConfigurationProfile.ACCURACY_OPTIMIZED: {
            "bert_classifier": {"batch_size": 8, "enable_preprocessing": True},
            "claim_extractor": {"max_claims_per_article": 10, "timeout_seconds": 60},
            "context_analyzer": {"enable_detailed_analysis": True, "timeout_seconds": 90},
            "evidence_evaluator": {"enable_detailed_analysis": True, "timeout_seconds": 120},
            "credible_source": {"max_sources_per_recommendation": 15, "timeout_seconds": 90},
            "llm_explanation": {"enable_detailed_analysis": True, "max_tokens": 3072}
        },
        ConfigurationProfile.BALANCED: {
            "bert_classifier": {"batch_size": 16, "enable_preprocessing": True},
            "claim_extractor": {"max_claims_per_article": 8, "timeout_seconds": 45},
            "context_analyzer": {"enable_detailed_analysis": True, "timeout_seconds": 60},
            "evidence_evaluator": {"enable_detailed_analysis": True, "timeout_seconds": 90},
            "credible_source": {"max_sources_per_recommendation": 10, "timeout_seconds": 60},
            "llm_explanation": {"enable_detailed_analysis": True, "max_tokens": 2500}
        }
    }

    settings = profile_settings[profile]
    logger.info(f"Applying {profile.value} configuration profile")

    for agent_name, agent_settings in settings.items():
        try:
            _config_manager.update_config(agent_name, **agent_settings)
        except Exception as e:
            logger.error(f"Failed to apply profile settings for {agent_name}: {e}")

    logger.info(f"Configuration profile '{profile.value}' applied successfully")


# Auto-initialize with environment-appropriate profile
def _auto_initialize():
    """Auto-initialize with appropriate profile based on environment."""
    try:
        environment = os.getenv('ENVIRONMENT', 'development')
        profile_map = {
            'development': ConfigurationProfile.BALANCED,
            'testing': ConfigurationProfile.PERFORMANCE_OPTIMIZED,
            'staging': ConfigurationProfile.BALANCED,
            'production': ConfigurationProfile.ACCURACY_OPTIMIZED
        }

        default_profile = profile_map.get(environment, ConfigurationProfile.BALANCED)
        
        # Only apply if CONFIGURATION_PROFILE env var is not set
        if not os.getenv('CONFIGURATION_PROFILE'):
            apply_configuration_profile(default_profile)
            logger.info(f"Auto-applied {default_profile.value} profile for {environment} environment")

    except Exception as e:
        logger.warning(f"Auto-initialization failed: {e}")


# Initialize on import
try:
    _auto_initialize()
except Exception as e:
    logger.warning(f"Configuration auto-initialization failed: {e}")
