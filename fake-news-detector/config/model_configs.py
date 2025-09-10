"""
Enhanced Model Configurations for Production Fake News Detection

Production-ready model configuration management with environment awareness,
performance optimization, and comprehensive agent settings.

Features:
- Environment-specific model configurations
- Performance and accuracy optimization profiles
- Enhanced safety settings management
- Dynamic configuration updates
- Comprehensive validation and monitoring
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfigs:
    """
    Enhanced Model Configuration Management
    
    Comprehensive configuration for all AI models and agents with:
    - Environment-aware settings
    - Performance optimization profiles
    - Enhanced safety configurations
    - Dynamic updates and validation
    """
    
    # Configuration metadata
    config_version: str = "3.2.0"
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Agent configurations
    bert_classifier: Dict[str, Any] = field(default_factory=dict)
    claim_extractor: Dict[str, Any] = field(default_factory=dict)
    context_analyzer: Dict[str, Any] = field(default_factory=dict)
    evidence_evaluator: Dict[str, Any] = field(default_factory=dict)
    credible_source: Dict[str, Any] = field(default_factory=dict)
    llm_explanation: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize all agent configurations with environment-aware defaults"""
        
        # Initialize each agent configuration if not provided
        if not self.bert_classifier:
            self.bert_classifier = self._get_bert_classifier_config()
        if not self.claim_extractor:
            self.claim_extractor = self._get_claim_extractor_config()
        if not self.context_analyzer:
            self.context_analyzer = self._get_context_analyzer_config()
        if not self.evidence_evaluator:
            self.evidence_evaluator = self._get_evidence_evaluator_config()
        if not self.credible_source:
            self.credible_source = self._get_credible_source_config()
        if not self.llm_explanation:
            self.llm_explanation = self._get_llm_explanation_config()
        
        # Apply environment-specific optimizations
        self._apply_environment_optimizations()
        
        # Validate all configurations
        self._validate_configurations()
        
        logger.info(f"Model configurations initialized for {self.environment} environment")

    def _get_bert_classifier_config(self) -> Dict[str, Any]:
        """Enhanced BERT Classifier configuration with environment awareness"""
        return {
            # Core model settings
            "model_name": "bert-base-uncased",
            "model_path": "models/bert_fake_news",
            "max_length": 512,
            "num_labels": 2,
            
            # Performance settings
            "batch_size": int(os.getenv('BERT_BATCH_SIZE', '16')),
            "device": os.getenv('BERT_DEVICE', 'auto'),
            "enable_preprocessing": True,
            "use_fast_tokenizer": True,
            
            # Preprocessing configuration
            "preprocessing_config": {
                "max_length": int(os.getenv('BERT_MAX_LENGTH', '2000')),
                "remove_urls": True,
                "remove_emails": True,
                "normalize_quotes": True,
                "remove_excessive_punctuation": True,
                "handle_special_characters": True,
                "lowercase_text": True,
                "remove_extra_whitespace": True
            },
            
            # Quality and confidence settings
            "prediction_threshold": float(os.getenv('BERT_THRESHOLD', '0.5')),
            "high_confidence_threshold": 0.8,
            "low_confidence_threshold": 0.6,
            "uncertain_prediction_threshold": 0.7,
            
            # Performance optimization
            "enable_metrics": True,
            "cache_predictions": self.environment in ["production", "staging"],
            "cache_ttl": 3600,  # 1 hour
            
            # LangGraph integration
            "state_key": "bert_classification",
            "next_agents": ["claim_extractor", "context_analyzer"],
            "processing_timeout": int(os.getenv('BERT_TIMEOUT', '30')),
            
            # Resource management
            "max_memory_usage_mb": int(os.getenv('BERT_MEMORY_LIMIT', '512')),
            "enable_gradient_checkpointing": self.environment == "production",
            
            # Error handling
            "enable_fallback_classification": True,
            "fallback_confidence": 0.5,
            "max_retries": 2
        }

    def _get_claim_extractor_config(self) -> Dict[str, Any]:
        """Enhanced Claim Extractor configuration"""
        return {
            # Model settings
            "model_name": os.getenv('CLAIM_MODEL', 'gemini-1.5-pro'),
            "temperature": float(os.getenv('CLAIM_TEMPERATURE', '0.3')),
            "max_tokens": int(os.getenv('CLAIM_MAX_TOKENS', '2048')),
            "top_p": 0.9,
            "top_k": 40,
            
            # Extraction settings
            "max_claims_per_article": int(os.getenv('MAX_CLAIMS', '8')),
            "min_claim_length": 10,
            "max_claim_length": 500,
            "enable_verification_analysis": True,
            "enable_claim_prioritization": True,
            "enable_claim_deduplication": True,
            
            # Pattern analysis
            "enable_pattern_preprocessing": True,
            "pattern_confidence_threshold": 0.5,
            "claim_richness_threshold": 5.0,
            "statistical_claim_detection": True,
            
            # Claim categorization
            "supported_claim_types": [
                "Statistical", "Event", "Attribution", "Research",
                "Policy", "Causal", "Medical", "Financial", "Other"
            ],
            "priority_levels": [1, 2, 3],  # 1=Critical, 2=Important, 3=Minor
            "auto_categorization_enabled": True,
            
            # Quality control
            "enable_fallback_parsing": True,
            "max_parsing_attempts": 3,
            "parsing_quality_threshold": 60,
            "validation_enabled": True,
            
            # API and performance
            "rate_limit_seconds": float(os.getenv('CLAIM_RATE_LIMIT', '1.0')),
            "max_retries": 3,
            "timeout_seconds": int(os.getenv('CLAIM_TIMEOUT', '45')),
            "enable_caching": True,
            "cache_ttl": 1800,  # 30 minutes
            
            # LangGraph integration
            "state_key": "extracted_claims",
            "next_agents": ["context_analyzer", "evidence_evaluator"],
            
            # Enhanced safety settings
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ],
            
            # Monitoring
            "enable_performance_tracking": True,
            "log_extraction_details": self.environment == "development"
        }

    def _get_context_analyzer_config(self) -> Dict[str, Any]:
        """Enhanced Context Analyzer configuration with LLM scoring"""
        return {
            # Model settings
            "model_name": os.getenv('CONTEXT_MODEL', 'gemini-1.5-pro'),
            "temperature": float(os.getenv('CONTEXT_TEMPERATURE', '0.4')),
            "max_tokens": int(os.getenv('CONTEXT_MAX_TOKENS', '3072')),
            "top_p": 0.9,
            
            # Analysis settings
            "enable_detailed_analysis": True,
            "enable_llm_scoring": True,  # Enhanced feature
            "llm_scoring_consistency_check": True,
            "bias_threshold": 5.0,
            "manipulation_threshold": 6.0,
            "enable_propaganda_analysis": True,
            "enable_sentiment_analysis": True,
            
            # Bias detection capabilities
            "bias_detection_modes": [
                "political_bias", "emotional_bias", "selection_bias",
                "linguistic_bias", "cultural_bias", "confirmation_bias"
            ],
            "emotional_analysis_depth": "comprehensive",
            "political_spectrum_analysis": True,
            
            # Manipulation detection
            "propaganda_techniques_count": 15,
            "fallacy_detection_enabled": True,
            "manipulation_scoring_algorithm": "weighted_average",
            "rhetorical_device_detection": True,
            
            # Enhanced scoring system
            "scoring_ranges": {
                "bias_score": [0, 100],
                "credibility_score": [0, 100],
                "risk_score": [0, 100],
                "manipulation_score": [0, 100]
            },
            "scoring_consistency_threshold": 0.15,  # Max deviation between scores
            
            # Pattern databases
            "pattern_database_size": {
                "bias_indicators": 250,
                "emotional_keywords": 200,
                "framing_patterns": 100,
                "linguistic_patterns": 80,
                "manipulation_techniques": 150
            },
            
            # Performance optimization
            "analysis_depth": "comprehensive" if self.environment == "development" else "standard",
            "enable_parallel_analysis": True,
            "max_analysis_time": int(os.getenv('CONTEXT_TIMEOUT', '60')),
            
            # API settings
            "rate_limit_seconds": float(os.getenv('CONTEXT_RATE_LIMIT', '1.0')),
            "max_retries": 3,
            "enable_caching": True,
            "cache_ttl": 2400,  # 40 minutes
            
            # LangGraph integration
            "state_key": "context_analysis",
            "next_agents": ["evidence_evaluator"],
            
            # Safety settings
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ],
            
            # Fallback and recovery
            "enable_safety_fallbacks": True,
            "fallback_scoring_enabled": True,
            "traditional_scoring_backup": True
        }

    def _get_evidence_evaluator_config(self) -> Dict[str, Any]:
        """Enhanced Evidence Evaluator with specific verification links"""
        return {
            # Model settings
            "model_name": os.getenv('EVIDENCE_MODEL', 'gemini-1.5-pro'),
            "temperature": float(os.getenv('EVIDENCE_TEMPERATURE', '0.3')),
            "max_tokens": int(os.getenv('EVIDENCE_MAX_TOKENS', '3072')),
            "top_p": 0.9,
            
            # Evaluation capabilities
            "enable_detailed_analysis": True,
            "enable_specific_verification_links": True,  # Enhanced feature
            "enable_institutional_fallbacks": True,  # Safety feature
            "evidence_threshold": 6.0,
            "enable_fallacy_detection": True,
            "enable_gap_analysis": True,
            "enable_source_cross_referencing": True,
            
            # Evidence analysis framework
            "evidence_types": [
                "statistical_evidence", "documentary_evidence",
                "testimonial_evidence", "circumstantial_evidence",
                "expert_opinion", "peer_reviewed_research",
                "government_data", "institutional_reports"
            ],
            
            # Source quality tiers
            "source_quality_tiers": {
                "tier_1": ["peer_reviewed", "government_official", "academic"],
                "tier_2": ["institutional", "expert_verified", "established_media"],
                "tier_3": ["journalistic", "verified_blogger", "citizen_journalism"],
                "tier_4": ["social_media", "unverified", "anonymous"]
            },
            
            # Verification link generation
            "verification_strategies": {
                "medical_claims": "pubmed_nejm_search",
                "statistical_claims": "government_data_search",
                "political_claims": "official_records_search",
                "scientific_claims": "academic_database_search",
                "business_claims": "financial_filings_search"
            },
            
            # Quality scoring weights
            "scoring_weights": {
                "source_quality": 0.4,
                "logical_consistency": 0.3,
                "evidence_completeness": 0.2,
                "verification_availability": 0.1
            },
            
            # Fallacy detection system
            "fallacy_types": [
                "ad_hominem", "straw_man", "false_dilemma", "appeal_to_authority",
                "circular_reasoning", "correlation_causation", "cherry_picking",
                "hasty_generalization", "appeal_to_emotion", "red_herring"
            ],
            "fallacy_confidence_threshold": 0.7,
            
            # Quality thresholds
            "quality_thresholds": {
                "excellent": 8.5,
                "good": 7.0,
                "acceptable": 5.5,
                "poor": 3.0,
                "very_poor": 1.0
            },
            
            # Performance settings
            "max_evaluation_time": int(os.getenv('EVIDENCE_TIMEOUT', '90')),
            "enable_parallel_evaluation": True,
            "max_concurrent_evaluations": 3,
            
            # API settings
            "rate_limit_seconds": float(os.getenv('EVIDENCE_RATE_LIMIT', '1.0')),
            "max_retries": 3,
            "enable_caching": True,
            "cache_ttl": 3600,  # 1 hour
            
            # LangGraph integration
            "state_key": "evidence_evaluation",
            "next_agents": ["credible_source"],
            
            # Safety settings
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ],
            
            # Enhanced safety handling
            "enable_safety_fallbacks": True,
            "institutional_fallback_sources": [
                "fact_checking_organizations",
                "academic_institutions",
                "government_agencies",
                "professional_journalism"
            ]
        }

    def _get_credible_source_config(self) -> Dict[str, Any]:
        """Enhanced Credible Source Agent with contextual recommendations"""
        return {
            # Model settings
            "model_name": os.getenv('SOURCE_MODEL', 'gemini-1.5-pro'),
            "temperature": float(os.getenv('SOURCE_TEMPERATURE', '0.2')),
            "max_tokens": int(os.getenv('SOURCE_MAX_TOKENS', '2048')),
            "top_p": 0.9,
            
            # Source recommendation capabilities
            "enable_contextual_recommendations": True,  # Enhanced feature
            "enable_domain_classification": True,
            "enable_expert_identification": True,
            "enable_cross_verification": True,
            "max_sources_per_recommendation": 10,
            "min_source_reliability_score": 6.0,
            
            # Contextual analysis
            "claim_context_matching": True,
            "domain_expertise_weighting": True,
            "geographical_relevance": True,
            "temporal_relevance": True,
            "language_preference_matching": True,
            
            # Source database management
            "source_database_size": 1000,  # Expanded database
            "enable_source_caching": True,
            "cache_ttl_hours": 24,
            "source_validation_frequency": "weekly",
            "auto_source_discovery": self.environment == "production",
            
            # Domain classification system
            "domain_categories": [
                "medical_health", "politics_government", "science_technology",
                "business_finance", "education", "environment", "sports",
                "entertainment", "social_issues", "international_affairs"
            ],
            "domain_confidence_threshold": 0.3,
            "max_domains_to_consider": 3,
            "cross_domain_validation": True,
            
            # Expert contact system
            "expert_contact_types": [
                "academic_researchers", "industry_professionals",
                "government_officials", "certified_experts",
                "fact_checkers", "investigative_journalists"
            ],
            "expert_verification_required": self.environment == "production",
            
            # Verification protocols
            "verification_protocols": {
                "basic": {"steps": 3, "sources": 2, "time_limit": 300},
                "standard": {"steps": 5, "sources": 3, "time_limit": 600},
                "comprehensive": {"steps": 8, "sources": 5, "time_limit": 1200}
            },
            "default_protocol": "standard",
            "protocol_auto_selection": True,
            
            # Quality assurance
            "source_quality_factors": [
                "domain_authority", "publication_frequency", "editorial_standards",
                "fact_checking_record", "expert_endorsements", "transparency_score"
            ],
            "min_quality_threshold": 7.0,
            "quality_decay_factor": 0.95,  # Annual quality decrease
            
            # Performance optimization
            "enable_parallel_processing": True,
            "max_concurrent_searches": 4,
            "search_timeout": int(os.getenv('SOURCE_TIMEOUT', '60')),
            "result_caching_enabled": True,
            
            # API settings
            "rate_limit_seconds": float(os.getenv('SOURCE_RATE_LIMIT', '1.0')),
            "max_retries": 3,
            "enable_caching": True,
            "cache_ttl": 1800,  # 30 minutes
            
            # LangGraph integration
            "state_key": "source_recommendations",
            "next_agents": ["llm_explanation"],
            
            # Safety settings
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ],
            
            # Enhanced safety handling
            "enable_safety_fallbacks": True,
            "contextual_safety_adaptation": True,
            "institutional_source_prioritization": True
        }

    def _get_llm_explanation_config(self) -> Dict[str, Any]:
        """Enhanced LLM Explanation configuration"""
        return {
            # Model settings
            "model_name": os.getenv('EXPLANATION_MODEL', 'gemini-1.5-pro'),
            "temperature": float(os.getenv('EXPLANATION_TEMPERATURE', '0.3')),
            "max_tokens": int(os.getenv('EXPLANATION_MAX_TOKENS', '3072')),
            "top_p": 0.9,
            "top_k": 40,
            
            # Explanation generation
            "enable_detailed_analysis": True,
            "enable_confidence_analysis": True,
            "enable_source_analysis": True,
            "enable_methodology_explanation": True,
            "confidence_threshold": 0.75,
            "explanation_depth": "comprehensive",
            
            # Content formatting
            "response_format": "structured_markdown",
            "include_evidence_summary": True,
            "include_verification_suggestions": True,
            "include_confidence_intervals": True,
            "include_limitations_section": True,
            "max_explanation_length": 2500,
            
            # Analysis integration
            "integrate_all_agent_results": True,
            "cross_reference_findings": True,
            "highlight_inconsistencies": True,
            "provide_uncertainty_analysis": True,
            
            # Quality assurance
            "explanation_quality_check": True,
            "factual_consistency_validation": True,
            "readability_optimization": True,
            "bias_neutrality_check": True,
            
            # Performance settings
            "generation_timeout": int(os.getenv('EXPLANATION_TIMEOUT', '75')),
            "enable_streaming": self.environment == "development",
            "enable_parallel_generation": False,  # Sequential for consistency
            
            # API settings
            "rate_limit_seconds": float(os.getenv('EXPLANATION_RATE_LIMIT', '1.0')),
            "max_retries": 3,
            "timeout_seconds": 90,
            "enable_caching": False,  # Fresh explanations preferred
            
            # LangGraph integration
            "state_key": "llm_explanation",
            "next_agents": [],  # Final agent
            "final_output_formatting": True,
            
            # Safety settings (most permissive for comprehensive analysis)
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ],
            
            # Monitoring and feedback
            "enable_explanation_feedback": True,
            "track_user_satisfaction": self.environment == "production",
            "continuous_improvement": True
        }

    def _apply_environment_optimizations(self):
        """Apply environment-specific optimizations to all agent configurations"""
        optimizations = {
            "development": {
                "enable_detailed_logging": True,
                "reduce_rate_limits": True,
                "enable_debug_output": True,
                "cache_ttl_multiplier": 0.5
            },
            "testing": {
                "disable_caching": True,
                "reduce_timeouts": True,
                "minimal_retries": True,
                "fast_fallbacks": True
            },
            "staging": {
                "moderate_caching": True,
                "standard_timeouts": True,
                "full_logging": True,
                "performance_monitoring": True
            },
            "production": {
                "optimize_for_speed": True,
                "enable_aggressive_caching": True,
                "minimize_logging": True,
                "maximize_reliability": True
            }
        }

        if self.environment in optimizations:
            env_opts = optimizations[self.environment]
            
            # Apply optimizations to all agent configs
            for agent_name in self.get_all_agent_names():
                agent_config = getattr(self, agent_name)
                
                if env_opts.get("reduce_rate_limits") and "rate_limit_seconds" in agent_config:
                    agent_config["rate_limit_seconds"] *= 0.5
                    
                if env_opts.get("reduce_timeouts") and "timeout_seconds" in agent_config:
                    agent_config["timeout_seconds"] = min(agent_config.get("timeout_seconds", 30), 15)
                    
                if env_opts.get("disable_caching"):
                    agent_config["enable_caching"] = False
                    
                if env_opts.get("minimal_retries"):
                    agent_config["max_retries"] = 0

    def _validate_configurations(self):
        """Comprehensive validation of all agent configurations"""
        validation_errors = []
        
        for agent_name in self.get_all_agent_names():
            agent_config = getattr(self, agent_name)
            
            # Validate required fields
            required_fields = ["model_name", "temperature", "max_tokens"]
            for field in required_fields:
                if field not in agent_config:
                    validation_errors.append(f"{agent_name}: Missing required field '{field}'")
            
            # Validate value ranges
            if "temperature" in agent_config:
                temp = agent_config["temperature"]
                if not 0.0 <= temp <= 2.0:
                    validation_errors.append(f"{agent_name}: Invalid temperature {temp} (must be 0.0-2.0)")
            
            if "max_tokens" in agent_config:
                tokens = agent_config["max_tokens"]
                if not 100 <= tokens <= 8192:
                    validation_errors.append(f"{agent_name}: Invalid max_tokens {tokens} (must be 100-8192)")
        
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
            logger.error(error_msg)
            if self.environment == "production":
                raise ValueError(error_msg)
            else:
                logger.warning("Continuing with invalid configuration in non-production environment")

    def get_all_agent_names(self) -> List[str]:
        """Get list of all agent names"""
        return [
            "bert_classifier", "claim_extractor", "context_analyzer",
            "evidence_evaluator", "credible_source", "llm_explanation"
        ]

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific agent with runtime validation
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Deep copy of agent configuration
        """
        if agent_name not in self.get_all_agent_names():
            raise ValueError(f"Unknown agent: {agent_name}. Available: {self.get_all_agent_names()}")
        
        config = getattr(self, agent_name).copy()
        
        # Add runtime metadata
        config["_runtime_metadata"] = {
            "retrieved_at": datetime.now().isoformat(),
            "config_version": self.config_version,
            "environment": self.environment
        }
        
        return config

    def update_agent_config(self, agent_name: str, **kwargs):
        """
        Update configuration for specific agent with validation
        
        Args:
            agent_name: Name of the agent
            **kwargs: Configuration updates
        """
        if agent_name not in self.get_all_agent_names():
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent_config = getattr(self, agent_name)
        
        # Validate updates
        for key, value in kwargs.items():
            if key in ["temperature"] and not 0.0 <= value <= 2.0:
                raise ValueError(f"Invalid {key} value: {value}")
            if key in ["max_tokens"] and not 100 <= value <= 8192:
                raise ValueError(f"Invalid {key} value: {value}")
        
        # Apply updates
        agent_config.update(kwargs)
        logger.info(f"Updated {agent_name} configuration with {len(kwargs)} changes")

    def get_performance_profile(self) -> Dict[str, Any]:
        """Get performance characteristics of current configuration"""
        profile = {
            "configuration_version": self.config_version,
            "environment": self.environment,
            "agents": {}
        }
        
        for agent_name in self.get_all_agent_names():
            agent_config = getattr(self, agent_name)
            profile["agents"][agent_name] = {
                "model": agent_config.get("model_name"),
                "timeout": agent_config.get("timeout_seconds", agent_config.get("max_analysis_time", 30)),
                "caching_enabled": agent_config.get("enable_caching", False),
                "rate_limit": agent_config.get("rate_limit_seconds", 1.0),
                "enhanced_features": self._get_enhanced_features(agent_name, agent_config)
            }
        
        return profile

    def _get_enhanced_features(self, agent_name: str, config: Dict[str, Any]) -> List[str]:
        """Get list of enhanced features enabled for an agent"""
        features = []
        
        # Common enhanced features
        if config.get("enable_detailed_analysis"):
            features.append("detailed_analysis")
        if config.get("enable_caching"):
            features.append("caching")
        if config.get("enable_safety_fallbacks"):
            features.append("safety_fallbacks")
        
        # Agent-specific enhanced features
        if agent_name == "context_analyzer":
            if config.get("enable_llm_scoring"):
                features.append("llm_scoring")
            if config.get("llm_scoring_consistency_check"):
                features.append("scoring_consistency")
        elif agent_name == "evidence_evaluator":
            if config.get("enable_specific_verification_links"):
                features.append("specific_verification_links")
            if config.get("enable_institutional_fallbacks"):
                features.append("institutional_fallbacks")
        elif agent_name == "credible_source":
            if config.get("enable_contextual_recommendations"):
                features.append("contextual_recommendations")
            if config.get("enable_expert_identification"):
                features.append("expert_identification")
        
        return features

    def to_dict(self) -> Dict[str, Any]:
        """Convert all configurations to dictionary format"""
        return {
            "config_metadata": {
                "version": self.config_version,
                "environment": self.environment,
                "created_at": self.created_at
            },
            "agent_configurations": {
                agent_name: getattr(self, agent_name)
                for agent_name in self.get_all_agent_names()
            }
        }


# Global model configs instance with thread-safe initialization
_model_configs_instance: Optional[ModelConfigs] = None

def get_model_config(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get model configuration for specific agent or all agents
    
    Args:
        agent_name: Name of specific agent, or None for all configs
        
    Returns:
        Configuration dictionary or complete configuration
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

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary of current configuration"""
    global _model_configs_instance
    
    if _model_configs_instance is None:
        _model_configs_instance = ModelConfigs()
    
    return _model_configs_instance.get_performance_profile()

def reset_model_configs():
    """Reset model configurations to defaults"""
    global _model_configs_instance
    _model_configs_instance = ModelConfigs()


# Configuration profiles for different use cases
CONFIGURATION_PROFILES = {
    "performance_optimized": {
        "description": "Optimized for speed and resource efficiency",
        "bert_classifier": {"batch_size": 32, "enable_preprocessing": False, "cache_predictions": True},
        "claim_extractor": {"max_claims_per_article": 5, "enable_detailed_analysis": False},
        "context_analyzer": {"analysis_depth": "standard", "enable_llm_scoring": False},
        "evidence_evaluator": {"enable_detailed_analysis": False, "max_evaluation_time": 45},
        "credible_source": {"max_sources_per_recommendation": 5, "default_protocol": "basic"},
        "llm_explanation": {"enable_detailed_analysis": False, "max_tokens": 1024}
    },
    
    "accuracy_optimized": {
        "description": "Optimized for maximum accuracy and thoroughness",
        "bert_classifier": {"batch_size": 8, "enable_preprocessing": True},
        "claim_extractor": {"max_claims_per_article": 10, "enable_verification_analysis": True},
        "context_analyzer": {"analysis_depth": "comprehensive", "enable_llm_scoring": True},
        "evidence_evaluator": {"enable_detailed_analysis": True, "enable_fallacy_detection": True},
        "credible_source": {"max_sources_per_recommendation": 15, "default_protocol": "comprehensive"},
        "llm_explanation": {"enable_detailed_analysis": True, "explanation_depth": "comprehensive"}
    },
    
    "balanced": {
        "description": "Balanced configuration for general use",
        "bert_classifier": {"batch_size": 16, "enable_preprocessing": True, "cache_predictions": True},
        "claim_extractor": {"max_claims_per_article": 8, "enable_verification_analysis": True},
        "context_analyzer": {"analysis_depth": "standard", "enable_llm_scoring": True},
        "evidence_evaluator": {"enable_detailed_analysis": True, "max_evaluation_time": 60},
        "credible_source": {"max_sources_per_recommendation": 10, "default_protocol": "standard"},
        "llm_explanation": {"enable_detailed_analysis": True, "max_tokens": 2500}
    }
}

def apply_configuration_profile(profile_name: str):
    """
    Apply a predefined configuration profile
    
    Args:
        profile_name: Name of the profile ("performance_optimized", "accuracy_optimized", "balanced")
    """
    if profile_name not in CONFIGURATION_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(CONFIGURATION_PROFILES.keys())}")
    
    profile = CONFIGURATION_PROFILES[profile_name]
    
    logger.info(f"Applying configuration profile: {profile_name}")
    logger.info(f"Description: {profile['description']}")
    
    # Apply profile settings to each agent
    for agent_name, agent_settings in profile.items():
        if agent_name != "description":
            update_model_config(agent_name, **agent_settings)
    
    logger.info(f"Configuration profile '{profile_name}' applied successfully")

def get_available_profiles() -> Dict[str, str]:
    """Get available configuration profiles with descriptions"""
    return {name: profile.get("description", "No description")
            for name, profile in CONFIGURATION_PROFILES.items()}


# Initialize with environment-appropriate profile
def _initialize_with_profile():
    """Initialize with appropriate profile based on environment"""
    try:
        environment = os.getenv('ENVIRONMENT', 'development')
        profile_map = {
            'development': 'balanced',
            'testing': 'performance_optimized',
            'staging': 'balanced',
            'production': 'accuracy_optimized'
        }
        
        default_profile = profile_map.get(environment, 'balanced')
        
        # Only apply if CONFIGURATION_PROFILE env var is not set
        if not os.getenv('CONFIGURATION_PROFILE'):
            apply_configuration_profile(default_profile)
            logger.info(f"Auto-applied {default_profile} profile for {environment} environment")
            
    except Exception as e:
        logger.warning(f"Could not auto-apply configuration profile: {e}")


# Auto-initialize on import
try:
    _initialize_with_profile()
except Exception as e:
    logger.warning(f"Configuration profile initialization failed: {e}")
