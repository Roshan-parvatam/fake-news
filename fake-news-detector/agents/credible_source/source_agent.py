# agents/credible_source/source_agent.py
"""
Enhanced Credible Source Agent - Main Implementation with Config Integration

This agent provides source recommendations and credibility assessments for fact-checking
and verification, analyzing source reliability and providing actionable verification guidance
with full configuration integration and modular architecture.

Features:
- Configuration integration from config files
- Centralized prompt management
- Multi-tier source reliability assessment
- Domain-specific source recommendations
- Cross-verification strategies
- Performance tracking and metrics
- LangGraph integration ready
"""

import os
import google.generativeai as genai
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import base agent functionality
from agents.base.base_agent import BaseAgent

# Import modular components
from .source_database import SourceReliabilityDatabase
from .domain_classifier import DomainClassifier

# ✅ IMPORT CONFIGURATION FILES
from config import get_model_config, get_prompt_template, get_settings
from utils.helpers import sanitize_text

class CredibleSourceAgent(BaseAgent):
    """
    🔍 ENHANCED CREDIBLE SOURCE AGENT WITH CONFIG INTEGRATION
    
    Modular source recommendation agent that inherits from BaseAgent
    for consistent interface and LangGraph compatibility.
    
    Features:
    - Inherits from BaseAgent for consistent interface
    - Configuration integration from config files
    - Modular component architecture (source database, domain classification)
    - AI-powered source analysis with systematic reliability assessment
    - Multi-tier source recommendations (primary, expert, institutional)
    - Cross-verification strategies and fact-checking guidance
    - Performance tracking and metrics
    - LangGraph integration ready
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced credible source agent with config integration
        
        Args:
            config: Configuration dictionary for runtime overrides
        """
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        source_config = get_model_config('credible_source')
        system_settings = get_settings()
        
        # Merge with runtime overrides
        if config:
            source_config.update(config)

        self.agent_name = "credible_source"
        
        # Initialize base agent with merged config
        super().__init__(source_config)
        
        # ✅ USE CONFIG VALUES FOR AI MODEL SETTINGS
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)  # Lower for consistent recommendations
        self.max_tokens = self.config.get('max_tokens', 2048)
        
        # ✅ SOURCE RECOMMENDATION SETTINGS FROM CONFIG
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.max_sources_per_recommendation = self.config.get('max_sources_per_recommendation', 10)
        self.enable_cross_verification = self.config.get('enable_cross_verification', True)
        self.enable_domain_specific_sources = self.config.get('enable_domain_specific_sources', True)
        
        # ✅ RELIABILITY ASSESSMENT SETTINGS FROM CONFIG
        self.reliability_tiers = self.config.get('reliability_tiers', [
            'primary_sources', 'expert_sources', 'institutional_sources',
            'journalistic_sources', 'secondary_sources'
        ])
        self.min_reliability_score = self.config.get('min_reliability_score', 6.0)
        self.preferred_source_types = self.config.get('preferred_source_types', [
            'academic', 'government', 'expert', 'institutional'
        ])
        
        # ✅ VERIFICATION STRATEGY SETTINGS FROM CONFIG
        self.verification_strategies = self.config.get('verification_strategies', [
            'primary_source_verification', 'expert_consultation',
            'institutional_confirmation', 'cross_referencing'
        ])
        self.fact_check_priority = self.config.get('fact_check_priority', 'high_impact_claims')
        
        # ✅ DOMAIN CLASSIFICATION SETTINGS FROM CONFIG
        self.enable_domain_classification = self.config.get('enable_domain_classification', True)
        self.domain_confidence_threshold = self.config.get('domain_confidence_threshold', 0.7)
        
        # ✅ GET API KEY FROM SYSTEM SETTINGS
        self.api_key = system_settings.gemini_api_key
        
        # ✅ LOAD PROMPTS FROM CONFIG INSTEAD OF HARDCODED
        self.source_recommendations_prompt = get_prompt_template('credible_source', 'source_recommendations')
        self.reliability_assessment_prompt = get_prompt_template('credible_source', 'reliability_assessment')
        self.verification_strategy_prompt = get_prompt_template('credible_source', 'verification_strategy')
        self.fact_check_guidance_prompt = get_prompt_template('credible_source', 'fact_check_guidance')
        
        # ✅ USE RATE LIMITING FROM CONFIG/SETTINGS
        self.rate_limit = self.config.get('rate_limit_seconds', system_settings.gemini_rate_limit)
        self.max_retries = self.config.get('max_retries', system_settings.max_retries)
        
        # Initialize Gemini API
        self._initialize_gemini_api()
        
        # Initialize modular components
        self.source_database = SourceReliabilityDatabase()
        self.domain_classifier = DomainClassifier()
        
        # Enhanced performance tracking with config awareness
        self.source_metrics = {
            'total_recommendations': 0,
            'successful_recommendations': 0,
            'source_analyses_generated': 0,
            'reliability_assessments_generated': 0,
            'verification_strategies_generated': 0,
            'fact_check_guidance_generated': 0,
            'high_reliability_sources_found': 0,
            'domain_classifications_performed': 0,
            'cross_verification_analyses': 0,
            'average_response_time': 0.0,
            'gemini_api_calls': 0,
            'config_integrated': True
        }
        
        # Rate limiting tracking
        self.last_request_time = None
        
        self.logger.info(f"✅ Enhanced Credible Source Agent initialized with config")
        self.logger.info(f"🤖 Model: {self.model_name}, Temperature: {self.temperature}")
        self.logger.info(f"🎯 Max Sources: {self.max_sources_per_recommendation}, Min Reliability: {self.min_reliability_score}")
        self.logger.info(f"🔍 Cross-Verification: {'On' if self.enable_cross_verification else 'Off'}, Domain Classification: {'On' if self.enable_domain_classification else 'Off'}")
    
    def _initialize_gemini_api(self):
        """
        🔐 INITIALIZE GEMINI API WITH CONFIG SETTINGS
        
        Sets up Gemini AI connection using configuration values optimized
        for source recommendation and analysis.
        """
        try:
            if not self.api_key:
                raise ValueError("Gemini API key not found in system settings")
            
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # ✅ USE GENERATION CONFIG FROM CONFIG FILES
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.config.get('top_p', 0.9),
                "top_k": self.config.get('top_k', 40),
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }
            
            # ✅ USE SAFETY SETTINGS FROM CONFIG
            safety_settings = self.config.get('safety_settings', [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ])
            
            # Create model instance
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info("🔐 Gemini API initialized for source recommendations")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Gemini API: {str(e)}")
            raise
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 MAIN PROCESSING METHOD - LANGGRAPH COMPATIBLE WITH CONFIG
        
        Process input according to BaseAgent interface for LangGraph compatibility.
        
        Args:
            input_data: Dictionary containing:
                - text: Article text to analyze
                - extracted_claims: Claims from claim extractor
                - evidence_evaluation: Results from evidence evaluator
                - include_detailed_analysis: Force detailed analysis
                
        Returns:
            Standardized output dictionary for LangGraph
        """
        # Validate input
        is_valid, error_msg = self.validate_input(input_data)
        if not is_valid:
            return self.format_error_output(ValueError(error_msg), input_data)
        
        # Start processing timer
        self._start_processing_timer()
        
        try:
            # Extract parameters
            article_text = input_data.get('text', '')
            extracted_claims = input_data.get('extracted_claims', [])
            evidence_evaluation = input_data.get('evidence_evaluation', {}) or {}
            include_detailed_analysis = input_data.get(
                'include_detailed_analysis', 
                self.enable_detailed_analysis
            )
            
            # ✅ USE CONFIG FOR PROCESSING DECISIONS
            evidence_score = evidence_evaluation.get('overall_evidence_score', 5.0)
            force_detailed = (
                include_detailed_analysis or
                evidence_score < self.min_reliability_score or  # Low evidence triggers detailed recommendations
                len(extracted_claims) > 5 or  # Many claims need more verification
                self.enable_detailed_analysis
            )
            
            # Perform source recommendation analysis
            recommendation_result = self.recommend_sources(
                article_text=article_text,
                extracted_claims=extracted_claims,
                evidence_evaluation=evidence_evaluation,
                include_detailed_analysis=force_detailed
            )
            
            # Extract overall recommendation score for metrics
            recommendation_score = recommendation_result['recommendation_scores']['overall_recommendation_score']
            
            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(recommendation_score / 10.0)  # Normalize to 0-1
            self.source_metrics['successful_recommendations'] += 1
            
            # Update specific recommendation metrics
            if recommendation_result.get('source_analysis'):
                self.source_metrics['source_analyses_generated'] += 1
            if recommendation_result.get('reliability_assessment'):
                self.source_metrics['reliability_assessments_generated'] += 1
            if recommendation_result.get('verification_strategies'):
                self.source_metrics['verification_strategies_generated'] += 1
            if recommendation_result.get('fact_check_guidance'):
                self.source_metrics['fact_check_guidance_generated'] += 1
            
            # Update domain classification metrics
            if recommendation_result.get('domain_analysis', {}).get('domain_classified'):
                self.source_metrics['domain_classifications_performed'] += 1
            
            # Update quality detection metrics
            high_reliability_count = len([s for s in recommendation_result.get('recommended_sources', [])
                                        if s.get('reliability_score', 0) >= 8.0])
            self.source_metrics['high_reliability_sources_found'] += high_reliability_count
            
            # Format output for LangGraph with config context
            return self.format_output(
                result=recommendation_result,
                confidence=recommendation_score / 10.0,  # Higher recommendation score = higher confidence
                metadata={
                    'response_time': recommendation_result['metadata']['response_time_seconds'],
                    'model_used': self.model_name,
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular',
                    'detailed_analysis_triggered': force_detailed,
                    'min_reliability_score_used': self.min_reliability_score,
                    'max_sources_limit': self.max_sources_per_recommendation
                }
            )
            
        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            return self.format_error_output(e, input_data)
    
    def recommend_sources(self,
                         article_text: str,
                         extracted_claims: List[Dict[str, Any]],
                         evidence_evaluation: Dict[str, Any],
                         include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        🔍 MAIN SOURCE RECOMMENDATION WITH CONFIG INTEGRATION
        
        Comprehensive source recommendation using config-driven parameters and analysis criteria.
        
        Args:
            article_text: The news article text to analyze
            extracted_claims: Claims from claim extractor agent
            evidence_evaluation: Results from evidence evaluator agent
            include_detailed_analysis: Enable detailed source analysis
            
        Returns:
            Dict containing comprehensive source recommendation results
        """
        self._respect_rate_limits()
        start_time = time.time()
        
        try:
            self.logger.info("Starting source recommendation with config integration...")
            
            # Step 1: Clean article text
            article_text = sanitize_text(article_text)
            
            # ✅ USE CONFIG FOR TEXT LENGTH LIMITS
            max_text_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_text_length:
                article_text = article_text[:max_text_length] + "..."
            
            # Step 2: Domain classification if enabled
            domain_analysis = {}
            if self.enable_domain_classification:
                domain_analysis = self.domain_classifier.classify_domain(article_text, extracted_claims)
                self.source_metrics['domain_classifications_performed'] += 1
                self.logger.info(f"🏷️ Domain classified as: {domain_analysis.get('primary_domain', 'general')}")
            
            # Step 3: Get systematic source recommendations using modular database
            database_recommendations = self.source_database.get_source_recommendations(
                article_text, 
                extracted_claims, 
                domain_analysis.get('primary_domain', 'general')
            )
            
            # Step 4: Generate AI-powered source analysis using config prompts
            source_analysis = self._generate_source_analysis(
                article_text, extracted_claims, evidence_evaluation, domain_analysis
            )
            
            # Step 5: Generate reliability assessment
            reliability_assessment = self._generate_reliability_assessment(
                article_text, database_recommendations
            )
            
            # Step 6: Generate verification strategies
            verification_strategies = self._generate_verification_strategies(
                extracted_claims, domain_analysis, evidence_evaluation
            )
            
            # Step 7: Optional fact-check guidance based on config
            fact_check_guidance = None
            if (self.enable_detailed_analysis and 
                (include_detailed_analysis or 
                 evidence_evaluation.get('overall_evidence_score', 10) < self.min_reliability_score)):
                fact_check_guidance = self._generate_fact_check_guidance(
                    extracted_claims, database_recommendations
                )
                self.source_metrics['fact_check_guidance_generated'] += 1
                self.logger.info("📋 Fact-check guidance generated due to evidence concerns")
            
            # Step 8: Cross-verification analysis if enabled
            cross_verification_analysis = None
            if self.enable_cross_verification and len(extracted_claims) > 0:
                cross_verification_analysis = self._generate_cross_verification_analysis(
                    extracted_claims, database_recommendations
                )
                self.source_metrics['cross_verification_analyses'] += 1
            
            # Step 9: Calculate comprehensive recommendation scores using config weights
            recommendation_scores = self._calculate_recommendation_scores(
                database_recommendations, source_analysis, reliability_assessment,
                verification_strategies, domain_analysis
            )
            
            # Step 10: Package results with config metadata
            response_time = time.time() - start_time
            result = {
                'source_analysis': source_analysis,
                'reliability_assessment': reliability_assessment,
                'verification_strategies': verification_strategies,
                'fact_check_guidance': fact_check_guidance,
                'cross_verification_analysis': cross_verification_analysis,
                'domain_analysis': domain_analysis,
                'recommended_sources': database_recommendations.get('recommended_sources', []),
                'source_categories': database_recommendations.get('source_categories', {}),
                'recommendation_scores': recommendation_scores,
                'source_summary': self._create_source_summary(
                    database_recommendations, recommendation_scores
                ),
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'response_time_seconds': round(response_time, 2),
                    'model_used': self.model_name,
                    'temperature_used': self.temperature,
                    'article_length_processed': len(article_text),
                    'claims_analyzed': len(extracted_claims),
                    'detailed_analysis_included': include_detailed_analysis,
                    'domain_classification_enabled': self.enable_domain_classification,
                    'cross_verification_enabled': self.enable_cross_verification,
                    'fact_check_guidance_triggered': fact_check_guidance is not None,
                    'domain_classified': domain_analysis.get('domain_classified', False),
                    'primary_domain': domain_analysis.get('primary_domain', 'general'),
                    'sources_recommended_count': len(database_recommendations.get('recommended_sources', [])),
                    'reliability_tiers_used': len(self.reliability_tiers),
                    'min_reliability_threshold': self.min_reliability_score,
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular'
                }
            }
            
            # Step 11: Update performance metrics
            self._update_recommendation_metrics(response_time, recommendation_scores['overall_recommendation_score'])
            
            self.logger.info(f"Successfully completed source recommendation in {response_time:.2f} seconds")
            self.logger.info(f"🔍 Overall recommendation score: {recommendation_scores['overall_recommendation_score']:.1f}/10")
            self.logger.info(f"📊 Sources recommended: {len(database_recommendations.get('recommended_sources', []))}")
            
            return result
            
        except Exception as e:
            self._update_recommendation_metrics(time.time() - start_time, 0, error=True)
            self.logger.error(f"Error in source recommendation: {str(e)}")
            raise
    
    def _generate_source_analysis(self, article_text: str, extracted_claims: List[Dict[str, Any]],
                                 evidence_evaluation: Dict[str, Any], domain_analysis: Dict[str, Any]) -> str:
        """
        Generate AI-powered source analysis using config prompt template
        
        Args:
            article_text: Article content
            extracted_claims: Claims from extractor
            evidence_evaluation: Evidence evaluation results
            domain_analysis: Domain classification results
            
        Returns:
            Source analysis text
        """
        try:
            # Prepare context from previous analyses
            evidence_summary = f"Evidence Score: {evidence_evaluation.get('overall_evidence_score', 'N/A')}/10"
            domain_summary = f"Domain: {domain_analysis.get('primary_domain', 'general')}"
            claims_summary = f"Claims Count: {len(extracted_claims)}"
            
            # ✅ USE SOURCE RECOMMENDATIONS PROMPT FROM CONFIG
            # Map available data to template variables
            topic_summary = article_text[:200] + "..." if len(article_text) > 200 else article_text
            news_domain = domain_analysis.get('primary_domain', 'general')
            original_source = "Unknown"  # Not available in current data flow
            
            # Extract prediction from context if available
            prediction = "UNKNOWN"
            if 'bert_results' in evidence_evaluation:
                prediction = evidence_evaluation['bert_results'].get('prediction', 'UNKNOWN')
            
            prompt = self.source_recommendations_prompt.format(
                topic_summary=topic_summary,
                extracted_claims=str(extracted_claims[:6]),  # Limit for prompt length
                prediction=prediction,
                news_domain=news_domain,
                original_source=original_source
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Source analysis blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Source analysis flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Source analysis unavailable."
            self.source_metrics['gemini_api_calls'] += 1
            self.source_metrics['source_analyses_generated'] += 1
            
            return getattr(response, 'text', None) or "Source analysis unavailable."
            
        except Exception as e:
            self.logger.error(f"Error in source analysis generation: {str(e)}")
            return f"Source analysis unavailable due to processing error: {str(e)}"
    
    def _generate_reliability_assessment(self, article_text: str, 
                                        database_recommendations: Dict[str, Any]) -> str:
        """
        Generate AI-powered reliability assessment using config prompt template
        
        Args:
            article_text: Article content
            database_recommendations: Systematic source recommendations
            
        Returns:
            Reliability assessment text
        """
        try:
            # Prepare source list for analysis
            source_list = []
            for source in database_recommendations.get('recommended_sources', [])[:8]:  # Limit sources
                source_name = source.get('name', 'Unknown')
                reliability_score = source.get('reliability_score', 0)
                source_type = source.get('type', 'Unknown')
                source_list.append(f"Source: {source_name} (Type: {source_type}, Reliability: {reliability_score}/10)")
            
            sources_text = "\n".join(source_list) if source_list else "No specific sources recommended"
            
            # ✅ USE RELIABILITY ASSESSMENT PROMPT FROM CONFIG
            # Map available data to template variables
            source_name = "Multiple Sources"  # Since we have multiple sources
            source_content = article_text[:300] + "..." if len(article_text) > 300 else article_text
            news_domain = "general"  # Default domain
            context = "Evidence evaluation context not available"
            
            prompt = self.reliability_assessment_prompt.format(
                source_name=source_name,
                source_content=source_content,
                news_domain=news_domain,
                context=context
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Reliability assessment blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Reliability assessment flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Reliability assessment unavailable."
            self.source_metrics['gemini_api_calls'] += 1
            self.source_metrics['reliability_assessments_generated'] += 1
            
            return getattr(response, 'text', None) or "Reliability assessment unavailable."
            
        except Exception as e:
            self.logger.error(f"Error in reliability assessment generation: {str(e)}")
            return f"Reliability assessment unavailable due to processing error: {str(e)}"
    
    def _generate_verification_strategies(self, extracted_claims: List[Dict[str, Any]],
                                         domain_analysis: Dict[str, Any],
                                         evidence_evaluation: Dict[str, Any]) -> str:
        """
        Generate AI-powered verification strategies using config prompt template
        
        Args:
            extracted_claims: Claims to verify
            domain_analysis: Domain classification results
            evidence_evaluation: Evidence evaluation results
            
        Returns:
            Verification strategies text
        """
        try:
            # Prepare high-priority claims for verification
            priority_claims = []
            for claim in extracted_claims[:6]:  # Limit for focused analysis
                claim_text = claim.get('text', 'Unknown claim')
                priority = claim.get('priority', 2)
                verifiability = claim.get('verifiability_score', 5)
                priority_claims.append(f"Priority {priority}: {claim_text} (Verifiability: {verifiability}/10)")
            
            claims_text = "\n".join(priority_claims) if priority_claims else "No claims available for verification"
            
            # Context from other analyses
            domain_context = f"Domain: {domain_analysis.get('primary_domain', 'general')}"
            evidence_context = f"Evidence Quality: {evidence_evaluation.get('overall_evidence_score', 'N/A')}/10"
            
            # ✅ USE VERIFICATION STRATEGY PROMPT FROM CONFIG
            # Map available data to template variables
            available_resources = "Standard fact-checking resources and databases"
            
            prompt = self.verification_strategy_prompt.format(
                priority_claims=claims_text,
                domain_context=domain_context,
                evidence_context=evidence_context,
                available_resources=available_resources
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Verification strategies blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Verification strategies flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Verification strategies unavailable."
            self.source_metrics['gemini_api_calls'] += 1
            self.source_metrics['verification_strategies_generated'] += 1
            
            return getattr(response, 'text', None) or "Verification strategies unavailable."
            
        except Exception as e:
            self.logger.error(f"Error in verification strategies generation: {str(e)}")
            return f"Verification strategies unavailable due to processing error: {str(e)}"
    
    def _generate_fact_check_guidance(self, extracted_claims: List[Dict[str, Any]],
                                     database_recommendations: Dict[str, Any]) -> str:
        """
        Generate AI-powered fact-check guidance using config prompt template
        
        Args:
            extracted_claims: Claims requiring fact-checking
            database_recommendations: Recommended sources for verification
            
        Returns:
            Fact-check guidance text
        """
        try:
            # ✅ USE FACT CHECK GUIDANCE PROMPT FROM CONFIG
            # Map available data to template variables
            verification_priority = "High"  # Default priority
            time_constraints = "Standard fact-checking timeline"
            
            prompt = self.fact_check_guidance_prompt.format(
                extracted_claims=str(extracted_claims[:5]),  # Limit for focused guidance
                recommended_sources=str(database_recommendations.get('recommended_sources', [])[:5]),
                verification_priority=verification_priority,
                time_constraints=time_constraints
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Fact-check guidance blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Fact-check guidance flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Fact-check guidance unavailable."
            self.source_metrics['gemini_api_calls'] += 1
            
            return getattr(response, 'text', None) or "Fact-check guidance unavailable."
            
        except Exception as e:
            self.logger.error(f"Error in fact-check guidance generation: {str(e)}")
            return f"Fact-check guidance unavailable due to processing error: {str(e)}"
    
    def _generate_cross_verification_analysis(self, extracted_claims: List[Dict[str, Any]],
                                             database_recommendations: Dict[str, Any]) -> str:
        """
        Generate cross-verification analysis for claims against multiple sources
        """
        try:
            # Simple cross-verification analysis
            analysis_lines = ["CROSS-VERIFICATION ANALYSIS:", ""]
            
            # Analyze each claim against available sources
            for i, claim in enumerate(extracted_claims[:5], 1):  # Limit claims
                claim_text = claim.get('text', 'Unknown claim')
                verifiability = claim.get('verifiability_score', 5)
                
                analysis_lines.extend([
                    f"Claim {i}: {claim_text[:80]}...",
                    f"  Verifiability Score: {verifiability}/10",
                    f"  Recommended Sources: {min(3, len(database_recommendations.get('recommended_sources', [])))} available",
                    ""
                ])
            
            # Add source categories summary
            source_categories = database_recommendations.get('source_categories', {})
            if source_categories:
                analysis_lines.extend([
                    "Available Source Categories:",
                    *[f"  • {category}: {count} sources" for category, count in source_categories.items()],
                    ""
                ])
            
            return "\n".join(analysis_lines)
            
        except Exception as e:
            self.logger.error(f"Error in cross-verification analysis: {str(e)}")
            return f"Cross-verification analysis unavailable: {str(e)}"
    
    def _calculate_recommendation_scores(self, database_recommendations: Dict, source_analysis: str,
                                        reliability_assessment: str, verification_strategies: str,
                                        domain_analysis: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive recommendation scores with config-aware weights
        """
        # 1. Source availability score
        recommended_sources = database_recommendations.get('recommended_sources', [])
        source_availability_score = min(10, len(recommended_sources) * 1.5)
        
        # 2. Source quality score (average reliability)
        if recommended_sources:
            avg_reliability = sum(s.get('reliability_score', 5) for s in recommended_sources) / len(recommended_sources)
            source_quality_score = avg_reliability
        else:
            source_quality_score = 3.0  # Low score for no sources
        
        # 3. Domain relevance score
        domain_confidence = domain_analysis.get('confidence', 0.5)
        domain_relevance_score = domain_confidence * 10
        
        # 4. Verification feasibility score
        verification_feasibility_score = self._estimate_verification_feasibility(verification_strategies)
        
        # ✅ CALCULATE WEIGHTED OVERALL SCORE USING CONFIG WEIGHTS
        scoring_weights = self.config.get('recommendation_scoring_weights', {
            'source_availability': 0.3,
            'source_quality': 0.4,
            'domain_relevance': 0.15,
            'verification_feasibility': 0.15
        })
        
        overall_score = (
            source_availability_score * scoring_weights['source_availability'] +
            source_quality_score * scoring_weights['source_quality'] +
            domain_relevance_score * scoring_weights['domain_relevance'] +
            verification_feasibility_score * scoring_weights['verification_feasibility']
        )
        
        # Recommendation level assessment with config thresholds
        recommendation_thresholds = self.config.get('recommendation_thresholds', {
            'excellent': 8.0,
            'good': 6.5,
            'fair': 5.0,
            'poor': 3.5
        })
        
        if overall_score >= recommendation_thresholds['excellent']:
            recommendation_level = "EXCELLENT"
        elif overall_score >= recommendation_thresholds['good']:
            recommendation_level = "GOOD"
        elif overall_score >= recommendation_thresholds['fair']:
            recommendation_level = "FAIR"
        elif overall_score >= recommendation_thresholds['poor']:
            recommendation_level = "POOR"
        else:
            recommendation_level = "VERY POOR"
        
        # Source availability factors
        availability_factors = []
        if len(recommended_sources) >= 5:
            availability_factors.append("Multiple sources available")
        if any(s.get('reliability_score', 0) >= 8 for s in recommended_sources):
            availability_factors.append("High-reliability sources available")
        if len(set(s.get('type', 'unknown') for s in recommended_sources)) >= 3:
            availability_factors.append("Diverse source types available")
        
        # Verification challenges
        verification_challenges = []
        if len(recommended_sources) < 3:
            verification_challenges.append("Limited source availability")
        if source_quality_score < 6.0:
            verification_challenges.append("Low average source reliability")
        if domain_confidence < 0.6:
            verification_challenges.append("Uncertain domain classification")
        
        return {
            'source_availability_score': round(source_availability_score, 2),
            'source_quality_score': round(source_quality_score, 2),
            'domain_relevance_score': round(domain_relevance_score, 2),
            'verification_feasibility_score': round(verification_feasibility_score, 2),
            'overall_recommendation_score': round(overall_score, 2),
            'recommendation_level': recommendation_level,
            'availability_factors': availability_factors,
            'verification_challenges': verification_challenges,
            'scoring_method': 'config_weighted',
            'weights_used': scoring_weights,
            'thresholds_used': recommendation_thresholds
        }
    
    def _estimate_verification_feasibility(self, verification_strategies: str) -> float:
        """Estimate verification feasibility from strategies text"""
        feasibility_indicators = [
            'primary sources', 'direct contact', 'official records',
            'expert consultation', 'multiple sources', 'public data',
            'verifiable', 'accessible', 'contactable'
        ]
        
        challenge_indicators = [
            'difficult to verify', 'limited access', 'confidential',
            'anonymous sources', 'unverifiable', 'no public record'
        ]
        
        strategies_lower = verification_strategies.lower()
        feasibility_count = sum(1 for indicator in feasibility_indicators if indicator in strategies_lower)
        challenge_count = sum(1 for indicator in challenge_indicators if indicator in strategies_lower)
        
        base_score = 5.0 + (feasibility_count * 0.8) - (challenge_count * 1.0)
        return max(0, min(10, base_score))
    
    def _create_source_summary(self, database_recommendations: Dict, recommendation_scores: Dict) -> str:
        """Create formatted source recommendation summary"""
        recommended_sources = database_recommendations.get('recommended_sources', [])
        
        if not recommended_sources:
            return "No credible sources could be identified for verification of this article."
        
        summary_lines = [
            f"SOURCE RECOMMENDATION SUMMARY",
            f"Overall Score: {recommendation_scores['overall_recommendation_score']:.1f}/10 ({recommendation_scores['recommendation_level']})",
            f"Sources Available: {len(recommended_sources)}",
            ""
        ]
        
        # Add score breakdown
        summary_lines.extend([
            f"Recommendation Breakdown:",
            f"  • Source Availability: {recommendation_scores['source_availability_score']:.1f}/10",
            f"  • Source Quality: {recommendation_scores['source_quality_score']:.1f}/10",
            f"  • Domain Relevance: {recommendation_scores['domain_relevance_score']:.1f}/10",
            f"  • Verification Feasibility: {recommendation_scores['verification_feasibility_score']:.1f}/10",
            ""
        ])
        
        # Add top sources
        top_sources = sorted(recommended_sources, key=lambda x: x.get('reliability_score', 0), reverse=True)[:5]
        if top_sources:
            summary_lines.append("Top Recommended Sources:")
            for i, source in enumerate(top_sources, 1):
                name = source.get('name', 'Unknown Source')
                reliability = source.get('reliability_score', 0)
                source_type = source.get('type', 'Unknown')
                summary_lines.append(f"  {i}. {name} (Type: {source_type}, Reliability: {reliability}/10)")
            summary_lines.append("")
        
        # Add availability factors
        if recommendation_scores['availability_factors']:
            summary_lines.append("Availability Strengths:")
            for factor in recommendation_scores['availability_factors']:
                summary_lines.append(f"  ✓ {factor}")
            summary_lines.append("")
        
        # Add challenges
        if recommendation_scores['verification_challenges']:
            summary_lines.append("Verification Challenges:")
            for challenge in recommendation_scores['verification_challenges']:
                summary_lines.append(f"  ⚠ {challenge}")
        
        return "\n".join(summary_lines)
    
    def _respect_rate_limits(self):
        """Rate limiting using config values"""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _update_recommendation_metrics(self, response_time: float, recommendation_score: float, error: bool = False):
        """Update recommendation-specific metrics with config awareness"""
        self.source_metrics['total_recommendations'] += 1
        
        if not error:
            # Update average response time
            total = self.source_metrics['total_recommendations']
            current_avg = self.source_metrics['average_response_time']
            self.source_metrics['average_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        📊 Get comprehensive performance metrics with config information
        
        Returns:
            Complete metrics dictionary including config details
        """
        # Get base metrics
        base_metrics = self.get_performance_metrics()
        
        # ✅ ADD CONFIG INFORMATION TO METRICS
        config_metrics = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_sources_per_recommendation': self.max_sources_per_recommendation,
            'min_reliability_score': self.min_reliability_score,
            'enable_detailed_analysis': self.enable_detailed_analysis,
            'enable_cross_verification': self.enable_cross_verification,
            'enable_domain_specific_sources': self.enable_domain_specific_sources,
            'enable_domain_classification': self.enable_domain_classification,
            'reliability_tiers_count': len(self.reliability_tiers),
            'preferred_source_types_count': len(self.preferred_source_types),
            'verification_strategies_count': len(self.verification_strategies),
            'domain_confidence_threshold': self.domain_confidence_threshold,
            'rate_limit_seconds': self.rate_limit,
            'config_version': '2.0_integrated'
        }
        
        # Get component metrics
        component_metrics = {
            'source_database_stats': self.source_database.get_database_statistics(),
            'domain_classifier_stats': self.domain_classifier.get_classifier_statistics(),
            'api_calls_made': self.source_metrics['gemini_api_calls']
        }
        
        return {
            **base_metrics,
            'source_specific_metrics': self.source_metrics,
            'config_metrics': config_metrics,
            'component_info': component_metrics,
            'agent_type': 'credible_source',
            'modular_architecture': True,
            'config_integrated': True,
            'prompt_source': 'centralized_config'
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_sources_per_recommendation': self.max_sources_per_recommendation,
            'min_reliability_score': self.min_reliability_score,
            'enable_detailed_analysis': self.enable_detailed_analysis,
            'enable_cross_verification': self.enable_cross_verification,
            'enable_domain_specific_sources': self.enable_domain_specific_sources,
            'enable_domain_classification': self.enable_domain_classification,
            'reliability_tiers': self.reliability_tiers,
            'preferred_source_types': self.preferred_source_types,
            'verification_strategies': self.verification_strategies,
            'fact_check_priority': self.fact_check_priority,
            'domain_confidence_threshold': self.domain_confidence_threshold,
            'rate_limit_seconds': self.rate_limit,
            'max_retries': self.max_retries,
            'config_source': 'config_files',
            'prompt_source': 'centralized_prompts_config'
        }

# Testing functionality with config integration
if __name__ == "__main__":
    """Test the modular credible source agent with config integration"""
    print("🧪 Testing Modular Credible Source Agent with Config Integration")
    print("=" * 75)
    
    try:
        # Initialize agent (will load from config files)
        agent = CredibleSourceAgent()
        print(f"✅ Agent initialized with config: {agent}")
        
        # Show config summary
        config_summary = agent.get_config_summary()
        print(f"\n⚙️ Configuration Summary:")
        for key, value in config_summary.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {value}")
        
        # Test source recommendation
        test_article = """
        Scientists at the University of Oxford have published a groundbreaking study
        in Nature Medicine showing that a new treatment reduces symptoms by 75%.
        The research was conducted over 3 years with 5,000 participants across 
        multiple countries. Dr. Sarah Johnson, the lead researcher, emphasized
        the statistical significance of the results. However, some experts have
        raised questions about the study's methodology and potential conflicts of interest.
        """
        
        test_claims = [
            {
                'text': 'Oxford University study in Nature Medicine shows 75% symptom reduction.',
                'claim_type': 'Research',
                'priority': 1,
                'verifiability_score': 8,
                'source': 'Oxford University researchers'
            },
            {
                'text': 'Study conducted over 3 years with 5,000 participants.',
                'claim_type': 'Statistical',
                'priority': 1,
                'verifiability_score': 9,
                'source': 'Research study data'
            },
            {
                'text': 'Some experts raised questions about methodology.',
                'claim_type': 'Attribution',
                'priority': 2,
                'verifiability_score': 6,
                'source': 'Expert opinions'
            }
        ]
        
        test_evidence = {
            'overall_evidence_score': 7.2,
            'source_quality_score': 8.1,
            'logical_consistency_score': 7.5
        }
        
        test_input = {
            "text": test_article,
            "extracted_claims": test_claims,
            "evidence_evaluation": test_evidence,
            "include_detailed_analysis": True
        }
        
        print(f"\n📝 Testing source recommendation...")
        print(f"Article preview: {test_article[:100]}...")
        print(f"Claims to verify: {len(test_claims)}")
        print(f"Evidence score: {test_evidence['overall_evidence_score']:.1f}/10")
        
        result = agent.process(test_input)
        
        if result['success']:
            recommendation_data = result['result']
            print(f"✅ Recommendation completed successfully")
            print(f"   Overall recommendation score: {recommendation_data['recommendation_scores']['overall_recommendation_score']:.1f}/10")
            print(f"   Recommendation level: {recommendation_data['recommendation_scores']['recommendation_level']}")
            print(f"   Sources recommended: {len(recommendation_data['recommended_sources'])}")
            print(f"   Source availability: {recommendation_data['recommendation_scores']['source_availability_score']:.1f}/10")
            print(f"   Source quality: {recommendation_data['recommendation_scores']['source_quality_score']:.1f}/10")
            print(f"   Response time: {recommendation_data['metadata']['response_time_seconds']}s")
            print(f"   Config version: {recommendation_data['metadata']['config_version']}")
            
            # Show analysis types generated
            analyses_generated = []
            if recommendation_data.get('source_analysis'):
                analyses_generated.append('Source Analysis')
            if recommendation_data.get('reliability_assessment'):
                analyses_generated.append('Reliability Assessment')
            if recommendation_data.get('verification_strategies'):
                analyses_generated.append('Verification Strategies')
            if recommendation_data.get('fact_check_guidance'):
                analyses_generated.append('Fact-Check Guidance')
            
            print(f"   Analyses generated: {', '.join(analyses_generated)}")
            
            # Show domain analysis
            domain_info = recommendation_data.get('domain_analysis', {})
            if domain_info.get('domain_classified'):
                print(f"   Domain classified: {domain_info.get('primary_domain', 'Unknown')}")
            
            # Show availability factors and challenges
            availability_factors = recommendation_data['recommendation_scores']['availability_factors']
            verification_challenges = recommendation_data['recommendation_scores']['verification_challenges']
            print(f"   Availability factors: {len(availability_factors)}")
            print(f"   Verification challenges: {len(verification_challenges)}")
            
        else:
            print(f"❌ Recommendation failed: {result['error']['message']}")
        
        # Show comprehensive metrics with config info
        print(f"\n📊 Comprehensive metrics with config info:")
        metrics = agent.get_comprehensive_metrics()
        print(f"Agent type: {metrics['agent_type']}")
        print(f"Config integrated: {metrics['config_integrated']}")
        print(f"Prompt source: {metrics['prompt_source']}")
        print(f"High reliability sources found: {metrics['source_specific_metrics']['high_reliability_sources_found']}")
        print(f"Domain classifications performed: {metrics['source_specific_metrics']['domain_classifications_performed']}")
        
        print(f"\n✅ Modular credible source agent with config integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure your GEMINI_API_KEY is set in your environment variables")
        import traceback
        traceback.print_exc()
