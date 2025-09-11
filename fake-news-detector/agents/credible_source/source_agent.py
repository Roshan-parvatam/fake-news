# agents/credible_source/source_agent.py

"""
Credible Source Agent - Production Ready

Enhanced credible source agent that provides contextual source recommendations
with robust error handling, structured logging, retry logic, and comprehensive
fallback mechanisms for production reliability.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import google.generativeai as genai

# Core imports
from agents.base import BaseAgent
from config import get_model_config, get_settings

# Utils with fallback
try:
    from utils.helpers import sanitize_text
except ImportError:
    def sanitize_text(text: str) -> str:
        """Basic text sanitization fallback."""
        if not isinstance(text, str):
            return ""
        return text.strip().replace('\x00', '').replace('\r\n', '\n')

# Local imports
from .source_database import SourceReliabilityDatabase
from .domain_classifier import DomainClassifier
from .prompts import get_source_prompt_template, get_domain_guidance
from .validators import InputValidator, OutputValidator
from .exceptions import (
    CredibleSourceError,
    InputValidationError,
    LLMResponseError,
    SafetyFilterError,
    ContextualRecommendationError,
    ConfigurationError,
    RateLimitError,
    ProcessingTimeoutError,
    raise_safety_filter_error,
    raise_contextual_recommendation_error
)


class CredibleSourceAgent(BaseAgent):
    """
    Production-ready credible source agent with contextual recommendations and safety handling.
    
    Provides domain-specific source recommendations with institutional fallbacks
    when AI analysis is blocked by safety filters, enhanced error handling,
    and comprehensive logging for production environments.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize credible source agent with production configuration."""
        
        # Setup structured logging first
        self._setup_logging()
        
        try:
            # Load configuration
            source_config = get_model_config('credible_source')
            system_settings = get_settings()
            
            if config:
                source_config.update(config)
            
            self.agent_name = "credible_source"
            super().__init__(source_config)

            # Core configuration
            self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
            self.temperature = self.config.get('temperature', 0.3)
            self.max_tokens = self.config.get('max_tokens', 2048)
            
            # Source recommendation settings
            self.max_sources_per_recommendation = self.config.get('max_sources_per_recommendation', 8)
            self.min_reliability_score = self.config.get('min_reliability_score', 6.0)
            self.enable_contextual_recommendations = self.config.get('enable_contextual_recommendations', True)
            self.enable_safety_fallbacks = self.config.get('enable_safety_fallbacks', True)
            
            # Domain classification settings
            self.enable_domain_classification = self.config.get('enable_domain_classification', True)
            self.domain_confidence_threshold = self.config.get('domain_confidence_threshold', 0.7)
            
            # API configuration with better error handling
            self._setup_api_config(system_settings)
            
            # Initialize components with error handling
            self._initialize_components()
            
            # Performance tracking
            self.source_metrics = {
                'total_recommendations': 0,
                'successful_recommendations': 0,
                'contextual_sources_generated': 0,
                'safety_fallbacks_used': 0,
                'safety_blocks_encountered': 0,
                'api_errors': 0,
                'successful_retries': 0
            }
            
            self.last_request_time = None
            self.logger.info(
                f"Credible Source Agent initialized successfully",
                extra={
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'safety_fallbacks': self.enable_safety_fallbacks,
                    'agent_version': '3.1.0'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Credible Source Agent: {str(e)}")
            raise ConfigurationError(f"Agent initialization failed: {str(e)}")

    def _setup_logging(self) -> None:
        """Setup structured logging for production use."""
        self.logger = logging.getLogger(f"{__name__}.CredibleSourceAgent")
        
        # Only add handler if none exists (avoid duplicates)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _setup_api_config(self, system_settings) -> None:
        """Setup API configuration with proper error handling."""
        # Load API key with multiple fallback options
        self.api_key = (
            os.getenv('GEMINI_API_KEY') or
            os.getenv('GOOGLE_API_KEY') or
            getattr(system_settings, 'gemini_api_key', None)
        )
        
        if not self.api_key:
            raise ConfigurationError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable."
            )
        
        # Rate limiting and retry configuration
        self.rate_limit = self.config.get('rate_limit_seconds', 
                                        getattr(system_settings, 'gemini_rate_limit', 1.0))
        self.max_retries = self.config.get('max_retries', 
                                         getattr(system_settings, 'max_retries', 3))
        self.retry_delay = self.config.get('retry_delay', 2.0)
        
        # Initialize Gemini API
        self._initialize_gemini_api()

    def _initialize_gemini_api(self) -> None:
        """Initialize Gemini API with production-ready configuration."""
        try:
            genai.configure(api_key=self.api_key)
            
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.config.get('top_p', 0.9),
                "top_k": self.config.get('top_k', 40),
                "max_output_tokens": self.max_tokens,
            }
            
            # Handle SafetySettings object properly - more permissive for institutional content
            safety_config = self.config.get('safety_settings')
            if hasattr(safety_config, 'to_dict'):
                safety_settings = safety_config.to_dict()
            else:
                safety_settings = safety_config or [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info(f"Gemini API initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise ConfigurationError(f"Gemini API initialization failed: {str(e)}")

    def _initialize_components(self) -> None:
        """Initialize agent components with error handling."""
        try:
            self.input_validator = InputValidator()
            self.output_validator = OutputValidator()
            self.source_database = SourceReliabilityDatabase()
            self.domain_classifier = DomainClassifier()
            
            self.logger.info("Agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise ConfigurationError(f"Component initialization failed: {str(e)}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method with comprehensive error handling.
        """
        session_id = f"source_{int(datetime.now().timestamp() * 1000)}"
        
        self.logger.info(
            "Starting source recommendation",
            extra={
                'session_id': session_id,
                'text_length': len(input_data.get('text', '')),
                'claims_count': len(input_data.get('extracted_claims', []))
            }
        )
        
        start_time = time.time()
        
        try:
            # Input validation
            validation_result = self.input_validator.validate_input_data(input_data)
            if not validation_result.is_valid:
                error_msg = f"Input validation failed: {validation_result.errors[0]}"
                self.logger.warning(error_msg, extra={'session_id': session_id})
                return self.format_error_output(
                    InputValidationError(error_msg), 
                    input_data
                )
            
            # Extract input data
            article_text = input_data.get('text', '')
            extracted_claims = input_data.get('extracted_claims', [])
            evidence_evaluation = input_data.get('evidence_evaluation', {})
            
            self.logger.info(
                "Processing source recommendation",
                extra={
                    'session_id': session_id,
                    'evidence_score': evidence_evaluation.get('overall_evidence_score', 0)
                }
            )
            
            # Perform source recommendation with safety handling
            recommendation_result = self.recommend_sources_with_safety_handling(
                article_text=article_text,
                extracted_claims=extracted_claims,
                evidence_evaluation=evidence_evaluation,
                session_id=session_id
            )
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            self.source_metrics['total_recommendations'] += 1
            self.source_metrics['successful_recommendations'] += 1
            
            if recommendation_result.get('contextual_sources'):
                self.source_metrics['contextual_sources_generated'] += len(
                    recommendation_result['contextual_sources']
                )
            
            if recommendation_result.get('safety_fallback_used'):
                self.source_metrics['safety_fallbacks_used'] += 1
            
            recommendation_score = recommendation_result['recommendation_scores']['overall_recommendation_score']
            confidence = recommendation_score / 10.0
            
            self.logger.info(
                "Source recommendation completed successfully",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time, 2),
                    'recommendation_score': recommendation_score,
                    'safety_fallback_used': recommendation_result.get('safety_fallback_used', False)
                }
            )
            
            return self.format_output(
                result=recommendation_result,
                session_id=session_id,
                confidence=confidence,
                metadata={
                    'processing_time': processing_time,
                    'model_used': self.model_name,
                    'safety_fallback_used': recommendation_result.get('safety_fallback_used', False),
                    'contextual_sources_count': len(recommendation_result.get('contextual_sources', [])),
                    'agent_version': '3.1.0',
                    'session_id': session_id
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"Source recommendation failed: {str(e)}",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time, 2),
                    'error_type': type(e).__name__
                }
            )
            return self.format_error_output(e, input_data)

    def recommend_sources_with_safety_handling(self,
                                             article_text: str,
                                             extracted_claims: List[Dict[str, Any]],
                                             evidence_evaluation: Dict[str, Any],
                                             session_id: str = None) -> Dict[str, Any]:
        """
        Generate contextual source recommendations with safety fallback handling.
        """
        start_time = time.time()
        
        try:
            # Clean and prepare article text
            article_text = sanitize_text(article_text)
            max_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_length:
                article_text = article_text[:max_length] + "..."
                self.logger.info(f"Article text truncated to {max_length} characters")
            
            # Perform domain classification with error handling
            domain_analysis = self._perform_domain_classification_with_retry(
                article_text, extracted_claims, session_id
            )
            
            # Generate contextual analysis with safety handling
            contextual_analysis = self._generate_contextual_analysis_with_fallback(
                article_text, extracted_claims, evidence_evaluation, domain_analysis, session_id
            )
            
            # Get database recommendations
            database_recommendations = self._get_database_recommendations_safe(
                article_text, extracted_claims, domain_analysis, session_id
            )
            
            # Generate reliability assessment with retry
            reliability_assessment = self._generate_reliability_assessment_with_retry(
                article_text, database_recommendations, session_id
            )
            
            # Generate verification strategies with retry
            verification_strategies = self._generate_verification_strategies_with_retry(
                extracted_claims, domain_analysis, evidence_evaluation, session_id
            )
            
            # Calculate recommendation scores
            recommendation_scores = self._calculate_recommendation_scores(
                contextual_analysis, database_recommendations, reliability_assessment
            )
            
            # Package results
            processing_time = time.time() - start_time
            
            result = {
                'contextual_analysis': contextual_analysis['analysis_text'],
                'contextual_sources': contextual_analysis['contextual_sources'],
                'database_recommendations': database_recommendations,
                'reliability_assessment': reliability_assessment,
                'verification_strategies': verification_strategies,
                'domain_analysis': domain_analysis,
                'recommendation_scores': recommendation_scores,
                'safety_fallback_used': contextual_analysis.get('safety_fallback_used', False),
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'response_time_seconds': round(processing_time, 2),
                    'model_used': self.model_name,
                    'claims_analyzed': len(extracted_claims),
                    'contextual_sources_count': len(contextual_analysis['contextual_sources']),
                    'safety_blocks_encountered': contextual_analysis.get('safety_blocks_encountered', 0),
                    'agent_version': '3.1.0',
                    'session_id': session_id
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Source recommendation with safety handling failed: {str(e)}", 
                            extra={'session_id': session_id})
            raise

    def _perform_domain_classification_with_retry(self,
                                                article_text: str,
                                                extracted_claims: List[Dict[str, Any]],
                                                session_id: str = None) -> Dict[str, Any]:
        """Perform domain classification with error handling."""
        try:
            if self.enable_domain_classification:
                domain_analysis = self.domain_classifier.classify_domain(article_text, extracted_claims)
                
                self.logger.info(f"Domain classified as: {domain_analysis.get('primary_domain', 'general')}", 
                               extra={'session_id': session_id})
                
                return domain_analysis
            else:
                return {
                    'primary_domain': 'general',
                    'confidence': 0.0,
                    'domain_classified': False,
                    'domain_scores': {},
                    'secondary_domains': []
                }
                
        except Exception as e:
            self.logger.warning(f"Domain classification failed, using general domain: {str(e)}", 
                              extra={'session_id': session_id})
            return {
                'primary_domain': 'general',
                'confidence': 0.0,
                'domain_classified': False,
                'domain_scores': {},
                'secondary_domains': [],
                'classification_error': str(e)
            }

    def _generate_contextual_analysis_with_fallback(self,
                                                  article_text: str,
                                                  extracted_claims: List[Dict[str, Any]],
                                                  evidence_evaluation: Dict[str, Any],
                                                  domain_analysis: Dict[str, Any],
                                                  session_id: str = None) -> Dict[str, Any]:
        """
        Generate contextual source analysis with institutional fallback.
        Addresses the main safety filter blocking issue from the original code.
        """
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limits()
                
                domain = domain_analysis.get('primary_domain', 'general')
                evidence_score = evidence_evaluation.get('overall_evidence_score', 5.0)
                
                # Generate contextual source analysis prompt
                prompt = get_source_prompt_template(
                    'contextual_analysis',
                    article_text=article_text,
                    extracted_claims=extracted_claims,
                    domain=domain,
                    evidence_score=evidence_score
                )
                
                response = self.model.generate_content(prompt)
                
                # Handle safety filter blocks
                if not self._is_valid_response(response):
                    self.logger.warning(
                        f"Contextual analysis blocked by safety filters (attempt {attempt + 1})",
                        extra={'session_id': session_id}
                    )
                    self.source_metrics['safety_blocks_encountered'] += 1
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        return self._generate_institutional_fallback(extracted_claims, domain, session_id)
                
                analysis_text = response.candidates[0].content.parts[0].text
                contextual_sources = self._parse_contextual_sources(analysis_text)
                
                self.logger.info(f"Generated {len(contextual_sources)} contextual sources", 
                               extra={'session_id': session_id, 'attempt': attempt + 1})
                
                return {
                    'analysis_text': analysis_text,
                    'contextual_sources': contextual_sources,
                    'safety_fallback_used': False,
                    'safety_blocks_encountered': 0
                }
                
            except Exception as e:
                self.source_metrics['api_errors'] += 1
                self.logger.warning(
                    f"Contextual analysis attempt {attempt + 1} failed: {str(e)}",
                    extra={'session_id': session_id}
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        # All retries failed, use institutional fallback
        self.logger.warning("All contextual analysis attempts failed, using institutional fallback")
        return self._generate_institutional_fallback(extracted_claims, domain_analysis.get('primary_domain', 'general'), session_id)

    def _generate_institutional_fallback(self, 
                                       extracted_claims: List[Dict[str, Any]], 
                                       domain: str,
                                       session_id: str = None) -> Dict[str, Any]:
        """
        Generate institutional fallback sources when AI analysis is blocked.
        Provides useful contextual sources instead of failing completely.
        """
        fallback_sources = []
        
        self.logger.info(f"Generating institutional fallback for domain: {domain}", 
                        extra={'session_id': session_id})
        
        # Analyze claims to determine appropriate institutional sources
        for claim in extracted_claims[:self.max_sources_per_recommendation]:
            claim_text = claim.get('text', 'Unknown claim')[:100]
            claim_type = claim.get('claim_type', 'General')
            
            # Domain-specific institutional sources
            if domain == 'health' or any(keyword in claim_text.lower()
                                       for keyword in ['medical', 'health', 'vaccine', 'drug']):
                fallback_sources.extend([
                    {
                        'name': 'Centers for Disease Control and Prevention (CDC)',
                        'details': f'Official US health agency for {claim_type.lower()} verification',
                        'relevance': f'Medical authority for: "{claim_text[:50]}..."',
                        'contact_method': 'Visit cdc.gov or call 1-800-CDC-INFO',
                        'type': 'institutional_health',
                        'relevance_score': 9,
                        'reliability_score': 10,
                        'url': 'https://www.cdc.gov/'
                    },
                    {
                        'name': 'World Health Organization (WHO)',
                        'details': f'International health authority for {claim_type.lower()} standards',
                        'relevance': f'Global health perspective on: "{claim_text[:50]}..."',
                        'contact_method': 'Visit who.int for official statements',
                        'type': 'institutional_international',
                        'relevance_score': 8,
                        'reliability_score': 10,
                        'url': 'https://www.who.int/'
                    }
                ])
                
            elif domain == 'science' or any(keyword in claim_text.lower()
                                          for keyword in ['study', 'research', 'scientist']):
                fallback_sources.extend([
                    {
                        'name': 'PubMed/National Library of Medicine',
                        'details': f'Peer-reviewed research database for {claim_type.lower()} studies',
                        'relevance': f'Academic verification for: "{claim_text[:50]}..."',
                        'contact_method': f'Search PubMed for: "{claim_text[:30]}"',
                        'type': 'academic_database',
                        'relevance_score': 8,
                        'reliability_score': 9,
                        'url': 'https://pubmed.ncbi.nlm.nih.gov/'
                    }
                ])
                
            elif domain == 'politics' or any(keyword in claim_text.lower()
                                           for keyword in ['government', 'policy', 'election']):
                fallback_sources.extend([
                    {
                        'name': 'Government Accountability Office (GAO)',
                        'details': f'Independent government oversight for {claim_type.lower()} verification',
                        'relevance': f'Official government information for: "{claim_text[:50]}..."',
                        'contact_method': 'Visit gao.gov or contact congressional liaisons',
                        'type': 'institutional_government',
                        'relevance_score': 8,
                        'reliability_score': 9,
                        'url': 'https://www.gao.gov/'
                    }
                ])
                
            # Generic high-quality sources
            else:
                fallback_sources.extend([
                    {
                        'name': f'Professional Fact-Checkers for {claim_type}',
                        'details': f'Established fact-checking organizations with {claim_type.lower()} expertise',
                        'relevance': f'Professional verification for: "{claim_text[:50]}..."',
                        'contact_method': 'FactCheck.org, Snopes.com, PolitiFact.com',
                        'type': 'fact_checker',
                        'relevance_score': 7,
                        'reliability_score': 8
                    }
                ])
        
        # Remove duplicates and limit results
        unique_sources = []
        seen_names = set()
        for source in fallback_sources:
            if source['name'] not in seen_names and len(unique_sources) < self.max_sources_per_recommendation:
                unique_sources.append(source)
                seen_names.add(source['name'])
        
        return {
            'analysis_text': f'Institutional source analysis provided due to content sensitivity. Domain-specific ({domain}) sources selected based on claim analysis.',
            'contextual_sources': unique_sources,
            'safety_fallback_used': True,
            'safety_blocks_encountered': 1
        }

    def _get_database_recommendations_safe(self,
                                         article_text: str,
                                         extracted_claims: List[Dict[str, Any]],
                                         domain_analysis: Dict[str, Any],
                                         session_id: str = None) -> Dict[str, Any]:
        """Get database recommendations with error handling."""
        try:
            database_recommendations = self.source_database.get_source_recommendations(
                article_text, extracted_claims, domain_analysis.get('primary_domain', 'general')
            )
            
            self.logger.info(f"Retrieved {database_recommendations['sources_recommended']} database sources", 
                           extra={'session_id': session_id})
            
            return database_recommendations
            
        except Exception as e:
            self.logger.error(f"Database recommendations failed: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'recommended_sources': [],
                'source_categories': {},
                'quality_metrics': {'average_reliability': 0.0},
                'domain_analyzed': domain_analysis.get('primary_domain', 'general'),
                'sources_recommended': 0,
                'error': str(e)
            }

    def _generate_reliability_assessment_with_retry(self,
                                                  article_text: str,
                                                  database_recommendations: Dict[str, Any],
                                                  session_id: str = None) -> str:
        """Generate reliability assessment with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limits()
                
                sources = database_recommendations.get('recommended_sources', [])[:5]
                source_list = [f"• {s.get('name', 'Unknown')}: {s.get('type', 'Unknown type')}"
                              for s in sources]
                sources_text = "\n".join(source_list) if source_list else "No database sources available"
                
                prompt = get_source_prompt_template(
                    'reliability_assessment',
                    article_text=article_text[:800],
                    recommended_sources=sources
                )
                
                response = self.model.generate_content(prompt)
                
                if self._is_valid_response(response):
                    self.logger.info(f"Generated reliability assessment", 
                                   extra={'session_id': session_id, 'attempt': attempt + 1})
                    return response.candidates[0].content.parts[0].text
                
            except Exception as e:
                self.source_metrics['api_errors'] += 1
                self.logger.warning(
                    f"Reliability assessment attempt {attempt + 1} failed: {str(e)}",
                    extra={'session_id': session_id}
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        # All retries failed, return fallback
        self.logger.warning("Reliability assessment failed, using fallback")
        return self._generate_safe_reliability_fallback(len(database_recommendations.get('recommended_sources', [])))

    def _generate_verification_strategies_with_retry(self,
                                                   extracted_claims: List[Dict[str, Any]],
                                                   domain_analysis: Dict[str, Any],
                                                   evidence_evaluation: Dict[str, Any],
                                                   session_id: str = None) -> str:
        """Generate verification strategies with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limits()
                
                prompt = get_source_prompt_template(
                    'verification_strategy',
                    extracted_claims=extracted_claims[:5],
                    domain_analysis=domain_analysis,
                    evidence_evaluation=evidence_evaluation
                )
                
                response = self.model.generate_content(prompt)
                
                if self._is_valid_response(response):
                    self.logger.info(f"Generated verification strategies", 
                                   extra={'session_id': session_id, 'attempt': attempt + 1})
                    return response.candidates[0].content.parts[0].text
                
            except Exception as e:
                self.source_metrics['api_errors'] += 1
                self.logger.warning(
                    f"Verification strategies attempt {attempt + 1} failed: {str(e)}",
                    extra={'session_id': session_id}
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        # All retries failed, return fallback
        self.logger.warning("Verification strategies failed, using fallback")
        return self._generate_safe_strategies_fallback(len(extracted_claims))

    def _parse_contextual_sources(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Parse contextual sources from LLM response with error handling."""
        import re
        sources = []
        
        try:
            # Extract structured sources from response
            source_patterns = [
                r'(\d+\.)\s*([^:]+):\s*([^\n]+)',  # Numbered sources
                r'•\s*([^:]+):\s*([^\n]+)',        # Bullet point sources
                r'-\s*([^:]+):\s*([^\n]+)'         # Dash sources
            ]
            
            for pattern in source_patterns:
                matches = re.findall(pattern, analysis_text, re.MULTILINE)
                for match in matches:
                    if len(match) >= 2:
                        name = match[-2].strip() if len(match) > 2 else match[0].strip()
                        details = match[-1].strip()
                        
                        if name and len(sources) < self.max_sources_per_recommendation:
                            source_data = {
                                'name': name,
                                'details': details[:200],
                                'type': 'contextual_primary',
                                'relevance_score': 8,
                                'reliability_score': 8
                            }
                            
                            # Extract URL if present
                            url_match = re.search(r'https?://[^\s]+', details)
                            if url_match:
                                source_data['url'] = url_match.group(0)
                            
                            sources.append(source_data)
            
            return sources[:self.max_sources_per_recommendation]
            
        except Exception as e:
            self.logger.error(f"Failed to parse contextual sources: {str(e)}")
            return []

    def _calculate_recommendation_scores(self,
                                       contextual_analysis: Dict[str, Any],
                                       database_recommendations: Dict[str, Any],
                                       reliability_assessment: str) -> Dict[str, Any]:
        """Calculate recommendation quality scores with error handling."""
        try:
            contextual_sources = contextual_analysis.get('contextual_sources', [])
            db_sources = database_recommendations.get('recommended_sources', [])
            
            # Source availability score
            source_availability_score = min(10, len(contextual_sources) * 1.5 + len(db_sources) * 0.5)
            
            # Source quality score
            if contextual_sources:
                avg_reliability = sum(s.get('reliability_score', 7) for s in contextual_sources) / len(contextual_sources)
                source_quality_score = avg_reliability
            else:
                source_quality_score = 6.0
            
            # Source relevance score
            source_relevance_score = min(10, len([s for s in contextual_sources if s.get('relevance_score', 0) >= 7]) * 2)
            
            # Overall recommendation score
            overall_score = (source_availability_score * 0.4 + source_quality_score * 0.4 + source_relevance_score * 0.2)
            
            # Apply penalty for safety fallback usage
            if contextual_analysis.get('safety_fallback_used', False):
                overall_score = max(overall_score - 1.0, 0)
            
            # Determine recommendation level
            if overall_score >= 8.5:
                recommendation_level = "EXCELLENT"
            elif overall_score >= 7.0:
                recommendation_level = "GOOD"
            elif overall_score >= 5.5:
                recommendation_level = "FAIR"
            else:
                recommendation_level = "POOR"
            
            return {
                'source_availability_score': round(source_availability_score, 2),
                'source_quality_score': round(source_quality_score, 2),
                'source_relevance_score': round(source_relevance_score, 2),
                'overall_recommendation_score': round(overall_score, 2),
                'recommendation_level': recommendation_level,
                'contextual_sources_count': len(contextual_sources),
                'database_sources_count': len(db_sources),
                'safety_fallback_applied': contextual_analysis.get('safety_fallback_used', False)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating recommendation scores: {str(e)}")
            return {
                'source_availability_score': 5.0,
                'source_quality_score': 5.0,
                'source_relevance_score': 5.0,
                'overall_recommendation_score': 5.0,
                'recommendation_level': "UNKNOWN",
                'contextual_sources_count': 0,
                'database_sources_count': 0,
                'safety_fallback_applied': False
            }

    def _generate_safe_reliability_fallback(self, sources_count: int) -> str:
        """Generate safe fallback reliability assessment."""
        return f"""
SOURCE RELIABILITY ASSESSMENT

Sources Analyzed: {sources_count}

Reliability Evaluation:
• Institutional source verification recommended
• Cross-reference with multiple independent authorities
• Verify source credentials and expertise areas
• Check for potential conflicts of interest

Recommended Approach:
1. Contact primary institutional sources directly
2. Verify through official channels and databases
3. Seek multiple independent confirmations
4. Document all source communications

Overall Assessment: Requires institutional verification through established channels.
"""

    def _generate_safe_strategies_fallback(self, claims_count: int) -> str:
        """Generate safe fallback verification strategies."""
        return f"""
VERIFICATION STRATEGIES

Claims to Verify: {claims_count}

Systematic Verification Approach:

1. Institutional Source Contact
   - Contact relevant government agencies
   - Reach out to academic institutions
   - Consult professional organizations

2. Expert Consultation
   - Identify subject matter experts
   - Contact university departments
   - Engage professional associations

3. Documentation Review
   - Request official documents
   - Check public records and databases
   - Verify through peer-reviewed sources

4. Cross-Reference Analysis
   - Compare multiple independent sources
   - Check for consistency across authorities
   - Identify and resolve discrepancies

Timeline: Allow 3-7 business days for thorough verification
Quality Check: Minimum 3 independent source confirmations recommended
"""

    def _is_valid_response(self, response) -> bool:
        """Check if LLM response is valid and not blocked by safety filters."""
        return (response and
                response.candidates and
                len(response.candidates) > 0 and
                response.candidates[0].finish_reason != 2 and  # Not SAFETY
                response.candidates[0].content and
                response.candidates[0].content.parts)

    def _respect_rate_limits(self) -> None:
        """Implement rate limiting for API calls."""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            base_metrics = super().get_performance_metrics()
        except:
            base_metrics = {}
        
        source_specific_metrics = {
            'total_recommendations': self.source_metrics['total_recommendations'],
            'successful_recommendations': self.source_metrics['successful_recommendations'],
            'contextual_sources_generated': self.source_metrics['contextual_sources_generated'],
            'safety_fallbacks_used': self.source_metrics['safety_fallbacks_used'],
            'safety_blocks_encountered': self.source_metrics['safety_blocks_encountered'],
            'api_errors': self.source_metrics['api_errors'],
            'successful_retries': self.source_metrics['successful_retries'],
            'success_rate': (
                self.source_metrics['successful_recommendations'] /
                max(self.source_metrics['total_recommendations'], 1) * 100
            ),
            'safety_fallback_rate': (
                self.source_metrics['safety_fallbacks_used'] /
                max(self.source_metrics['total_recommendations'], 1) * 100
            ),
            'model_config': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'max_retries': self.max_retries
            }
        }
        
        return {**base_metrics, **source_specific_metrics}

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data for source recommendations."""
        if not isinstance(input_data, dict):
            return False, "Input must be a dictionary"
        
        if 'text' not in input_data:
            return False, "Missing required 'text' field"
        
        validation_result = self.input_validator.validate_input_data(input_data)
        if not validation_result.is_valid:
            return False, validation_result.errors[0]
        
        return True, ""

    async def _process_internal(self, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Internal processing method for BaseAgent compatibility.

        Args:
            input_data: Input data dictionary containing text, claims, and evidence evaluation
            session_id: Optional session identifier for tracking

        Returns:
            Processing result dictionary
        """
        try:
            # Extract input data
            article_text = input_data.get('text', '')
            extracted_claims = input_data.get('extracted_claims', [])
            evidence_evaluation = input_data.get('evidence_evaluation', {})

            # Perform source recommendation with safety handling
            recommendation_result = self.recommend_sources_with_safety_handling(
                article_text=article_text,
                extracted_claims=extracted_claims,
                evidence_evaluation=evidence_evaluation,
                session_id=session_id
            )

            return recommendation_result

        except Exception as e:
            self.logger.error(f"Error in _process_internal: {str(e)}", extra={'session_id': session_id})
            raise CredibleSourceError(f"Internal processing failed: {str(e)}")


# Testing functionality
if __name__ == "__main__":
    """Test the credible source agent."""
    try:
        agent = CredibleSourceAgent()
        
        test_input = {
            "text": """
            A recent study from Johns Hopkins University shows promising results
            for a new treatment approach. The research, published in Nature Medicine,
            demonstrates significant improvements in patient outcomes.
            """,
            "extracted_claims": [
                {
                    "text": "Study from Johns Hopkins shows promising results",
                    "claim_type": "Research",
                    "priority": 1,
                    "verifiability_score": 8
                }
            ],
            "evidence_evaluation": {
                "overall_evidence_score": 7.5
            }
        }
        
        result = agent.process(test_input)
        
        if result['success']:
            recommendation_data = result['result']
            print("✅ Credible Source Recommendations:")
            print(f"  Contextual Sources Found: {len(recommendation_data.get('contextual_sources', []))}")
            print(f"  Safety Fallback Used: {recommendation_data.get('safety_fallback_used', False)}")
            print(f"  Recommendation Score: {recommendation_data['recommendation_scores']['overall_recommendation_score']}/10")
            print(f"  Processing Time: {recommendation_data['metadata']['response_time_seconds']}s")
        else:
            print(f"❌ Recommendation failed: {result['error']['message']}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
