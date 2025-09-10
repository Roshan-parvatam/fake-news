# agents/credible_source/source_agent.py

"""
Credible Source Agent

Production-ready credible source agent that provides contextual source
recommendations with robust safety filter handling and institutional fallbacks.
Enhanced with clean architecture and reliable source database integration.
"""

import time
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import google.generativeai as genai

from agents.base.base_agent import BaseAgent
from config import get_model_config, get_settings
from utils.helpers import sanitize_text

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
    raise_safety_filter_error,
    raise_contextual_recommendation_error
)

class CredibleSourceAgent(BaseAgent):
    """
    Credible source agent with contextual recommendations and safety handling.
    
    Provides domain-specific source recommendations with institutional fallbacks
    when AI analysis is blocked by safety filters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize credible source agent with safety handling."""
        
        # Load configuration
        source_config = get_model_config('credible_source')
        system_settings = get_settings()
        
        if config:
            source_config.update(config)

        self.agent_name = "credible_source"
        super().__init__(source_config)

        # AI Model Configuration
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 2048)

        # Source Recommendation Configuration
        self.max_sources_per_recommendation = self.config.get('max_sources_per_recommendation', 8)
        self.min_reliability_score = self.config.get('min_reliability_score', 6.0)
        self.enable_contextual_recommendations = self.config.get('enable_contextual_recommendations', True)
        self.enable_safety_fallbacks = self.config.get('enable_safety_fallbacks', True)

        # Domain Classification Configuration
        self.enable_domain_classification = self.config.get('enable_domain_classification', True)
        self.domain_confidence_threshold = self.config.get('domain_confidence_threshold', 0.7)

        # ✅ FIXED: Enhanced API key loading from .env
        self.api_key = (
            os.getenv('GEMINI_API_KEY') or 
            os.getenv('GOOGLE_API_KEY') or 
            getattr(system_settings, 'gemini_api_key', None)
        )
        
        if not self.api_key:
            raise ConfigurationError(
                "Gemini API key not found. Please set GEMINI_API_KEY in your .env file"
            )

        # Rate limiting configuration
        self.rate_limit = self.config.get('rate_limit_seconds', getattr(system_settings, 'gemini_rate_limit', 1.0))
        self.max_retries = self.config.get('max_retries', getattr(system_settings, 'max_retries', 3))

        # Initialize components
        self._initialize_gemini_api()
        self.input_validator = InputValidator()
        self.output_validator = OutputValidator()
        self.source_database = SourceReliabilityDatabase()
        self.domain_classifier = DomainClassifier()

        # Performance metrics
        self.source_metrics = {
            'total_recommendations': 0,
            'successful_recommendations': 0,
            'contextual_sources_generated': 0,
            'safety_fallbacks_used': 0,
            'safety_blocks_encountered': 0
        }

        self.last_request_time = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Credible Source Agent initialized with safety handling enabled")

    def _initialize_gemini_api(self) -> None:
        """Initialize Gemini API with safety settings."""
        try:
            genai.configure(api_key=self.api_key)

            generation_config = {
                "temperature": self.temperature,
                "top_p": self.config.get('top_p', 0.9),
                "top_k": self.config.get('top_k', 40),
                "max_output_tokens": self.max_tokens,
            }

            # More permissive safety settings for institutional content
            safety_settings = [
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
            raise ConfigurationError(f"Failed to initialize Gemini API: {str(e)}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for credible source recommendations.

        Args:
            input_data: Dictionary containing article text, claims, and evidence evaluation

        Returns:
            Dictionary with source recommendations and analysis
        """
        # ✅ FIXED: Enhanced input validation
        validation_result = self.input_validator.validate_article_text(input_data.get('text', ''))
        if not validation_result.is_valid:
            return self.format_error_output(
                InputValidationError(f"Input validation failed: {validation_result.errors[0]}"),
                input_data
            )

        # ✅ FIXED: Session management compatibility
        start_time = time.time()
        
        if hasattr(self, '_start_processing_session'):
            self._start_processing_session(input_data)

        try:
            article_text = input_data.get('text', '')
            extracted_claims = input_data.get('extracted_claims', [])
            evidence_evaluation = input_data.get('evidence_evaluation', {})

            # Perform source recommendation with safety handling
            recommendation_result = self.recommend_sources_with_safety_handling(
                article_text=article_text,
                extracted_claims=extracted_claims,
                evidence_evaluation=evidence_evaluation
            )

            # Update metrics
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

            return self.format_output(
                result=recommendation_result,
                confidence=confidence,
                metadata={
                    'processing_time': processing_time,
                    'model_used': self.model_name,
                    'safety_fallback_used': recommendation_result.get('safety_fallback_used', False),
                    'agent_version': '2.0.0'
                }
            )

        except Exception as e:
            self.logger.error(f"Source recommendation failed: {str(e)}")
            return self.format_error_output(e, input_data)
        
        finally:
            # ✅ FIXED: Session cleanup compatibility
            if hasattr(self, '_end_processing_session'):
                self._end_processing_session()

    def recommend_sources_with_safety_handling(self,
                                               article_text: str,
                                               extracted_claims: List[Dict[str, Any]],
                                               evidence_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate contextual source recommendations with safety fallback handling.

        Args:
            article_text: Article content to analyze
            extracted_claims: Claims extracted from the article
            evidence_evaluation: Evidence quality assessment

        Returns:
            Dictionary containing source recommendations and analysis
        """
        self._respect_rate_limits()
        start_time = time.time()

        try:
            # Clean and prepare article text
            article_text = sanitize_text(article_text)
            max_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_length:
                article_text = article_text[:max_length] + "..."

            # Perform domain classification
            domain_analysis = {}
            if self.enable_domain_classification:
                domain_analysis = self.domain_classifier.classify_domain(article_text, extracted_claims)

            # Generate contextual source analysis with safety handling
            contextual_analysis = self._generate_contextual_analysis_with_fallback(
                article_text, extracted_claims, evidence_evaluation, domain_analysis
            )

            # Get database recommendations
            database_recommendations = self.source_database.get_source_recommendations(
                article_text, extracted_claims, domain_analysis.get('primary_domain', 'general')
            )

            # Generate additional analyses
            reliability_assessment = self._generate_reliability_assessment_safe(
                article_text, database_recommendations
            )

            verification_strategies = self._generate_verification_strategies_safe(
                extracted_claims, domain_analysis, evidence_evaluation
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
                    'agent_version': '2.0.0'
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Source recommendation with safety handling failed: {str(e)}")
            raise

    def _generate_contextual_analysis_with_fallback(self,
                                                    article_text: str,
                                                    extracted_claims: List[Dict[str, Any]],
                                                    evidence_evaluation: Dict[str, Any],
                                                    domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate contextual source analysis with institutional fallback.
        Addresses the main safety filter blocking issue from the original code.
        """
        try:
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
                self.logger.warning("Contextual analysis blocked by safety filters - using institutional fallback")
                self.source_metrics['safety_blocks_encountered'] += 1
                return self._generate_institutional_fallback(extracted_claims, domain)

            analysis_text = response.candidates[0].content.parts[0].text
            contextual_sources = self._parse_contextual_sources(analysis_text)

            return {
                'analysis_text': analysis_text,
                'contextual_sources': contextual_sources,
                'safety_fallback_used': False,
                'safety_blocks_encountered': 0
            }

        except Exception as e:
            self.logger.error(f"Contextual analysis generation failed: {str(e)}")
            self.source_metrics['safety_blocks_encountered'] += 1
            return self._generate_institutional_fallback(extracted_claims, domain_analysis.get('primary_domain', 'general'))

    def _generate_institutional_fallback(self, extracted_claims: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """
        Generate institutional fallback sources when AI analysis is blocked.
        Provides useful contextual sources instead of failing completely.
        """
        fallback_sources = []

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

    def _parse_contextual_sources(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Parse contextual sources from LLM response."""
        import re
        sources = []

        # Extract structured sources from response
        source_patterns = [
            r'(\d+\.)\s*([^:]+):\s*([^\n]+)',  # Numbered sources
            r'•\s*([^:]+):\s*([^\n]+)',       # Bullet point sources  
            r'-\s*([^:]+):\s*([^\n]+)'        # Dash sources
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

    def _generate_reliability_assessment_safe(self, article_text: str, 
                                            database_recommendations: Dict[str, Any]) -> str:
        """Generate reliability assessment with safety handling."""
        try:
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
            
            if not self._is_valid_response(response):
                return self._generate_safe_reliability_fallback(len(sources))

            return response.candidates[0].content.parts[0].text

        except Exception as e:
            self.logger.error(f"Reliability assessment generation failed: {str(e)}")
            return self._generate_safe_reliability_fallback(len(database_recommendations.get('recommended_sources', [])))

    def _generate_verification_strategies_safe(self, extracted_claims: List[Dict[str, Any]],
                                             domain_analysis: Dict[str, Any],
                                             evidence_evaluation: Dict[str, Any]) -> str:
        """Generate verification strategies with safety handling."""
        try:
            prompt = get_source_prompt_template(
                'verification_strategy',
                extracted_claims=extracted_claims[:5],
                domain_analysis=domain_analysis,
                evidence_evaluation=evidence_evaluation
            )

            response = self.model.generate_content(prompt)
            
            if not self._is_valid_response(response):
                return self._generate_safe_strategies_fallback(len(extracted_claims))

            return response.candidates[0].content.parts[0].text

        except Exception as e:
            self.logger.error(f"Verification strategies generation failed: {str(e)}")
            return self._generate_safe_strategies_fallback(len(extracted_claims))

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

    def _calculate_recommendation_scores(self, contextual_analysis: Dict[str, Any],
                                       database_recommendations: Dict[str, Any],
                                       reliability_assessment: str) -> Dict[str, Any]:
        """Calculate recommendation quality scores."""
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
        base_metrics = super().get_performance_metrics()
        source_specific_metrics = {
            'total_recommendations': self.source_metrics['total_recommendations'],
            'successful_recommendations': self.source_metrics['successful_recommendations'],
            'contextual_sources_generated': self.source_metrics['contextual_sources_generated'],
            'safety_fallbacks_used': self.source_metrics['safety_fallbacks_used'],
            'safety_blocks_encountered': self.source_metrics['safety_blocks_encountered'],
            'success_rate': (
                self.source_metrics['successful_recommendations'] /
                max(self.source_metrics['total_recommendations'], 1) * 100
            ),
            'safety_fallback_rate': (
                self.source_metrics['safety_fallbacks_used'] /
                max(self.source_metrics['total_recommendations'], 1) * 100
            )
        }

        return {**base_metrics, **source_specific_metrics}

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data for source recommendations."""
        if not isinstance(input_data, dict):
            return False, "Input must be a dictionary"
        
        if 'text' not in input_data:
            return False, "Missing required 'text' field"

        validation_result = self.input_validator.validate_article_text(input_data['text'])
        if not validation_result.is_valid:
            return False, validation_result.errors[0]

        return True, ""


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
            print(f"   Contextual Sources Found: {len(recommendation_data.get('contextual_sources', []))}")
            print(f"   Safety Fallback Used: {recommendation_data.get('safety_fallback_used', False)}")
            print(f"   Recommendation Score: {recommendation_data['recommendation_scores']['overall_recommendation_score']}/10")
            print(f"   Processing Time: {recommendation_data['metadata']['response_time_seconds']}s")
        else:
            print(f"❌ Recommendation failed: {result['error']['message']}")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
