# agents/evidence_evaluator/evaluator_agent.py

"""
Evidence Evaluator Agent - Production Ready

Production-ready evidence evaluation agent that assesses evidence quality,
source credibility, and logical consistency with proper error handling,
structured logging, and graceful degradation.
"""

import os
import time
import logging
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

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
from .criteria import EvidenceQualityCriteria
from .fallacy_detection import LogicalFallacyDetector
from .prompts import get_prompt_template, PromptValidator
from .validators import validate_evidence_input
from .exceptions import (
    EvidenceEvaluatorError,
    InputValidationError,
    LLMResponseError,
    VerificationSourceError,
    ConfigurationError,
    RateLimitError,
    ProcessingTimeoutError
)


class EvidenceEvaluatorAgent(BaseAgent):
    """
    Production-ready evidence evaluation agent with robust error handling,
    structured logging, and graceful degradation capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evidence evaluator agent with production configuration."""
        
        # Setup structured logging first
        self._setup_logging()
        
        try:
            # Load configuration
            evidence_config = get_model_config('evidence_evaluator')
            system_settings = get_settings()
            
            if config:
                evidence_config.update(config)
            
            self.agent_name = "evidence_evaluator"
            super().__init__(evidence_config)
            
            # Core configuration
            self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
            self.temperature = self.config.get('temperature', 0.3)
            self.max_tokens = self.config.get('max_tokens', 3072)
            
            # Processing settings
            self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
            self.evidence_threshold = self.config.get('evidence_threshold', 6.0)
            self.enable_fallacy_detection = self.config.get('enable_fallacy_detection', True)
            self.max_verification_sources = self.config.get('max_verification_sources', 50)  # Much higher limit for RAG
            
            # Scoring weights
            self.scoring_weights = self.config.get('scoring_weights', {
                'source_quality': 0.35,
                'logical_consistency': 0.3,
                'evidence_completeness': 0.25,
                'verification_quality': 0.1
            })
            
            # API configuration with better error handling
            self._setup_api_config(system_settings)
            
            # Initialize components with error handling
            self._initialize_components()
            
            # Performance tracking
            self.evaluation_metrics = {
                'evaluations_completed': 0,
                'verification_sources_generated': 0,
                'high_quality_sources_found': 0,
                'fallacies_detected': 0,
                'evidence_quality_assessments': 0,
                'api_errors': 0,
                'successful_retries': 0
            }
            
            self.last_request_time = None
            self.logger.info(
                f"Evidence Evaluator Agent initialized successfully",
                extra={
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'fallacy_detection': self.enable_fallacy_detection,
                    'agent_version': '3.1.0'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Evidence Evaluator Agent: {str(e)}")
            raise ConfigurationError(f"Agent initialization failed: {str(e)}")

    def _setup_logging(self) -> None:
        """Setup structured logging for production use."""
        self.logger = logging.getLogger(f"{__name__}.EvidenceEvaluatorAgent")
        
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
            
            # Handle SafetySettings object properly
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
            self.quality_criteria = EvidenceQualityCriteria(self.config)
            self.fallacy_detector = LogicalFallacyDetector(self.config)
            self.prompt_validator = PromptValidator()
            
            self.logger.info("Agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise ConfigurationError(f"Component initialization failed: {str(e)}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method with comprehensive error handling.
        """
        session_id = f"eval_{int(datetime.now().timestamp() * 1000)}"
        
        self.logger.info(
            "Starting evidence evaluation",
            extra={
                'session_id': session_id,
                'text_length': len(input_data.get('text', '')),
                'claims_count': len(input_data.get('extracted_claims', []))
            }
        )
        
        start_time = time.time()
        
        try:
            # Input validation
            validation_result = validate_evidence_input(input_data)
            if not validation_result.is_valid:
                error_msg = f"Input validation failed: {validation_result.errors[0]}"
                self.logger.warning(error_msg, extra={'session_id': session_id})
                return self.format_error_output(
                    InputValidationError(error_msg), 
                    input_data,
                    session_id
                )
            
            # Extract input data
            article_text = input_data.get('text', '')
            extracted_claims = input_data.get('extracted_claims', [])
            context_analysis = input_data.get('context_analysis', {})
            
            # Get the prediction from bert_results
            bert_results = input_data.get('bert_results', {})
            prediction = bert_results.get('prediction', 'UNKNOWN')
            
            # Determine analysis depth
            context_score = context_analysis.get('overall_context_score', 5.0)
            include_detailed_analysis = (
                self.enable_detailed_analysis or
                context_score > 7.0 or
                len(extracted_claims) < 2
            )
            
            self.logger.info(
                "Processing evidence evaluation",
                extra={
                    'session_id': session_id,
                    'detailed_analysis': include_detailed_analysis,
                    'context_score': context_score
                }
            )
            
            # Perform evidence evaluation
            evaluation_result = self.evaluate_evidence(
                article_text=article_text,
                extracted_claims=extracted_claims,
                context_analysis=context_analysis,
                prediction=prediction,
                include_detailed_analysis=include_detailed_analysis,
                session_id=session_id
            )
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            self.evaluation_metrics['evaluations_completed'] += 1
            
            evidence_score = evaluation_result['evidence_scores'].get('overall_evidence_score', 5.0)
            confidence = evidence_score / 10.0
            
            self.logger.info(
                "Evidence evaluation completed successfully",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time, 2),
                    'evidence_score': evidence_score,
                    'verification_sources': len(evaluation_result.get('verification_sources', []))
                }
            )
            
            return self.format_output(
                result=evaluation_result,
                session_id=session_id,
                confidence=confidence,
                metadata={
                    'processing_time': processing_time,
                    'model_used': self.model_name,
                    'detailed_analysis': include_detailed_analysis,
                    'verification_sources_count': len(evaluation_result.get('verification_sources', [])),
                    'agent_version': '3.1.0',
                    'session_id': session_id
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"Evidence evaluation failed: {str(e)}",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time, 2),
                    'error_type': type(e).__name__
                }
            )
            return self.format_error_output(e, input_data, session_id)

    def evaluate_evidence(self,
                         article_text: str,
                         extracted_claims: List[Dict[str, Any]],
                         context_analysis: Dict[str, Any],
                         prediction: str = 'UNKNOWN',
                         include_detailed_analysis: bool = True,
                         session_id: str = None) -> Dict[str, Any]:
        """
        Comprehensive evidence evaluation with robust error handling.
        """
        start_time = time.time()
        
        # Clean and prepare article text
        article_text = sanitize_text(article_text)
        max_length = self.config.get('max_article_length', 4000)
        if len(article_text) > max_length:
            article_text = article_text[:max_length] + "..."
            self.logger.info(f"Article text truncated to {max_length} characters")
        
        # Step 1: Generate verification sources with retry logic
        verification_sources = self._generate_verification_sources_with_retry(
            article_text, extracted_claims, prediction, session_id
        )

        # Step 1.5: Verify and enrich the generated links
        if verification_sources:
            self.logger.info(f"Verifying {len(verification_sources)} generated links...", extra={'session_id': session_id})
            verification_sources = self._verify_and_enrich_links(verification_sources, session_id)
            
            # Filter out unreachable links for a cleaner result
            original_count = len(verification_sources)
            verification_sources = [link for link in verification_sources if link.get('is_verified')]
            self.logger.info(f"Retained {len(verification_sources)} of {original_count} links after verification.", extra={'session_id': session_id})
        
        # Step 2: Assess source quality with retry logic
        source_quality_analysis = self._assess_source_quality_with_retry(
            article_text, extracted_claims, session_id
        )
        
        # Step 3: Analyze logical consistency with retry logic
        logical_consistency_analysis = self._analyze_logical_consistency_with_retry(
            article_text, extracted_claims, session_id
        )
        
        # Step 4: Run quality assessment (traditional method - no API)
        try:
            quality_assessment = self.quality_criteria.assess_evidence_quality(
                article_text, extracted_claims
            )
        except Exception as e:
            self.logger.warning(f"Quality assessment failed, using defaults: {str(e)}")
            quality_assessment = self._create_fallback_quality_assessment()
        
        # Step 5: Detect logical fallacies (traditional method - no API)
        fallacy_report = {}
        if self.enable_fallacy_detection:
            try:
                fallacy_report = self.fallacy_detector.detect_fallacies(article_text)
            except Exception as e:
                self.logger.warning(f"Fallacy detection failed: {str(e)}")
                fallacy_report = {'detected_fallacies': [], 'fallacy_summary': 'Fallacy detection unavailable'}
        
        # Step 6: Identify evidence gaps (optional, with retry)
        evidence_gaps_analysis = None
        if include_detailed_analysis:
            evidence_gaps_analysis = self._identify_evidence_gaps_with_retry(
                article_text, extracted_claims, session_id
            )
        
        # Step 7: Calculate comprehensive scores
        evidence_scores = self._calculate_evidence_scores(
            quality_assessment, verification_sources, source_quality_analysis,
            logical_consistency_analysis, fallacy_report
        )
        
        # Step 8: Create summary
        evidence_summary = self._create_evidence_summary(
            extracted_claims, evidence_scores, verification_sources
        )
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(verification_sources, fallacy_report)
        
        return {
            'verification_links': verification_sources,  # Map verification_sources to verification_links for API compatibility
            'verification_sources': verification_sources,  # Keep both for backward compatibility
            'source_quality_analysis': source_quality_analysis,
            'logical_consistency_analysis': logical_consistency_analysis,
            'evidence_gaps_analysis': evidence_gaps_analysis,
            'quality_assessment': quality_assessment,
            'fallacy_report': fallacy_report,
            'evidence_scores': evidence_scores,
            'evidence_summary': evidence_summary,
            'metadata': {
                'processing_time_seconds': round(processing_time, 2),
                'model_used': self.model_name,
                'claims_evaluated': len(extracted_claims),
                'detailed_analysis_included': include_detailed_analysis,
                'verification_sources_found': len(verification_sources),
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            }
        }

    def _search_with_duckduckgo(self, claims: List[Dict[str, Any]], session_id: str = None) -> List[Dict[str, Any]]:
        """
        Performs a DuckDuckGo search for each claim to find verification sources.

        Args:
            claims: A list of claims to be verified.
            session_id: Optional session ID for tracking.

        Returns:
            A list of dictionaries, where each dictionary represents a verification source.
        """
        verification_sources = []
        with DDGS() as ddgs:
            for claim in claims:
                claim_text = claim.get('text')
                if not claim_text:
                    continue

                self.logger.info(f"Searching for claim: '{claim_text}'", extra={'session_id': session_id})
                try:
                    search_results = list(ddgs.text(claim_text, max_results=3))
                    for result in search_results:
                        verification_sources.append({
                            'claim': claim_text,
                            'institution': result.get('title', 'N/A'),
                            'url': result.get('href', ''),
                            'verification_type': 'search_result',
                            'confidence': 0.7,  # Assign a default confidence
                            'quality_score': self._calculate_source_quality_score({'url': result.get('href', '')}),
                            'source_type': 'duckduckgo_search',
                            'page_title': result.get('title', 'N/A'),
                            'snippet': result.get('body', '')
                        })
                except Exception as e:
                    self.logger.error(f"DuckDuckGo search failed for claim '{claim_text}': {e}", extra={'session_id': session_id})

                # Respectful delay to avoid rate limiting
                time.sleep(2)  # 2-second delay between queries

        return verification_sources

    def _generate_verification_sources_with_retry(self, 
                                                article_text: str, 
                                                claims: List[Dict[str, Any]],
                                                prediction: str,
                                                session_id: str = None) -> List[Dict[str, Any]]:
        """
        Generates verification sources using DuckDuckGo search with retry logic.
        """
        for attempt in range(self.max_retries):
            try:
                # The new RAG approach using DuckDuckGo search
                verification_sources = self._search_with_duckduckgo(claims, session_id)

                validated_sources = []
                from .validators import URLValidator
                url_validator = URLValidator()
                for source in verification_sources:
                    url = source.get('url', '')
                    if url and url_validator.validate_url_specificity(url).is_valid:
                        validated_sources.append(source)
                    else:
                        self.logger.warning(f"Rejected URL for its lack of specificity: {url}", extra={'session_id': session_id})

                self.logger.info(
                    f"Generated {len(validated_sources)} verification sources via DuckDuckGo search",
                    extra={'session_id': session_id, 'attempt': attempt + 1}
                )

                return validated_sources[:self.max_verification_sources]

            except Exception as e:
                self.evaluation_metrics['api_errors'] += 1
                self.logger.warning(
                    f"Verification source generation attempt {attempt + 1} failed: {str(e)}",
                    extra={'session_id': session_id}
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue

        self.logger.warning("All verification source generation attempts failed, using fallback.")
        return self._create_fallback_sources(claims)

    def _assess_source_quality_with_retry(self, 
                                        article_text: str, 
                                        claims: List[Dict[str, Any]],
                                        session_id: str = None) -> str:
        """Assess source quality with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limits()
                
                sources = [claim.get('source', 'Not specified') for claim in claims]
                sources = [s for s in sources if s != 'Not specified'][:10]
                
                # Provide fallback if no sources available
                if not sources:
                    sources = ['No specific sources identified in claims']
                
                prompt = get_prompt_template(
                    'source_quality',
                    article_text=article_text,
                    sources=sources
                )
                
                response = self.model.generate_content(prompt)
                
                if self._is_valid_response(response):
                    return response.candidates[0].content.parts[0].text
                
            except Exception as e:
                self.logger.warning(
                    f"Source quality assessment attempt {attempt + 1} failed: {str(e)}",
                    extra={'session_id': session_id}
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        # All retries failed, return fallback
        self.logger.warning("Source quality assessment failed, using fallback")
        return self._create_fallback_source_analysis(len(claims))

    def _analyze_logical_consistency_with_retry(self, 
                                              article_text: str, 
                                              claims: List[Dict[str, Any]],
                                              session_id: str = None) -> str:
        """Analyze logical consistency with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limits()
                
                claim_texts = [claim.get('text', '') for claim in claims[:6]]
                
                prompt = get_prompt_template(
                    'logical_consistency',
                    article_text=article_text,
                    claims=claim_texts
                )
                
                response = self.model.generate_content(prompt)
                
                if self._is_valid_response(response):
                    return response.candidates[0].content.parts[0].text
                
            except Exception as e:
                self.logger.warning(
                    f"Logical consistency analysis attempt {attempt + 1} failed: {str(e)}",
                    extra={'session_id': session_id}
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        # All retries failed, return fallback
        self.logger.warning("Logical consistency analysis failed, using fallback")
        return self._create_fallback_logical_analysis(len(claims))

    def _identify_evidence_gaps_with_retry(self, 
                                         article_text: str, 
                                         claims: List[Dict[str, Any]],
                                         session_id: str = None) -> str:
        """Identify evidence gaps with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limits()
                
                prompt = get_prompt_template(
                    'evidence_gaps',
                    article_text=article_text,
                    claims=claims
                )
                
                response = self.model.generate_content(prompt)
                
                if self._is_valid_response(response):
                    return response.candidates[0].content.parts[0].text
                
            except Exception as e:
                self.logger.warning(
                    f"Evidence gaps analysis attempt {attempt + 1} failed: {str(e)}",
                    extra={'session_id': session_id}
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        # All retries failed, return fallback
        self.logger.warning("Evidence gaps analysis failed, using fallback")
        return "Evidence gap analysis unavailable due to API limitations."

    # Keep all existing helper methods unchanged
    def _parse_verification_sources(self, response_text: str) -> List[Dict[str, Any]]:
        """
        [JSON VERSION] Parses a JSON string from the LLM response into structured
        verification sources.
        """
        self.logger.debug(f"Attempting to parse JSON response ({len(response_text)} chars)")
        try:
            # Use regex to find the JSON array, ignoring any leading/trailing text from the LLM
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if not json_match:
                self.logger.error("No valid JSON array found in the LLM response.")
                self.logger.debug(f"Full response received: {response_text}")
                return []
            
            json_string = json_match.group(0)
            parsed_data = json.loads(json_string)

            if isinstance(parsed_data, list):
                # Add the source_type for internal tracking
                for source in parsed_data:
                    if isinstance(source, dict):
                         source['source_type'] = 'llm_generated'
                self.logger.info(f"Successfully parsed {len(parsed_data)} sources from JSON.")
                return parsed_data
            else:
                self.logger.warning(f"Parsed JSON is not a list. Type: {type(parsed_data)}")
                return []

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from LLM response: {e}")
            self.logger.debug(f"Response text that failed parsing: {response_text}")
            return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during JSON parsing: {e}")
            return []

    def _calculate_source_quality_score(self, source: Dict[str, Any]) -> float:
        """Calculate quality score for a verification source."""
        score = 0.5  # Base score
        
        institution = source.get('institution', '').lower()
        url = source.get('url', '').lower()
        verification_type = source.get('verification_type', '').lower()
        
        # Institution quality bonus
        high_quality_institutions = [
            'cdc', 'who', 'nih', 'fda', 'harvard', 'stanford', 'mit',
            'pubmed', 'nature', 'science', 'reuters', 'ap news'
        ]
        
        if any(inst in institution for inst in high_quality_institutions):
            score += 0.3
        
        # Domain quality bonus
        high_quality_domains = ['.gov', '.edu', 'pubmed', 'nature.com', 'science.org']
        if any(domain in url for domain in high_quality_domains):
            score += 0.2
        
        # Verification type bonus
        if verification_type in ['primary_source', 'official_data']:
            score += 0.15
        elif verification_type in ['expert_analysis', 'research_study']:
            score += 0.1
        
        # Confidence factor
        confidence = source.get('confidence', 0.5)
        score = score * (0.7 + 0.3 * confidence)
        
        return min(1.0, max(0.1, score))

    def _calculate_evidence_scores(self,
                                 quality_assessment: Dict[str, Any],
                                 verification_sources: List[Dict[str, Any]],
                                 source_quality_analysis: str,
                                 logical_consistency_analysis: str,
                                 fallacy_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive evidence scores."""
        
        # Extract base scores
        source_quality_score = quality_assessment.get('source_quality_score', 5.0)
        completeness_score = quality_assessment.get('completeness_score', 5.0)
        
        # Calculate verification quality score
        verification_score = 3.0
        if verification_sources:
            avg_quality = sum(s.get('quality_score', 0.5) for s in verification_sources) / len(verification_sources)
            verification_score = 3.0 + (avg_quality * 7.0)
            
            if len(verification_sources) >= 3:
                verification_score += 1.0
            if len([s for s in verification_sources if s.get('quality_score', 0) >= 0.8]) >= 2:
                verification_score += 1.0
        
        verification_score = max(0, min(10, verification_score))
        
        # Calculate logical consistency score
        fallacy_count = len(fallacy_report.get('detected_fallacies', []))
        logical_score = max(1.0, min(10.0, 8.0 - fallacy_count))
        
        # Calculate overall score with weights
        overall_score = (
            (source_quality_score * self.scoring_weights.get('source_quality', 0.35)) +
            (logical_score * self.scoring_weights.get('logical_consistency', 0.3)) +
            (completeness_score * self.scoring_weights.get('evidence_completeness', 0.25)) +
            (verification_score * self.scoring_weights.get('verification_quality', 0.1))
        )
        
        overall_score = max(0, min(10, overall_score))
        
        # Determine quality level
        if overall_score >= 8.0:
            quality_level = "HIGH QUALITY"
        elif overall_score >= 6.0:
            quality_level = "MODERATE QUALITY"
        elif overall_score >= 4.0:
            quality_level = "LOW QUALITY"
        else:
            quality_level = "POOR QUALITY"
        
        return {
            'source_quality_score': round(source_quality_score, 2),
            'logical_consistency_score': round(logical_score, 2),
            'evidence_completeness_score': round(completeness_score, 2),
            'verification_quality_score': round(verification_score, 2),
            'overall_evidence_score': round(overall_score, 2),
            'quality_level': quality_level,
            'verification_sources_count': len(verification_sources),
            'high_quality_sources_count': len([s for s in verification_sources if s.get('quality_score', 0) >= 0.8]),
            'fallacy_count': fallacy_count
        }

    def _create_evidence_summary(self,
                               claims: List[Dict[str, Any]],
                               evidence_scores: Dict[str, Any],
                               verification_sources: List[Dict[str, Any]]) -> str:
        """Create comprehensive evidence summary."""
        if not claims:
            return "No claims available for evidence evaluation."
        
        overall_score = evidence_scores.get('overall_evidence_score', 5.0)
        quality_level = evidence_scores.get('quality_level', 'UNKNOWN')
        source_count = len(verification_sources)
        high_quality_count = evidence_scores.get('high_quality_sources_count', 0)
        
        summary_lines = [
            "EVIDENCE EVALUATION SUMMARY",
            f"Overall Score: {overall_score:.1f}/10 ({quality_level})",
            f"Verification Sources: {source_count} total ({high_quality_count} high-quality)",
            "",
            "Component Scores:",
            f"• Source Quality: {evidence_scores.get('source_quality_score', 5.0):.1f}/10",
            f"• Logical Consistency: {evidence_scores.get('logical_consistency_score', 5.0):.1f}/10",
            f"• Evidence Completeness: {evidence_scores.get('evidence_completeness_score', 5.0):.1f}/10",
            f"• Verification Quality: {evidence_scores.get('verification_quality_score', 5.0):.1f}/10",
            "",
            "Claims Analysis:",
            f"• Total Claims Evaluated: {len(claims)}",
            f"• High Verifiability Claims: {len([c for c in claims if c.get('verifiability_score', 0) >= 7])}",
            f"• Fallacies Detected: {evidence_scores.get('fallacy_count', 0)}"
        ]
        
        if high_quality_count >= 3:
            summary_lines.append("✓ Excellent verification sources available")
        elif high_quality_count >= 1:
            summary_lines.append("⚠ Some high-quality verification sources available")
        elif source_count > 0:
            summary_lines.append("⚠ Basic verification sources provided")
        else:
            summary_lines.append("❌ Limited verification sources available")
        
        return "\n".join(summary_lines)

    def _update_metrics(self, verification_sources: List[Dict[str, Any]], fallacy_report: Dict[str, Any]) -> None:
        """Update performance metrics."""
        self.evaluation_metrics['verification_sources_generated'] += len(verification_sources)
        self.evaluation_metrics['high_quality_sources_found'] += len([
            s for s in verification_sources if s.get('quality_score', 0) >= 0.8
        ])
        
        if fallacy_report:
            self.evaluation_metrics['fallacies_detected'] += len(fallacy_report.get('detected_fallacies', []))
        
        self.evaluation_metrics['evidence_quality_assessments'] += 1

    # Fallback methods
    def _create_fallback_sources(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create fallback verification sources when LLM generation fails."""
        fallback_sources = []
        for claim in claims[:3]:
            claim_text = claim.get('text', 'Unknown claim')
            fallback_sources.append({
                'claim': claim_text,
                'institution': 'Snopes Fact Checking',
                'url': 'https://www.snopes.com/fact-check/',
                'verification_type': 'fact_checker',
                'confidence': 0.6,
                'quality_score': 0.6,
                'source_type': 'fallback'
            })
        return fallback_sources

    def _create_fallback_source_analysis(self, source_count: int) -> str:
        """Create fallback source analysis when LLM analysis fails."""
        return f"""
SOURCE QUALITY ASSESSMENT
Sources Identified: {source_count}
Assessment: Automated evaluation indicates sources require human verification.
Recommendation: Cross-reference claims with established authorities and verify source credentials independently.
Quality Check: Verify sources through official channels and check for conflicts of interest.
"""

    def _create_fallback_logical_analysis(self, claim_count: int) -> str:
        """Create fallback logical analysis when LLM analysis fails."""
        return f"""
LOGICAL CONSISTENCY ASSESSMENT
Claims Analyzed: {claim_count}
Assessment: Logical structure requires human review for comprehensive evaluation.
Recommendation: Check claim-to-claim consistency, verify evidence-to-conclusion logic, and assess reasoning quality independently.
Review Focus: Look for logical fallacies, verify statistical claims, and ensure temporal consistency.
"""

    def _create_fallback_quality_assessment(self) -> Dict[str, Any]:
        """Create fallback quality assessment when components fail."""
        return {
            'source_quality_score': 5.0,
            'completeness_score': 5.0,
            'overall_quality_score': 5.0,
            'quality_summary': 'Quality assessment unavailable - using default values'
        }

    def _is_valid_response(self, response) -> bool:
        """Check if LLM response is valid and not blocked."""
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

    def _verify_and_enrich_links(self, links: List[Dict[str, Any]], session_id: str = None) -> List[Dict[str, Any]]:
        """
        Verifies URLs are reachable and enriches them with page titles.
        
        Args:
            links: List of link dictionaries with 'url' field
            session_id: Optional session ID for logging
            
        Returns:
            List of enriched link dictionaries with verification status and page titles
        """
        verified_links = []
        
        self.logger.info(f"Starting verification of {len(links)} links", extra={'session_id': session_id})
        
        for i, link in enumerate(links):
            url = link.get('url')
            if not url:
                self.logger.warning(f"Link {i+1} has no URL, skipping verification", extra={'session_id': session_id})
                continue

            try:
                # Use a timeout and a common user-agent
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                }
                
                self.logger.debug(f"Verifying URL {i+1}: {url}", extra={'session_id': session_id})
                response = requests.get(url, timeout=10, headers=headers)
                
                # Check for successful response
                if response.status_code == 200:
                    link['is_verified'] = True
                    link['status_code'] = 200
                    
                    # Try to parse the page title for context
                    try:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title_tag = soup.find('title')
                        if title_tag and title_tag.string:
                            link['page_title'] = title_tag.string.strip()
                        else:
                            link['page_title'] = "No Title Found"
                    except Exception as parse_error:
                        self.logger.debug(f"Failed to parse title for {url}: {parse_error}", extra={'session_id': session_id})
                        link['page_title'] = "Title Parse Failed"
                    
                    self.logger.debug(f"✅ URL {i+1} verified successfully: {link.get('page_title', 'No title')}", extra={'session_id': session_id})
                else:
                    link['is_verified'] = False
                    link['status_code'] = response.status_code
                    link['page_title'] = "Unreachable"
                    self.logger.warning(f"❌ URL {i+1} returned status {response.status_code}: {url}", extra={'session_id': session_id})
                
            except requests.RequestException as e:
                self.logger.warning(f"❌ Failed to verify URL {i+1} {url}: {e}", extra={'session_id': session_id})
                link['is_verified'] = False
                link['status_code'] = "Error"
                link['page_title'] = f"Verification Failed: {type(e).__name__}"
            except Exception as e:
                self.logger.error(f"❌ Unexpected error verifying URL {i+1} {url}: {e}", extra={'session_id': session_id})
                link['is_verified'] = False
                link['status_code'] = "Error"
                link['page_title'] = f"Verification Error: {type(e).__name__}"

            verified_links.append(link)
        
        verified_count = sum(1 for link in verified_links if link.get('is_verified', False))
        self.logger.info(f"Link verification completed: {verified_count}/{len(verified_links)} links verified", extra={'session_id': session_id})
        
        return verified_links

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            base_metrics = super().get_performance_metrics()
        except:
            base_metrics = {}
        
        evidence_metrics = {
            'evaluations_completed': self.evaluation_metrics['evaluations_completed'],
            'verification_sources_generated': self.evaluation_metrics['verification_sources_generated'],
            'high_quality_sources_found': self.evaluation_metrics['high_quality_sources_found'],
            'fallacies_detected': self.evaluation_metrics['fallacies_detected'],
            'evidence_quality_assessments': self.evaluation_metrics['evidence_quality_assessments'],
            'api_errors': self.evaluation_metrics['api_errors'],
            'successful_retries': self.evaluation_metrics['successful_retries'],
            'model_config': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'max_retries': self.max_retries
            }
        }
        
        return {**base_metrics, **evidence_metrics}

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data for evidence evaluation."""
        if not isinstance(input_data, dict):
            return False, "Input must be a dictionary"
        
        if 'text' not in input_data:
            return False, "Missing required 'text' field"
        
        if not input_data['text'].strip():
            return False, "Article text cannot be empty"
        
        if len(input_data['text']) < 50:
            return False, "Article text too short for meaningful analysis"
        
        return True, ""

    async def _process_internal(self, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Internal processing method for BaseAgent compatibility.

        Args:
            input_data: Input data dictionary containing text, claims, and context analysis
            session_id: Optional session identifier for tracking

        Returns:
            Processing result dictionary
        """
        try:
            # Extract input data
            article_text = input_data.get('text', '')
            extracted_claims = input_data.get('extracted_claims', [])
            context_analysis = input_data.get('context_analysis', {})
            
            # Get the prediction from bert_results
            bert_results = input_data.get('bert_results', {})
            prediction = bert_results.get('prediction', 'UNKNOWN')
            
            # Determine analysis depth
            context_score = context_analysis.get('overall_context_score', 5.0)
            include_detailed_analysis = (
                self.enable_detailed_analysis or
                context_score > 7.0 or
                len(extracted_claims) < 2
            )

            # Perform evidence evaluation
            evaluation_result = self.evaluate_evidence(
                article_text=article_text,
                extracted_claims=extracted_claims,
                context_analysis=context_analysis,
                prediction=prediction,
                include_detailed_analysis=include_detailed_analysis,
                session_id=session_id
            )

            return evaluation_result

        except Exception as e:
            self.logger.error(f"Error in _process_internal: {str(e)}", extra={'session_id': session_id})
            raise EvidenceEvaluatorError(f"Internal processing failed: {str(e)}")


# Testing functionality
if __name__ == "__main__":
    """Test the evidence evaluator agent."""
    try:
        agent = EvidenceEvaluatorAgent()
        
        test_input = {
            "text": """
            According to a study published by researchers at Harvard University,
            the new treatment shows promising results in clinical trials with
            over 1,000 participants. The peer-reviewed research, published in
            Nature Medicine, demonstrates significant efficacy rates.
            """,
            "extracted_claims": [
                {
                    "text": "Harvard University study shows promising treatment results",
                    "verifiability_score": 8,
                    "priority": 1
                },
                {
                    "text": "Clinical trials included over 1,000 participants",
                    "verifiability_score": 7,
                    "priority": 2
                }
            ],
            "context_analysis": {
                "overall_context_score": 6.5,
                "risk_level": "MEDIUM"
            }
        }
        
        result = agent.process(test_input)
        
        if result['success']:
            evaluation_data = result['result']
            evidence_scores = evaluation_data.get('evidence_scores', {})
            print("✅ Evidence Evaluation Results:")
            print(f" Overall Evidence Score: {evidence_scores.get('overall_evidence_score', 0):.1f}/10")
            print(f" Quality Level: {evidence_scores.get('quality_level', 'UNKNOWN')}")
            print(f" Verification Sources: {evidence_scores.get('verification_sources_count', 0)}")
            print(f" High Quality Sources: {evidence_scores.get('high_quality_sources_count', 0)}")
            print(f" Processing Time: {evaluation_data['metadata']['processing_time_seconds']}s")
        else:
            print(f"❌ Evaluation failed: {result['error']['message']}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
