# agents/context_analyzer/analyzer_agent.py

"""
Context Analyzer Agent - Production Ready

Production-ready context analysis agent that examines articles for bias,
emotional manipulation, framing techniques, and propaganda methods with
enhanced error handling, session tracking, and comprehensive logging.
"""

import os
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

import google.generativeai as genai

from agents.base import BaseAgent
from config import get_model_config, get_settings

# Utils import with fallback
try:
    from utils import sanitize_text
except ImportError:
    def sanitize_text(text: str) -> str:
        """Basic text sanitization fallback."""
        if not isinstance(text, str):
            return ""
        return text.strip().replace('\x00', '').replace('\r\n', '\n')

from .bias_patterns import BiasPatternDatabase
from .manipulation_detection import ManipulationDetector
from .prompts import get_context_prompt_template, validate_context_analysis_output
from .validators import InputValidator, OutputValidator, validate_context_input
from .exceptions import (
    ContextAnalyzerError,
    LLMResponseError,
    BiasDetectionError,
    ManipulationDetectionError,
    ScoringConsistencyError,
    ConfigurationError,
    InputValidationError,
    ProcessingTimeoutError,
    handle_context_analyzer_exception,
    is_recoverable_error,
    get_retry_delay
)


class ContextAnalyzerAgent(BaseAgent):
    """
    Production-ready Context Analyzer Agent with enhanced reliability.
    
    Examines articles for bias, manipulation, framing, and propaganda while
    ensuring numerical scores match textual explanations for consistency.
    Enhanced with comprehensive error handling, session tracking, and
    performance monitoring for production use.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the context analyzer agent with production configuration."""
        
        # Load configuration from config files
        try:
            system_settings = get_settings()
            context_config = get_model_config('context_analyzer')
            
            # Merge configurations with precedence: passed config > model config > system settings
            self.config = {**context_config, **(config or {})}
            
            self.agent_name = "context_analyzer"
            super().__init__(self.config)
            
            # Enhanced API key loading with multiple fallbacks
            self.api_key = (
                os.getenv('GEMINI_API_KEY') or
                os.getenv('GOOGLE_API_KEY') or
                getattr(system_settings, 'gemini_api_key', None) or
                self.config.get('api_key')
            )
            
            if not self.api_key:
                raise ConfigurationError(
                    "Gemini API key not found. Please set GEMINI_API_KEY in your .env file"
                )
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

        # Model Configuration with production defaults
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.4)
        self.max_tokens = self.config.get('max_tokens', 3072)
        
        # Analysis Configuration
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.bias_threshold = self.config.get('bias_threshold', 70.0)
        self.manipulation_threshold = self.config.get('manipulation_threshold', 70.0)
        self.enable_propaganda_analysis = self.config.get('enable_propaganda_analysis', True)
        
        # Rate limiting and retry configuration
        self.rate_limit = self.config.get('rate_limit_seconds', 1.0)
        self.max_retries = self.config.get('max_retries', 3)
        self.request_timeout = self.config.get('request_timeout_seconds', 30.0)
        
        # Initialize components with error handling
        try:
            self._initialize_gemini_api()
            self.input_validator = InputValidator(self.config.get('validation'))
            self.output_validator = OutputValidator(self.config.get('validation'))
            self.bias_database = BiasPatternDatabase(self.config.get('bias_patterns'))
            self.manipulation_detector = ManipulationDetector(self.config.get('manipulation_detection'))
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize components: {str(e)}")

        # Performance and monitoring metrics
        self.metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'llm_scores_generated': 0,
            'high_bias_detected': 0,
            'high_manipulation_detected': 0,
            'score_consistency_checks': 0,
            'retry_attempts': 0,
            'fallback_analyses': 0,
            'processing_times': []
        }

        self.last_request_time = None
        self.logger = logging.getLogger(f"{__name__}.ContextAnalyzerAgent")
        
        self.logger.info(
            f"Context Analyzer Agent initialized successfully",
            extra={
                'model_name': self.model_name,
                'temperature': self.temperature,
                'bias_threshold': self.bias_threshold,
                'manipulation_threshold': self.manipulation_threshold,
                'detailed_analysis': self.enable_detailed_analysis
            }
        )

    def _initialize_gemini_api(self) -> None:
        """Initialize Gemini API with enhanced safety settings and error handling."""
        try:
            genai.configure(api_key=self.api_key)
            
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.config.get('top_p', 0.9),
                "top_k": self.config.get('top_k', 40),
                "max_output_tokens": self.max_tokens,
            }

            # Enhanced safety settings for content analysis
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
            self.logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise ConfigurationError(f"Gemini API initialization failed: {str(e)}")

    def process(self, input_data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """
        Main processing method for context analysis with comprehensive error handling.

        Args:
            input_data: Dictionary containing article text and previous analysis
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with context analysis results and metadata
        """
        session_id = session_id or f"context_analysis_{int(time.time() * 1000)}"
        start_time = time.time()
        
        self.metrics['total_analyses'] += 1
        
        self.logger.info(
            f"Starting context analysis",
            extra={
                'session_id': session_id,
                'input_keys': list(input_data.keys()) if isinstance(input_data, dict) else None,
                'article_length': len(input_data.get('text', '')) if isinstance(input_data, dict) else 0
            }
        )

        try:
            # Enhanced input validation with detailed feedback
            validation_result = self.input_validator.validate_processing_input(input_data)
            if not validation_result.is_valid:
                error_msg = f"Input validation failed: {'; '.join(validation_result.errors)}"
                self.logger.warning(f"Input validation failed for session {session_id}: {error_msg}")
                return self._format_error_response(
                    InputValidationError(error_msg),
                    session_id,
                    time.time() - start_time
                )

            # Extract and validate input parameters
            article_text = input_data.get('text', '')
            previous_analysis = input_data.get('previous_analysis', {})
            include_detailed_analysis = input_data.get(
                'include_detailed_analysis',
                self.enable_detailed_analysis
            )

            # Determine analysis depth based on previous analysis confidence
            bert_confidence = previous_analysis.get('confidence', 1.0)
            force_detailed = (
                include_detailed_analysis or
                bert_confidence < 0.7 or
                self.enable_detailed_analysis
            )

            # Perform comprehensive context analysis with retry logic
            analysis_result = self._analyze_context_with_retry(
                article_text=article_text,
                previous_analysis=previous_analysis,
                include_detailed_analysis=force_detailed,
                session_id=session_id
            )

            # Update success metrics
            processing_time = time.time() - start_time
            self.metrics['successful_analyses'] += 1
            self.metrics['processing_times'].append(processing_time)

            # Calculate confidence score from risk assessment
            risk_score = analysis_result.get('context_scores', {}).get('risk_score', 50)
            confidence = max(0.1, 1.0 - (risk_score / 100.0))

            self.logger.info(
                f"Context analysis completed successfully",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time, 3),
                    'confidence': round(confidence, 3),
                    'detailed_analysis': force_detailed,
                    'bias_score': analysis_result.get('context_scores', {}).get('bias_score', 0),
                    'manipulation_score': analysis_result.get('context_scores', {}).get('manipulation_score', 0)
                }
            )

            return self._format_success_response(
                analysis_result,
                confidence,
                processing_time,
                session_id,
                {
                    'detailed_analysis': force_detailed,
                    'scoring_method': 'llm_driven_with_consistency_validation',
                    'agent_version': '3.1.0'
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics['failed_analyses'] += 1
            
            self.logger.error(
                f"Context analysis failed: {str(e)}",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time, 3),
                    'error_type': type(e).__name__
                }
            )
            
            return self._format_error_response(e, session_id, processing_time)

    def _analyze_context_with_retry(self,
                                  article_text: str,
                                  previous_analysis: Dict[str, Any],
                                  include_detailed_analysis: bool = True,
                                  session_id: str = None) -> Dict[str, Any]:
        """
        Comprehensive context analysis with retry logic and fallback mechanisms.

        Args:
            article_text: Article content to analyze
            previous_analysis: Results from previous agents
            include_detailed_analysis: Whether to include detailed analysis
            session_id: Session ID for tracking

        Returns:
            Dictionary containing comprehensive context analysis with consistent scores
        """
        max_attempts = self.max_retries + 1
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    self.metrics['retry_attempts'] += 1
                    self.logger.info(f"Retry attempt {attempt} for session {session_id}")
                
                return self._perform_context_analysis(
                    article_text,
                    previous_analysis,
                    include_detailed_analysis,
                    session_id
                )
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Final attempt failed, return fallback analysis
                    self.logger.warning(f"All retry attempts failed for session {session_id}, using fallback")
                    self.metrics['fallback_analyses'] += 1
                    return self._create_fallback_analysis(article_text, previous_analysis, session_id)
                
                if not is_recoverable_error(e):
                    # Non-recoverable error, don't retry
                    raise
                
                # Wait before retry
                retry_delay = get_retry_delay(e, attempt + 1)
                if retry_delay:
                    self.logger.info(f"Waiting {retry_delay:.1f}s before retry {attempt + 1}")
                    time.sleep(retry_delay)

    def _perform_context_analysis(self,
                                article_text: str,
                                previous_analysis: Dict[str, Any],
                                include_detailed_analysis: bool,
                                session_id: str) -> Dict[str, Any]:
        """Perform the actual context analysis with LLM integration."""
        
        self._respect_rate_limits()
        analysis_start_time = time.time()

        try:
            # Extract information from previous analysis
            prediction = previous_analysis.get('prediction', 'Unknown')
            confidence = previous_analysis.get('confidence', 0.0)
            source = previous_analysis.get('source', 'Unknown Source')
            topic_domain = previous_analysis.get('topic_domain', 'general')

            # Clean and prepare article text
            article_text = sanitize_text(article_text)
            max_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_length:
                article_text = article_text[:max_length] + "..."
                self.logger.debug(f"Article truncated to {max_length} characters for session {session_id}")

            # Run pattern-based analysis (provides backup data)
            pattern_analysis = self.bias_database.analyze_bias_patterns(article_text)
            manipulation_report = self.manipulation_detector.get_manipulation_report(article_text)

            # Generate LLM analysis with consistent scores
            llm_analysis_result = self._generate_llm_analysis_with_consistent_scores(
                article_text, source, topic_domain, prediction, confidence, session_id
            )

            # Generate additional analyses if needed
            additional_analyses = {}
            llm_scores = llm_analysis_result.get('scores', {})
            
            if (include_detailed_analysis or
                llm_scores.get('bias', 0) > self.bias_threshold or
                llm_scores.get('manipulation', 0) > self.manipulation_threshold):
                
                additional_analyses = self._generate_additional_analyses(
                    article_text, previous_analysis, llm_scores, session_id
                )

            # Calculate processing time
            processing_time = time.time() - analysis_start_time

            # Package comprehensive results
            result = {
                'llm_analysis': llm_analysis_result['analysis_text'],
                'llm_scores': llm_analysis_result['scores'],
                'context_scores': {
                    'bias_score': llm_scores.get('bias', 0),
                    'manipulation_score': llm_scores.get('manipulation', 0),
                    'credibility': llm_scores.get('credibility', 50),
                    'risk_level': self._get_risk_level_from_scores(llm_scores),
                    'risk_score': llm_scores.get('risk', 50),
                    'overall_context_score': llm_scores.get('risk', 50) / 10.0
                },
                'bias_analysis': additional_analyses.get('bias_analysis'),
                'framing_analysis': additional_analyses.get('framing_analysis'),
                'emotional_analysis': additional_analyses.get('emotional_analysis'),
                'propaganda_analysis': additional_analyses.get('propaganda_analysis'),
                'manipulation_report': manipulation_report,
                'pattern_analysis': {
                    'bias_counts': pattern_analysis.get('bias_counts', {}),
                    'emotional_counts': pattern_analysis.get('emotional_counts', {}),
                    'indicators_found': pattern_analysis.get('indicators_found', [])
                },
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'response_time_seconds': round(processing_time, 3),
                    'model_used': self.model_name,
                    'article_length_processed': len(article_text),
                    'detailed_analysis_included': include_detailed_analysis,
                    'scoring_method': 'llm_driven_with_validation',
                    'consistency_validated': llm_analysis_result.get('consistency_validated', False),
                    'agent_version': '3.1.0',
                    'session_id': session_id
                }
            }

            # Update specific detection metrics
            if llm_scores.get('bias', 0) >= self.bias_threshold:
                self.metrics['high_bias_detected'] += 1
            if llm_scores.get('manipulation', 0) >= self.manipulation_threshold:
                self.metrics['high_manipulation_detected'] += 1

            self.metrics['llm_scores_generated'] += 1
            self.metrics['score_consistency_checks'] += 1

            return result

        except Exception as e:
            self.logger.error(f"Context analysis failed for session {session_id}: {str(e)}")
            raise

    def _generate_llm_analysis_with_consistent_scores(self,
                                                    article_text: str,
                                                    source: str,
                                                    topic_domain: str,
                                                    prediction: str,
                                                    confidence: float,
                                                    session_id: str) -> Dict[str, Any]:
        """
        Generate LLM analysis with consistent numerical scores.
        
        Addresses the main issue where text analysis doesn't match numerical scores.
        """
        try:
            # Use structured prompt for comprehensive analysis
            prompt = get_context_prompt_template(
                'comprehensive_analysis',
                article_text=article_text,
                source=source,
                prediction=prediction,
                confidence=confidence,
                session_id=session_id
            )

            # Generate content with timeout handling
            response = self._generate_content_with_timeout(prompt, session_id)
            
            if not self._is_valid_response(response):
                self.logger.warning(f"LLM response blocked by safety filters for session {session_id}, using fallback")
                return self._create_fallback_llm_analysis(article_text, session_id)

            analysis_text = response.candidates[0].content.parts[0].text

            # Parse scores from response with error handling
            scores = self._parse_llm_scores(analysis_text, session_id)

            # Validate score consistency and adjust if needed
            consistency_validated = validate_context_analysis_output(analysis_text, scores)
            if not consistency_validated:
                self.logger.warning(f"Score consistency validation failed for session {session_id}, adjusting scores")
                scores = self._adjust_inconsistent_scores(analysis_text, scores, session_id)

            return {
                'analysis_text': analysis_text,
                'scores': scores,
                'scoring_method': 'llm_generated_with_validation',
                'consistency_validated': consistency_validated,
                'session_id': session_id
            }

        except Exception as e:
            self.logger.error(f"LLM analysis generation failed for session {session_id}: {str(e)}")
            return self._create_fallback_llm_analysis(article_text, session_id)

    def _generate_content_with_timeout(self, prompt: str, session_id: str):
        """Generate content with timeout handling using threading."""
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def generate_content():
            try:
                response = self.model.generate_content(prompt)
                result_queue.put(response)
            except Exception as e:
                exception_queue.put(e)
        
        # Start generation in a separate thread
        thread = threading.Thread(target=generate_content)
        thread.daemon = True
        thread.start()
        
        # Wait for result with timeout
        thread.join(timeout=self.request_timeout)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            raise ProcessingTimeoutError(
                f"LLM request timed out after {self.request_timeout} seconds",
                timeout_seconds=self.request_timeout,
                operation="llm_generation",
                session_id=session_id
            )
        
        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()
        
        # Get result
        if not result_queue.empty():
            return result_queue.get()
        
        # This shouldn't happen, but just in case
        raise ProcessingTimeoutError(
            f"LLM request failed unexpectedly",
            timeout_seconds=self.request_timeout,
            operation="llm_generation",
            session_id=session_id
        )

    def _parse_llm_scores(self, analysis_text: str, session_id: str) -> Dict[str, int]:
        """Parse numerical scores from LLM response with enhanced error handling."""
        import re
        
        scores = {}
        
        # Enhanced regex patterns for score extraction
        patterns = {
            'bias': [
                r'(?:bias.*?score|bias):\s*(\d+)',
                r'BIAS:\s*(\d+)',
                r'bias.*?(\d+)',
                r'political.*?bias.*?(\d+)'
            ],
            'manipulation': [
                r'(?:manipulation.*?score|manipulation):\s*(\d+)',
                r'MANIPULATION:\s*(\d+)',
                r'manipulation.*?(\d+)',
                r'emotional.*?manipulation.*?(\d+)'
            ],
            'credibility': [
                r'(?:credibility.*?score|credibility):\s*(\d+)',
                r'CREDIBILITY:\s*(\d+)',
                r'credibility.*?(\d+)',
                r'reliability.*?(\d+)'
            ],
            'risk': [
                r'(?:risk.*?score|risk):\s*(\d+)',
                r'RISK:\s*(\d+)',
                r'risk.*?(\d+)',
                r'danger.*?(\d+)'
            ]
        }

        for score_type, pattern_list in patterns.items():
            score_found = False
            for pattern in pattern_list:
                try:
                    match = re.search(pattern, analysis_text, re.IGNORECASE)
                    if match:
                        score = int(match.group(1))
                        scores[score_type] = max(0, min(100, score))
                        score_found = True
                        break
                except (ValueError, IndexError):
                    continue
            
            if not score_found:
                # Estimate score from text analysis
                estimated_score = self._estimate_score_from_text(analysis_text, score_type, session_id)
                scores[score_type] = estimated_score
                self.logger.debug(f"Estimated {score_type} score: {estimated_score} for session {session_id}")

        return scores

    def _estimate_score_from_text(self, text: str, score_type: str, session_id: str) -> int:
        """Estimate score from text analysis when explicit score not provided."""
        text_lower = text.lower()
        
        # Enhanced text analysis patterns
        score_indicators = {
            'bias': {
                'high': ['extreme bias', 'highly biased', 'very biased', 'severe bias', 'heavy bias'],
                'moderate': ['biased', 'partisan', 'slanted', 'some bias', 'moderate bias'],
                'low': ['minimal bias', 'slight bias', 'little bias'],
                'neutral': ['neutral', 'balanced', 'unbiased', 'objective', 'fair']
            },
            'manipulation': {
                'high': ['extreme manipulation', 'highly manipulative', 'severe manipulation', 'heavy manipulation'],
                'moderate': ['manipulation', 'manipulative', 'misleading', 'some manipulation'],
                'low': ['minimal manipulation', 'slight manipulation', 'little manipulation'],
                'neutral': ['straightforward', 'direct', 'honest', 'transparent']
            },
            'credibility': {
                'high': ['highly credible', 'very reliable', 'trustworthy', 'authoritative', 'credible'],
                'moderate': ['somewhat credible', 'moderately reliable', 'fairly reliable'],
                'low': ['questionable', 'unreliable', 'dubious', 'suspect'],
                'neutral': ['standard', 'typical', 'average']
            },
            'risk': {
                'high': ['high risk', 'dangerous', 'harmful', 'critical risk', 'severe risk'],
                'moderate': ['moderate risk', 'concerning', 'some risk', 'potential risk'],
                'low': ['low risk', 'minimal risk', 'little risk'],
                'neutral': ['safe', 'harmless', 'benign']
            }
        }
        
        if score_type not in score_indicators:
            return 50  # Default neutral score
        
        indicators = score_indicators[score_type]
        
        # Check for indicators in order of severity
        for level, phrases in [('high', indicators['high']), ('moderate', indicators['moderate']), 
                              ('low', indicators['low']), ('neutral', indicators['neutral'])]:
            if any(phrase in text_lower for phrase in phrases):
                score_mapping = {
                    'high': 85 if score_type != 'credibility' else 85,
                    'moderate': 65 if score_type != 'credibility' else 65,
                    'low': 25 if score_type != 'credibility' else 75,  # Credibility is inverted
                    'neutral': 50 if score_type != 'credibility' else 75
                }
                
                # Special handling for credibility (higher is better)
                if score_type == 'credibility':
                    credibility_mapping = {'high': 85, 'moderate': 65, 'low': 25, 'neutral': 50}
                    return credibility_mapping[level]
                
                return score_mapping[level]
        
        return 50  # Default if no indicators found

    def _adjust_inconsistent_scores(self, analysis_text: str, scores: Dict[str, int], session_id: str) -> Dict[str, int]:
        """Adjust scores that don't match textual analysis."""
        adjusted_scores = scores.copy()
        text_lower = analysis_text.lower()
        
        # Bias consistency adjustments
        if any(phrase in text_lower for phrase in ['minimal bias', 'neutral', 'balanced', 'unbiased']):
            if scores.get('bias', 0) > 30:
                adjusted_scores['bias'] = 20
                self.logger.debug(f"Adjusted bias score to 20 for session {session_id}")
        elif any(phrase in text_lower for phrase in ['extreme bias', 'highly biased', 'severe bias']):
            if scores.get('bias', 0) < 70:
                adjusted_scores['bias'] = 80
                self.logger.debug(f"Adjusted bias score to 80 for session {session_id}")

        # Manipulation consistency adjustments
        if any(phrase in text_lower for phrase in ['no manipulation', 'straightforward', 'honest']):
            if scores.get('manipulation', 0) > 30:
                adjusted_scores['manipulation'] = 15
                self.logger.debug(f"Adjusted manipulation score to 15 for session {session_id}")
        elif any(phrase in text_lower for phrase in ['extreme manipulation', 'highly manipulative']):
            if scores.get('manipulation', 0) < 70:
                adjusted_scores['manipulation'] = 80
                self.logger.debug(f"Adjusted manipulation score to 80 for session {session_id}")

        return adjusted_scores

    def _generate_additional_analyses(self,
                                    article_text: str,
                                    previous_analysis: Dict[str, Any],
                                    llm_scores: Dict[str, int],
                                    session_id: str) -> Dict[str, str]:
        """Generate additional detailed analyses when needed."""
        additional_analyses = {}

        try:
            # Generate bias analysis if high bias detected
            if llm_scores.get('bias', 0) > self.bias_threshold:
                try:
                    bias_prompt = get_context_prompt_template(
                        'bias_detection',
                        article_text=article_text,
                        source=previous_analysis.get('source', 'Unknown'),
                        topic_domain=previous_analysis.get('topic_domain', 'general'),
                        prediction=previous_analysis.get('prediction', 'Unknown'),
                        confidence=previous_analysis.get('confidence', 0.0),
                        session_id=session_id
                    )

                    response = self.model.generate_content(bias_prompt)
                    if self._is_valid_response(response):
                        additional_analyses['bias_analysis'] = response.candidates[0].content.parts[0].text
                except Exception as e:
                    self.logger.warning(f"Failed to generate bias analysis for session {session_id}: {str(e)}")

            # Generate manipulation analysis if high manipulation detected
            if llm_scores.get('manipulation', 0) > self.manipulation_threshold:
                try:
                    manipulation_prompt = get_context_prompt_template(
                        'emotional_manipulation',
                        article_text=article_text,
                        emotional_indicators={'high_manipulation_detected': True},
                        session_id=session_id
                    )

                    response = self.model.generate_content(manipulation_prompt)
                    if self._is_valid_response(response):
                        additional_analyses['emotional_analysis'] = response.candidates[0].content.parts[0].text
                except Exception as e:
                    self.logger.warning(f"Failed to generate manipulation analysis for session {session_id}: {str(e)}")

            # Generate propaganda analysis for high risk content
            if self.enable_propaganda_analysis and llm_scores.get('risk', 0) > 70:
                try:
                    propaganda_prompt = get_context_prompt_template(
                        'propaganda_detection',
                        article_text=article_text,
                        session_id=session_id
                    )

                    response = self.model.generate_content(propaganda_prompt)
                    if self._is_valid_response(response):
                        additional_analyses['propaganda_analysis'] = response.candidates[0].content.parts[0].text
                except Exception as e:
                    self.logger.warning(f"Failed to generate propaganda analysis for session {session_id}: {str(e)}")

        except Exception as e:
            self.logger.warning(f"Additional analysis generation failed for session {session_id}: {str(e)}")

        return additional_analyses

    def _create_fallback_analysis(self, article_text: str, previous_analysis: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Create fallback analysis when LLM fails completely."""
        self.logger.info(f"Creating fallback analysis for session {session_id}")
        
        # Use pattern-based analysis as fallback
        try:
            bias_analysis = self.bias_database.analyze_bias_patterns(article_text)
            manipulation_report = self.manipulation_detector.get_manipulation_report(article_text)
            
            # Generate conservative scores based on pattern analysis
            bias_score = min(70, bias_analysis.get('bias_intensity_score', 50))
            manipulation_score = min(70, manipulation_report.get('overall_manipulation_score', 5) * 10)
            
            return {
                'llm_analysis': f"Automated fallback analysis for article of {len(article_text)} characters. Pattern-based analysis completed. Manual review recommended for comprehensive assessment.",
                'llm_scores': {
                    'bias': int(bias_score),
                    'manipulation': int(manipulation_score),
                    'credibility': 50,  # Neutral credibility
                    'risk': max(int(bias_score), int(manipulation_score))
                },
                'context_scores': {
                    'bias_score': int(bias_score),
                    'manipulation_score': int(manipulation_score),
                    'credibility': 50,
                    'risk_level': 'MEDIUM',
                    'risk_score': max(int(bias_score), int(manipulation_score)),
                    'overall_context_score': max(bias_score, manipulation_score) / 10.0
                },
                'pattern_analysis': bias_analysis,
                'manipulation_report': manipulation_report,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'scoring_method': 'pattern_based_fallback',
                    'model_used': 'pattern_analysis',
                    'consistency_validated': False,
                    'is_fallback': True,
                    'session_id': session_id
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fallback analysis creation failed for session {session_id}: {str(e)}")
            # Ultimate fallback - neutral scores
            return {
                'llm_analysis': f"Basic analysis for article of {len(article_text)} characters. Analysis incomplete - manual review required.",
                'llm_scores': {'bias': 50, 'manipulation': 50, 'credibility': 50, 'risk': 50},
                'context_scores': {
                    'bias_score': 50, 'manipulation_score': 50, 'credibility': 50,
                    'risk_level': 'MEDIUM', 'risk_score': 50, 'overall_context_score': 5.0
                },
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'scoring_method': 'emergency_fallback',
                    'is_fallback': True,
                    'session_id': session_id
                }
            }

    def _create_fallback_llm_analysis(self, article_text: str, session_id: str) -> Dict[str, Any]:
        """Create fallback LLM analysis when main analysis fails."""
        return {
            'analysis_text': f"Automated analysis for article of {len(article_text)} characters. Unable to complete full LLM analysis - using pattern-based assessment. Manual review recommended.",
            'scores': {'bias': 50, 'manipulation': 50, 'credibility': 50, 'risk': 50},
            'scoring_method': 'fallback_analysis',
            'consistency_validated': False,
            'is_fallback': True,
            'session_id': session_id
        }

    def _get_risk_level_from_scores(self, scores: Dict[str, int]) -> str:
        """Convert numerical risk score to risk level classification."""
        risk_score = scores.get('risk', 50)
        
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"

    def _is_valid_response(self, response) -> bool:
        """Check if LLM response is valid and not blocked by safety filters."""
        try:
            return (response and
                    hasattr(response, 'candidates') and
                    response.candidates and
                    len(response.candidates) > 0 and
                    response.candidates[0].finish_reason != 'SAFETY' and
                    response.candidates[0].content and
                    response.candidates[0].content.parts and
                    len(response.candidates[0].content.parts) > 0)
        except (AttributeError, IndexError):
            return False

    def _respect_rate_limits(self) -> None:
        """Implement rate limiting for API calls with configurable delay."""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                sleep_time = self.rate_limit - time_since_last
                self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _format_success_response(self, result: Dict[str, Any], confidence: float, 
                                processing_time: float, session_id: str, 
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format successful response with comprehensive metadata."""
        return {
            'success': True,
            'result': result,
            'confidence': round(confidence, 3),
            'processing_time': round(processing_time, 3),
            'session_id': session_id,
            'metadata': {
                **metadata,
                'timestamp': datetime.now().isoformat(),
                'agent_name': self.agent_name
            }
        }

    def _format_error_response(self, error: Exception, session_id: str, 
                              processing_time: float) -> Dict[str, Any]:
        """Format error response with comprehensive error information."""
        error_info = handle_context_analyzer_exception(error)
        error_info['session_id'] = session_id
        
        return {
            'success': False,
            'error': error_info,
            'processing_time': round(processing_time, 3),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for monitoring."""
        base_metrics = super().get_performance_metrics() if hasattr(super(), 'get_performance_metrics') else {}
        
        avg_processing_time = (
            sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            if self.metrics['processing_times'] else 0
        )
        
        success_rate = (
            (self.metrics['successful_analyses'] / self.metrics['total_analyses'] * 100)
            if self.metrics['total_analyses'] > 0 else 0
        )

        context_metrics = {
            'analyses_completed': self.metrics['total_analyses'],
            'successful_analyses': self.metrics['successful_analyses'],
            'failed_analyses': self.metrics['failed_analyses'],
            'success_rate_percent': round(success_rate, 2),
            'average_processing_time_seconds': round(avg_processing_time, 3),
            'llm_scores_generated': self.metrics['llm_scores_generated'],
            'high_bias_detected': self.metrics['high_bias_detected'],
            'high_manipulation_detected': self.metrics['high_manipulation_detected'],
            'score_consistency_checks': self.metrics['score_consistency_checks'],
            'retry_attempts': self.metrics['retry_attempts'],
            'fallback_analyses': self.metrics['fallback_analyses'],
            'model_config': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'bias_threshold': self.bias_threshold,
                'manipulation_threshold': self.manipulation_threshold
            }
        }

        return {**base_metrics, **context_metrics}

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data for context analysis with detailed feedback."""
        try:
            validation_result = self.input_validator.validate_processing_input(input_data)
            if validation_result.is_valid:
                return True, "Input validation passed"
            else:
                return False, f"Validation failed: {'; '.join(validation_result.errors)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def _process_internal(self, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Internal processing method for BaseAgent compatibility.

        Args:
            input_data: Input data dictionary containing text and previous analysis
            session_id: Optional session identifier for tracking

        Returns:
            Processing result dictionary
        """
        try:
            # Extract and validate input parameters
            article_text = input_data.get('text', '')
            previous_analysis = input_data.get('previous_analysis', {})
            include_detailed_analysis = input_data.get(
                'include_detailed_analysis',
                self.enable_detailed_analysis
            )

            # Determine analysis depth based on previous analysis confidence
            bert_confidence = previous_analysis.get('confidence', 1.0)
            force_detailed = (
                include_detailed_analysis or
                bert_confidence < 0.7 or
                self.enable_detailed_analysis
            )

            # Perform comprehensive context analysis with retry logic
            analysis_result = self._analyze_context_with_retry(
                article_text=article_text,
                previous_analysis=previous_analysis,
                include_detailed_analysis=force_detailed,
                session_id=session_id
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"Error in _process_internal: {str(e)}", extra={'session_id': session_id})
            raise ContextAnalyzerError(f"Internal processing failed: {str(e)}")


# Testing functionality
if __name__ == "__main__":
    """Test the context analyzer agent with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize agent
        agent = ContextAnalyzerAgent()
        
        # Test input with potential bias and manipulation
        test_input = {
            "text": """
            The corrupt establishment politicians are once again betraying hardworking Americans!
            This outrageous scandal exposes their lies while patriots demand justice.
            Every real American must wake up to this crisis before it's too late.
            The biased mainstream media refuses to report the truth about this dangerous situation.
            """,
            "previous_analysis": {
                "prediction": "FAKE",
                "confidence": 0.75,
                "source": "Social Media Post",
                "topic_domain": "political"
            },
            "include_detailed_analysis": True
        }

        print("=== CONTEXT ANALYZER TEST ===")
        result = agent.process(test_input, session_id="test_session_context_001")

        if result['success']:
            analysis_data = result['result']
            llm_scores = analysis_data.get('llm_scores', {})
            context_scores = analysis_data.get('context_scores', {})
            
            print("✅ Context Analysis Results:")
            print(f"   Bias Score: {llm_scores.get('bias', 0)}/100")
            print(f"   Manipulation Score: {llm_scores.get('manipulation', 0)}/100")
            print(f"   Credibility Score: {llm_scores.get('credibility', 50)}/100")
            print(f"   Risk Level: {context_scores.get('risk_level', 'UNKNOWN')}")
            print(f"   Overall Context Score: {context_scores.get('overall_context_score', 0):.1f}/10")
            print(f"   Processing Time: {result['processing_time']:.3f}s")
            print(f"   Confidence: {result['confidence']:.3f}")
            
            # Show additional analyses if available
            if analysis_data.get('bias_analysis'):
                print("   ✓ Detailed bias analysis included")
            if analysis_data.get('emotional_analysis'):
                print("   ✓ Emotional manipulation analysis included")
            if analysis_data.get('propaganda_analysis'):
                print("   ✓ Propaganda analysis included")
                
        else:
            print(f"❌ Analysis failed: {result['error']}")

        # Show performance metrics
        metrics = agent.get_performance_metrics()
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Total analyses: {metrics['analyses_completed']}")
        print(f"Success rate: {metrics['success_rate_percent']:.1f}%")
        print(f"Average processing time: {metrics['average_processing_time_seconds']:.3f}s")
        print(f"High bias detections: {metrics['high_bias_detected']}")
        print(f"High manipulation detections: {metrics['high_manipulation_detected']}")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
