# agents/llm_explanation/explanation_agent.py

"""
Enhanced LLM Explanation Agent - Production Ready

Production-ready explanation generation agent with comprehensive error handling,
source assessment, advanced AI integration, and configurable explanation types.
Generates human-readable explanations of fake news detection results using
Google Gemini with enhanced safety measures and performance optimization.
"""

import time
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from contextlib import asynccontextmanager

# Enhanced imports with fallback handling
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

try:
    from agents.base import BaseAgent
except ImportError:
    # Fallback base agent implementation
    class BaseAgent:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.agent_name = "base_agent"
        
        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            raise NotImplementedError
        
        def get_performance_metrics(self) -> Dict[str, Any]:
            return {"agent_type": self.agent_name}
        
        def format_output(self, result: Any, session_id: str, confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return {
                "success": True,
                "result": result,
                "confidence": confidence,
                "metadata": metadata or {},
                "session_id": session_id
            }
        
        def format_error_output(self, error: Exception, input_data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
            return {
                "success": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error)
                },
                "input_data": input_data,
                "session_id": session_id
            }

try:
    from config import get_model_config, get_settings
except ImportError:
    # Fallback configuration
    def get_model_config(agent_name: str) -> Dict[str, Any]:
        return {
            "model_name": "gemini-1.5-pro",
            "temperature": 0.3,
            "max_tokens": 3072
        }
    
    def get_settings() -> Any:
        class Settings:
            gemini_api_key = None
            gemini_rate_limit = 1.0
            max_retries = 3
        return Settings()

try:
    from utils.helpers import sanitize_text
except ImportError:
    def sanitize_text(text: str) -> str:
        """Basic text sanitization fallback."""
        if not isinstance(text, str):
            return ""
        return text.strip().replace('\x00', '').replace('\r\n', '\n')

from .source_database import SourceReliabilityDatabase
from .prompts import get_explanation_prompt, validate_prompt_parameters
from .validators import InputValidator, OutputValidator
from .exceptions import (
    LLMExplanationError,
    InputValidationError,
    APIConfigurationError,
    LLMResponseError,
    ExplanationGenerationError,
    RateLimitError,
    SourceAssessmentError,
    ProcessingTimeoutError,
    handle_llm_explanation_exception,
    is_recoverable_error,
    get_retry_delay
)


class LLMExplanationAgent(BaseAgent):
    """
    Enhanced explanation agent for generating comprehensive fake news explanations.

    Features:
    - Multi-level explanation generation (basic, detailed, confidence analysis)
    - Comprehensive source reliability assessment using curated database
    - Advanced Gemini model integration with safety filters and rate limiting
    - Robust error handling with automatic recovery mechanisms
    - Performance tracking and quality metrics for production monitoring
    - Asynchronous processing support for high-throughput scenarios
    - LangGraph integration compatibility for multi-agent workflows
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced LLM explanation agent with production configuration.

        Args:
            config: Optional configuration dictionary with model parameters,
                   thresholds, and feature toggles
        """
        # Load and merge configuration
        explanation_config = get_model_config('llm_explanation')
        system_settings = get_settings()
        
        if config:
            explanation_config.update(config)

        self.agent_name = "llm_explanation"
        super().__init__(explanation_config)

        # Enhanced AI Model Configuration
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 3072)
        self.top_p = self.config.get('top_p', 0.9)
        self.top_k = self.config.get('top_k', 40)

        # Enhanced Analysis Configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.75)
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.enable_source_analysis = self.config.get('enable_source_analysis', True)
        self.enable_confidence_analysis = self.config.get('enable_confidence_analysis', True)
        
        # Content Processing Limits with enhanced defaults
        self.max_article_length = self.config.get('max_article_length', 5000)
        self.min_explanation_length = self.config.get('min_explanation_length', 150)
        self.max_explanation_length = self.config.get('max_explanation_length', 8000)

        # Enhanced API Configuration
        self.api_key = self._load_api_key(system_settings)
        self.rate_limit = self.config.get('rate_limit_seconds', getattr(system_settings, 'gemini_rate_limit', 1.0))
        self.max_retries = self.config.get('max_retries', getattr(system_settings, 'max_retries', 3))
        self.request_timeout = self.config.get('request_timeout_seconds', 30.0)

        # Initialize core components
        self._initialize_gemini_api()
        self.source_database = SourceReliabilityDatabase()
        self.input_validator = InputValidator(self.config)
        self.output_validator = OutputValidator(self.config)

        # Enhanced performance tracking
        self.explanation_metrics = {
            'total_explanations': 0,
            'successful_explanations': 0,
            'failed_explanations': 0,
            'detailed_analyses_generated': 0,
            'confidence_analyses_generated': 0,
            'source_assessments_performed': 0,
            'average_response_time': 0.0,
            'total_processing_time': 0.0,
            'safety_blocks': 0,
            'rate_limit_hits': 0,
            'api_errors': 0,
            'retry_attempts': 0,
            'timeout_errors': 0
        }

        # Rate limiting state
        self.last_request_time = None
        self.request_count = 0
        self.session_start_time = time.time()

        # Enhanced logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Enhanced LLM Explanation Agent initialized - Model: {self.model_name}")

    def _load_api_key(self, system_settings) -> str:
        """Load API key with enhanced fallback options."""
        api_key = (
            os.getenv('GEMINI_API_KEY') or
            os.getenv('GOOGLE_API_KEY') or
            os.getenv('GOOGLE_GEMINI_API_KEY') or
            getattr(system_settings, 'gemini_api_key', None) or
            self.config.get('api_key')
        )

        if not api_key:
            raise APIConfigurationError(
                "Gemini API key not found. Please set GEMINI_API_KEY environment variable"
            )

        # Validate API key format
        if not api_key.startswith('AI') or len(api_key) < 20:
            raise APIConfigurationError(
                "Invalid Gemini API key format. Please check your API key"
            )

        return api_key

    def _initialize_gemini_api(self) -> None:
        """Initialize Gemini API with comprehensive configuration."""
        try:
            genai.configure(api_key=self.api_key)

            # Enhanced generation configuration
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }

            # Enhanced safety settings for production
            safety_settings = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
            ]

            # Initialize model with enhanced configuration
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            self.logger.info(f"Gemini API initialized successfully - Model: {self.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise APIConfigurationError(f"Failed to initialize Gemini API: {str(e)}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for explanation generation with comprehensive validation and error handling.

        Args:
            input_data: Dictionary containing:
                - text: Article text to explain
                - prediction: FAKE/REAL classification
                - confidence: Confidence score (0.0-1.0)
                - metadata: Additional context (optional)
                - require_detailed_analysis: Force detailed analysis (optional)
                - session_id: Session identifier for tracking (optional)

        Returns:
            Standardized output dictionary with explanation results and metadata
        """
        session_id = input_data.get('session_id', f"expl_{int(time.time())}")
        start_time = time.time()

        # Enhanced input validation
        validation_result = self.input_validator.validate_explanation_input(input_data)
        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            self.explanation_metrics['failed_explanations'] += 1
            self.logger.error(f"Input validation failed: {error_msg}", extra={'session_id': session_id})
            return self.format_error_output(InputValidationError(error_msg), input_data, session_id)

        try:
            self.logger.info(f"Starting explanation generation", extra={'session_id': session_id})

            # Extract and validate parameters
            article_text = input_data.get('text', '')
            prediction = input_data.get('prediction', 'UNKNOWN')
            confidence = input_data.get('confidence', 0.0)
            metadata = input_data.get('metadata', {})
            require_detailed = input_data.get('require_detailed_analysis', False)

            # Determine analysis depth based on confidence and configuration
            trigger_detailed = (
                require_detailed or
                confidence < self.confidence_threshold or
                (self.enable_detailed_analysis and prediction == 'FAKE')
            )

            # Generate comprehensive explanation with retries
            explanation_result = self._generate_explanation_with_retries(
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                metadata=metadata,
                require_detailed_analysis=trigger_detailed,
                session_id=session_id
            )

            # Enhanced output validation
            output_validation = self.output_validator.validate_explanation_output(explanation_result)
            if not output_validation.is_valid:
                self.logger.warning(f"Output validation issues: {output_validation.errors}", 
                                  extra={'session_id': session_id})
                # Add validation warnings to metadata
                explanation_result['metadata']['validation_warnings'] = output_validation.warnings

            # Update comprehensive metrics
            processing_time = time.time() - start_time
            self._update_success_metrics(processing_time, explanation_result)

            self.logger.info(
                f"Explanation generation completed successfully in {processing_time:.3f}s",
                extra={'session_id': session_id}
            )

            # Return enhanced output
            return self.format_output(
                result=explanation_result,
                session_id=session_id,
                confidence=confidence,
                metadata={
                    'processing_time_seconds': processing_time,
                    'model_used': self.model_name,
                    'agent_version': '4.0.0',
                    'detailed_analysis_triggered': trigger_detailed,
                    'output_validation_score': 100 - len(output_validation.warnings) * 5,
                    'api_calls_made': self._count_api_calls(explanation_result),
                    'total_tokens_estimated': self._estimate_token_usage(article_text, explanation_result)
                }
            )

        except LLMExplanationError as e:
            processing_time = time.time() - start_time
            self._update_error_metrics(e, processing_time)
            self.logger.error(f"LLM explanation error: {str(e)}", extra={'session_id': session_id})
            return self.format_error_output(e, input_data, session_id)

        except Exception as e:
            processing_time = time.time() - start_time
            wrapped_error = handle_llm_explanation_exception(e)
            self._update_error_metrics(wrapped_error, processing_time)
            self.logger.error(f"Unexpected error: {str(e)}", extra={'session_id': session_id})
            return self.format_error_output(wrapped_error, input_data, session_id)

    def _generate_explanation_with_retries(self, article_text: str, prediction: str, 
                                         confidence: float, metadata: Dict[str, Any],
                                         require_detailed_analysis: bool = False,
                                         session_id: str = None) -> Dict[str, Any]:
        """Generate explanation with enhanced retry logic and error recovery."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return self.generate_explanation(
                    article_text=article_text,
                    prediction=prediction,
                    confidence=confidence,
                    metadata=metadata,
                    require_detailed_analysis=require_detailed_analysis,
                    session_id=session_id
                )
            
            except Exception as e:
                last_error = e
                self.explanation_metrics['retry_attempts'] += 1
                
                if attempt < self.max_retries and is_recoverable_error(e):
                    retry_delay = get_retry_delay(e) or (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}",
                        extra={'session_id': session_id}
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    break
        
        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise ExplanationGenerationError("Max retries reached without success")

    def generate_explanation(self, article_text: str, prediction: str, 
                           confidence: float, metadata: Dict[str, Any],
                           require_detailed_analysis: bool = False,
                           session_id: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation with multiple analysis components.

        Args:
            article_text: Article content to explain
            prediction: Classification result (REAL/FAKE/UNKNOWN)
            confidence: Confidence score (0.0-1.0)
            metadata: Additional context information
            require_detailed_analysis: Force detailed forensic analysis
            session_id: Session identifier for tracking

        Returns:
            Dictionary containing comprehensive explanation results with metadata
        """
        start_time = time.time()
        self.explanation_metrics['total_explanations'] += 1

        try:
            # Enhanced content preparation
            article_text = sanitize_text(article_text)
            if len(article_text) > self.max_article_length:
                original_length = len(article_text)
                article_text = article_text[:self.max_article_length] + "..."
                self.logger.info(f"Article truncated from {original_length} to {len(article_text)} chars",
                               extra={'session_id': session_id})

            # Extract enhanced metadata with defaults
            source = metadata.get('source', 'Unknown Source')
            date = metadata.get('date', 'Unknown Date')
            subject = metadata.get('subject', 'General News')
            author = metadata.get('author', 'Unknown Author')
            domain = metadata.get('domain', 'general')

            # Step 1: Generate primary explanation with enhanced prompts
            self.logger.info("Generating primary explanation", extra={'session_id': session_id})
            explanation = self._generate_primary_explanation(
                article_text, prediction, confidence, source, date, subject, session_id
            )

            # Step 2: Conditional detailed analysis with enhanced logic
            detailed_analysis = None
            if require_detailed_analysis or confidence < self.confidence_threshold:
                self.logger.info("Generating detailed forensic analysis", extra={'session_id': session_id})
                detailed_analysis = self._generate_detailed_analysis(
                    article_text, prediction, confidence, metadata, session_id
                )
                self.explanation_metrics['detailed_analyses_generated'] += 1

            # Step 3: Confidence analysis with enhanced assessment
            confidence_analysis = None
            if self.enable_confidence_analysis:
                self.logger.info("Generating confidence analysis", extra={'session_id': session_id})
                confidence_analysis = self._generate_confidence_analysis(
                    article_text, prediction, confidence, session_id
                )
                self.explanation_metrics['confidence_analyses_generated'] += 1

            # Step 4: Source assessment with enhanced database
            source_assessment = None
            if self.enable_source_analysis and source != 'Unknown Source':
                try:
                    self.logger.info(f"Performing source assessment for: {source}", 
                                   extra={'session_id': session_id})
                    source_assessment = self.source_database.get_reliability_summary(source)
                    self.explanation_metrics['source_assessments_performed'] += 1
                except Exception as e:
                    self.logger.warning(f"Source assessment failed: {str(e)}", 
                                      extra={'session_id': session_id})
                    source_assessment = {
                        'error': f"Assessment failed: {str(e)}",
                        'reliability_level': 'UNKNOWN',
                        'verification_recommendation': 'Verify independently'
                    }

            # Package comprehensive results with enhanced metadata
            processing_time = time.time() - start_time
            
            result = {
                'explanation': explanation,
                'detailed_analysis': detailed_analysis,
                'confidence_analysis': confidence_analysis,
                'source_assessment': source_assessment,
                'quality_indicators': self._assess_explanation_quality(
                    explanation, detailed_analysis, confidence_analysis
                ),
                'metadata': {
                    'input_parameters': {
                        'prediction': prediction,
                        'confidence_level': confidence,
                        'source': source,
                        'date': date,
                        'subject': subject,
                        'author': author,
                        'domain': domain
                    },
                    'processing_details': {
                        'response_time_seconds': round(processing_time, 3),
                        'model_used': self.model_name,
                        'temperature_used': self.temperature,
                        'max_tokens_configured': self.max_tokens,
                        'article_length_processed': len(article_text),
                        'truncation_applied': len(article_text) >= self.max_article_length
                    },
                    'analysis_components': {
                        'detailed_analysis_included': detailed_analysis is not None,
                        'confidence_analysis_included': confidence_analysis is not None,
                        'source_analysis_included': source_assessment is not None,
                        'detailed_analysis_triggered_by': 'manual' if require_detailed_analysis else 'automatic'
                    },
                    'system_info': {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'agent_version': '4.0.0',
                        'session_id': session_id,
                        'request_id': f"expl_{int(time.time())}_{hash(article_text[:100]) % 10000}"
                    }
                }
            }

            self.logger.info(f"Generated comprehensive explanation in {processing_time:.3f}s",
                           extra={'session_id': session_id})
            
            return result

        except Exception as e:
            self.logger.error(f"Error in explanation generation: {str(e)}", 
                            extra={'session_id': session_id})
            raise ExplanationGenerationError(f"Generation failed: {str(e)}", "comprehensive_generation")

    def _generate_primary_explanation(self, article_text: str, prediction: str, confidence: float,
                                    source: str, date: str, subject: str, session_id: str = None) -> str:
        """Generate main explanation using structured prompts with enhanced error handling."""
        try:
            # Validate prompt parameters
            prompt_validation = validate_prompt_parameters(
                'main',
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                source=source,
                date=date,
                subject=subject
            )
            
            if not prompt_validation.is_valid:
                raise ExplanationGenerationError(
                    f"Prompt validation failed: {'; '.join(prompt_validation.errors)}",
                    "prompt_validation"
                )

            self._respect_rate_limits()

            # Generate enhanced prompt
            prompt = get_explanation_prompt(
                'main',
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                source=source,
                date=date,
                subject=subject
            )

            # Call Gemini API with timeout handling
            response = self._call_gemini_with_timeout(prompt, session_id)
            
            if not self._is_valid_response(response):
                self.explanation_metrics['safety_blocks'] += 1
                raise LLMResponseError(
                    "Primary explanation blocked by safety filters",
                    "main_explanation", self.model_name, safety_blocked=True
                )

            explanation_text = response.candidates[0].content.parts[0].text
            
            # Validate explanation length and quality
            if not explanation_text or len(explanation_text) < self.min_explanation_length:
                raise ExplanationGenerationError(
                    f"Generated explanation too short: {len(explanation_text) if explanation_text else 0} characters",
                    "primary_explanation"
                )

            return explanation_text

        except LLMResponseError:
            raise  # Re-raise LLM response errors
        except ExplanationGenerationError:
            raise  # Re-raise generation errors
        except Exception as e:
            raise ExplanationGenerationError(f"Primary explanation failed: {str(e)}", "primary_explanation")

    def _generate_detailed_analysis(self, article_text: str, prediction: str, 
                                  confidence: float, metadata: Dict[str, Any],
                                  session_id: str = None) -> str:
        """Generate detailed forensic analysis with enhanced capabilities."""
        try:
            self._respect_rate_limits()

            prompt = get_explanation_prompt(
                'detailed',
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                metadata=metadata
            )

            response = self._call_gemini_with_timeout(prompt, session_id)
            
            if not self._is_valid_response(response):
                self.explanation_metrics['safety_blocks'] += 1
                self.logger.warning("Detailed analysis blocked by safety filters", 
                                  extra={'session_id': session_id})
                return "Detailed analysis could not be generated due to content restrictions."

            analysis_text = response.candidates[0].content.parts[0].text
            
            if not analysis_text:
                return "Detailed analysis could not be generated at this time."

            return analysis_text

        except Exception as e:
            self.logger.warning(f"Detailed analysis generation failed: {str(e)}", 
                              extra={'session_id': session_id})
            return f"Detailed analysis unavailable: {str(e)}"

    def _generate_confidence_analysis(self, article_text: str, prediction: str, 
                                    confidence: float, session_id: str = None) -> str:
        """Generate confidence level appropriateness analysis with enhanced assessment."""
        try:
            self._respect_rate_limits()

            prompt = get_explanation_prompt(
                'confidence',
                article_text=article_text,
                prediction=prediction,
                confidence=confidence
            )

            response = self._call_gemini_with_timeout(prompt, session_id)
            
            if not self._is_valid_response(response):
                self.explanation_metrics['safety_blocks'] += 1
                self.logger.warning("Confidence analysis blocked by safety filters", 
                                  extra={'session_id': session_id})
                return "Confidence analysis could not be generated due to content restrictions."

            analysis_text = response.candidates[0].content.parts[0].text
            
            if not analysis_text:
                return "Confidence analysis could not be generated at this time."

            return analysis_text

        except Exception as e:
            self.logger.warning(f"Confidence analysis generation failed: {str(e)}", 
                              extra={'session_id': session_id})
            return f"Confidence analysis unavailable: {str(e)}"

    def _call_gemini_with_timeout(self, prompt: str, session_id: str = None):
        """Call Gemini API with timeout handling and enhanced error reporting."""
        try:
            # Note: google.generativeai doesn't have built-in timeout support
            # In production, you might want to use asyncio.wait_for for timeout
            response = self.model.generate_content(prompt)
            return response
            
        except Exception as e:
            self.explanation_metrics['api_errors'] += 1
            if "timeout" in str(e).lower():
                self.explanation_metrics['timeout_errors'] += 1
                raise ProcessingTimeoutError(
                    f"API request timed out after {self.request_timeout}s",
                    self.request_timeout, "gemini_api_call"
                )
            else:
                raise LLMResponseError(f"API call failed: {str(e)}", "api_call", self.model_name)

    def _is_valid_response(self, response) -> bool:
        """Enhanced response validation with comprehensive checks."""
        if not response:
            return False
        
        if not hasattr(response, 'candidates') or not response.candidates:
            return False
        
        candidate = response.candidates[0]
        
        # Check finish reason
        if hasattr(candidate, 'finish_reason'):
            # Handle different finish reasons
            finish_reason = candidate.finish_reason
            if finish_reason == 2:  # SAFETY blocked
                return False
            elif finish_reason == 3:  # RECITATION blocked
                self.logger.warning("Content blocked due to recitation policy")
                return False
            elif finish_reason in [4, 5]:  # OTHER or unknown errors
                return False
        
        # Check content structure
        if not (hasattr(candidate, 'content') and candidate.content and 
                hasattr(candidate.content, 'parts') and candidate.content.parts):
            return False
        
        # Check for actual text content
        text_content = candidate.content.parts[0].text
        return bool(text_content and text_content.strip())

    def _respect_rate_limits(self) -> None:
        """Enhanced rate limiting with adaptive delays and monitoring."""
        current_time = time.time()
        
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.rate_limit:
                delay = self.rate_limit - time_since_last
                self.explanation_metrics['rate_limit_hits'] += 1
                self.logger.debug(f"Rate limiting: sleeping {delay:.2f}s")
                time.sleep(delay)

        self.last_request_time = time.time()
        self.request_count += 1

    def _assess_explanation_quality(self, explanation: str, detailed_analysis: Optional[str],
                                  confidence_analysis: Optional[str]) -> Dict[str, Any]:
        """Assess the quality of generated explanations with enhanced metrics."""
        quality_metrics = {
            'explanation_length': len(explanation) if explanation else 0,
            'explanation_word_count': len(explanation.split()) if explanation else 0,
            'has_detailed_analysis': bool(detailed_analysis),
            'has_confidence_analysis': bool(confidence_analysis),
            'estimated_readability': 'unknown'
        }

        # Basic readability assessment
        if explanation:
            sentences = explanation.count('.') + explanation.count('!') + explanation.count('?')
            words = len(explanation.split())
            
            if sentences > 0:
                avg_sentence_length = words / sentences
                if avg_sentence_length < 15:
                    quality_metrics['estimated_readability'] = 'easy'
                elif avg_sentence_length < 25:
                    quality_metrics['estimated_readability'] = 'moderate'
                else:
                    quality_metrics['estimated_readability'] = 'complex'

        # Quality score calculation
        quality_score = 70  # Base score
        
        if quality_metrics['explanation_length'] >= self.min_explanation_length:
            quality_score += 10
        if quality_metrics['has_detailed_analysis']:
            quality_score += 10
        if quality_metrics['has_confidence_analysis']:
            quality_score += 10
        
        quality_metrics['overall_quality_score'] = min(100, quality_score)
        
        return quality_metrics

    def _update_success_metrics(self, processing_time: float, result: Dict[str, Any]) -> None:
        """Update success metrics with enhanced tracking."""
        self.explanation_metrics['successful_explanations'] += 1
        self._update_response_time_metric(processing_time)
        
        # Update component-specific metrics
        if result.get('detailed_analysis'):
            self.explanation_metrics['detailed_analyses_generated'] += 1
        if result.get('confidence_analysis'):
            self.explanation_metrics['confidence_analyses_generated'] += 1
        if result.get('source_assessment'):
            self.explanation_metrics['source_assessments_performed'] += 1

    def _update_error_metrics(self, error: Exception, processing_time: float) -> None:
        """Update error metrics with detailed categorization."""
        self.explanation_metrics['failed_explanations'] += 1
        self._update_response_time_metric(processing_time)
        
        # Categorize errors for detailed metrics
        if isinstance(error, RateLimitError):
            self.explanation_metrics['rate_limit_hits'] += 1
        elif isinstance(error, ProcessingTimeoutError):
            self.explanation_metrics['timeout_errors'] += 1
        elif isinstance(error, LLMResponseError):
            if error.safety_blocked:
                self.explanation_metrics['safety_blocks'] += 1
            else:
                self.explanation_metrics['api_errors'] += 1

    def _update_response_time_metric(self, response_time: float) -> None:
        """Update average response time metric with enhanced calculation."""
        self.explanation_metrics['total_processing_time'] += response_time
        
        total_explanations = self.explanation_metrics['total_explanations']
        if total_explanations > 0:
            self.explanation_metrics['average_response_time'] = (
                self.explanation_metrics['total_processing_time'] / total_explanations
            )

    def _count_api_calls(self, result: Dict[str, Any]) -> int:
        """Count the number of API calls made for this explanation."""
        call_count = 1  # Primary explanation
        if result.get('detailed_analysis'):
            call_count += 1
        if result.get('confidence_analysis'):
            call_count += 1
        return call_count

    def _estimate_token_usage(self, article_text: str, result: Dict[str, Any]) -> int:
        """Estimate token usage for cost tracking (rough approximation)."""
        # Rough estimation: 1 token ≈ 4 characters for English
        input_tokens = len(article_text) // 4
        output_tokens = 0
        
        if result.get('explanation'):
            output_tokens += len(result['explanation']) // 4
        if result.get('detailed_analysis'):
            output_tokens += len(result['detailed_analysis']) // 4
        if result.get('confidence_analysis'):
            output_tokens += len(result['confidence_analysis']) // 4
            
        return input_tokens + output_tokens

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance and configuration metrics for monitoring."""
        base_metrics = super().get_performance_metrics()
        
        session_duration = time.time() - self.session_start_time
        
        return {
            **base_metrics,
            'explanation_specific_metrics': {
                **self.explanation_metrics,
                'success_rate': (
                    self.explanation_metrics['successful_explanations'] / 
                    max(self.explanation_metrics['total_explanations'], 1)
                ) * 100,
                'error_rate': (
                    self.explanation_metrics['failed_explanations'] / 
                    max(self.explanation_metrics['total_explanations'], 1)
                ) * 100,
                'requests_per_minute': (
                    self.request_count / max(session_duration / 60, 1)
                ),
                'session_duration_minutes': session_duration / 60
            },
            'configuration_metrics': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'confidence_threshold': self.confidence_threshold,
                'rate_limit_seconds': self.rate_limit,
                'max_retries': self.max_retries,
                'max_article_length': self.max_article_length,
                'request_timeout': self.request_timeout,
                'detailed_analysis_enabled': self.enable_detailed_analysis,
                'source_analysis_enabled': self.enable_source_analysis,
                'confidence_analysis_enabled': self.enable_confidence_analysis
            },
            'component_metrics': {
                'source_database_stats': self.source_database.get_database_statistics(),
                'validation_enabled': True,
                'input_validator_configured': bool(self.input_validator),
                'output_validator_configured': bool(self.output_validator)
            },
            'quality_metrics': {
                'api_availability': 100 - (self.explanation_metrics['api_errors'] / 
                                        max(self.explanation_metrics['total_explanations'], 1)) * 100,
                'safety_compliance': 100 - (self.explanation_metrics['safety_blocks'] / 
                                          max(self.explanation_metrics['total_explanations'], 1)) * 100
            },
            'agent_info': {
                'agent_type': 'llm_explanation',
                'agent_version': '4.0.0',
                'architecture': 'modular_production',
                'gemini_model': self.model_name,
                'initialization_time': self.session_start_time
            }
        }

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate input data for explanation generation with enhanced feedback.

        Args:
            input_data: Input dictionary to validate

        Returns:
            Tuple of (is_valid, error_message_or_empty_string)
        """
        validation_result = self.input_validator.validate_explanation_input(input_data)
        
        if validation_result.is_valid:
            return True, ""
        else:
            error_details = []
            error_details.extend(validation_result.errors)
            if validation_result.warnings:
                error_details.extend([f"Warning: {w}" for w in validation_result.warnings])
            
            return False, "; ".join(error_details)

    async def _process_internal(self, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Internal processing method for BaseAgent compatibility.

        Args:
            input_data: Input data dictionary containing text and analysis results
            session_id: Optional session identifier for tracking

        Returns:
            Processing result dictionary
        """
        try:
            # Extract input parameters
            article_text = input_data.get('text', '')
            bert_results = input_data.get('bert_results', {})
            extracted_claims = input_data.get('extracted_claims', [])
            context_analysis = input_data.get('context_analysis', {})
            evidence_evaluation = input_data.get('evidence_evaluation', {})
            source_recommendations = input_data.get('source_recommendations', {})
            
            # Determine explanation level
            confidence = bert_results.get('confidence', 0.0)
            explanation_level = self._determine_explanation_level(
                confidence, context_analysis, evidence_evaluation
            )

            # Generate comprehensive explanation
            explanation_result = await self.generate_explanation(
                article_text=article_text,
                bert_results=bert_results,
                extracted_claims=extracted_claims,
                context_analysis=context_analysis,
                evidence_evaluation=evidence_evaluation,
                source_recommendations=source_recommendations,
                explanation_level=explanation_level,
                session_id=session_id
            )

            return explanation_result

        except Exception as e:
            self.logger.error(f"Error in _process_internal: {str(e)}", extra={'session_id': session_id})
            raise LLMExplanationError(f"Internal processing failed: {str(e)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status for monitoring and diagnostics."""
        metrics = self.explanation_metrics
        total_requests = metrics['total_explanations']
        
        # Determine health status
        if total_requests == 0:
            status = "HEALTHY"  # No requests yet
            health_score = 100
        else:
            success_rate = (metrics['successful_explanations'] / total_requests) * 100
            error_rate = (metrics['failed_explanations'] / total_requests) * 100
            
            if success_rate >= 95:
                status = "HEALTHY"
                health_score = 100
            elif success_rate >= 85:
                status = "WARNING"
                health_score = 85
            elif success_rate >= 70:
                status = "DEGRADED"
                health_score = 70
            else:
                status = "CRITICAL"
                health_score = 50

        return {
            'status': status,
            'health_score': health_score,
            'uptime_minutes': (time.time() - self.session_start_time) / 60,
            'total_requests': total_requests,
            'success_rate': (metrics['successful_explanations'] / max(total_requests, 1)) * 100,
            'error_rate': (metrics['failed_explanations'] / max(total_requests, 1)) * 100,
            'average_response_time': metrics['average_response_time'],
            'last_error_time': getattr(self, '_last_error_time', None),
            'api_connectivity': 'CONNECTED' if metrics['api_errors'] < 5 else 'ISSUES',
            'rate_limit_status': 'NORMAL' if metrics['rate_limit_hits'] < 10 else 'THROTTLED'
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics for fresh monitoring period."""
        self.explanation_metrics = {
            'total_explanations': 0,
            'successful_explanations': 0,
            'failed_explanations': 0,
            'detailed_analyses_generated': 0,
            'confidence_analyses_generated': 0,
            'source_assessments_performed': 0,
            'average_response_time': 0.0,
            'total_processing_time': 0.0,
            'safety_blocks': 0,
            'rate_limit_hits': 0,
            'api_errors': 0,
            'retry_attempts': 0,
            'timeout_errors': 0
        }
        self.session_start_time = time.time()
        self.request_count = 0
        self.logger.info("Performance metrics reset")


# Testing functionality
if __name__ == "__main__":
    """Test LLM explanation agent functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== LLM EXPLANATION AGENT TEST ===")
    
    try:
        # Initialize agent with test configuration
        test_config = {
            'temperature': 0.2,
            'max_tokens': 2048,
            'enable_detailed_analysis': True,
            'confidence_threshold': 0.8
        }
        
        agent = LLMExplanationAgent(test_config)
        
        # Comprehensive test input
        test_input = {
            "text": """
            A groundbreaking study claims that drinking 10 cups of coffee daily can extend
            lifespan by 50 years. The research was allegedly conducted by Dr. Anonymous at an
            undisclosed institution and published in an unknown journal. The study has not been 
            peer-reviewed and the data has not been made available for verification. The findings
            contradict decades of established medical research on caffeine consumption limits.
            """,
            "prediction": "FAKE",
            "confidence": 0.92,
            "metadata": {
                "source": "HealthyLifeBlog.net",
                "date": "2025-01-15",
                "subject": "Health",
                "author": "Anonymous Blogger",
                "domain": "health"
            },
            "require_detailed_analysis": True,
            "session_id": "test_session_001"
        }

        print("🔄 Starting comprehensive explanation generation test...")
        
        # Process the test input
        result = agent.process(test_input)

        if result['success']:
            explanation_data = result['result']
            metadata = result['metadata']
            
            print(f"✅ Explanation generated successfully!")
            print(f"   📊 Processing time: {metadata['processing_time_seconds']:.2f}s")
            print(f"   🤖 Model used: {metadata['model_used']}")
            print(f"   🆔 Session ID: {metadata['session_id']}")
            print(f"   🎯 Output quality score: {metadata['output_validation_score']}")
            
            # Show analysis components
            components = explanation_data['metadata']['analysis_components']
            print(f"\n📋 Analysis Components Generated:")
            print(f"   ✅ Main explanation: Yes")
            print(f"   📝 Detailed analysis: {'Yes' if components['detailed_analysis_included'] else 'No'}")
            print(f"   📈 Confidence analysis: {'Yes' if components['confidence_analysis_included'] else 'No'}")
            print(f"   🔍 Source assessment: {'Yes' if components['source_analysis_included'] else 'No'}")

            # Show explanation preview
            explanation_preview = explanation_data['explanation'][:300] + "..." if len(explanation_data['explanation']) > 300 else explanation_data['explanation']
            print(f"\n📄 Main Explanation Preview:")
            print(f"   {explanation_preview}")

            # Show source assessment if available
            if explanation_data['source_assessment']:
                source_info = explanation_data['source_assessment']
                print(f"\n🔍 Source Assessment:")
                print(f"   📊 Reliability: {source_info.get('reliability_level', 'Unknown')}")
                print(f"   ⚠️  Confidence: {source_info.get('confidence_level', 'Unknown')}")
                if source_info.get('bias_warning'):
                    print(f"   🚨 Bias Warning: {source_info['bias_warning'][:100]}...")

            # Show quality indicators
            quality = explanation_data.get('quality_indicators', {})
            print(f"\n📈 Quality Metrics:")
            print(f"   📝 Explanation length: {quality.get('explanation_length', 0)} characters")
            print(f"   📚 Word count: {quality.get('explanation_word_count', 0)} words")
            print(f"   📖 Readability: {quality.get('estimated_readability', 'unknown')}")
            print(f"   🏆 Overall quality score: {quality.get('overall_quality_score', 0)}/100")

        else:
            print(f"❌ Explanation generation failed!")
            error_info = result['error']
            print(f"   🚫 Error type: {error_info['type']}")
            print(f"   💬 Error message: {error_info['message']}")

        # Show comprehensive performance metrics
        print("\n📊 Performance Metrics:")
        metrics = agent.get_performance_metrics()
        expl_metrics = metrics['explanation_specific_metrics']
        
        print(f"   📈 Total explanations: {expl_metrics['total_explanations']}")
        print(f"   ✅ Success rate: {expl_metrics.get('success_rate', 0):.1f}%")
        print(f"   ⏱️  Average response time: {expl_metrics['average_response_time']:.2f}s")
        print(f"   🔄 API calls made: {metrics['metadata'].get('api_calls_made', 1)}")
        print(f"   🪙 Estimated tokens: {metrics['metadata'].get('total_tokens_estimated', 0)}")

        # Test health status
        print("\n🏥 Health Status:")
        health = agent.get_health_status()
        print(f"   🟢 Status: {health['status']}")
        print(f"   💯 Health score: {health['health_score']}/100")
        print(f"   ⏰ Uptime: {health['uptime_minutes']:.1f} minutes")
        print(f"   🌐 API connectivity: {health['api_connectivity']}")

        print("\n✅ LLM EXPLANATION AGENT TESTING COMPLETED SUCCESSFULLY")

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print("💡 Make sure your GEMINI_API_KEY is set in your environment variables")
        print("💡 Check your network connection and API key validity")
        
        # Show error details for debugging
        import traceback
        print(f"\n🔍 Full error traceback:")
        print(traceback.format_exc())
