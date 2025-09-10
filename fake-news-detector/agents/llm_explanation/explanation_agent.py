# agents/llm_explanation/explanation_agent.py

"""
Enhanced LLM Explanation Agent

Production-ready explanation generation agent with comprehensive error handling,
source assessment, and configurable explanation types. Generates human-readable
explanations of fake news detection results using advanced AI models.
"""

import time
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import google.generativeai as genai

# ✅ FIXED: Correct import path for BaseAgent
from agents.base import BaseAgent
from config import get_model_config, get_settings

# ✅ FIXED: Utils import with fallback
try:
    from utils.helpers import sanitize_text
except ImportError:
    def sanitize_text(text: str) -> str:
        """Basic text sanitization fallback."""
        if not isinstance(text, str):
            return ""
        return text.strip().replace('\x00', '').replace('\r\n', '\n')

from .source_database import SourceReliabilityDatabase
from .prompts import get_explanation_prompt
from .validators import InputValidator, OutputValidator
from .exceptions import (
    LLMExplanationError,
    InputValidationError,
    APIConfigurationError,
    LLMResponseError,
    ExplanationGenerationError,
    RateLimitError,
    SourceAssessmentError,
    handle_llm_explanation_exception
)

class LLMExplanationAgent(BaseAgent):
    """
    Enhanced explanation agent for generating human-readable fake news explanations.
    
    Features:
    - Multi-level explanation generation (basic, detailed, confidence analysis)
    - Comprehensive source reliability assessment
    - Configurable AI model parameters and safety settings
    - Robust error handling and recovery mechanisms
    - Performance tracking and quality metrics
    - LangGraph integration compatibility
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced LLM explanation agent with configuration."""
        
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        explanation_config = get_model_config('llm_explanation')
        system_settings = get_settings()
        
        if config:
            explanation_config.update(config)

        self.agent_name = "llm_explanation"
        super().__init__(explanation_config)

        # AI Model Configuration
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 3072)
        self.top_p = self.config.get('top_p', 0.9)
        self.top_k = self.config.get('top_k', 40)

        # Analysis Configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.75)
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.enable_source_analysis = self.config.get('enable_source_analysis', True)
        self.enable_confidence_analysis = self.config.get('enable_confidence_analysis', True)

        # Content Processing Limits
        self.max_article_length = self.config.get('max_article_length', 4000)
        self.min_explanation_length = self.config.get('min_explanation_length', 100)

        # ✅ FIXED: Enhanced API key loading from .env
        self.api_key = (
            os.getenv('GEMINI_API_KEY') or 
            os.getenv('GOOGLE_API_KEY') or 
            getattr(system_settings, 'gemini_api_key', None)
        )
        
        if not self.api_key:
            raise APIConfigurationError(
                "Gemini API key not found. Please set GEMINI_API_KEY in your .env file"
            )

        # Rate limiting configuration
        self.rate_limit = self.config.get('rate_limit_seconds', getattr(system_settings, 'gemini_rate_limit', 1.0))
        self.max_retries = self.config.get('max_retries', getattr(system_settings, 'max_retries', 3))

        # Initialize components
        self._initialize_gemini_api()
        self.source_database = SourceReliabilityDatabase()
        self.input_validator = InputValidator(self.config)
        self.output_validator = OutputValidator(self.config)

        # Performance tracking
        self.explanation_metrics = {
            'total_explanations': 0,
            'successful_explanations': 0,
            'detailed_analyses_generated': 0,
            'confidence_analyses_generated': 0,
            'source_assessments_performed': 0,
            'average_response_time': 0.0,
            'safety_blocks': 0,
            'rate_limit_hits': 0
        }

        self.last_request_time = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Enhanced LLM Explanation Agent initialized - Model: {self.model_name}")

    def _initialize_gemini_api(self) -> None:
        """Initialize Gemini API with comprehensive configuration."""
        try:
            genai.configure(api_key=self.api_key)

            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }

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
            raise APIConfigurationError(f"Failed to initialize Gemini API: {str(e)}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for explanation generation with comprehensive validation.
        
        Args:
            input_data: Dictionary containing:
                - text: Article text to explain
                - prediction: FAKE/REAL classification
                - confidence: Confidence score (0.0-1.0)
                - metadata: Additional context (optional)
                - require_detailed_analysis: Force detailed analysis (optional)
                
        Returns:
            Standardized output dictionary with explanation results
        """
        # ✅ FIXED: Enhanced input validation
        validation_result = self.input_validator.validate_explanation_input(input_data)
        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            return self.format_error_output(InputValidationError(error_msg), input_data)

        # ✅ FIXED: Session management compatibility
        if hasattr(self, '_start_processing_session'):
            self._start_processing_session(input_data)

        start_time = time.time()
        
        try:
            # Extract and process parameters
            article_text = input_data.get('text', '')
            prediction = input_data.get('prediction', 'UNKNOWN')
            confidence = input_data.get('confidence', 0.0)
            metadata = input_data.get('metadata', {})
            require_detailed = input_data.get('require_detailed_analysis', False)

            # Determine analysis depth
            trigger_detailed = (
                require_detailed or 
                confidence < self.confidence_threshold or 
                self.enable_detailed_analysis
            )

            # Generate comprehensive explanation
            explanation_result = self.generate_explanation(
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                metadata=metadata,
                require_detailed_analysis=trigger_detailed
            )

            # Validate output
            output_validation = self.output_validator.validate_explanation_output(explanation_result)
            if not output_validation.is_valid:
                self.logger.warning(f"Output validation issues: {output_validation.errors}")

            # Update metrics
            processing_time = time.time() - start_time
            self.explanation_metrics['successful_explanations'] += 1
            self._update_response_time_metric(processing_time)

            # Update specific metrics
            if explanation_result.get('detailed_analysis'):
                self.explanation_metrics['detailed_analyses_generated'] += 1
            if explanation_result.get('confidence_analysis'):
                self.explanation_metrics['confidence_analyses_generated'] += 1
            if explanation_result.get('source_assessment'):
                self.explanation_metrics['source_assessments_performed'] += 1

            return self.format_output(
                result=explanation_result,
                confidence=confidence,
                metadata={
                    'processing_time': processing_time,
                    'model_used': self.model_name,
                    'agent_version': '3.0.0',
                    'detailed_analysis_triggered': trigger_detailed,
                    'output_validation_warnings': len(output_validation.warnings)
                }
            )

        except LLMExplanationError as e:
            processing_time = time.time() - start_time
            self.logger.error(f"LLM explanation error: {str(e)}")
            return self.format_error_output(e, input_data)
        except Exception as e:
            processing_time = time.time() - start_time
            wrapped_error = handle_llm_explanation_exception(e)
            self.logger.error(f"Unexpected error: {str(e)}")
            return self.format_error_output(wrapped_error, input_data)
        
        finally:
            # ✅ FIXED: Session cleanup compatibility
            if hasattr(self, '_end_processing_session'):
                self._end_processing_session()

    def generate_explanation(self,
                           article_text: str,
                           prediction: str,
                           confidence: float,
                           metadata: Dict[str, Any],
                           require_detailed_analysis: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive explanation with multiple analysis components.
        
        Args:
            article_text: Article content to explain
            prediction: Classification result (REAL/FAKE)
            confidence: Confidence score (0.0-1.0)
            metadata: Additional context information
            require_detailed_analysis: Force detailed forensic analysis
            
        Returns:
            Dictionary containing comprehensive explanation results
        """
        start_time = time.time()
        self.explanation_metrics['total_explanations'] += 1

        try:
            # Clean and prepare content
            article_text = sanitize_text(article_text)
            if len(article_text) > self.max_article_length:
                article_text = article_text[:self.max_article_length] + "..."

            # Extract metadata with defaults
            source = metadata.get('source', 'Unknown Source')
            date = metadata.get('date', 'Unknown Date')
            subject = metadata.get('subject', 'General News')

            # Step 1: Generate primary explanation
            explanation = self._generate_primary_explanation(
                article_text, prediction, confidence, source, date, subject
            )

            # Step 2: Conditional detailed analysis
            detailed_analysis = None
            if require_detailed_analysis or confidence < self.confidence_threshold:
                detailed_analysis = self._generate_detailed_analysis(
                    article_text, prediction, confidence, metadata
                )

            # Step 3: Confidence analysis if enabled
            confidence_analysis = None
            if self.enable_confidence_analysis:
                confidence_analysis = self._generate_confidence_analysis(
                    article_text, prediction, confidence
                )

            # Step 4: Source assessment if enabled
            source_assessment = None
            if self.enable_source_analysis and source != 'Unknown Source':
                try:
                    source_assessment = self.source_database.get_reliability_summary(source)
                    self.explanation_metrics['source_assessments_performed'] += 1
                except Exception as e:
                    self.logger.warning(f"Source assessment failed: {str(e)}")
                    source_assessment = {'error': f"Assessment failed: {str(e)}"}

            # Package comprehensive results
            processing_time = time.time() - start_time

            result = {
                'explanation': explanation,
                'detailed_analysis': detailed_analysis,
                'confidence_analysis': confidence_analysis,
                'source_assessment': source_assessment,
                'metadata': {
                    'prediction': prediction,
                    'confidence_level': confidence,
                    'source': source,
                    'date': date,
                    'subject': subject,
                    'response_time_seconds': round(processing_time, 2),
                    'model_used': self.model_name,
                    'temperature_used': self.temperature,
                    'detailed_analysis_included': detailed_analysis is not None,
                    'confidence_analysis_included': confidence_analysis is not None,
                    'source_analysis_included': source_assessment is not None,
                    'article_length_processed': len(article_text),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'agent_version': '3.0.0'
                }
            }

            self.logger.info(f"Generated explanation in {processing_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in explanation generation: {str(e)}")
            raise ExplanationGenerationError(f"Generation failed: {str(e)}", "explanation_generation")

    def _generate_primary_explanation(self, article_text: str, prediction: str,
                                    confidence: float, source: str, date: str, subject: str) -> str:
        """Generate main explanation using structured prompts."""
        try:
            self._respect_rate_limits()

            prompt = get_explanation_prompt(
                'main',
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                source=source,
                date=date,
                subject=subject
            )

            response = self.model.generate_content(prompt)
            
            if not self._is_valid_response(response):
                self.explanation_metrics['safety_blocks'] += 1
                raise LLMResponseError("Primary explanation blocked by safety filters",
                                     "main_explanation", self.model_name, safety_blocked=True)

            return response.candidates[0].content.parts[0].text or "Explanation generation failed"

        except LLMResponseError:
            raise  # Re-raise LLM response errors
        except Exception as e:
            raise ExplanationGenerationError(f"Primary explanation failed: {str(e)}", "primary_explanation")

    def _generate_detailed_analysis(self, article_text: str, prediction: str,
                                  confidence: float, metadata: Dict[str, Any]) -> str:
        """Generate detailed forensic analysis."""
        try:
            self._respect_rate_limits()

            prompt = get_explanation_prompt(
                'detailed',
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                metadata=metadata
            )

            response = self.model.generate_content(prompt)
            
            if not self._is_valid_response(response):
                self.explanation_metrics['safety_blocks'] += 1
                return "Detailed analysis blocked by safety filters"

            return response.candidates[0].content.parts[0].text or "Detailed analysis generation failed"

        except Exception as e:
            self.logger.warning(f"Detailed analysis generation failed: {str(e)}")
            return f"Detailed analysis unavailable: {str(e)}"

    def _generate_confidence_analysis(self, article_text: str, prediction: str, confidence: float) -> str:
        """Generate confidence level appropriateness analysis."""
        try:
            self._respect_rate_limits()

            prompt = get_explanation_prompt(
                'confidence',
                article_text=article_text,
                prediction=prediction,
                confidence=confidence
            )

            response = self.model.generate_content(prompt)
            
            if not self._is_valid_response(response):
                self.explanation_metrics['safety_blocks'] += 1
                return "Confidence analysis blocked by safety filters"

            return response.candidates[0].content.parts[0].text or "Confidence analysis generation failed"

        except Exception as e:
            self.logger.warning(f"Confidence analysis generation failed: {str(e)}")
            return f"Confidence analysis unavailable: {str(e)}"

    def _is_valid_response(self, response) -> bool:
        """Check if LLM response is valid and not blocked."""
        return (response and 
                hasattr(response, 'candidates') and 
                response.candidates and 
                len(response.candidates) > 0 and 
                hasattr(response.candidates[0], 'finish_reason') and 
                response.candidates[0].finish_reason != 2 and  # Not SAFETY blocked
                hasattr(response.candidates[0], 'content') and 
                response.candidates[0].content and 
                hasattr(response.candidates[0].content, 'parts') and 
                response.candidates[0].content.parts)

    def _respect_rate_limits(self) -> None:
        """Implement API rate limiting with tracking."""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                delay = self.rate_limit - time_since_last
                time.sleep(delay)
                self.explanation_metrics['rate_limit_hits'] += 1
        self.last_request_time = time.time()

    def _update_response_time_metric(self, response_time: float) -> None:
        """Update average response time metric."""
        total_explanations = self.explanation_metrics['total_explanations']
        current_avg = self.explanation_metrics['average_response_time']
        
        if total_explanations > 1:
            new_avg = ((current_avg * (total_explanations - 1)) + response_time) / total_explanations
            self.explanation_metrics['average_response_time'] = new_avg
        else:
            self.explanation_metrics['average_response_time'] = response_time

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance and configuration metrics."""
        base_metrics = super().get_performance_metrics()
        return {
            **base_metrics,
            'explanation_specific_metrics': self.explanation_metrics,
            'configuration_metrics': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'confidence_threshold': self.confidence_threshold,
                'rate_limit_seconds': self.rate_limit,
                'max_article_length': self.max_article_length,
                'detailed_analysis_enabled': self.enable_detailed_analysis,
                'source_analysis_enabled': self.enable_source_analysis,
                'confidence_analysis_enabled': self.enable_confidence_analysis
            },
            'component_metrics': {
                'source_database_stats': self.source_database.get_database_statistics(),
                'validation_enabled': True
            },
            'agent_type': 'llm_explanation',
            'agent_version': '3.0.0',
            'architecture': 'modular_production'
        }

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate input data for explanation generation.
        
        Args:
            input_data: Input dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        validation_result = self.input_validator.validate_explanation_input(input_data)
        if validation_result.is_valid:
            return True, ""
        else:
            return False, "; ".join(validation_result.errors)

# Testing functionality
if __name__ == "__main__":
    """Test LLM explanation agent functionality."""
    try:
        # Initialize agent
        agent = LLMExplanationAgent()

        # Test input
        test_input = {
            "text": """
            A groundbreaking study claims that drinking 10 cups of coffee daily can extend
            lifespan by 50 years. The research was conducted by Dr. Anonymous at an
            undisclosed institution. The study has not been peer-reviewed and the data
            has not been made available for verification.
            """,
            "prediction": "FAKE",
            "confidence": 0.89,
            "metadata": {
                "source": "HealthyLifeBlog.net",
                "date": "2025-01-15",
                "subject": "Health"
            },
            "require_detailed_analysis": True
        }

        print("=== LLM EXPLANATION AGENT TEST ===")
        result = agent.process(test_input)
        
        if result['success']:
            explanation_data = result['result']
            print(f"✅ Explanation generated successfully")
            print(f"   Response time: {explanation_data['metadata']['response_time_seconds']:.2f}s")
            print(f"   Model used: {explanation_data['metadata']['model_used']}")
            print(f"   Detailed analysis: {'Yes' if explanation_data['detailed_analysis'] else 'No'}")
            print(f"   Confidence analysis: {'Yes' if explanation_data['confidence_analysis'] else 'No'}")
            print(f"   Source assessment: {'Yes' if explanation_data['source_assessment'] else 'No'}")

            # Show explanation preview
            explanation_preview = explanation_data['explanation'][:200] + "..."
            print(f"\n📄 Explanation preview:")
            print(f"   {explanation_preview}")

            # Show source assessment if available
            if explanation_data['source_assessment']:
                source_info = explanation_data['source_assessment']
                print(f"\n🔍 Source Assessment:")
                print(f"   Reliability: {source_info.get('reliability_level', 'Unknown')}")
                print(f"   Recommendation: {source_info.get('verification_recommendation', 'N/A')[:100]}...")
        else:
            print(f"❌ Explanation generation failed: {result['error']['message']}")

        # Show comprehensive metrics
        metrics = agent.get_performance_metrics()
        print(f"\n📊 Performance Metrics:")
        print(f"   Total explanations: {metrics['explanation_specific_metrics']['total_explanations']}")
        print(f"   Success rate: {(metrics['explanation_specific_metrics']['successful_explanations'] / max(metrics['explanation_specific_metrics']['total_explanations'], 1)) * 100:.1f}%")
        print(f"   Average response time: {metrics['explanation_specific_metrics']['average_response_time']:.2f}s")
        print(f"   Safety blocks: {metrics['explanation_specific_metrics']['safety_blocks']}")

        print("\n✅ LLM EXPLANATION AGENT TESTING COMPLETED")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure your GEMINI_API_KEY is set in your environment variables")
