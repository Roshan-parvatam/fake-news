# agents/context_analyzer/analyzer_agent.py

"""
Context Analyzer Agent

Production-ready context analysis agent that examines articles for bias,
emotional manipulation, framing techniques, and propaganda methods with
LLM-driven consistent scoring and configuration integration.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import google.generativeai as genai

# ✅ FIXED: Correct import path for BaseAgent
from agents.base import BaseAgent
from config import get_model_config, get_settings

# ✅ FIXED: Utils import with fallback
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
from .validators import InputValidator, OutputValidator
from .exceptions import (
    ContextAnalyzerError,
    LLMResponseError,
    BiasDetectionError,
    ManipulationDetectionError,
    ScoringConsistencyError,
    ConfigurationError
)

class ContextAnalyzerAgent(BaseAgent):
    """
    Context analysis agent with LLM-driven scoring and modular architecture.
    
    Examines articles for bias, manipulation, framing, and propaganda while
    ensuring numerical scores match textual explanations for consistency.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the context analyzer agent with configuration."""
        
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        context_config = get_model_config('context_analyzer')
        system_settings = get_settings()
        
        if config:
            context_config.update(config)

        self.agent_name = "context_analyzer"
        super().__init__(context_config)

        # AI Model Configuration
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.4)
        self.max_tokens = self.config.get('max_tokens', 3072)

        # Analysis Configuration
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.bias_threshold = self.config.get('bias_threshold', 70.0)
        self.manipulation_threshold = self.config.get('manipulation_threshold', 70.0)
        self.enable_propaganda_analysis = self.config.get('enable_propaganda_analysis', True)

        # Detection Modes
        self.bias_detection_modes = self.config.get('bias_detection_modes', [
            'political_bias', 'emotional_bias', 'selection_bias',
            'linguistic_bias', 'cultural_bias'
        ])
        self.emotional_analysis_depth = self.config.get('emotional_analysis_depth', 'comprehensive')

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
        self.bias_database = BiasPatternDatabase()
        self.manipulation_detector = ManipulationDetector()

        # Performance metrics
        self.analysis_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'llm_scores_generated': 0,
            'high_bias_detected': 0,
            'high_manipulation_detected': 0,
            'score_consistency_checks': 0
        }

        self.last_request_time = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Context Analyzer Agent initialized with model {self.model_name}")

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

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
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
        Main processing method for context analysis.
        
        Args:
            input_data: Dictionary containing article text and previous analysis
            
        Returns:
            Dictionary with context analysis results and LLM scores
        """
        # ✅ FIXED: Enhanced input validation
        validation_result = self.input_validator.validate_processing_input(input_data)
        if not validation_result.is_valid:
            return self.format_error_output(
                ValueError(f"Input validation failed: {validation_result.errors[0]}"),
                input_data
            )

        # ✅ FIXED: Session management compatibility
        if hasattr(self, '_start_processing_session'):
            self._start_processing_session(input_data)

        start_time = time.time()
        
        try:
            article_text = input_data.get('text', '')
            previous_analysis = input_data.get('previous_analysis', {})
            include_detailed_analysis = input_data.get(
                'include_detailed_analysis', 
                self.enable_detailed_analysis
            )

            # Determine analysis depth
            bert_confidence = previous_analysis.get('confidence', 1.0)
            force_detailed = (
                include_detailed_analysis or 
                bert_confidence < 0.7 or 
                self.enable_detailed_analysis
            )

            # Perform context analysis
            analysis_result = self.analyze_context_with_llm_scoring(
                article_text=article_text,
                previous_analysis=previous_analysis,
                include_detailed_analysis=force_detailed
            )

            # Update metrics
            processing_time = time.time() - start_time
            self.analysis_metrics['total_analyses'] += 1
            self.analysis_metrics['successful_analyses'] += 1
            
            llm_scores = analysis_result.get('llm_scores', {})
            
            # Update detection metrics
            if llm_scores.get('bias', 0) >= self.bias_threshold:
                self.analysis_metrics['high_bias_detected'] += 1
            if llm_scores.get('manipulation', 0) >= self.manipulation_threshold:
                self.analysis_metrics['high_manipulation_detected'] += 1

            # Calculate confidence score
            risk_score = llm_scores.get('risk', 50)
            confidence = 1.0 - (risk_score / 100.0)

            return self.format_output(
                result=analysis_result,
                confidence=confidence,
                metadata={
                    'processing_time': processing_time,
                    'model_used': self.model_name,
                    'detailed_analysis': force_detailed,
                    'scoring_method': 'llm_driven',
                    'agent_version': '3.0.0'
                }
            )

        except Exception as e:
            self.logger.error(f"Context analysis failed: {str(e)}")
            return self.format_error_output(e, input_data)
        
        finally:
            # ✅ FIXED: Session cleanup compatibility
            if hasattr(self, '_end_processing_session'):
                self._end_processing_session()

    def analyze_context_with_llm_scoring(self,
                                        article_text: str,
                                        previous_analysis: Dict[str, Any],
                                        include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Comprehensive context analysis with LLM-driven consistent scoring.
        
        Args:
            article_text: Article content to analyze
            previous_analysis: Results from previous agents
            include_detailed_analysis: Whether to include detailed analysis
            
        Returns:
            Dictionary containing comprehensive context analysis with consistent scores
        """
        self._respect_rate_limits()
        start_time = time.time()

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

            # Run pattern-based analysis (backup data)
            pattern_analysis = self.bias_database.analyze_bias_patterns(article_text)
            manipulation_report = self.manipulation_detector.get_manipulation_report(article_text)

            # Generate LLM analysis with consistent scores
            llm_analysis_result = self._generate_llm_analysis_with_consistent_scores(
                article_text, source, topic_domain, prediction, confidence
            )

            # Generate additional analyses if needed
            additional_analyses = {}
            llm_scores = llm_analysis_result.get('scores', {})
            
            if (include_detailed_analysis or 
                llm_scores.get('bias', 0) > 60 or 
                llm_scores.get('manipulation', 0) > 60):
                additional_analyses = self._generate_additional_analyses(
                    article_text, previous_analysis, llm_scores
                )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Package results
            result = {
                'llm_analysis': llm_analysis_result['analysis_text'],
                'llm_scores': llm_analysis_result['scores'],
                'context_scores': {
                    'bias_score': llm_scores.get('bias', 0),
                    'manipulation_score': llm_scores.get('manipulation', 0),
                    'credibility': llm_scores.get('credibility', 50),
                    'risk_level': self._get_risk_level_from_scores(llm_scores),
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
                    'response_time_seconds': round(processing_time, 2),
                    'model_used': self.model_name,
                    'article_length_processed': len(article_text),
                    'detailed_analysis_included': include_detailed_analysis,
                    'scoring_method': 'llm_driven',
                    'consistency_validated': True,
                    'agent_version': '3.0.0'
                }
            }

            # Update metrics
            self.analysis_metrics['llm_scores_generated'] += 1
            self.analysis_metrics['score_consistency_checks'] += 1

            return result

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {str(e)}")
            raise

    def _generate_llm_analysis_with_consistent_scores(self,
                                                     article_text: str,
                                                     source: str,
                                                     topic_domain: str,
                                                     prediction: str,
                                                     confidence: float) -> Dict[str, Any]:
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
                confidence=confidence
            )

            response = self.model.generate_content(prompt)
            
            if not self._is_valid_response(response):
                self.logger.warning("LLM response blocked by safety filters, using fallback")
                return self._create_fallback_analysis(article_text)

            analysis_text = response.candidates[0].content.parts[0].text

            # Parse scores from response
            scores = self._parse_llm_scores(analysis_text)

            # Validate score consistency
            if not validate_context_analysis_output(analysis_text, scores):
                self.logger.warning("Score consistency validation failed, adjusting scores")
                scores = self._adjust_inconsistent_scores(analysis_text, scores)

            return {
                'analysis_text': analysis_text,
                'scores': scores,
                'scoring_method': 'llm_generated',
                'consistency_validated': True
            }

        except Exception as e:
            self.logger.error(f"LLM analysis generation failed: {str(e)}")
            return self._create_fallback_analysis(article_text)

    def _parse_llm_scores(self, analysis_text: str) -> Dict[str, int]:
        """Parse numerical scores from LLM response."""
        import re
        scores = {}
        
        # ✅ FIXED: Use raw strings to fix regex escape sequence warnings
        patterns = {
            'bias': r'(?:bias.*?score|bias):\s*(\d+)',
            'manipulation': r'(?:manipulation.*?score|manipulation):\s*(\d+)',
            'credibility': r'(?:credibility.*?score|credibility):\s*(\d+)',
            'risk': r'(?:risk.*?score|risk):\s*(\d+)'
        }

        for score_type, pattern in patterns.items():
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                scores[score_type] = max(0, min(100, score))
            else:
                scores[score_type] = self._estimate_score_from_text(analysis_text, score_type)

        return scores

    def _estimate_score_from_text(self, text: str, score_type: str) -> int:
        """Estimate score from text analysis when explicit score not provided."""
        text_lower = text.lower()
        
        if score_type == 'bias':
            if any(word in text_lower for word in ['extreme bias', 'highly biased', 'very biased']):
                return 85
            elif any(word in text_lower for word in ['biased', 'partisan', 'slanted']):
                return 65
            elif any(word in text_lower for word in ['some bias', 'slightly biased']):
                return 35
            elif any(word in text_lower for word in ['minimal bias', 'neutral', 'balanced']):
                return 15
            else:
                return 50
                
        elif score_type == 'manipulation':
            if any(word in text_lower for word in ['extreme manipulation', 'highly manipulative']):
                return 85
            elif any(word in text_lower for word in ['manipulation', 'manipulative', 'misleading']):
                return 65
            elif any(word in text_lower for word in ['some manipulation', 'emotional appeals']):
                return 35
            elif any(word in text_lower for word in ['minimal manipulation', 'straightforward']):
                return 15
            else:
                return 50
                
        elif score_type == 'credibility':
            if any(word in text_lower for word in ['highly credible', 'very reliable']):
                return 85
            elif any(word in text_lower for word in ['credible', 'reliable']):
                return 70
            elif any(word in text_lower for word in ['somewhat credible', 'moderately reliable']):
                return 55
            elif any(word in text_lower for word in ['questionable', 'unreliable']):
                return 25
            else:
                return 50
                
        elif score_type == 'risk':
            if any(word in text_lower for word in ['high risk', 'dangerous', 'harmful']):
                return 80
            elif any(word in text_lower for word in ['moderate risk', 'concerning']):
                return 60
            elif any(word in text_lower for word in ['some risk', 'minor concerns']):
                return 40
            elif any(word in text_lower for word in ['low risk', 'minimal risk']):
                return 20
            else:
                return 50

        return 50

    def _adjust_inconsistent_scores(self, analysis_text: str, scores: Dict[str, int]) -> Dict[str, int]:
        """Adjust scores that don't match textual analysis."""
        adjusted_scores = scores.copy()
        text_lower = analysis_text.lower()

        # Bias consistency adjustments
        if any(phrase in text_lower for phrase in ['minimal bias', 'neutral', 'balanced']):
            if scores.get('bias', 0) > 30:
                adjusted_scores['bias'] = 20
        elif any(phrase in text_lower for phrase in ['extreme bias', 'highly biased']):
            if scores.get('bias', 0) < 70:
                adjusted_scores['bias'] = 80

        # Manipulation consistency adjustments
        if any(phrase in text_lower for phrase in ['no manipulation', 'straightforward']):
            if scores.get('manipulation', 0) > 30:
                adjusted_scores['manipulation'] = 15
        elif any(phrase in text_lower for phrase in ['extreme manipulation', 'highly manipulative']):
            if scores.get('manipulation', 0) < 70:
                adjusted_scores['manipulation'] = 80

        return adjusted_scores

    def _generate_additional_analyses(self,
                                    article_text: str,
                                    previous_analysis: Dict[str, Any],
                                    llm_scores: Dict[str, int]) -> Dict[str, str]:
        """Generate additional detailed analyses when needed."""
        additional_analyses = {}

        try:
            # Generate bias analysis if high bias detected
            if llm_scores.get('bias', 0) > 60:
                bias_prompt = get_context_prompt_template(
                    'bias_detection',
                    article_text=article_text,
                    source=previous_analysis.get('source', 'Unknown'),
                    topic_domain=previous_analysis.get('topic_domain', 'general'),
                    prediction=previous_analysis.get('prediction', 'Unknown'),
                    confidence=previous_analysis.get('confidence', 0.0)
                )
                
                response = self.model.generate_content(bias_prompt)
                if self._is_valid_response(response):
                    additional_analyses['bias_analysis'] = response.candidates[0].content.parts[0].text

            # Generate manipulation analysis if high manipulation detected
            if llm_scores.get('manipulation', 0) > 60:
                manipulation_prompt = get_context_prompt_template(
                    'emotional_manipulation',
                    article_text=article_text,
                    emotional_indicators={'high_manipulation_detected': True}
                )
                
                response = self.model.generate_content(manipulation_prompt)
                if self._is_valid_response(response):
                    additional_analyses['emotional_analysis'] = response.candidates[0].content.parts[0].text

            # Generate propaganda analysis for high risk content
            if self.enable_propaganda_analysis and llm_scores.get('risk', 0) > 70:
                propaganda_prompt = get_context_prompt_template(
                    'propaganda_detection',
                    article_text=article_text
                )
                
                response = self.model.generate_content(propaganda_prompt)
                if self._is_valid_response(response):
                    additional_analyses['propaganda_analysis'] = response.candidates[0].content.parts[0].text

        except Exception as e:
            self.logger.warning(f"Additional analysis generation failed: {str(e)}")

        return additional_analyses

    def _create_fallback_analysis(self, article_text: str) -> Dict[str, Any]:
        """Create fallback analysis when LLM fails."""
        return {
            'analysis_text': f"Automated analysis for article of {len(article_text)} characters. Manual review recommended.",
            'scores': {
                'bias': 50,
                'manipulation': 50,
                'credibility': 50,
                'risk': 50
            },
            'scoring_method': 'fallback',
            'consistency_validated': False
        }

    def _get_risk_level_from_scores(self, scores: Dict[str, int]) -> str:
        """Convert numerical risk score to risk level."""
        risk_score = scores.get('risk', 50)
        
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        else:
            return "LOW"

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

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = super().get_performance_metrics()
        context_metrics = {
            'analyses_completed': self.analysis_metrics['total_analyses'],
            'successful_analyses': self.analysis_metrics['successful_analyses'],
            'llm_scores_generated': self.analysis_metrics['llm_scores_generated'],
            'high_bias_detected': self.analysis_metrics['high_bias_detected'],
            'high_manipulation_detected': self.analysis_metrics['high_manipulation_detected'],
            'score_consistency_checks': self.analysis_metrics['score_consistency_checks'],
            'model_config': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        }
        
        return {**base_metrics, **context_metrics}

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data for context analysis."""
        if not isinstance(input_data, dict):
            return False, "Input must be a dictionary"
        
        if 'text' not in input_data:
            return False, "Missing required 'text' field"
            
        if not input_data['text'].strip():
            return False, "Article text cannot be empty"
            
        if len(input_data['text']) < 50:
            return False, "Article text too short for meaningful analysis"
        
        return True, ""

# Testing functionality
if __name__ == "__main__":
    """Test the context analyzer agent."""
    try:
        agent = ContextAnalyzerAgent()
        
        test_input = {
            "text": """
            Recent studies from Harvard University show that the new policy is
            highly effective based on comprehensive data analysis with strong
            statistical significance and peer review validation.
            """,
            "previous_analysis": {
                "prediction": "REAL",
                "confidence": 0.94,
                "source": "Harvard University",
                "topic_domain": "policy"
            }
        }

        result = agent.process(test_input)
        
        if result['success']:
            analysis_data = result['result']
            llm_scores = analysis_data.get('llm_scores', {})
            print("✅ Context Analysis Results:")
            print(f"   Bias Score: {llm_scores.get('bias', 0)}/100")
            print(f"   Manipulation Score: {llm_scores.get('manipulation', 0)}/100")
            print(f"   Credibility Score: {llm_scores.get('credibility', 50)}/100")
            print(f"   Risk Level: {analysis_data['context_scores']['risk_level']}")
            print(f"   Processing Time: {analysis_data['metadata']['response_time_seconds']}s")
        else:
            print(f"❌ Analysis failed: {result['error']['message']}")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
