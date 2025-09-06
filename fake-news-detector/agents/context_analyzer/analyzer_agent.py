# agents/context_analyzer/analyzer_agent.py
"""
Enhanced Context Analyzer Agent - Main Implementation with Config Integration

This agent examines articles for bias, emotional manipulation, framing techniques,
and propaganda methods with full configuration integration and modular architecture.

Features:
- Configuration integration from config files
- Centralized prompt management
- Pattern-based bias detection with AI enhancement
- Multi-dimensional analysis (bias, framing, emotional, propaganda)
- Comprehensive scoring and risk assessment
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
from .bias_patterns import BiasPatternDatabase
from .manipulation_detection import ManipulationDetector

# âœ… IMPORT CONFIGURATION FILES
from config import get_model_config, get_prompt_template, get_settings
from utils.helpers import sanitize_text

class ContextAnalyzerAgent(BaseAgent):
    """
    ðŸ” ENHANCED CONTEXT ANALYZER AGENT WITH CONFIG INTEGRATION
    
    Modular context analysis agent that inherits from BaseAgent
    for consistent interface and LangGraph compatibility.
    
    Features:
    - Inherits from BaseAgent for consistent interface
    - Configuration integration from config files
    - Modular component architecture (bias patterns, manipulation detection)
    - AI-powered analysis with pattern-based pre-screening
    - Multi-dimensional context analysis (bias, framing, emotional, propaganda)
    - Comprehensive scoring and risk assessment
    - Performance tracking and metrics
    - LangGraph integration ready
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced context analyzer agent with config integration
        
        Args:
            config: Configuration dictionary for runtime overrides
        """
        # âœ… GET CONFIGURATION FROM CONFIG FILES
        context_config = get_model_config('context_analyzer')
        system_settings = get_settings()
        
        # Merge with runtime overrides
        if config:
            context_config.update(config)

        self.agent_name = "context_analyzer"
        
        # Initialize base agent with merged config
        super().__init__(context_config)
        
        # âœ… USE CONFIG VALUES FOR AI MODEL SETTINGS
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.4)  # Higher for nuanced analysis
        self.max_tokens = self.config.get('max_tokens', 3072)
        
        # âœ… ANALYSIS SETTINGS FROM CONFIG
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.bias_threshold = self.config.get('bias_threshold', 5.0)
        self.manipulation_threshold = self.config.get('manipulation_threshold', 6.0)
        self.enable_propaganda_analysis = self.config.get('enable_propaganda_analysis', True)
        
        # âœ… BIAS DETECTION SETTINGS FROM CONFIG
        self.bias_detection_modes = self.config.get('bias_detection_modes', [
            'political_bias', 'emotional_bias', 'selection_bias', 
            'linguistic_bias', 'cultural_bias'
        ])
        self.emotional_analysis_depth = self.config.get('emotional_analysis_depth', 'comprehensive')
        
        # âœ… MANIPULATION DETECTION SETTINGS FROM CONFIG
        self.propaganda_techniques_count = self.config.get('propaganda_techniques_count', 10)
        self.fallacy_detection_enabled = self.config.get('fallacy_detection_enabled', True)
        self.manipulation_scoring_algorithm = self.config.get('manipulation_scoring_algorithm', 'weighted_average')
        
        # âœ… GET API KEY FROM SYSTEM SETTINGS
        self.api_key = system_settings.gemini_api_key
        
        # âœ… LOAD PROMPTS FROM CONFIG INSTEAD OF HARDCODED
        self.bias_prompt = get_prompt_template('context_analyzer', 'bias_detection')
        self.framing_prompt = get_prompt_template('context_analyzer', 'framing_analysis')
        self.emotional_prompt = get_prompt_template('context_analyzer', 'emotional_manipulation')
        self.propaganda_prompt = get_prompt_template('context_analyzer', 'propaganda_detection')
        
        # âœ… USE RATE LIMITING FROM CONFIG/SETTINGS
        self.rate_limit = self.config.get('rate_limit_seconds', system_settings.gemini_rate_limit)
        self.max_retries = self.config.get('max_retries', system_settings.max_retries)
        
        # Initialize Gemini API
        self._initialize_gemini_api()
        
        # Initialize modular components
        self.bias_database = BiasPatternDatabase()
        self.manipulation_detector = ManipulationDetector()
        
        # Enhanced performance tracking with config awareness
        self.analysis_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'bias_analyses_generated': 0,
            'framing_analyses_generated': 0,
            'emotional_analyses_generated': 0,
            'propaganda_analyses_generated': 0,
            'average_response_time': 0.0,
            'gemini_api_calls': 0,
            'high_bias_detected': 0,
            'high_manipulation_detected': 0,
            'config_integrated': True
        }
        
        # Rate limiting tracking
        self.last_request_time = None
        
        self.logger.info(f"âœ… Enhanced Context Analyzer Agent initialized with config")
        self.logger.info(f"ðŸ¤– Model: {self.model_name}, Temperature: {self.temperature}")
        self.logger.info(f"ðŸŽ¯ Bias Threshold: {self.bias_threshold}, Manipulation Threshold: {self.manipulation_threshold}")
        self.logger.info(f"ðŸ” Analysis Modes: {len(self.bias_detection_modes)} bias types, Propaganda: {'On' if self.enable_propaganda_analysis else 'Off'}")
    
    def _initialize_gemini_api(self):
        """
        ðŸ” INITIALIZE GEMINI API WITH CONFIG SETTINGS
        
        Sets up Gemini AI connection using configuration values optimized
        for nuanced context analysis.
        """
        try:
            if not self.api_key:
                raise ValueError("Gemini API key not found in system settings")
            
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # âœ… USE GENERATION CONFIG FROM CONFIG FILES
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.config.get('top_p', 0.9),
                "top_k": self.config.get('top_k', 40),
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }
            
            # âœ… USE SAFETY SETTINGS FROM CONFIG
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
            
            self.logger.info("ðŸ” Gemini API initialized for context analysis")
            
        except Exception as e:
            self.logger.error(f"âŒ Failed to initialize Gemini API: {str(e)}")
            raise
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ðŸŽ¯ MAIN PROCESSING METHOD - LANGGRAPH COMPATIBLE WITH CONFIG
        
        Process input according to BaseAgent interface for LangGraph compatibility.
        
        Args:
            input_data: Dictionary containing:
                - text: Article text to analyze
                - previous_analysis: Results from previous agents (BERT, Claims)
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
            previous_analysis = input_data.get('previous_analysis', {})
            include_detailed_analysis = input_data.get(
                'include_detailed_analysis', 
                self.enable_detailed_analysis
            )
            
            # âœ… USE CONFIG FOR PROCESSING DECISIONS
            bert_confidence = previous_analysis.get('confidence', 1.0)
            force_detailed = (
                include_detailed_analysis or
                bert_confidence < 0.7 or  # Low confidence triggers detailed analysis
                self.enable_detailed_analysis
            )
            
            # Perform context analysis
            analysis_result = self.analyze_context(
                article_text=article_text,
                previous_analysis=previous_analysis,
                include_detailed_analysis=force_detailed
            )
            
            # Extract overall context score for metrics
            context_score = analysis_result['context_scores']['overall_context_score']
            
            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(context_score / 10.0)  # Normalize to 0-1
            self.analysis_metrics['successful_analyses'] += 1
            
            # Update specific analysis metrics
            if analysis_result.get('bias_analysis'):
                self.analysis_metrics['bias_analyses_generated'] += 1
            if analysis_result.get('framing_analysis'):
                self.analysis_metrics['framing_analyses_generated'] += 1
            if analysis_result.get('emotional_analysis'):
                self.analysis_metrics['emotional_analyses_generated'] += 1
            if analysis_result.get('propaganda_analysis'):
                self.analysis_metrics['propaganda_analyses_generated'] += 1
            
            # Update risk detection metrics
            if context_score >= self.bias_threshold:
                self.analysis_metrics['high_bias_detected'] += 1
            if analysis_result['manipulation_report']['overall_manipulation_score'] >= self.manipulation_threshold:
                self.analysis_metrics['high_manipulation_detected'] += 1
            
            # Format output for LangGraph with config context
            return self.format_output(
                result=analysis_result,
                confidence=1.0 - (context_score / 10.0),  # Lower context score = higher confidence
                metadata={
                    'response_time': analysis_result['metadata']['response_time_seconds'],
                    'model_used': self.model_name,
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular',
                    'detailed_analysis_triggered': force_detailed,
                    'bias_threshold_used': self.bias_threshold,
                    'manipulation_threshold_used': self.manipulation_threshold
                }
            )
            
        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            return self.format_error_output(e, input_data)
    
    def analyze_context(self,
                       article_text: str,
                       previous_analysis: Dict[str, Any],
                       include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        ðŸ” MAIN CONTEXT ANALYSIS WITH CONFIG INTEGRATION
        
        Comprehensive context analysis using config-driven parameters and analysis modes.
        
        Args:
            article_text: The news article text to analyze
            previous_analysis: Results from previous agents (BERT, Claims)
            include_detailed_analysis: Enable detailed forensic analysis
            
        Returns:
            Dict containing comprehensive context analysis results
        """
        self._respect_rate_limits()
        start_time = time.time()
        
        try:
            self.logger.info("Starting context analysis with config integration...")
            
            # Step 1: Extract information from previous analysis
            prediction = previous_analysis.get('prediction', 'Unknown')
            confidence = previous_analysis.get('confidence', 0.0)
            source = previous_analysis.get('source', 'Unknown Source')
            topic_domain = previous_analysis.get('topic_domain', 'general')
            
            # Step 2: Clean article text
            article_text = sanitize_text(article_text)
            
            # âœ… USE CONFIG FOR TEXT LENGTH LIMITS
            max_text_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_text_length:
                article_text = article_text[:max_text_length] + "..."
            
            # Step 3: Run pattern-based analysis using modular components
            pattern_analysis = self.bias_database.analyze_bias_patterns(article_text)
            manipulation_report = self.manipulation_detector.get_manipulation_report(article_text)
            
            # Step 4: Generate AI-powered bias analysis using config prompts
            bias_analysis = self._generate_bias_analysis(
                article_text, source, topic_domain, prediction, confidence
            )
            
            # Step 5: Generate framing analysis
            framing_analysis = self._generate_framing_analysis(article_text, previous_analysis)
            
            # Step 6: Generate emotional manipulation analysis
            emotional_indicators = pattern_analysis.get('emotional_counts', {})
            emotional_analysis = self._generate_emotional_analysis(article_text, emotional_indicators)
            
            # Step 7: Optional propaganda analysis based on config and complexity
            propaganda_analysis = None
            if (self.enable_propaganda_analysis and 
                (include_detailed_analysis or 
                 confidence < 0.7 or 
                 manipulation_report['overall_manipulation_score'] > self.manipulation_threshold)):
                propaganda_analysis = self._generate_propaganda_analysis(article_text)
                self.analysis_metrics['propaganda_analyses_generated'] += 1
                self.logger.info("ðŸš¨ Propaganda analysis generated due to high risk indicators")
            
            # Step 8: Calculate comprehensive context scores using config weights
            context_scores = self._calculate_context_scores(
                pattern_analysis, manipulation_report, bias_analysis, 
                framing_analysis, emotional_analysis, propaganda_analysis
            )
            
            # Step 9: Package results with config metadata
            response_time = time.time() - start_time
            result = {
                'bias_analysis': bias_analysis,
                'framing_analysis': framing_analysis,
                'emotional_analysis': emotional_analysis,
                'propaganda_analysis': propaganda_analysis,
                'manipulation_report': manipulation_report,
                'context_scores': context_scores,
                'pattern_analysis': {
                    'bias_counts': pattern_analysis.get('bias_counts', {}),
                    'emotional_counts': pattern_analysis.get('emotional_counts', {}),
                    'framing_counts': pattern_analysis.get('framing_counts', {}),
                    'indicators_found': pattern_analysis.get('indicators_found', [])
                },
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'response_time_seconds': round(response_time, 2),
                    'model_used': self.model_name,
                    'temperature_used': self.temperature,
                    'article_length_processed': len(article_text),
                    'detailed_analysis_included': include_detailed_analysis,
                    'propaganda_analysis_triggered': propaganda_analysis is not None,
                    'bias_threshold': self.bias_threshold,
                    'manipulation_threshold': self.manipulation_threshold,
                    'analysis_modes_used': self.bias_detection_modes,
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular',
                    'pattern_indicators_found': len(pattern_analysis.get('indicators_found', []))
                }
            }
            
            # Step 10: Update performance metrics
            self._update_analysis_metrics(response_time, context_scores['overall_context_score'])
            
            self.logger.info(f"Successfully completed context analysis in {response_time:.2f} seconds")
            self.logger.info(f"ðŸŽ¯ Overall context score: {context_scores['overall_context_score']:.1f}/10 ({context_scores['risk_level']})")
            
            return result
            
        except Exception as e:
            self._update_analysis_metrics(time.time() - start_time, 0, error=True)
            self.logger.error(f"Error in context analysis: {str(e)}")
            raise
    
    def _generate_bias_analysis(self, article_text: str, source: str, topic_domain: str,
                               prediction: str, confidence: float) -> str:
        """
        Generate AI-powered bias analysis using config prompt template
        
        Args:
            article_text: Article content
            source: Article source
            topic_domain: Article domain
            prediction: BERT classification result
            confidence: BERT confidence score
            
        Returns:
            Bias analysis text
        """
        try:
            # âœ… USE BIAS PROMPT FROM CONFIG
            prompt = self.bias_prompt.format(
                article_text=article_text,
                source=source,
                topic_domain=topic_domain,
                prediction=prediction,
                confidence=confidence
            )
            
            response = self.model.generate_content(prompt)
            self.analysis_metrics['gemini_api_calls'] += 1
            self.analysis_metrics['bias_analyses_generated'] += 1
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in bias analysis generation: {str(e)}")
            return f"Bias analysis unavailable due to processing error: {str(e)}"
    
    def _generate_framing_analysis(self, article_text: str, previous_analysis: Dict[str, Any]) -> str:
        """
        Generate AI-powered framing analysis using config prompt template
        
        Args:
            article_text: Article content
            previous_analysis: Context from previous agents
            
        Returns:
            Framing analysis text
        """
        try:
            # âœ… USE FRAMING PROMPT FROM CONFIG
            prompt = self.framing_prompt.format(
                article_text=article_text,
                previous_analysis=str(previous_analysis)
            )
            
            response = self.model.generate_content(prompt)
            self.analysis_metrics['gemini_api_calls'] += 1
            self.analysis_metrics['framing_analyses_generated'] += 1
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in framing analysis generation: {str(e)}")
            return f"Framing analysis unavailable due to processing error: {str(e)}"
    
    def _generate_emotional_analysis(self, article_text: str, emotional_indicators: Dict[str, int]) -> str:
        """
        Generate AI-powered emotional manipulation analysis using config prompt template
        
        Args:
            article_text: Article content
            emotional_indicators: Pattern-based emotional indicators
            
        Returns:
            Emotional analysis text
        """
        try:
            # âœ… USE EMOTIONAL PROMPT FROM CONFIG
            prompt = self.emotional_prompt.format(
                article_text=article_text,
                emotional_indicators=str(emotional_indicators)
            )
            
            response = self.model.generate_content(prompt)
            self.analysis_metrics['gemini_api_calls'] += 1
            self.analysis_metrics['emotional_analyses_generated'] += 1
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in emotional analysis generation: {str(e)}")
            return f"Emotional analysis unavailable due to processing error: {str(e)}"
    
    def _generate_propaganda_analysis(self, article_text: str) -> str:
        """
        Generate AI-powered propaganda analysis using config prompt template
        
        Args:
            article_text: Article content
            
        Returns:
            Propaganda analysis text
        """
        try:
            # âœ… USE PROPAGANDA PROMPT FROM CONFIG
            prompt = self.propaganda_prompt.format(
                article_text=article_text,
                prediction="Analysis pending",
                confidence=0.0
            )
            
            response = self.model.generate_content(prompt)
            self.analysis_metrics['gemini_api_calls'] += 1
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in propaganda analysis generation: {str(e)}")
            return f"Propaganda analysis unavailable due to processing error: {str(e)}"
    
    def _calculate_context_scores(self, pattern_analysis: Dict, manipulation_report: Dict,
                                 bias_analysis: str, framing_analysis: str, 
                                 emotional_analysis: str, propaganda_analysis: Optional[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive context scores with config-aware weights
        
        Args:
            pattern_analysis: Pattern-based analysis results
            manipulation_report: Manipulation detection results
            bias_analysis: AI bias analysis
            framing_analysis: AI framing analysis
            emotional_analysis: AI emotional analysis
            propaganda_analysis: Optional AI propaganda analysis
            
        Returns:
            Comprehensive context scoring
        """
        # âœ… USE CONFIG FOR SCORING WEIGHTS
        scoring_weights = self.config.get('scoring_weights', {
            'pattern_score_weight': 0.3,
            'ai_analysis_weight': 0.4,
            'manipulation_weight': 0.3
        })
        
        # Component scoring
        # 1. Pattern-based bias score
        bias_counts = pattern_analysis.get('bias_counts', {})
        total_bias_indicators = sum(bias_counts.values())
        pattern_bias_score = min(10, total_bias_indicators * 0.5)
        
        # 2. Emotional manipulation score
        emotional_counts = pattern_analysis.get('emotional_counts', {})
        total_emotional = sum(emotional_counts.values())
        emotional_score = min(10, total_emotional * 0.3)
        
        # 3. AI-based analysis score estimation
        ai_score = self._estimate_ai_analysis_score(bias_analysis, framing_analysis, emotional_analysis)
        
        # 4. Manipulation detector score
        manipulation_score = manipulation_report.get('overall_manipulation_score', 0)
        
        # 5. Propaganda score (if available)
        propaganda_score = 0
        if propaganda_analysis and self.enable_propaganda_analysis:
            propaganda_score = self._estimate_propaganda_score(propaganda_analysis)
        
        # âœ… CALCULATE WEIGHTED OVERALL SCORE
        pattern_component = (pattern_bias_score + emotional_score) / 2
        ai_component = (ai_score + propaganda_score) / 2 if propaganda_score > 0 else ai_score
        manipulation_component = manipulation_score
        
        overall_score = (
            pattern_component * scoring_weights['pattern_score_weight'] +
            ai_component * scoring_weights['ai_analysis_weight'] +
            manipulation_component * scoring_weights['manipulation_weight']
        )
        
        # Risk level assessment with config thresholds
        risk_thresholds = self.config.get('risk_thresholds', {
            'low': 3.0,
            'medium': 5.0,
            'high': 7.0,
            'critical': 9.0
        })
        
        if overall_score <= risk_thresholds['low']:
            risk_level = "LOW"
        elif overall_score <= risk_thresholds['medium']:
            risk_level = "MEDIUM"
        elif overall_score <= risk_thresholds['high']:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Primary concerns identification
        primary_concerns = []
        if pattern_bias_score > self.bias_threshold:
            primary_concerns.append(f"High pattern bias ({pattern_bias_score:.1f}/10)")
        if emotional_score > 5.0:
            primary_concerns.append(f"Emotional manipulation ({emotional_score:.1f}/10)")
        if manipulation_score > self.manipulation_threshold:
            primary_concerns.append(f"Manipulation techniques ({manipulation_score:.1f}/10)")
        if propaganda_score > 6.0:
            primary_concerns.append(f"Propaganda methods ({propaganda_score:.1f}/10)")
        
        return {
            'bias_score': round(pattern_bias_score, 2),
            'emotional_manipulation_score': round(emotional_score, 2),
            'framing_score': round(ai_score, 2),
            'propaganda_score': round(propaganda_score, 2),
            'manipulation_score': round(manipulation_score, 2),
            'overall_context_score': round(overall_score, 2),
            'risk_level': risk_level,
            'primary_concerns': primary_concerns,
            'scoring_method': 'config_weighted',
            'thresholds_used': {
                'bias_threshold': self.bias_threshold,
                'manipulation_threshold': self.manipulation_threshold
            }
        }
    
    def _estimate_ai_analysis_score(self, bias_analysis: str, framing_analysis: str, emotional_analysis: str) -> float:
        """Estimate score from AI analysis text content"""
        # Simple heuristic based on concerning keywords
        concerning_keywords = [
            'heavily biased', 'extremely biased', 'propaganda', 'manipulation',
            'misleading', 'deceptive', 'inflammatory', 'polarizing',
            'one-sided', 'partisan', 'agenda-driven', 'distorted'
        ]
        
        all_analysis = f"{bias_analysis} {framing_analysis} {emotional_analysis}".lower()
        
        concern_count = sum(1 for keyword in concerning_keywords if keyword in all_analysis)
        return min(10, concern_count * 1.2)
    
    def _estimate_propaganda_score(self, propaganda_analysis: str) -> float:
        """Estimate propaganda score from AI analysis"""
        propaganda_indicators = [
            'propaganda technique', 'manipulation', 'false flag', 'astroturfing',
            'gaslighting', 'whataboutism', 'strawman', 'ad hominem'
        ]
        
        analysis_lower = propaganda_analysis.lower()
        indicator_count = sum(1 for indicator in propaganda_indicators if indicator in analysis_lower)
        return min(10, indicator_count * 1.5)
    
    def _respect_rate_limits(self):
        """Rate limiting using config values"""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _update_analysis_metrics(self, response_time: float, context_score: float, error: bool = False):
        """Update analysis-specific metrics with config awareness"""
        self.analysis_metrics['total_analyses'] += 1
        
        if not error:
            # Update average response time
            total = self.analysis_metrics['total_analyses']
            current_avg = self.analysis_metrics['average_response_time']
            self.analysis_metrics['average_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        ðŸ“Š Get comprehensive performance metrics with config information
        
        Returns:
            Complete metrics dictionary including config details
        """
        # Get base metrics
        base_metrics = self.get_performance_metrics()
        
        # âœ… ADD CONFIG INFORMATION TO METRICS
        config_metrics = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'bias_threshold': self.bias_threshold,
            'manipulation_threshold': self.manipulation_threshold,
            'enable_detailed_analysis': self.enable_detailed_analysis,
            'enable_propaganda_analysis': self.enable_propaganda_analysis,
            'bias_detection_modes': self.bias_detection_modes,
            'emotional_analysis_depth': self.emotional_analysis_depth,
            'manipulation_scoring_algorithm': self.manipulation_scoring_algorithm,
            'rate_limit_seconds': self.rate_limit,
            'config_version': '2.0_integrated'
        }
        
        # Get component metrics
        component_metrics = {
            'bias_database_stats': self.bias_database.get_bias_statistics(),
            'manipulation_detector_stats': self.manipulation_detector.get_detection_statistics(),
            'api_calls_made': self.analysis_metrics['gemini_api_calls']
        }
        
        return {
            **base_metrics,
            'analysis_specific_metrics': self.analysis_metrics,
            'config_metrics': config_metrics,
            'component_info': component_metrics,
            'agent_type': 'context_analyzer',
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
            'bias_threshold': self.bias_threshold,
            'manipulation_threshold': self.manipulation_threshold,
            'enable_detailed_analysis': self.enable_detailed_analysis,
            'enable_propaganda_analysis': self.enable_propaganda_analysis,
            'bias_detection_modes': self.bias_detection_modes,
            'emotional_analysis_depth': self.emotional_analysis_depth,
            'propaganda_techniques_count': self.propaganda_techniques_count,
            'fallacy_detection_enabled': self.fallacy_detection_enabled,
            'manipulation_scoring_algorithm': self.manipulation_scoring_algorithm,
            'rate_limit_seconds': self.rate_limit,
            'max_retries': self.max_retries,
            'config_source': 'config_files',
            'prompt_source': 'centralized_prompts_config'
        }

# Testing functionality with config integration
if __name__ == "__main__":
    """Test the modular context analyzer agent with config integration"""
    print("ðŸ§ª Testing Modular Context Analyzer Agent with Config Integration")
    print("=" * 75)
    
    try:
        # Initialize agent (will load from config files)
        agent = ContextAnalyzerAgent()
        print(f"âœ… Agent initialized with config: {agent}")
        
        # Show config summary
        config_summary = agent.get_config_summary()
        print(f"\nâš™ï¸ Configuration Summary:")
        for key, value in config_summary.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {value}")
        
        # Test context analysis
        test_article = """
        This outrageous scandal exposes the corrupt establishment's lies! 
        The so-called "experts" are hiding the truth from hardworking Americans
        while pushing their radical agenda. Every patriot must wake up and see
        how they're manipulating us with fake statistics and biased reporting.
        Don't let them fool you - the mainstream media is the enemy of the people!
        """
        
        test_input = {
            "text": test_article,
            "previous_analysis": {
                "prediction": "FAKE",
                "confidence": 0.65,
                "source": "UnknownBlog.com",
                "topic_domain": "politics"
            },
            "include_detailed_analysis": True
        }
        
        print(f"\nðŸ“ Testing context analysis...")
        print(f"Article preview: {test_article[:100]}...")
        print(f"Previous analysis: {test_input['previous_analysis']['prediction']} ({test_input['previous_analysis']['confidence']:.2%})")
        
        result = agent.process(test_input)
        
        if result['success']:
            analysis_data = result['result']
            print(f"âœ… Analysis completed successfully")
            print(f"   Overall context score: {analysis_data['context_scores']['overall_context_score']:.1f}/10")
            print(f"   Risk level: {analysis_data['context_scores']['risk_level']}")
            print(f"   Primary concerns: {len(analysis_data['context_scores']['primary_concerns'])}")
            print(f"   Response time: {analysis_data['metadata']['response_time_seconds']}s")
            print(f"   Config version: {analysis_data['metadata']['config_version']}")
            
            # Show analysis types generated
            analyses_generated = []
            if analysis_data.get('bias_analysis'):
                analyses_generated.append('Bias Analysis')
            if analysis_data.get('framing_analysis'):
                analyses_generated.append('Framing Analysis')
            if analysis_data.get('emotional_analysis'):
                analyses_generated.append('Emotional Analysis')
            if analysis_data.get('propaganda_analysis'):
                analyses_generated.append('Propaganda Analysis')
            
            print(f"   Analyses generated: {', '.join(analyses_generated)}")
            
            # Show pattern analysis results
            pattern_stats = analysis_data['pattern_analysis']
            total_bias_indicators = sum(pattern_stats['bias_counts'].values())
            total_emotional_indicators = sum(pattern_stats['emotional_counts'].values())
            print(f"   Pattern indicators: {total_bias_indicators} bias, {total_emotional_indicators} emotional")
            
        else:
            print(f"âŒ Analysis failed: {result['error']['message']}")
        
        # Show comprehensive metrics with config info
        print(f"\nðŸ“Š Comprehensive metrics with config info:")
        metrics = agent.get_comprehensive_metrics()
        print(f"Agent type: {metrics['agent_type']}")
        print(f"Config integrated: {metrics['config_integrated']}")
        print(f"Prompt source: {metrics['prompt_source']}")
        print(f"High bias detections: {metrics['analysis_specific_metrics']['high_bias_detected']}")
        print(f"High manipulation detections: {metrics['analysis_specific_metrics']['high_manipulation_detected']}")
        
        print(f"\nâœ… Modular context analyzer agent with config integration test completed!")
        
    except Exception as e:
        print(f"âŒ Test failed: {str(e)}")
        print("Make sure your GEMINI_API_KEY is set in your environment variables")
        import traceback
        traceback.print_exc()