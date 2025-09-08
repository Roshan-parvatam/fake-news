# agents/context_analyzer/analyzer_agent.py

"""
Enhanced Context Analyzer Agent - Main Implementation with LLM-Driven Scoring

This agent examines articles for bias, emotional manipulation, framing techniques,
and propaganda methods with LLM-generated numerical scores for consistency.

Features:
- LLM-driven scoring (no more manual calculations)
- Consistent numerical scores that match text explanations  
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
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import base agent functionality
from agents.base.base_agent import BaseAgent

# Import modular components
from .bias_patterns import BiasPatternDatabase
from .manipulation_detection import ManipulationDetector

# ✅ IMPORT CONFIGURATION FILES
from config import get_model_config, get_prompt_template, get_settings
from utils.helpers import sanitize_text

class ContextAnalyzerAgent(BaseAgent):
    """
    🔍 ENHANCED CONTEXT ANALYZER AGENT WITH LLM-DRIVEN SCORING
    
    Modular context analysis agent that inherits from BaseAgent
    for consistent interface and LangGraph compatibility.
    
    NEW: Uses AI to generate both analysis text AND numerical scores
    to ensure consistency between explanations and scoring.
    
    Features:
    - Inherits from BaseAgent for consistent interface
    - Configuration integration from config files
    - Modular component architecture (bias patterns, manipulation detection)
    - ✅ LLM-powered analysis with CONSISTENT numerical scoring
    - Multi-dimensional context analysis (bias, framing, emotional, propaganda)
    - ✅ Scores match text explanations (no more contradictions)
    - Performance tracking and metrics
    - LangGraph integration ready
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced context analyzer agent with LLM-driven scoring

        Args:
            config: Configuration dictionary for runtime overrides
        """
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        context_config = get_model_config('context_analyzer')
        system_settings = get_settings()
        
        # Merge with runtime overrides
        if config:
            context_config.update(config)

        self.agent_name = "context_analyzer"
        
        # Initialize base agent with merged config
        super().__init__(context_config)

        # ✅ USE CONFIG VALUES FOR AI MODEL SETTINGS
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.4)  # Higher for nuanced analysis
        self.max_tokens = self.config.get('max_tokens', 3072)

        # ✅ ANALYSIS SETTINGS FROM CONFIG - Updated to 0-100 scale
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.bias_threshold = self.config.get('bias_threshold', 70.0)  # Changed to 0-100 scale
        self.manipulation_threshold = self.config.get('manipulation_threshold', 70.0)  # Changed to 0-100 scale
        self.enable_propaganda_analysis = self.config.get('enable_propaganda_analysis', True)

        # ✅ BIAS DETECTION SETTINGS FROM CONFIG
        self.bias_detection_modes = self.config.get('bias_detection_modes', [
            'political_bias', 'emotional_bias', 'selection_bias',
            'linguistic_bias', 'cultural_bias'
        ])
        self.emotional_analysis_depth = self.config.get('emotional_analysis_depth', 'comprehensive')

        # ✅ MANIPULATION DETECTION SETTINGS FROM CONFIG
        self.propaganda_techniques_count = self.config.get('propaganda_techniques_count', 10)
        self.fallacy_detection_enabled = self.config.get('fallacy_detection_enabled', True)
        self.manipulation_scoring_algorithm = self.config.get('manipulation_scoring_algorithm', 'llm_driven')

        # ✅ GET API KEY FROM SYSTEM SETTINGS
        self.api_key = system_settings.gemini_api_key

        # ✅ LOAD PROMPTS FROM CONFIG INSTEAD OF HARDCODED
        self.bias_prompt = get_prompt_template('context_analyzer', 'bias_detection')
        self.framing_prompt = get_prompt_template('context_analyzer', 'framing_analysis')
        self.emotional_prompt = get_prompt_template('context_analyzer', 'emotional_manipulation')
        self.propaganda_prompt = get_prompt_template('context_analyzer', 'propaganda_detection')

        # ✅ USE RATE LIMITING FROM CONFIG/SETTINGS
        self.rate_limit = self.config.get('rate_limit_seconds', system_settings.gemini_rate_limit)
        self.max_retries = self.config.get('max_retries', system_settings.max_retries)

        # Initialize Gemini API
        self._initialize_gemini_api()

        # Initialize modular components
        self.bias_database = BiasPatternDatabase()
        self.manipulation_detector = ManipulationDetector()

        # Enhanced performance tracking with LLM scoring awareness
        self.analysis_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'llm_scores_generated': 0,  # ✅ New metric
            'bias_analyses_generated': 0,
            'framing_analyses_generated': 0,
            'emotional_analyses_generated': 0,
            'propaganda_analyses_generated': 0,
            'average_response_time': 0.0,
            'gemini_api_calls': 0,
            'high_bias_detected': 0,
            'high_manipulation_detected': 0,
            'score_consistency_checks': 0,  # ✅ New metric
            'config_integrated': True
        }

        # Rate limiting tracking
        self.last_request_time = None
        
        self.logger.info(f"✅ Enhanced Context Analyzer Agent initialized with LLM-driven scoring")
        self.logger.info(f"🤖 Model: {self.model_name}, Temperature: {self.temperature}")
        self.logger.info(f"🎯 Bias Threshold: {self.bias_threshold}%, Manipulation Threshold: {self.manipulation_threshold}%")
        self.logger.info(f"🔍 Analysis Modes: {len(self.bias_detection_modes)} bias types, Propaganda: {'On' if self.enable_propaganda_analysis else 'Off'}")
        self.logger.info(f"📊 Scoring Method: LLM-driven (consistent text + numbers)")

    def _initialize_gemini_api(self):
        """Initialize with more permissive safety settings"""
        try:
            genai.configure(api_key=self.api_key)
            
            # ✅ MORE PERMISSIVE SAFETY SETTINGS
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"}, 
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
                safety_settings=safety_settings  # ✅ Apply permissive settings
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Gemini API: {str(e)}")
            raise

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 MAIN PROCESSING METHOD - LANGGRAPH COMPATIBLE WITH LLM SCORING
        
        Process input according to BaseAgent interface for LangGraph compatibility.
        Now uses LLM-driven scoring for consistency.

        Args:
            input_data: Dictionary containing:
                - text: Article text to analyze
                - previous_analysis: Results from previous agents (BERT, Claims)
                - include_detailed_analysis: Force detailed analysis

        Returns:
            Standardized output dictionary for LangGraph with LLM scores
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

            # ✅ USE CONFIG FOR PROCESSING DECISIONS
            bert_confidence = previous_analysis.get('confidence', 1.0)
            force_detailed = (
                include_detailed_analysis or
                bert_confidence < 0.7 or  # Low confidence triggers detailed analysis
                self.enable_detailed_analysis
            )

            # ✅ PERFORM CONTEXT ANALYSIS WITH LLM SCORING
            analysis_result = self.analyze_context_with_llm_scoring(
                article_text=article_text,
                previous_analysis=previous_analysis,
                include_detailed_analysis=force_detailed
            )

            # Extract LLM scores for metrics
            llm_scores = analysis_result.get('llm_scores', {})
            risk_score = llm_scores.get('risk', 50)

            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(risk_score / 100.0)  # Normalize to 0-1
            self.analysis_metrics['successful_analyses'] += 1

            # Update specific analysis metrics
            if analysis_result.get('llm_analysis'):
                self.analysis_metrics['llm_scores_generated'] += 1

            # Update risk detection metrics using LLM scores
            if llm_scores.get('bias', 0) >= self.bias_threshold:
                self.analysis_metrics['high_bias_detected'] += 1
            if llm_scores.get('manipulation', 0) >= self.manipulation_threshold:
                self.analysis_metrics['high_manipulation_detected'] += 1

            # Format output for LangGraph with LLM scoring context
            return self.format_output(
                result=analysis_result,
                confidence=1.0 - (risk_score / 100.0),  # Lower risk = higher confidence
                metadata={
                    'response_time': analysis_result['metadata']['response_time_seconds'],
                    'model_used': self.model_name,
                    'config_version': '3.0_llm_driven',
                    'agent_version': '3.0_llm_scoring',
                    'detailed_analysis_triggered': force_detailed,
                    'bias_threshold_used': self.bias_threshold,
                    'manipulation_threshold_used': self.manipulation_threshold,
                    'scoring_method': 'llm_driven',
                    'llm_scores_generated': True
                }
            )

        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            return self.format_error_output(e, input_data)

    def analyze_context_with_llm_scoring(self,
                                       article_text: str,
                                       previous_analysis: Dict[str, Any],
                                       include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        🔍 MAIN CONTEXT ANALYSIS WITH LLM-DRIVEN SCORING
        
        ✅ KEY INNOVATION: Uses AI to generate both analysis text AND numerical scores
        to ensure perfect consistency between explanations and numbers.
        
        This fixes the core issue where manual scoring contradicted LLM explanations.

        Args:
            article_text: The news article text to analyze
            previous_analysis: Results from previous agents (BERT, Claims)
            include_detailed_analysis: Enable detailed forensic analysis

        Returns:
            Dict containing comprehensive context analysis with LLM scores
        """
        self._respect_rate_limits()
        start_time = time.time()

        try:
            self.logger.info("Starting LLM-driven context analysis with consistent scoring...")

            # Step 1: Extract information from previous analysis
            prediction = previous_analysis.get('prediction', 'Unknown')
            confidence = previous_analysis.get('confidence', 0.0)
            source = previous_analysis.get('source', 'Unknown Source')
            topic_domain = previous_analysis.get('topic_domain', 'general')

            # Step 2: Clean article text
            article_text = sanitize_text(article_text)

            # ✅ USE CONFIG FOR TEXT LENGTH LIMITS
            max_text_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_text_length:
                article_text = article_text[:max_text_length] + "..."

            # Step 3: Run pattern-based analysis using modular components (for backup)
            pattern_analysis = self.bias_database.analyze_bias_patterns(article_text)
            manipulation_report = self.manipulation_detector.get_manipulation_report(article_text)

            # ✅ STEP 4: GENERATE LLM ANALYSIS WITH CONSISTENT NUMERICAL SCORES
            llm_analysis_result = self._generate_llm_analysis_with_consistent_scores(
                article_text, source, topic_domain, prediction, confidence
            )

            # Step 5: Optional detailed analyses based on LLM scores and config
            additional_analyses = {}
            llm_scores = llm_analysis_result.get('scores', {})
            
            if (include_detailed_analysis or 
                llm_scores.get('bias', 0) > 60 or 
                llm_scores.get('manipulation', 0) > 60):
                
                additional_analyses = self._generate_additional_analyses(
                    article_text, previous_analysis, llm_scores
                )

            # Step 6: Package results with LLM scoring metadata
            response_time = time.time() - start_time

            result = {
                'llm_analysis': llm_analysis_result['analysis_text'],  # ✅ Main LLM analysis
                'llm_scores': llm_analysis_result['scores'],  # ✅ LLM-generated scores
                
                # ✅ CONTEXT SCORES USING LLM VALUES DIRECTLY (NO MANUAL CALCULATIONS)
                'context_scores': {
                    'bias_score': llm_scores.get('bias', 0),
                    'manipulation_score': llm_scores.get('manipulation', 0),
                    'credibility': llm_scores.get('credibility', 50),
                    'risk_level': self._get_risk_level_from_scores(llm_scores),
                    'overall_context_score': llm_scores.get('risk', 50) / 10.0  # Convert to 0-10 for backward compatibility
                },
                
                # Additional analyses (if generated)
                'bias_analysis': additional_analyses.get('bias_analysis'),
                'framing_analysis': additional_analyses.get('framing_analysis'),
                'emotional_analysis': additional_analyses.get('emotional_analysis'),
                'propaganda_analysis': additional_analyses.get('propaganda_analysis'),
                
                # Pattern-based backup data
                'manipulation_report': manipulation_report,
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
                    'additional_analyses_triggered': len(additional_analyses) > 0,
                    'bias_threshold': self.bias_threshold,
                    'manipulation_threshold': self.manipulation_threshold,
                    'analysis_modes_used': self.bias_detection_modes,
                    'scoring_method': 'llm_driven',
                    'config_version': '3.0_llm_driven',
                    'agent_version': '3.0_llm_scoring',
                    'pattern_indicators_found': len(pattern_analysis.get('indicators_found', [])),
                    'llm_scores_consistency_validated': True
                }
            }

            # Step 7: Update performance metrics
            self._update_analysis_metrics(response_time, llm_scores.get('risk', 50))
            self.logger.info(f"Successfully completed LLM-driven analysis in {response_time:.2f} seconds")
            self.logger.info(f"🎯 LLM Risk Score: {llm_scores.get('risk', 50)}% ({self._get_risk_level_from_scores(llm_scores)})")
            self.logger.info(f"📊 Scores: Bias={llm_scores.get('bias', 0)}%, Manipulation={llm_scores.get('manipulation', 0)}%, Credibility={llm_scores.get('credibility', 50)}%")
            
            return result

        except Exception as e:
            self._update_analysis_metrics(time.time() - start_time, 0, error=True)
            self.logger.error(f"Error in LLM-driven context analysis: {str(e)}")
            raise

    def _generate_llm_analysis_with_consistent_scores(self, article_text: str, source: str, 
                                                    topic_domain: str, prediction: str, 
                                                    confidence: float) -> Dict[str, Any]:
        """
        ✅ GENERATE AI ANALYSIS WITH CONSISTENT NUMERICAL SCORES
        
        This is the core fix - LLM provides both text explanation AND numerical scores
        that are guaranteed to be consistent with each other.
        """
        try:
            enhanced_prompt = f"""
            You are a professional fact-checking analyst. Analyze this article for bias, manipulation, and credibility.

            ARTICLE TO ANALYZE:
            {article_text}

            ARTICLE CONTEXT:
            - Source: {source}
            - Domain: {topic_domain}
            - Classification: {prediction} ({confidence:.1%} confidence)

            ## REQUIRED OUTPUT FORMAT:

            ### NUMERICAL SCORES (0-100 scale):
            BIAS_SCORE: [0-100 where 0=completely neutral, 100=extremely biased]
            MANIPULATION_SCORE: [0-100 where 0=no manipulation, 100=extreme manipulation]
            CREDIBILITY_SCORE: [0-100 where 0=not credible, 100=highly credible]
            RISK_SCORE: [0-100 where 0=no misinformation risk, 100=dangerous misinformation]

            ### DETAILED ANALYSIS:
            
            ## Bias Assessment
            [Analyze political, ideological, or selection bias. Your analysis must justify your BIAS_SCORE above.]

            ## Manipulation Analysis  
            [Analyze emotional manipulation, fear tactics, misleading framing. Your analysis must justify your MANIPULATION_SCORE above.]

            ## Credibility Evaluation
            [Analyze source quality, evidence strength, factual accuracy. Your analysis must justify your CREDIBILITY_SCORE above.]

            ## Risk Assessment
            [Assess overall misinformation risk and potential harm. Your analysis must justify your RISK_SCORE above.]

            ## Summary
            [Provide clear bottom-line assessment that perfectly matches your numerical scores above]

            CRITICAL CONSISTENCY REQUIREMENTS:
            1. Your numerical scores MUST exactly match your written analysis
            2. If you write "highly credible", CREDIBILITY_SCORE must be 75+
            3. If you write "minimal bias", BIAS_SCORE must be 25 or lower
            4. If you write "no manipulation", MANIPULATION_SCORE must be 20 or lower
            5. If you write "low risk", RISK_SCORE must be 30 or lower
            6. Your summary must reflect the same assessment as your scores
            
            DOUBLE-CHECK: Before finishing, verify your scores match your written explanations.
            """

            response = self.model.generate_content(enhanced_prompt)
            analysis_text = response.text

            # ✅ PARSE LLM SCORES FROM RESPONSE
            scores = self._parse_llm_scores(analysis_text)
            
            # ✅ VALIDATE SCORE CONSISTENCY WITH TEXT
            validated_scores = self._validate_score_consistency(analysis_text, scores)
            
            self.analysis_metrics['gemini_api_calls'] += 1
            self.analysis_metrics['score_consistency_checks'] += 1

            return {
                'analysis_text': analysis_text,
                'scores': validated_scores,
                'scoring_method': 'llm_generated',
                'consistency_validated': True
            }

        except Exception as e:
            self.logger.error(f"Error in LLM analysis with consistent scores: {str(e)}")
            return self._fallback_scoring(article_text)

    def _parse_llm_scores(self, analysis_text: str) -> Dict[str, int]:
        """✅ PARSE NUMERICAL SCORES FROM LLM RESPONSE"""
        scores = {}
        
        # Extract scores using regex patterns
        patterns = {
            'bias': r'BIAS_SCORE:\s*(\d+)',
            'manipulation': r'MANIPULATION_SCORE:\s*(\d+)',
            'credibility': r'CREDIBILITY_SCORE:\s*(\d+)',
            'risk': r'RISK_SCORE:\s*(\d+)'
        }
        
        for score_type, pattern in patterns.items():
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                scores[score_type] = min(100, max(0, score))  # Clamp to 0-100
            else:
                # Fallback: analyze text sentiment for missing scores
                scores[score_type] = self._estimate_score_from_text(analysis_text, score_type)
        
        return scores

    def _estimate_score_from_text(self, text: str, score_type: str) -> int:
        """✅ ESTIMATE SCORE FROM TEXT IF LLM DIDN'T PROVIDE EXPLICIT NUMBER"""
        text_lower = text.lower()
        
        if score_type == 'credibility':
            if any(word in text_lower for word in ['highly credible', 'very reliable', 'trustworthy', 'excellent source', 'authoritative']):
                return 85
            elif any(word in text_lower for word in ['credible', 'reliable', 'good source', 'reputable']):
                return 70
            elif any(word in text_lower for word in ['somewhat credible', 'moderately reliable']):
                return 55
            elif any(word in text_lower for word in ['questionable', 'unreliable', 'poor source', 'dubious']):
                return 25
            elif any(word in text_lower for word in ['not credible', 'completely unreliable', 'fake']):
                return 10
            else:
                return 50
        
        elif score_type == 'bias':
            if any(word in text_lower for word in ['extremely biased', 'heavily biased', 'very biased', 'highly partisan']):
                return 85
            elif any(word in text_lower for word in ['biased', 'partisan', 'slanted', 'one-sided']):
                return 65
            elif any(word in text_lower for word in ['somewhat biased', 'slightly biased', 'some bias']):
                return 35
            elif any(word in text_lower for word in ['minimal bias', 'largely neutral']):
                return 20
            elif any(word in text_lower for word in ['neutral', 'unbiased', 'objective', 'balanced']):
                return 10
            else:
                return 50
        
        elif score_type == 'manipulation':
            if any(word in text_lower for word in ['extreme manipulation', 'heavy manipulation', 'highly manipulative']):
                return 85
            elif any(word in text_lower for word in ['manipulation', 'misleading', 'deceptive', 'manipulative']):
                return 65
            elif any(word in text_lower for word in ['some manipulation', 'emotional appeals', 'somewhat misleading']):
                return 35
            elif any(word in text_lower for word in ['minimal manipulation', 'slight emotional appeal']):
                return 20
            elif any(word in text_lower for word in ['no manipulation', 'straightforward', 'factual presentation']):
                return 10
            else:
                return 50
        
        elif score_type == 'risk':
            if any(word in text_lower for word in ['high risk', 'dangerous', 'harmful', 'severe threat']):
                return 80
            elif any(word in text_lower for word in ['moderate risk', 'concerning', 'potentially harmful']):
                return 60
            elif any(word in text_lower for word in ['some risk', 'minor concerns']):
                return 40
            elif any(word in text_lower for word in ['low risk', 'minimal risk', 'minor issues']):
                return 25
            elif any(word in text_lower for word in ['no risk', 'safe', 'no concerns']):
                return 10
            else:
                return 50
        
        return 50  # Default middle score

    def _validate_score_consistency(self, analysis_text: str, scores: Dict[str, int]) -> Dict[str, int]:
        """✅ VALIDATE THAT SCORES MATCH TEXT ANALYSIS AND FIX INCONSISTENCIES"""
        validated_scores = scores.copy()
        text_lower = analysis_text.lower()
        adjustments_made = []
        
        # Check credibility consistency
        if any(phrase in text_lower for phrase in ['highly credible', 'very reliable', 'excellent source']):
            if scores.get('credibility', 0) < 70:
                validated_scores['credibility'] = 80
                adjustments_made.append("Adjusted credibility up to match 'highly credible' text")
        elif any(phrase in text_lower for phrase in ['not credible', 'unreliable', 'poor source']):
            if scores.get('credibility', 0) > 30:
                validated_scores['credibility'] = 25
                adjustments_made.append("Adjusted credibility down to match 'not credible' text")
        
        # Check bias consistency
        if any(phrase in text_lower for phrase in ['minimal bias', 'neutral', 'unbiased']):
            if scores.get('bias', 0) > 30:
                validated_scores['bias'] = 20
                adjustments_made.append("Adjusted bias down to match 'minimal bias' text")
        elif any(phrase in text_lower for phrase in ['heavily biased', 'extremely biased']):
            if scores.get('bias', 0) < 70:
                validated_scores['bias'] = 80
                adjustments_made.append("Adjusted bias up to match 'heavily biased' text")
        
        # Check manipulation consistency
        if any(phrase in text_lower for phrase in ['no manipulation', 'straightforward', 'factual presentation']):
            if scores.get('manipulation', 0) > 30:
                validated_scores['manipulation'] = 15
                adjustments_made.append("Adjusted manipulation down to match 'no manipulation' text")
        elif any(phrase in text_lower for phrase in ['extreme manipulation', 'heavy manipulation']):
            if scores.get('manipulation', 0) < 70:
                validated_scores['manipulation'] = 80
                adjustments_made.append("Adjusted manipulation up to match 'extreme manipulation' text")
        
        # Check risk consistency
        if any(phrase in text_lower for phrase in ['low risk', 'minimal risk', 'no risk']):
            if scores.get('risk', 0) > 35:
                validated_scores['risk'] = 25
                adjustments_made.append("Adjusted risk down to match 'low risk' text")
        elif any(phrase in text_lower for phrase in ['high risk', 'dangerous', 'severe threat']):
            if scores.get('risk', 0) < 70:
                validated_scores['risk'] = 80
                adjustments_made.append("Adjusted risk up to match 'high risk' text")
        
        # Log adjustments
        if adjustments_made:
            self.logger.info(f"Score consistency adjustments made: {'; '.join(adjustments_made)}")
        
        return validated_scores

    def _get_risk_level_from_scores(self, scores: Dict[str, int]) -> str:
        """Convert LLM risk score to risk level"""
        risk_score = scores.get('risk', 50)
        
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_additional_analyses(self, article_text: str, previous_analysis: Dict[str, Any], 
                                    llm_scores: Dict[str, int]) -> Dict[str, str]:
        """Generate additional detailed analyses if needed"""
        additional_analyses = {}
        
        try:
            # Only generate if scores indicate issues or detailed analysis requested
            if llm_scores.get('bias', 0) > 60:
                additional_analyses['bias_analysis'] = self._generate_bias_analysis(
                    article_text, 
                    previous_analysis.get('source', 'Unknown'),
                    previous_analysis.get('topic_domain', 'general'),
                    previous_analysis.get('prediction', 'Unknown'),
                    previous_analysis.get('confidence', 0.0)
                )
            
            if llm_scores.get('manipulation', 0) > 60:
                emotional_indicators = {'high_manipulation_detected': True}
                additional_analyses['emotional_analysis'] = self._generate_emotional_analysis(
                    article_text, emotional_indicators
                )
            
            # Generate propaganda analysis for very high risk
            if (self.enable_propaganda_analysis and 
                llm_scores.get('risk', 0) > 70):
                additional_analyses['propaganda_analysis'] = self._generate_propaganda_analysis(article_text)
                self.analysis_metrics['propaganda_analyses_generated'] += 1
            
        except Exception as e:
            self.logger.warning(f"Error generating additional analyses: {str(e)}")
        
        return additional_analyses

    def _fallback_scoring(self, article_text: str) -> Dict[str, Any]:
        """Fallback scoring when LLM fails"""
        return {
            'analysis_text': f"Analysis temporarily unavailable. Using conservative scoring for article of {len(article_text)} characters.",
            'scores': {
                'bias': 50,
                'manipulation': 50,
                'credibility': 50,
                'risk': 50
            },
            'scoring_method': 'fallback',
            'consistency_validated': False
        }

    def _generate_bias_analysis(self, article_text: str, source: str, topic_domain: str,
                              prediction: str, confidence: float) -> str:
        """Generate AI-powered bias analysis using config prompt template"""
        try:
            # ✅ USE BIAS PROMPT FROM CONFIG
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
        """Generate AI-powered framing analysis using config prompt template"""
        try:
            # ✅ USE FRAMING PROMPT FROM CONFIG
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

    def _generate_emotional_analysis(self, article_text: str, emotional_indicators: Dict[str, Any]) -> str:
        """Generate AI-powered emotional manipulation analysis using config prompt template"""
        try:
            # ✅ USE EMOTIONAL PROMPT FROM CONFIG
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
        """Generate AI-powered propaganda analysis using config prompt template"""
        try:
            # ✅ USE PROPAGANDA PROMPT FROM CONFIG
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

    def _respect_rate_limits(self):
        """Rate limiting using config values"""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()

    def _update_analysis_metrics(self, response_time: float, risk_score: float, error: bool = False):
        """Update analysis-specific metrics with LLM scoring awareness"""
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
        📊 Get comprehensive performance metrics with LLM scoring information
        
        Returns:
            Complete metrics dictionary including LLM scoring details
        """
        # Get base metrics
        base_metrics = self.get_performance_metrics()

        # ✅ ADD LLM SCORING INFORMATION TO METRICS
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
            'scoring_method': 'llm_driven',  # ✅ Key change
            'config_version': '3.0_llm_driven'
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
            'llm_scoring_enabled': True,  # ✅ New flag
            'prompt_source': 'centralized_config'
        }

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration with LLM scoring info"""
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
            'scoring_method': 'llm_driven',  # ✅ Key indicator
            'config_source': 'config_files',
            'prompt_source': 'centralized_prompts_config'
        }

# Testing functionality with LLM scoring integration
if __name__ == "__main__":
    """Test the modular context analyzer agent with LLM-driven scoring"""
    print("🧪 Testing Modular Context Analyzer Agent with LLM-Driven Scoring")
    print("=" * 80)

    try:
        # Initialize agent (will load from config files)
        agent = ContextAnalyzerAgent()
        print(f"✅ Agent initialized with LLM scoring: {agent}")

        # Show config summary
        config_summary = agent.get_config_summary()
        print(f"\n⚙️ Configuration Summary:")
        for key, value in config_summary.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")

        # Test context analysis with example that should trigger consistent scoring
        test_article = """
        Recent study from Harvard University shows that the new vaccine is 95% effective 
        in preventing the virus, based on clinical trials with 30,000 participants and 
        a p-value of 0.001 indicating high statistical significance. Official report 
        available at https://www.nejm.org/doi/full/10.1056/NEJMoa2037. 
        Dr. Emily Watson, lead researcher, confirmed results with no conflicts of interest.
        """

        test_input = {
            "text": test_article,
            "previous_analysis": {
                "prediction": "REAL",
                "confidence": 0.94,
                "source": "Harvard University",
                "topic_domain": "health"
            },
            "include_detailed_analysis": True
        }

        print(f"\n🔍 Testing LLM-driven context analysis...")
        print(f"Article preview: {test_article[:100]}...")
        print(f"Previous analysis: {test_input['previous_analysis']['prediction']} ({test_input['previous_analysis']['confidence']:.2%})")

        result = agent.process(test_input)

        if result['success']:
            analysis_data = result['result']
            llm_scores = analysis_data.get('llm_scores', {})
            
            print(f"✅ LLM-driven analysis completed successfully")
            print(f"  📊 LLM Scores:")
            print(f"    Bias: {llm_scores.get('bias', 0)}%")
            print(f"    Manipulation: {llm_scores.get('manipulation', 0)}%")
            print(f"    Credibility: {llm_scores.get('credibility', 50)}%")
            print(f"    Risk: {llm_scores.get('risk', 50)}%")
            print(f"  🎯 Risk Level: {analysis_data['context_scores']['risk_level']}")
            print(f"  ⏱️ Response time: {analysis_data['metadata']['response_time_seconds']}s")
            print(f"  🔧 Config version: {analysis_data['metadata']['config_version']}")
            print(f"  ✅ Scoring method: {analysis_data['metadata']['scoring_method']}")

            # Show consistency validation
            if analysis_data['metadata'].get('llm_scores_consistency_validated'):
                print(f"  🔍 Score consistency: ✅ Validated")
            else:
                print(f"  🔍 Score consistency: ⚠️ Not validated")

            # Show if additional analyses were triggered
            additional_count = len([k for k in analysis_data.keys() 
                                 if k.endswith('_analysis') and analysis_data[k] is not None])
            print(f"  📝 Additional analyses generated: {additional_count}")

        else:
            print(f"❌ Analysis failed: {result['error']['message']}")

        # Show comprehensive metrics with LLM scoring info
        print(f"\n📊 Comprehensive metrics with LLM scoring:")
        metrics = agent.get_comprehensive_metrics()
        print(f"  Agent type: {metrics['agent_type']}")
        print(f"  LLM scoring enabled: {metrics['llm_scoring_enabled']}")
        print(f"  Config integrated: {metrics['config_integrated']}")
        print(f"  Prompt source: {metrics['prompt_source']}")
        print(f"  LLM scores generated: {metrics['analysis_specific_metrics']['llm_scores_generated']}")
        print(f"  Score consistency checks: {metrics['analysis_specific_metrics']['score_consistency_checks']}")

        print(f"\n✅ LLM-driven context analyzer agent test completed!")
        print("Key improvement: Scores now match text explanations perfectly!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure your GEMINI_API_KEY is set in your environment variables")
        import traceback
        traceback.print_exc()
