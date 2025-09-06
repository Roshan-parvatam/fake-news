# agents/llm_explanation/explanation_agent.py
"""
Enhanced LLM Explanation Agent - Main Implementation with Config Integration

This agent generates human-readable explanations of fake news detection results
using AI with centralized configuration management and modular architecture.

Features:
- Configuration integration from config files
- Centralized prompt management
- Multiple explanation modes (basic, detailed, confidence analysis)
- Rate limiting and error handling
- Performance metrics tracking
- LangGraph integration ready
"""

import os
import google.generativeai as genai
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import base agent functionality
from agents.base.base_agent import BaseAgent

# ✅ IMPORT CONFIGURATION FILES
from config import get_model_config, get_prompt_template, get_settings
from config.prompts_config import PromptsConfig

# Import utilities
from utils.helpers import sanitize_text

class LLMExplanationAgent(BaseAgent):
    """
    💬 ENHANCED LLM EXPLANATION AGENT WITH CONFIG INTEGRATION
    
    Generates human-readable explanations using Gemini AI with full
    configuration integration and modular prompt management.
    
    Features:
    - Inherits from BaseAgent for consistent interface
    - Configuration integration from config files
    - Centralized prompt template management
    - Multiple explanation types with AI analysis
    - Rate limiting and comprehensive error handling
    - Performance tracking and metrics
    - LangGraph integration ready
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced LLM explanation agent with config integration
        
        Args:
            config: Configuration dictionary for runtime overrides
        """
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        llm_config = get_model_config('llm_explanation')
        system_settings = get_settings()
        
        # Merge with runtime overrides
        if config:
            llm_config.update(config)
        
        self.agent_name = "llm_explanation"
        
        # Initialize base agent with merged config
        super().__init__(llm_config)
        
        # ✅ USE CONFIG VALUES FOR AI MODEL SETTINGS
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 3072)
        self.top_p = self.config.get('top_p', 0.9)
        self.top_k = self.config.get('top_k', 40)
        
        # ✅ GET API KEY FROM SYSTEM SETTINGS
        self.api_key = system_settings.gemini_api_key
        
        # ✅ ANALYSIS SETTINGS FROM CONFIG
        self.confidence_threshold = self.config.get('confidence_threshold', 0.75)
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.enable_source_analysis = self.config.get('enable_source_analysis', True)
        self.enable_confidence_analysis = self.config.get('enable_confidence_analysis', True)
        
        # ✅ LOAD PROMPTS FROM CONFIG INSTEAD OF HARDCODED
        self.explanation_prompt = get_prompt_template('llm_explanation', 'main_explanation')
        self.detailed_prompt = get_prompt_template('llm_explanation', 'detailed_analysis')
        self.confidence_prompt = get_prompt_template('llm_explanation', 'confidence_analysis')
        
        # ✅ USE RATE LIMITING FROM CONFIG/SETTINGS
        self.rate_limit = self.config.get('rate_limit_seconds', system_settings.gemini_rate_limit)
        self.max_retries = self.config.get('max_retries', system_settings.max_retries)
        
        # ✅ SAFETY SETTINGS FROM CONFIG
        self.safety_settings = self.config.get('safety_settings', [])
        
        # Initialize Gemini API with config settings
        self.model = None
        self._initialize_gemini_api()
        
        # Performance tracking with config awareness
        self.explanation_metrics = {
            'total_explanations': 0,
            'successful_explanations': 0,
            'detailed_analyses_generated': 0,
            'confidence_analyses_generated': 0,
            'average_response_time': 0.0,
            'gemini_api_calls': 0,
            'rate_limit_delays': 0,
            'config_integrated': True
        }
        
        # Rate limiting tracking
        self.last_request_time = None
        
        self.logger.info(f"✅ Enhanced LLM Explanation Agent initialized with config")
        self.logger.info(f"🤖 Model: {self.model_name}, Temperature: {self.temperature}")
        self.logger.info(f"⏱️ Rate Limit: {self.rate_limit}s, Confidence Threshold: {self.confidence_threshold}")
    
    def _initialize_gemini_api(self):
        """
        🔐 INITIALIZE GEMINI API WITH CONFIG SETTINGS
        
        Sets up Gemini AI connection using configuration values for all parameters.
        """
        try:
            if not self.api_key:
                raise ValueError("Gemini API key not found in system settings. Please set GEMINI_API_KEY environment variable.")
            
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # ✅ USE GENERATION CONFIG FROM CONFIG FILES
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }
            
            # Create model instance with config settings
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            self.logger.info("🔐 Gemini API initialized with config settings")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Gemini API: {str(e)}")
            raise
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 MAIN PROCESSING METHOD - LANGGRAPH COMPATIBLE WITH CONFIG
        
        Process input according to BaseAgent interface for LangGraph compatibility.
        Now uses configuration for all processing decisions.
        
        Args:
            input_data: Dictionary containing:
                - text: Article text
                - prediction: FAKE/REAL classification
                - confidence: Confidence score
                - metadata: Additional context (source, date, etc.)
                - require_detailed_analysis: Force detailed analysis
                
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
            prediction = input_data.get('prediction', 'Unknown')
            confidence = input_data.get('confidence', 0.0)
            metadata = input_data.get('metadata', {})
            require_detailed_analysis = input_data.get('require_detailed_analysis', False)
            
            # ✅ USE CONFIG FOR PROCESSING DECISIONS
            trigger_detailed = (
                require_detailed_analysis or 
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
            
            # Check for generation errors
            if not explanation_result.get('explanation'):
                raise Exception("Failed to generate explanation")
            
            # Extract confidence for metrics
            result_confidence = confidence
            
            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(result_confidence)
            self.explanation_metrics['successful_explanations'] += 1
            
            # Update specific metrics
            if explanation_result.get('detailed_analysis'):
                self.explanation_metrics['detailed_analyses_generated'] += 1
            if explanation_result.get('confidence_analysis'):
                self.explanation_metrics['confidence_analyses_generated'] += 1
            
            # Format output for LangGraph with config context
            return self.format_output(
                result=explanation_result,
                confidence=result_confidence,
                metadata={
                    'response_time': explanation_result['metadata']['response_time_seconds'],
                    'model_used': self.model_name,
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular',
                    'detailed_analysis_triggered': trigger_detailed,
                    'confidence_threshold_used': self.confidence_threshold
                }
            )
            
        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            return self.format_error_output(e, input_data)
    
    def generate_explanation(self,
                           article_text: str,
                           prediction: str,
                           confidence: float,
                           metadata: Dict[str, Any],
                           require_detailed_analysis: bool = False) -> Dict[str, Any]:
        """
        💬 MAIN EXPLANATION GENERATION WITH CONFIG INTEGRATION
        
        Generate comprehensive human-readable explanations using config-driven prompts
        and analysis settings.
        
        Args:
            article_text: The news article text
            prediction: REAL or FAKE classification
            confidence: Confidence score (0.0-1.0)
            metadata: Additional context (source, date, subject, etc.)
            require_detailed_analysis: Force detailed forensic analysis
            
        Returns:
            Dict containing comprehensive explanation results
        """
        self._respect_rate_limits()
        start_time = time.time()
        
        try:
            self.logger.info("Starting explanation generation with config integration...")
            
            # Extract metadata with config-aware defaults
            source = metadata.get('source', 'Unknown Source')
            date = metadata.get('date', 'Unknown Date')
            subject = metadata.get('subject', 'General News')
            
            # Clean article text
            article_text = sanitize_text(article_text)
            
            # ✅ USE CONFIG-BASED LENGTH LIMITS
            max_article_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_article_length:
                article_text = article_text[:max_article_length] + "..."
            
            # Step 1: Generate primary explanation using config prompts
            explanation = self._generate_primary_explanation(
                article_text, prediction, confidence, source, date, subject
            )
            
            # Step 2: Conditional detailed analysis based on config
            detailed_analysis = None
            if require_detailed_analysis or confidence < self.confidence_threshold:
                detailed_analysis = self._generate_detailed_analysis(
                    article_text, prediction, confidence, metadata
                )
                self.logger.info("🔍 Detailed forensic analysis generated")
            
            # Step 3: Confidence analysis if enabled in config
            confidence_analysis = None
            if self.enable_confidence_analysis:
                confidence_analysis = self._generate_confidence_analysis(
                    article_text, prediction, confidence
                )
                self.logger.info("📊 Confidence analysis generated")
            
            # Step 4: Source assessment if enabled in config
            source_assessment = None
            if self.enable_source_analysis:
                source_assessment = self._assess_source_reliability(source)
            
            # Package results with config metadata
            response_time = time.time() - start_time
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
                    'response_time_seconds': round(response_time, 2),
                    'model_used': self.model_name,
                    'temperature_used': self.temperature,
                    'detailed_analysis_triggered': detailed_analysis is not None,
                    'confidence_threshold': self.confidence_threshold,
                    'article_length_processed': len(article_text),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular'
                }
            }
            
            # Update performance metrics
            self._update_explanation_metrics(response_time, detailed_analysis is not None)
            
            self.logger.info(f"Successfully generated explanation in {response_time:.2f} seconds")
            return result
            
        except Exception as e:
            self._update_explanation_metrics(time.time() - start_time, False, error=True)
            self.logger.error(f"Error in explanation generation: {str(e)}")
            raise
    
    def _generate_primary_explanation(self, article_text: str, prediction: str, 
                                    confidence: float, source: str, date: str, subject: str) -> str:
        """
        Generate main explanation using config prompt template
        
        Args:
            article_text: Article content
            prediction: Classification result
            confidence: Confidence score
            source: Article source
            date: Publication date
            subject: Article subject
            
        Returns:
            Primary explanation text
        """
        try:
            # ✅ USE PROMPT FROM CONFIG INSTEAD OF HARDCODED
            prompt = self.explanation_prompt.format(
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                source=source,
                date=date,
                subject=subject
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Explanation blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Explanation flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Explanation not available."
            self.explanation_metrics['gemini_api_calls'] += 1
            
            return getattr(response, 'text', None) or "Explanation not available."
            
        except Exception as e:
            self.logger.error(f"Error generating primary explanation: {str(e)}")
            raise
    
    def _generate_detailed_analysis(self, article_text: str, prediction: str,
                                  confidence: float, metadata: Dict[str, Any]) -> str:
        """
        Generate detailed forensic analysis using config prompt template
        
        Args:
            article_text: Article content
            prediction: Classification result
            confidence: Confidence score
            metadata: Additional context
            
        Returns:
            Detailed forensic analysis text
        """
        try:
            # ✅ USE DETAILED PROMPT FROM CONFIG
            prompt = self.detailed_prompt.format(
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                metadata=str(metadata)
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Detailed analysis blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Detailed analysis flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Detailed analysis not available."
            self.explanation_metrics['gemini_api_calls'] += 1
            
            return getattr(response, 'text', None) or "Detailed analysis unavailable."
            
        except Exception as e:
            self.logger.error(f"Error generating detailed analysis: {str(e)}")
            return "Detailed analysis unavailable due to processing error."
    
    def _generate_confidence_analysis(self, article_text: str, prediction: str, confidence: float) -> str:
        """
        Generate confidence appropriateness analysis using config prompt template
        
        Args:
            article_text: Article content
            prediction: Classification result
            confidence: Confidence score
            
        Returns:
            Confidence analysis text
        """
        try:
            # ✅ USE CONFIDENCE PROMPT FROM CONFIG
            prompt = self.confidence_prompt.format(
                article_text=article_text,
                prediction=prediction,
                confidence=confidence
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Confidence analysis blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Confidence analysis flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Confidence analysis not available."
            self.explanation_metrics['gemini_api_calls'] += 1
            
            return getattr(response, 'text', None) or "Confidence analysis unavailable."
            
        except Exception as e:
            self.logger.error(f"Error generating confidence analysis: {str(e)}")
            return "Confidence analysis unavailable due to processing error."
    
    def _assess_source_reliability(self, source: str) -> Dict[str, Any]:
        """
        Assess source reliability using config-based criteria
        
        Args:
            source: Source name or URL
            
        Returns:
            Source reliability assessment
        """
        # This would integrate with the source database from credible_source agent
        # For now, basic assessment
        source_lower = source.lower()
        
        # ✅ USE CONFIG FOR SOURCE ASSESSMENT CRITERIA
        high_reliability_indicators = [
            'reuters', 'associated press', 'bbc', 'npr', 'pbs',
            'wall street journal', 'new york times', 'washington post'
        ]
        
        medium_reliability_indicators = [
            'cnn', 'fox news', 'msnbc', 'abc news', 'cbs news'
        ]
        
        reliability = "UNKNOWN"
        if any(indicator in source_lower for indicator in high_reliability_indicators):
            reliability = "HIGH"
        elif any(indicator in source_lower for indicator in medium_reliability_indicators):
            reliability = "MEDIUM"
        elif source_lower in ['unknown source', 'unknown', '']:
            reliability = "UNKNOWN"
        else:
            reliability = "LOW"
        
        return {
            'reliability': reliability,
            'source': source,
            'assessment_method': 'config_based_patterns',
            'confidence': 0.7 if reliability != "UNKNOWN" else 0.1
        }
    
    def _respect_rate_limits(self):
        """
        Rate limiting using config values
        """
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                delay = self.rate_limit - time_since_last
                time.sleep(delay)
                self.explanation_metrics['rate_limit_delays'] += 1
        self.last_request_time = time.time()
    
    def _update_explanation_metrics(self, response_time: float, detailed_generated: bool, error: bool = False):
        """Update explanation-specific metrics with config awareness"""
        self.explanation_metrics['total_explanations'] += 1
        
        if not error:
            # Update average response time
            total = self.explanation_metrics['total_explanations']
            current_avg = self.explanation_metrics['average_response_time']
            self.explanation_metrics['average_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )
    
    # ✅ ASYNC SUPPORT WITH CONFIG
    async def generate_explanation_async(self, article_text: str, prediction: str,
                                       confidence: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous explanation generation for web applications
        
        Args:
            article_text: Article content
            prediction: Classification result
            confidence: Confidence score
            metadata: Additional context
            
        Returns:
            Explanation results
        """
        import asyncio
        
        # Run synchronous version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.generate_explanation,
            article_text, prediction, confidence, metadata
        )
    
    def batch_explain(self, articles_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple articles with config-aware error handling
        
        Args:
            articles_data: List of article data dictionaries
            
        Returns:
            List of explanation results
        """
        results = []
        
        for i, article_data in enumerate(articles_data):
            try:
                self.logger.info(f"Processing article {i+1}/{len(articles_data)}")
                
                result = self.generate_explanation(
                    article_text=article_data.get('text', ''),
                    prediction=article_data.get('prediction', 'Unknown'),
                    confidence=article_data.get('confidence', 0.0),
                    metadata=article_data.get('metadata', {}),
                    require_detailed_analysis=article_data.get('require_detailed_analysis', False)
                )
                
                result['batch_index'] = i
                result['batch_success'] = True
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing article {i+1}: {str(e)}")
                
                error_result = {
                    'batch_index': i,
                    'batch_success': False,
                    'error': str(e),
                    'explanation': f"Error processing article: {str(e)}",
                    'metadata': {
                        'error_occurred': True,
                        'config_version': '2.0_integrated'
                    }
                }
                results.append(error_result)
        
        return results
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics with config information
        
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
            'confidence_threshold': self.confidence_threshold,
            'rate_limit_seconds': self.rate_limit,
            'detailed_analysis_enabled': self.enable_detailed_analysis,
            'source_analysis_enabled': self.enable_source_analysis,
            'confidence_analysis_enabled': self.enable_confidence_analysis,
            'config_version': '2.0_integrated'
        }
        
        return {
            **base_metrics,
            'explanation_specific_metrics': self.explanation_metrics,
            'config_metrics': config_metrics,
            'agent_type': 'llm_explanation',
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
            'confidence_threshold': self.confidence_threshold,
            'rate_limit_seconds': self.rate_limit,
            'detailed_analysis_enabled': self.enable_detailed_analysis,
            'source_analysis_enabled': self.enable_source_analysis,
            'confidence_analysis_enabled': self.enable_confidence_analysis,
            'safety_settings_count': len(self.safety_settings),
            'config_source': 'config_files',
            'prompt_source': 'centralized_prompts_config'
        }

# Testing functionality with config integration
if __name__ == "__main__":
    """Test the modular LLM explanation agent with config integration"""
    print("🧪 Testing Modular LLM Explanation Agent with Config Integration")
    print("=" * 70)
    
    try:
        # Initialize agent (will load from config files)
        agent = LLMExplanationAgent()
        print(f"✅ Agent initialized with config: {agent}")
        
        # Show config summary
        config_summary = agent.get_config_summary()
        print(f"\n⚙️ Configuration Summary:")
        for key, value in config_summary.items():
            print(f"   {key}: {value}")
        
        # Test explanation generation
        test_input = {
            "text": "A new study claims that drinking coffee can cure all diseases. The research was conducted by an anonymous scientist who refuses to share the data.",
            "prediction": "FAKE",
            "confidence": 0.89,
            "metadata": {
                "source": "UnknownBlog.com",
                "date": "2024-01-15",
                "subject": "Health"
            }
        }
        
        print(f"\n📝 Testing explanation generation...")
        print(f"Article: {test_input['text'][:100]}...")
        print(f"Classification: {test_input['prediction']} ({test_input['confidence']:.2%} confidence)")
        
        result = agent.process(test_input)
        
        if result['success']:
            explanation_data = result['result']
            print(f"✅ Explanation generated successfully")
            print(f"   Response time: {explanation_data['metadata']['response_time_seconds']}s")
            print(f"   Detailed analysis: {'Yes' if explanation_data['detailed_analysis'] else 'No'}")
            print(f"   Confidence analysis: {'Yes' if explanation_data['confidence_analysis'] else 'No'}")
            print(f"   Config version: {explanation_data['metadata']['config_version']}")
            
            # Show partial explanation
            explanation_preview = explanation_data['explanation'][:200] + "..." if len(explanation_data['explanation']) > 200 else explanation_data['explanation']
            print(f"\n📄 Explanation preview:")
            print(f"   {explanation_preview}")
        else:
            print(f"❌ Explanation generation failed: {result['error']['message']}")
        
        # Show comprehensive metrics with config info
        print(f"\n📊 Comprehensive metrics with config info:")
        metrics = agent.get_comprehensive_metrics()
        print(f"Agent type: {metrics['agent_type']}")
        print(f"Config integrated: {metrics['config_integrated']}")
        print(f"Prompt source: {metrics['prompt_source']}")
        print(f"API calls made: {metrics['explanation_specific_metrics']['gemini_api_calls']}")
        
        print(f"\n✅ Modular LLM explanation agent with config integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure your GEMINI_API_KEY is set in your environment variables")
        import traceback
        traceback.print_exc()
