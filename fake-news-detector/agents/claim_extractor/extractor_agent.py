# agents/claim_extractor/extractor_agent.py
"""
Enhanced Claim Extractor Agent - Main Implementation with Config Integration

This agent identifies and extracts specific, verifiable claims from news articles
that can be fact-checked, with full configuration integration and modular architecture.

Features:
- Configuration integration from config files
- Centralized prompt management
- Pattern-based pre-analysis with AI enhancement
- Multiple extraction modes for different complexity levels
- Comprehensive claim parsing and validation
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
from .patterns import ClaimPatternDatabase
from .parsers import ClaimParser

# ✅ IMPORT CONFIGURATION FILES
from config import get_model_config, get_prompt_template, get_settings
from utils.helpers import sanitize_text

class ClaimExtractorAgent(BaseAgent):
    """
    📋 ENHANCED CLAIM EXTRACTOR AGENT WITH CONFIG INTEGRATION
    
    Modular claim extraction agent that inherits from BaseAgent
    for consistent interface and LangGraph compatibility.
    
    Features:
    - Inherits from BaseAgent for consistent interface
    - Configuration integration from config files
    - Modular component architecture (patterns, parsers)
    - AI-powered claim extraction with pattern-based pre-screening
    - Multiple extraction modes for different use cases
    - Comprehensive claim categorization and prioritization
    - Performance tracking and metrics
    - LangGraph integration ready
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced claim extractor agent with config integration
        
        Args:
            config: Configuration dictionary for runtime overrides
        """
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        claim_config = get_model_config('claim_extractor')
        system_settings = get_settings()
        
        # Merge with runtime overrides
        if config:
            claim_config.update(config)
        
        self.agent_name = "claim_extractor"
        
        # Initialize base agent with merged config
        super().__init__(claim_config)
        
        # ✅ USE CONFIG VALUES FOR EXTRACTION SETTINGS
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 2048)
        
        # ✅ EXTRACTION SETTINGS FROM CONFIG
        self.max_claims = self.config.get('max_claims_per_article', 8)
        self.min_claim_length = self.config.get('min_claim_length', 10)
        self.enable_verification = self.config.get('enable_verification_analysis', True)
        self.enable_prioritization = self.config.get('enable_claim_prioritization', True)
        
        # ✅ PATTERN ANALYSIS SETTINGS FROM CONFIG
        self.enable_pattern_preprocessing = self.config.get('enable_pattern_preprocessing', True)
        self.pattern_confidence_threshold = self.config.get('pattern_confidence_threshold', 0.5)
        self.claim_richness_threshold = self.config.get('claim_richness_threshold', 5.0)
        
        # ✅ PARSING SETTINGS FROM CONFIG
        self.enable_fallback_parsing = self.config.get('enable_fallback_parsing', True)
        self.max_parsing_attempts = self.config.get('max_parsing_attempts', 3)
        self.parsing_quality_threshold = self.config.get('parsing_quality_threshold', 60)
        
        # ✅ GET API KEY FROM SYSTEM SETTINGS
        self.api_key = system_settings.gemini_api_key
        
        # ✅ LOAD PROMPTS FROM CONFIG INSTEAD OF HARDCODED
        self.claim_extraction_prompt = get_prompt_template('claim_extractor', 'claim_extraction')
        self.verification_prompt = get_prompt_template('claim_extractor', 'verification_analysis')
        self.prioritization_prompt = get_prompt_template('claim_extractor', 'claim_prioritization')
        
        # ✅ USE RATE LIMITING FROM CONFIG/SETTINGS
        self.rate_limit = self.config.get('rate_limit_seconds', system_settings.gemini_rate_limit)
        self.max_retries = self.config.get('max_retries', system_settings.max_retries)
        
        # Initialize Gemini API
        self._initialize_gemini_api()
        
        # Initialize modular components
        self.pattern_database = ClaimPatternDatabase()
        self.claim_parser = ClaimParser()
        
        # Enhanced performance tracking with config awareness
        self.extraction_metrics = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'total_claims_extracted': 0,
            'average_claims_per_article': 0.0,
            'average_response_time': 0.0,
            'gemini_api_calls': 0,
            'pattern_analyses_performed': 0,
            'verification_analyses_generated': 0,
            'prioritization_analyses_generated': 0,
            'config_integrated': True
        }
        
        # Rate limiting tracking
        self.last_request_time = None
        
        self.logger.info(f"✅ Enhanced Claim Extractor Agent initialized with config")
        self.logger.info(f"🤖 Model: {self.model_name}, Max Claims: {self.max_claims}")
        self.logger.info(f"🔍 Pattern Analysis: {'Enabled' if self.enable_pattern_preprocessing else 'Disabled'}")
        self.logger.info(f"📊 Verification: {'Enabled' if self.enable_verification else 'Disabled'}")
    
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
        🎯 MAIN PROCESSING METHOD - LANGGRAPH COMPATIBLE WITH CONFIG
        
        Process input according to BaseAgent interface for LangGraph compatibility.
        
        Args:
            input_data: Dictionary containing:
                - text: Article text to analyze
                - bert_results: BERT classification results
                - topic_domain: News category (optional)
                - include_verification_analysis: Force verification analysis
                
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
            bert_results = input_data.get('bert_results', {})
            topic_domain = input_data.get('topic_domain', 'general')
            include_verification_analysis = input_data.get(
                'include_verification_analysis', 
                self.enable_verification
            )
            
            # ✅ USE CONFIG FOR PROCESSING DECISIONS
            force_detailed = (
                bert_results.get('confidence', 0.0) < 0.7 or
                include_verification_analysis
            )
            
            # Perform claim extraction
            extraction_result = self.extract_claims(
                article_text=article_text,
                bert_results=bert_results,
                topic_domain=topic_domain,
                include_verification_analysis=force_detailed
            )
            
            # Extract confidence for metrics
            confidence = bert_results.get('confidence', 0.0)
            
            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(confidence)
            self.extraction_metrics['successful_extractions'] += 1
            self.extraction_metrics['total_claims_extracted'] += extraction_result['metadata']['total_claims_found']
            
            # Update average claims per article
            total_successful = self.extraction_metrics['successful_extractions']
            total_claims = self.extraction_metrics['total_claims_extracted']
            if total_successful > 0:
                self.extraction_metrics['average_claims_per_article'] = total_claims / total_successful
            
            # Format output for LangGraph with config context
            return self.format_output(
                result=extraction_result,
                confidence=confidence,
                metadata={
                    'response_time': extraction_result['metadata']['response_time_seconds'],
                    'model_used': self.model_name,
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular',
                    'pattern_analysis_enabled': self.enable_pattern_preprocessing,
                    'verification_analysis_included': extraction_result['metadata']['verification_analysis_included']
                }
            )
            
        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            return self.format_error_output(e, input_data)
    
    def extract_claims(self,
                      article_text: str,
                      bert_results: Dict[str, Any],
                      topic_domain: str = "general",
                      include_verification_analysis: bool = True) -> Dict[str, Any]:
        """
        🎯 MAIN METHOD - Extract Claims from Article with Config Integration
        
        This is the primary method that coordinates the entire claim extraction process
        using configuration-driven settings and parameters.
        
        Args:
            article_text: The full news article text to analyze
            bert_results: Results from BERT classifier (prediction, confidence)
            topic_domain: Category of news (politics, health, science, etc.)
            include_verification_analysis: Whether to analyze verifiability
            
        Returns:
            Dict containing comprehensive claim extraction results
        """
        self._respect_rate_limits()
        start_time = time.time()
        
        try:
            self.logger.info("Starting claim extraction analysis with config integration...")
            
            # Step 1: Extract information from BERT results
            prediction = bert_results.get('prediction', 'Unknown')
            confidence = bert_results.get('confidence', 0.0)
            
            # Step 2: Clean the article text
            article_text = sanitize_text(article_text)
            
            # ✅ USE CONFIG FOR TEXT LENGTH LIMITS
            max_text_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_text_length:
                article_text = article_text[:max_text_length] + "..."
            
            # Step 3: Run pattern-based claim detection if enabled
            pattern_analysis = {}
            if self.enable_pattern_preprocessing:
                pattern_analysis = self.pattern_database.analyze_claim_patterns(article_text)
                self.extraction_metrics['pattern_analyses_performed'] += 1
                self.logger.info(f"🔍 Pattern analysis: {pattern_analysis['total_claim_indicators']} indicators found")
            
            # Step 4: Generate topic summary for context
            topic_summary = self._generate_topic_summary(article_text, topic_domain)
            
            # Step 5: Generate AI-powered claim extraction using config prompts
            raw_extraction = self._generate_claim_extraction(
                article_text, prediction, confidence, topic_domain
            )
            if not raw_extraction or raw_extraction.startswith("Content ") or raw_extraction.startswith("No response"):
                # Fallback quick prompt path to bypass strict prompts when blocked
                fallback_prompt = f"Extract 3 main claims from this text: {article_text[:500]}"
                try:
                    fallback_response = self.model.generate_content(fallback_prompt)
                    if getattr(fallback_response, 'candidates', None) and (
                        getattr(fallback_response.candidates[0], 'finish_reason', None) != 2
                    ):
                        raw_extraction = getattr(fallback_response, 'text', None) or raw_extraction
                except Exception as _:
                    pass
            
            # Step 6: Parse extracted claims into structured format using modular parser
            structured_claims = self.claim_parser.parse_extracted_claims(raw_extraction)
            
            # ✅ APPLY CONFIG LIMITS
            if len(structured_claims) > self.max_claims:
                structured_claims = structured_claims[:self.max_claims]
                self.logger.info(f"⚠️ Limited claims to configured maximum: {self.max_claims}")
            
            # Step 7: Optionally generate verification analysis based on config
            verification_analysis = None
            if include_verification_analysis and self.enable_verification and structured_claims:
                verification_analysis = self._generate_verification_analysis(structured_claims)
                self.extraction_metrics['verification_analyses_generated'] += 1
            
            # Step 8: Generate claim prioritization if enabled
            claim_priorities = None
            if self.enable_prioritization and structured_claims:
                claim_priorities = self._generate_claim_prioritization(structured_claims)
                self.extraction_metrics['prioritization_analyses_generated'] += 1
            
            # Step 9: Package results with config metadata
            response_time = time.time() - start_time
            result = {
                'extracted_claims': structured_claims,
                'raw_extraction': raw_extraction,
                'verification_analysis': verification_analysis,
                'claim_priorities': claim_priorities,
                'topic_summary': topic_summary,
                'pattern_analysis': pattern_analysis,
                'claims_summary': self.claim_parser.format_claims_summary(structured_claims),
                'metadata': {
                    'total_claims_found': len(structured_claims),
                    'critical_claims': len(self.claim_parser.get_claims_by_priority(structured_claims, 1)) if structured_claims else 0,
                    'verifiable_claims': len(self.claim_parser.get_most_verifiable_claims(structured_claims)) if structured_claims else 0,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'response_time_seconds': round(response_time, 2),
                    'model_used': self.model_name,
                    'temperature_used': self.temperature,
                    'prompt_version': 'v2.0_config_integrated',
                    'verification_analysis_included': verification_analysis is not None,
                    'prioritization_included': claim_priorities is not None,
                    'article_length': len(article_text),
                    'topic_domain': topic_domain,
                    'bert_prediction': prediction,
                    'bert_confidence': confidence,
                    'claim_richness_score': pattern_analysis.get('claim_richness_score', 0) if pattern_analysis else 0,
                    'parsing_quality': self.claim_parser.calculate_parsing_quality(structured_claims) if structured_claims else {'quality': 'none', 'score': 0},
                    'agent_version': '2.0_modular',
                    'config_version': '2.0_integrated',
                    'max_claims_limit': self.max_claims,
                    'pattern_analysis_enabled': self.enable_pattern_preprocessing
                }
            }
            
            # Step 10: Update performance metrics
            self._update_extraction_metrics(response_time, len(structured_claims), success=True)
            
            self.logger.info(f"Successfully extracted {len(structured_claims)} claims in {response_time:.2f} seconds")
            return result
            
        except Exception as e:
            self._update_extraction_metrics(time.time() - start_time, 0, success=False)
            self.logger.error(f"Error in claim extraction: {str(e)}")
            raise
    
    def _generate_topic_summary(self, article_text: str, topic_domain: str) -> str:
        """Generate brief topic summary for context with config limits"""
        # ✅ USE CONFIG FOR SUMMARY LENGTH
        summary_length = self.config.get('topic_summary_length', 200)
        summary = article_text[:summary_length].strip()
        
        if len(article_text) > summary_length:
            # Find last complete sentence
            last_period = summary.rfind('.')
            if last_period > summary_length * 0.7:  # Ensure reasonable length
                summary = summary[:last_period + 1]
            else:
                summary += "..."
        
        return f"{topic_domain.title()} article: {summary}"
    
    def _generate_claim_extraction(self, article_text: str, prediction: str,
                                  confidence: float, topic_domain: str) -> str:
        """
        Generate AI-powered claim extraction using config prompt template
        
        Args:
            article_text: Article content
            prediction: BERT classification result
            confidence: BERT confidence score
            topic_domain: Article domain/category
            
        Returns:
            Raw AI extraction text
        """
        try:
            # ✅ USE PROMPT FROM CONFIG INSTEAD OF HARDCODED
            prompt = self.claim_extraction_prompt.format(
                article_text=article_text,
                prediction=prediction,
                confidence=confidence,
                topic_domain=topic_domain
            )

            response = self.model.generate_content(prompt)

            # Defensive handling for safety blocks/empty candidates
            if not getattr(response, 'candidates', None):
                self.logger.warning("Content blocked or empty response; falling back to conservative analysis")
                return "Content blocked by safety filters. Using fallback analysis."

            candidate = response.candidates[0]
            # finish_reason 2 corresponds to SAFETY block in Gemini API
            if getattr(candidate, 'finish_reason', None) == 2:
                self.logger.warning("Response flagged by safety filters; returning conservative analysis")
                return "Content flagged by safety filters. Proceeding with conservative analysis."

            # Ensure parts exist
            content = getattr(candidate, 'content', None)
            if not content or not getattr(content, 'parts', None):
                self.logger.warning("No content parts in response; using default analysis")
                return "No response generated. Using default analysis."

            self.extraction_metrics['gemini_api_calls'] += 1
            return getattr(response, 'text', None) or "No response text available."

        except Exception as e:
            self.logger.error(f"Error in claim extraction generation: {str(e)}")
            return f"Analysis temporarily unavailable. Error: {str(e)}"
    
    def _generate_verification_analysis(self, structured_claims: List[Dict]) -> str:
        """
        Generate AI-powered verification analysis using config prompt template
        
        Args:
            structured_claims: List of extracted claims
            
        Returns:
            Verification analysis text
        """
        try:
            # ✅ LIMIT CLAIMS FOR ANALYSIS BASED ON CONFIG
            analysis_claim_limit = self.config.get('verification_analysis_claim_limit', 5)
            
            # Prepare claims summary for analysis
            claims_summary = "\n".join([
                f"Claim {i+1}: {claim['text']} (Type: {claim['claim_type']}, Priority: {claim['priority']})"
                for i, claim in enumerate(structured_claims[:analysis_claim_limit])
            ])
            
            # ✅ USE VERIFICATION PROMPT FROM CONFIG
            prompt = self.verification_prompt.format(
                extracted_claims=claims_summary
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Verification analysis blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Verification analysis flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Verification analysis not available."
            self.extraction_metrics['gemini_api_calls'] += 1
            return getattr(response, 'text', None) or "Verification analysis unavailable."
            
        except Exception as e:
            self.logger.error(f"Error in verification analysis generation: {str(e)}")
            return "Verification analysis unavailable due to processing error."
    
    def _generate_claim_prioritization(self, structured_claims: List[Dict]) -> str:
        """
        Generate AI-powered claim prioritization using config prompt template
        
        Args:
            structured_claims: List of extracted claims
            
        Returns:
            Prioritization analysis text
        """
        try:
            if not structured_claims:
                return "No claims available for prioritization."
            
            # Prepare claims summary for prioritization
            claims_summary = "\n".join([
                f"Claim {i+1}: {claim['text']} "
                f"(Type: {claim['claim_type']}, Current Priority: {claim['priority']}, "
                f"Verifiability: {claim['verifiability_score']}/10)"
                for i, claim in enumerate(structured_claims)
            ])
            
            # ✅ USE PRIORITIZATION PROMPT FROM CONFIG
            prompt = self.prioritization_prompt.format(
                extracted_claims=claims_summary
            )
            
            response = self.model.generate_content(prompt)
            if not getattr(response, 'candidates', None):
                return "Prioritization blocked by safety filters."
            candidate = response.candidates[0]
            if getattr(candidate, 'finish_reason', None) == 2:
                return "Prioritization flagged by safety filters."
            if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
                return "Prioritization not available."
            self.extraction_metrics['gemini_api_calls'] += 1
            return getattr(response, 'text', None) or "Prioritization unavailable."
            
        except Exception as e:
            self.logger.error(f"Error in claim prioritization: {str(e)}")
            return "Claim prioritization unavailable due to processing error."
    
    def quick_extract(self, article_text: str, max_claims: Optional[int] = None) -> List[str]:
        """
        ⚡ QUICK CLAIM EXTRACTION WITH CONFIG
        
        Fast extraction method using only pattern-based detection with config limits.
        
        Args:
            article_text: Article to extract claims from
            max_claims: Maximum number of claims (uses config default if None)
            
        Returns:
            List of claim texts (strings)
        """
        try:
            # ✅ USE CONFIG FOR MAX CLAIMS IF NOT SPECIFIED
            max_claims = max_claims or self.config.get('quick_extract_max_claims', 5)
            
            # Use modular pattern database for quick extraction
            return self.pattern_database.extract_potential_claims(article_text, max_claims)
            
        except Exception as e:
            self.logger.error(f"Error in quick extraction: {str(e)}")
            return [f"Error extracting claims: {str(e)} - please try full extraction method"]
    
    def _respect_rate_limits(self):
        """Rate limiting using config values"""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _update_extraction_metrics(self, response_time: float, claims_found: int, success: bool):
        """Update extraction-specific metrics with config awareness"""
        self.extraction_metrics['total_extractions'] += 1
        
        if success:
            # Update average response time
            total = self.extraction_metrics['total_extractions']
            current_avg = self.extraction_metrics['average_response_time']
            self.extraction_metrics['average_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        📊 Get comprehensive performance metrics with config information
        
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
            'max_claims_per_article': self.max_claims,
            'min_claim_length': self.min_claim_length,
            'enable_verification': self.enable_verification,
            'enable_prioritization': self.enable_prioritization,
            'enable_pattern_preprocessing': self.enable_pattern_preprocessing,
            'pattern_confidence_threshold': self.pattern_confidence_threshold,
            'enable_fallback_parsing': self.enable_fallback_parsing,
            'parsing_quality_threshold': self.parsing_quality_threshold,
            'rate_limit_seconds': self.rate_limit,
            'config_version': '2.0_integrated'
        }
        
        # Get component metrics
        component_metrics = {
            'pattern_stats': self.pattern_database.get_pattern_statistics(),
            'api_calls_made': self.extraction_metrics['gemini_api_calls']
        }
        
        return {
            **base_metrics,
            'extraction_specific_metrics': self.extraction_metrics,
            'config_metrics': config_metrics,
            'component_info': component_metrics,
            'agent_type': 'claim_extractor',
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
            'max_claims_per_article': self.max_claims,
            'min_claim_length': self.min_claim_length,
            'enable_verification': self.enable_verification,
            'enable_prioritization': self.enable_prioritization,
            'enable_pattern_preprocessing': self.enable_pattern_preprocessing,
            'pattern_confidence_threshold': self.pattern_confidence_threshold,
            'enable_fallback_parsing': self.enable_fallback_parsing,
            'parsing_quality_threshold': self.parsing_quality_threshold,
            'rate_limit_seconds': self.rate_limit,
            'max_retries': self.max_retries,
            'config_source': 'config_files',
            'prompt_source': 'centralized_prompts_config'
        }

# Testing functionality with config integration
if __name__ == "__main__":
    """Test the modular claim extractor agent with config integration"""
    print("🧪 Testing Modular Claim Extractor Agent with Config Integration")
    print("=" * 70)
    
    try:
        # Initialize agent (will load from config files)
        agent = ClaimExtractorAgent()
        print(f"✅ Agent initialized with config: {agent}")
        
        # Show config summary
        config_summary = agent.get_config_summary()
        print(f"\n⚙️ Configuration Summary:")
        for key, value in config_summary.items():
            print(f"   {key}: {value}")
        
        # Test claim extraction
        test_article = """
        A new study published in Nature Medicine found that 85% of patients who received
        the experimental drug showed significant improvement within 30 days. Dr. Sarah Johnson,
        lead researcher at Harvard Medical School, announced the results at yesterday's conference.
        The clinical trial included 1,200 participants across 15 hospitals worldwide.
        According to the research team, the drug reduced symptoms by an average of 60%
        compared to the placebo group. The FDA is expected to fast-track approval by March 2024.
        """
        
        test_input = {
            "text": test_article,
            "bert_results": {
                "prediction": "REAL",
                "confidence": 0.78
            },
            "topic_domain": "health",
            "include_verification_analysis": True
        }
        
        print(f"\n📝 Testing claim extraction...")
        print(f"Article preview: {test_article[:100]}...")
        print(f"BERT result: {test_input['bert_results']['prediction']} ({test_input['bert_results']['confidence']:.2%})")
        
        result = agent.process(test_input)
        
        if result['success']:
            extraction_data = result['result']
            print(f"✅ Extraction completed successfully")
            print(f"   Total claims found: {extraction_data['metadata']['total_claims_found']}")
            print(f"   Critical claims: {extraction_data['metadata']['critical_claims']}")
            print(f"   Verifiable claims: {extraction_data['metadata']['verifiable_claims']}")
            print(f"   Claim richness score: {extraction_data['metadata']['claim_richness_score']}/10")
            print(f"   Response time: {extraction_data['metadata']['response_time_seconds']}s")
            print(f"   Config version: {extraction_data['metadata']['config_version']}")
            
            # Show sample claims
            if extraction_data['extracted_claims']:
                print(f"\n📋 Sample extracted claims:")
                for i, claim in enumerate(extraction_data['extracted_claims'][:3], 1):
                    print(f"   {i}. {claim['text'][:80]}...")
                    print(f"      Type: {claim['claim_type']}, Priority: {claim['priority']}")
        else:
            print(f"❌ Extraction failed: {result['error']['message']}")
        
        # Test quick extraction
        print(f"\n⚡ Testing quick extraction...")
        quick_claims = agent.quick_extract(test_article, max_claims=3)
        for i, claim in enumerate(quick_claims, 1):
            print(f"   {i}. {claim[:80]}...")
        
        # Show comprehensive metrics with config info
        print(f"\n📊 Comprehensive metrics with config info:")
        metrics = agent.get_comprehensive_metrics()
        print(f"Agent type: {metrics['agent_type']}")
        print(f"Config integrated: {metrics['config_integrated']}")
        print(f"Prompt source: {metrics['prompt_source']}")
        print(f"Pattern types: {metrics['component_info']['pattern_stats']['claim_pattern_types']}")
        print(f"Average claims per article: {metrics['extraction_specific_metrics']['average_claims_per_article']:.1f}")
        
        print(f"\n✅ Modular claim extractor agent with config integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure your GEMINI_API_KEY is set in your environment variables")
        import traceback
        traceback.print_exc()
