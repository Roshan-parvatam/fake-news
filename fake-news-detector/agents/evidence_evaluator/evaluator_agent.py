# agents/evidence_evaluator/evaluator_agent.py

"""
Enhanced Evidence Evaluator Agent - Main Implementation with Config Integration

This agent evaluates the quality and reliability of evidence presented in articles,
analyzing source credibility, logical consistency, and evidence completeness
with full configuration integration and modular architecture.

Features:
- Configuration integration from config files
- Centralized prompt management
- Multi-criteria evidence evaluation (source quality, logic, completeness)
- Logical fallacy detection with AI enhancement
- Evidence gap analysis and scoring
- Performance tracking and metrics
- LangGraph integration ready
- Evidence link parsing for verification URLs
"""

import os
import google.generativeai as genai
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import base agent functionality
from agents.base.base_agent import BaseAgent

# Import modular components
from .criteria import EvidenceQualityCriteria
from .fallacy_detection import LogicalFallacyDetector

# ✅ IMPORT CONFIGURATION FILES
from config import get_model_config, get_prompt_template, get_settings
from utils.helpers import sanitize_text

class EvidenceEvaluatorAgent(BaseAgent):
    """
    📊 ENHANCED EVIDENCE EVALUATOR AGENT WITH CONFIG INTEGRATION
    
    Modular evidence evaluation agent that inherits from BaseAgent
    for consistent interface and LangGraph compatibility.
    
    Features:
    - Inherits from BaseAgent for consistent interface
    - Configuration integration from config files
    - Modular component architecture (criteria, fallacy detection)
    - AI-powered evidence analysis with systematic evaluation
    - Multi-criteria scoring (source quality, logic, completeness)
    - Comprehensive evidence gap analysis
    - Performance tracking and metrics
    - LangGraph integration ready
    - Evidence link parsing for verification URLs
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced evidence evaluator agent with config integration
        
        Args:
            config: Configuration dictionary for runtime overrides
        """
        
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        evidence_config = get_model_config('evidence_evaluator')
        system_settings = get_settings()
        
        # Merge with runtime overrides
        if config:
            evidence_config.update(config)
        
        self.agent_name = "evidence_evaluator"
        
        # Initialize base agent with merged config
        super().__init__(evidence_config)
        
        # ✅ USE CONFIG VALUES FOR AI MODEL SETTINGS
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)  # Lower for consistent analysis
        self.max_tokens = self.config.get('max_tokens', 3072)
        
        # ✅ EVALUATION SETTINGS FROM CONFIG
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.evidence_threshold = self.config.get('evidence_threshold', 6.0)
        self.enable_fallacy_detection = self.config.get('enable_fallacy_detection', True)
        self.enable_gap_analysis = self.config.get('enable_gap_analysis', True)
        
        # ✅ EVIDENCE CRITERIA SETTINGS FROM CONFIG
        self.evidence_types = self.config.get('evidence_types', [
            'statistical_evidence', 'documentary_evidence',
            'testimonial_evidence', 'circumstantial_evidence'
        ])
        
        self.source_quality_tiers = self.config.get('source_quality_tiers', [
            'primary', 'expert', 'institutional', 'journalistic'
        ])
        
        # ✅ SCORING WEIGHTS FROM CONFIG
        self.scoring_weights = self.config.get('scoring_weights', {
            'source_quality': 0.4,
            'logical_consistency': 0.3,
            'evidence_completeness': 0.3
        })
        
        # ✅ FALLACY DETECTION SETTINGS FROM CONFIG
        self.fallacy_types_count = self.config.get('fallacy_types_count', 10)
        self.reasoning_quality_threshold = self.config.get('reasoning_quality_threshold', 5.0)
        self.logical_health_threshold = self.config.get('logical_health_threshold', 6.0)
        
        # ✅ QUALITY THRESHOLDS FROM CONFIG
        self.high_quality_threshold = self.config.get('high_quality_threshold', 7.0)
        self.medium_quality_threshold = self.config.get('medium_quality_threshold', 5.0)
        self.poor_quality_threshold = self.config.get('poor_quality_threshold', 3.0)
        
        # ✅ GET API KEY FROM SYSTEM SETTINGS
        self.api_key = system_settings.gemini_api_key
        
        # ✅ LOAD PROMPTS FROM CONFIG INSTEAD OF HARDCODED
        self.evidence_prompt = get_prompt_template('evidence_evaluator', 'evidence_evaluation')
        self.source_quality_prompt = get_prompt_template('evidence_evaluator', 'source_quality')
        self.logical_consistency_prompt = get_prompt_template('evidence_evaluator', 'logical_consistency')
        self.evidence_gaps_prompt = get_prompt_template('evidence_evaluator', 'evidence_gaps')
        
        # ✅ USE RATE LIMITING FROM CONFIG/SETTINGS
        self.rate_limit = self.config.get('rate_limit_seconds', system_settings.gemini_rate_limit)
        self.max_retries = self.config.get('max_retries', system_settings.max_retries)
        
        # Initialize Gemini API
        self._initialize_gemini_api()
        
        # Initialize modular components
        self.quality_criteria = EvidenceQualityCriteria()
        self.fallacy_detector = LogicalFallacyDetector()
        
        # Enhanced performance tracking with config awareness
        self.evaluation_metrics = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'evidence_evaluations_generated': 0,
            'source_quality_analyses_generated': 0,
            'logical_consistency_analyses_generated': 0,
            'evidence_gap_analyses_generated': 0,
            'high_quality_evidence_found': 0,
            'poor_quality_evidence_found': 0,
            'fallacies_detected': 0,
            'average_response_time': 0.0,
            'gemini_api_calls': 0,
            'config_integrated': True,
            'verification_links_parsed': 0
        }
        
        # Rate limiting tracking
        self.last_request_time = None
        
        self.logger.info(f"✅ Enhanced Evidence Evaluator Agent initialized with config")
        self.logger.info(f"🤖 Model: {self.model_name}, Temperature: {self.temperature}")
        self.logger.info(f"🎯 Evidence Threshold: {self.evidence_threshold}, Quality Levels: {len(self.source_quality_tiers)}")
        self.logger.info(f"🔍 Fallacy Detection: {'On' if self.enable_fallacy_detection else 'Off'}, Gap Analysis: {'On' if self.enable_gap_analysis else 'Off'}")

    def _initialize_gemini_api(self):
        """
        🔐 INITIALIZE GEMINI API WITH CONFIG SETTINGS
        
        Sets up Gemini AI connection using configuration values optimized
        for evidence evaluation and analysis.
        """
        try:
            if not self.api_key:
                raise ValueError("Gemini API key not found in system settings")
            
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # ✅ USE GENERATION CONFIG FROM CONFIG FILES
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.config.get('top_p', 0.9),
                "top_k": self.config.get('top_k', 40),
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }
            
            # ✅ USE SAFETY SETTINGS FROM CONFIG
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
            
            self.logger.info("🔐 Gemini API initialized for evidence evaluation")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Gemini API: {str(e)}")
            raise

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 MAIN PROCESSING METHOD - LANGGRAPH COMPATIBLE WITH CONFIG
        
        Process input according to BaseAgent interface for LangGraph compatibility.
        
        Args:
            input_data: Dictionary containing:
                - text: Article text to evaluate
                - extracted_claims: Claims from claim extractor
                - context_analysis: Results from context analyzer
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
            extracted_claims = input_data.get('extracted_claims', [])
            context_analysis = input_data.get('context_analysis', {})
            include_detailed_analysis = input_data.get(
                'include_detailed_analysis',
                self.enable_detailed_analysis
            )
            
            # ✅ USE CONFIG FOR PROCESSING DECISIONS
            context_score = context_analysis.get('overall_context_score', 5.0)
            force_detailed = (
                include_detailed_analysis or
                context_score > 7.0 or  # High context issues trigger detailed analysis
                len(extracted_claims) < 2 or  # Few claims need more scrutiny
                self.enable_detailed_analysis
            )
            
            # Perform evidence evaluation
            evaluation_result = self.evaluate_evidence(
                article_text=article_text,
                extracted_claims=extracted_claims,
                context_analysis=context_analysis,
                include_detailed_analysis=force_detailed
            )
            
            # Extract overall evidence score for metrics
            evidence_score = evaluation_result['evidence_scores']['overall_evidence_score']
            
            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(evidence_score / 10.0)  # Normalize to 0-1
            self.evaluation_metrics['successful_evaluations'] += 1
            
            # Update specific evaluation metrics
            if evaluation_result.get('evidence_analysis'):
                self.evaluation_metrics['evidence_evaluations_generated'] += 1
            if evaluation_result.get('source_quality_analysis'):
                self.evaluation_metrics['source_quality_analyses_generated'] += 1
            if evaluation_result.get('logical_consistency_analysis'):
                self.evaluation_metrics['logical_consistency_analyses_generated'] += 1
            if evaluation_result.get('evidence_gaps_analysis'):
                self.evaluation_metrics['evidence_gap_analyses_generated'] += 1
            
            # Update quality detection metrics
            if evidence_score >= self.high_quality_threshold:
                self.evaluation_metrics['high_quality_evidence_found'] += 1
            elif evidence_score <= self.poor_quality_threshold:
                self.evaluation_metrics['poor_quality_evidence_found'] += 1
            
            # Format output for LangGraph with config context
            return self.format_output(
                result=evaluation_result,
                confidence=evidence_score / 10.0,  # Higher evidence score = higher confidence
                metadata={
                    'response_time': evaluation_result['metadata']['response_time_seconds'],
                    'model_used': self.model_name,
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular',
                    'detailed_analysis_triggered': force_detailed,
                    'evidence_threshold_used': self.evidence_threshold,
                    'scoring_weights_used': self.scoring_weights
                }
            )
            
        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            return self.format_error_output(e, input_data)

    def evaluate_evidence(self,
                         article_text: str,
                         extracted_claims: List[Dict[str, Any]],
                         context_analysis: Dict[str, Any],
                         include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        📊 MAIN EVIDENCE EVALUATION WITH CONFIG INTEGRATION
        
        Comprehensive evidence evaluation using config-driven parameters and analysis criteria.
        
        Args:
            article_text: The news article text to evaluate
            extracted_claims: Claims from claim extractor agent
            context_analysis: Results from context analyzer agent
            include_detailed_analysis: Enable detailed forensic evaluation
            
        Returns:
            Dict containing comprehensive evidence evaluation results
        """
        
        self._respect_rate_limits()
        start_time = time.time()
        
        try:
            self.logger.info("Starting evidence evaluation with config integration...")
            
            # Step 1: Clean article text
            article_text = sanitize_text(article_text)
            
            # ✅ USE CONFIG FOR TEXT LENGTH LIMITS
            max_text_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_text_length:
                article_text = article_text[:max_text_length] + "..."
            
            # Step 2: Prepare evidence context from previous analyses
            evidence_context = self._prepare_evidence_context(extracted_claims, context_analysis)
            
            # Step 3: Run systematic evidence quality assessment using modular components
            quality_assessment = self.quality_criteria.assess_evidence_quality(
                article_text, extracted_claims
            )
            
            # Step 4: Generate AI-powered comprehensive evidence analysis using config prompts
            evidence_analysis = self._generate_evidence_analysis(
                article_text, extracted_claims, context_analysis
            )
            
            # Step 5: Generate source quality analysis
            source_quality_analysis = self._generate_source_quality_analysis(
                article_text, extracted_claims
            )
            
            # Step 6: Generate logical consistency analysis
            logical_consistency_analysis = self._generate_logical_consistency_analysis(
                article_text, extracted_claims
            )
            
            # Step 7: Optional evidence gaps analysis based on config
            evidence_gaps_analysis = None
            if (self.enable_gap_analysis and
                (include_detailed_analysis or
                 quality_assessment['overall_quality_score'] < self.evidence_threshold)):
                evidence_gaps_analysis = self._generate_evidence_gaps_analysis(
                    article_text, extracted_claims
                )
                self.evaluation_metrics['evidence_gap_analyses_generated'] += 1
                self.logger.info("🔍 Evidence gaps analysis generated due to quality concerns")
            
            # Step 8: Run logical fallacy detection if enabled
            fallacy_report = {}
            if self.enable_fallacy_detection:
                fallacy_report = self.fallacy_detector.detect_fallacies(article_text)
                self.evaluation_metrics['fallacies_detected'] += len(fallacy_report.get('detected_fallacies', []))
            
            # Step 9: Parse verification links from evidence analysis
            verification_links = self._parse_evidence_links(evidence_analysis)
            if verification_links:
                self.evaluation_metrics['verification_links_parsed'] += len(verification_links)
            
            # Step 10: Calculate comprehensive evidence scores using config weights
            evidence_scores = self._calculate_evidence_scores(
                quality_assessment, evidence_analysis, source_quality_analysis,
                logical_consistency_analysis, fallacy_report
            )
            
            # Step 11: Package results with config metadata
            response_time = time.time() - start_time
            result = {
                'evidence_analysis': evidence_analysis,
                'source_quality_analysis': source_quality_analysis,
                'logical_consistency_analysis': logical_consistency_analysis,
                'evidence_gaps_analysis': evidence_gaps_analysis,
                'quality_assessment': quality_assessment,
                'fallacy_report': fallacy_report,
                'evidence_scores': evidence_scores,
                'verification_links': verification_links,  # NEW: Add parsed verification links
                'evidence_summary': self._create_evidence_summary(
                    extracted_claims, quality_assessment, evidence_scores
                ),
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'response_time_seconds': round(response_time, 2),
                    'model_used': self.model_name,
                    'temperature_used': self.temperature,
                    'article_length_processed': len(article_text),
                    'claims_evaluated': len(extracted_claims),
                    'detailed_analysis_included': include_detailed_analysis,
                    'gaps_analysis_triggered': evidence_gaps_analysis is not None,
                    'fallacy_detection_enabled': self.enable_fallacy_detection,
                    'evidence_threshold': self.evidence_threshold,
                    'scoring_weights': self.scoring_weights,
                    'verification_links_found': len(verification_links),
                    'quality_thresholds': {
                        'high': self.high_quality_threshold,
                        'medium': self.medium_quality_threshold,
                        'poor': self.poor_quality_threshold
                    },
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular',
                    'criteria_evaluated': len(self.evidence_types),
                    'source_tiers_assessed': len(self.source_quality_tiers)
                }
            }
            
            # Step 12: Update performance metrics
            self._update_evaluation_metrics(response_time, evidence_scores['overall_evidence_score'])
            
            self.logger.info(f"Successfully completed evidence evaluation in {response_time:.2f} seconds")
            self.logger.info(f"📊 Overall evidence score: {evidence_scores['overall_evidence_score']:.1f}/10 ({evidence_scores['quality_level']})")
            self.logger.info(f"🔗 Verification links found: {len(verification_links)}")
            
            return result
            
        except Exception as e:
            self._update_evaluation_metrics(time.time() - start_time, 0, error=True)
            self.logger.error(f"Error in evidence evaluation: {str(e)}")
            raise

    def _parse_evidence_links(self, evidence_analysis: str) -> List[Dict[str, str]]:
        """Parse specific verification links from AI analysis"""
        links = []
        lines = evidence_analysis.split('\n')
        
        current_claim = ""
        current_url = ""
        current_explanation = ""
        
        for line in lines:
            if "**Specific Claim**:" in line:
                current_claim = line.split(":", 1)[1].strip()
            elif "**Verification URL**:" in line:
                current_url = line.split(":", 1)[1].strip()
            elif "**What It Proves**:" in line:
                current_explanation = line.split(":", 1)[1].strip()
                
                # Add complete link when we have all components
                if current_claim and current_url and current_explanation:
                    links.append({
                        "claim": current_claim,
                        "url": current_url if current_url.startswith('http') else f"https://{current_url}",
                        "explanation": current_explanation,
                        "type": "verification"
                    })
                    current_claim = current_url = current_explanation = ""
        
        return links[:5]  # Limit to top 5

    def _prepare_evidence_context(self, extracted_claims: List[Dict[str, Any]],
                                context_analysis: Dict[str, Any]) -> str:
        """Prepare context from previous agent analyses"""
        context_parts = []
        
        # Add claims context
        if extracted_claims:
            claims_text = "\n".join([
                f"- {claim.get('text', 'Unknown claim')}"
                for claim in extracted_claims[:5]  # Limit for brevity
            ])
            context_parts.append(f"Key Claims:\n{claims_text}")
        
        # Add context analysis summary
        if context_analysis:
            bias_score = context_analysis.get('overall_context_score', 0)
            risk_level = context_analysis.get('risk_level', 'Unknown')
            context_parts.append(f"Context Analysis: {bias_score}/10 bias score, {risk_level} risk")
        
        return "\n\n".join(context_parts) if context_parts else "No previous analysis available"

    def _generate_evidence_analysis(self, article_text: str, extracted_claims: List[Dict[str, Any]],
                                  context_analysis: Dict[str, Any]) -> str:
        """
        Generate AI-powered evidence analysis using config prompt template
        
        Args:
            article_text: Article content
            extracted_claims: Claims from extractor
            context_analysis: Context from analyzer
            
        Returns:
            Evidence analysis text
        """
        try:
            # Prepare source list from claims
            source_list = []
            for claim in extracted_claims[:10]:  # Limit sources
                source = claim.get('source', 'Not specified')
                if source != 'Not specified' and source not in source_list:
                    source_list.append(source)
            
            sources_text = "; ".join(source_list) if source_list else "No specific sources identified"
            
            # ✅ USE EVIDENCE PROMPT FROM CONFIG
            # Extract prediction and confidence from available data
            bert_results = context_analysis.get('bert_results', {})
            prediction = bert_results.get('prediction', 'UNKNOWN')
            confidence = bert_results.get('confidence', 0.0)
            
            prompt = self.evidence_prompt.format(
                article_text=article_text,
                extracted_claims=str(extracted_claims[:8]),  # Limit for prompt length
                prediction=prediction,
                confidence=confidence,
                source_recommendations=sources_text
            )
            
            response = self.model.generate_content(prompt)
            self.evaluation_metrics['gemini_api_calls'] += 1
            self.evaluation_metrics['evidence_evaluations_generated'] += 1
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in evidence analysis generation: {str(e)}")
            return f"Evidence analysis unavailable due to processing error: {str(e)}"

    def _generate_source_quality_analysis(self, article_text: str,
                                        extracted_claims: List[Dict[str, Any]]) -> str:
        """
        Generate AI-powered source quality analysis using config prompt template
        
        Args:
            article_text: Article content
            extracted_claims: Claims with source attributions
            
        Returns:
            Source quality analysis text
        """
        try:
            # Prepare source list for analysis
            source_list = []
            for claim in extracted_claims:
                source = claim.get('source', 'Not specified')
                verification_strategy = claim.get('verification_strategy', 'Standard fact-checking')
                if source != 'Not specified':
                    source_list.append(f"Source: {source} (Strategy: {verification_strategy})")
            
            sources_text = "\n".join(source_list[:10]) if source_list else "No specific sources identified in claims"
            
            # ✅ USE SOURCE QUALITY PROMPT FROM CONFIG
            prompt = self.source_quality_prompt.format(
                article_text=article_text,
                source_list=sources_text
            )
            
            response = self.model.generate_content(prompt)
            self.evaluation_metrics['gemini_api_calls'] += 1
            self.evaluation_metrics['source_quality_analyses_generated'] += 1
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in source quality analysis generation: {str(e)}")
            return f"Source quality analysis unavailable due to processing error: {str(e)}"

    def _generate_logical_consistency_analysis(self, article_text: str,
                                             extracted_claims: List[Dict[str, Any]]) -> str:
        """
        Generate AI-powered logical consistency analysis using config prompt template
        
        Args:
            article_text: Article content
            extracted_claims: Claims to analyze for logical consistency
            
        Returns:
            Logical consistency analysis text
        """
        try:
            # Prepare key claims for analysis
            key_claims = []
            for claim in extracted_claims[:8]:  # Limit for analysis depth
                claim_text = claim.get('text', 'Unknown claim')
                claim_type = claim.get('claim_type', 'Other')
                priority = claim.get('priority', 2)
                key_claims.append(f"Priority {priority} {claim_type}: {claim_text}")
            
            claims_text = "\n".join(key_claims) if key_claims else "No claims available for analysis"
            
            # ✅ USE LOGICAL CONSISTENCY PROMPT FROM CONFIG
            prompt = self.logical_consistency_prompt.format(
                article_text=article_text,
                key_claims=claims_text
            )
            
            response = self.model.generate_content(prompt)
            self.evaluation_metrics['gemini_api_calls'] += 1
            self.evaluation_metrics['logical_consistency_analyses_generated'] += 1
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in logical consistency analysis generation: {str(e)}")
            return f"Logical consistency analysis unavailable due to processing error: {str(e)}"

    def _generate_evidence_gaps_analysis(self, article_text: str,
                                       extracted_claims: List[Dict[str, Any]]) -> str:
        """
        Generate AI-powered evidence gaps analysis using config prompt template
        
        Args:
            article_text: Article content
            extracted_claims: Claims to analyze for evidence gaps
            
        Returns:
            Evidence gaps analysis text
        """
        try:
            # ✅ USE EVIDENCE GAPS PROMPT FROM CONFIG
            prompt = self.evidence_gaps_prompt.format(
                article_text=article_text,
                extracted_claims=str(extracted_claims[:6])  # Limit for focused analysis
            )
            
            response = self.model.generate_content(prompt)
            self.evaluation_metrics['gemini_api_calls'] += 1
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in evidence gaps analysis generation: {str(e)}")
            return f"Evidence gaps analysis unavailable due to processing error: {str(e)}"

    def _calculate_evidence_scores(self, quality_assessment: Dict, evidence_analysis: str,
                                 source_quality_analysis: str, logical_consistency_analysis: str,
                                 fallacy_report: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive evidence scores with config-aware weights
        
        Args:
            quality_assessment: Systematic quality assessment results
            evidence_analysis: AI evidence analysis
            source_quality_analysis: AI source analysis
            logical_consistency_analysis: AI logical analysis
            fallacy_report: Fallacy detection results
            
        Returns:
            Comprehensive evidence scoring
        """
        
        # 1. Source quality score from systematic assessment
        source_quality_score = quality_assessment.get('source_quality_score', 5.0)
        
        # 2. Evidence completeness score
        completeness_score = quality_assessment.get('completeness_score', 5.0)
        
        # 3. Logical consistency score estimation from AI analysis
        logical_score = self._estimate_logical_score(logical_consistency_analysis, fallacy_report)
        
        # 4. Evidence strength score from AI analysis
        evidence_strength_score = self._estimate_evidence_strength(evidence_analysis)
        
        # ✅ CALCULATE WEIGHTED OVERALL SCORE USING CONFIG WEIGHTS
        source_component = source_quality_score * self.scoring_weights['source_quality']
        logical_component = logical_score * self.scoring_weights['logical_consistency']
        completeness_component = completeness_score * self.scoring_weights['evidence_completeness']
        
        overall_score = source_component + logical_component + completeness_component
        
        # Quality level assessment with config thresholds
        if overall_score >= self.high_quality_threshold:
            quality_level = "HIGH QUALITY"
        elif overall_score >= self.medium_quality_threshold:
            quality_level = "MEDIUM QUALITY"
        elif overall_score >= self.poor_quality_threshold:
            quality_level = "POOR QUALITY"
        else:
            quality_level = "VERY POOR QUALITY"
        
        # Evidence reliability assessment
        reliability_factors = []
        if source_quality_score >= 7.0:
            reliability_factors.append("Strong source quality")
        if logical_score >= 7.0:
            reliability_factors.append("Logical consistency")
        if completeness_score >= 7.0:
            reliability_factors.append("Complete evidence")
        if len(fallacy_report.get('detected_fallacies', [])) == 0:
            reliability_factors.append("No logical fallacies")
        
        # Evidence concerns
        evidence_concerns = []
        if source_quality_score < 4.0:
            evidence_concerns.append(f"Poor source quality ({source_quality_score:.1f}/10)")
        if logical_score < 4.0:
            evidence_concerns.append(f"Logical inconsistencies ({logical_score:.1f}/10)")
        if completeness_score < 4.0:
            evidence_concerns.append(f"Incomplete evidence ({completeness_score:.1f}/10)")
        if len(fallacy_report.get('detected_fallacies', [])) > 2:
            evidence_concerns.append(f"Multiple logical fallacies ({len(fallacy_report.get('detected_fallacies', []))} found)")
        
        return {
            'source_quality_score': round(source_quality_score, 2),
            'logical_consistency_score': round(logical_score, 2),
            'evidence_completeness_score': round(completeness_score, 2),
            'evidence_strength_score': round(evidence_strength_score, 2),
            'overall_evidence_score': round(overall_score, 2),
            'quality_level': quality_level,
            'reliability_factors': reliability_factors,
            'evidence_concerns': evidence_concerns,
            'scoring_method': 'config_weighted',
            'weights_used': self.scoring_weights,
            'thresholds_used': {
                'high_quality': self.high_quality_threshold,
                'medium_quality': self.medium_quality_threshold,
                'poor_quality': self.poor_quality_threshold
            }
        }

    def _estimate_logical_score(self, logical_analysis: str, fallacy_report: Dict) -> float:
        """Estimate logical consistency score from AI analysis and fallacy detection"""
        
        # Base score from fallacy count
        fallacy_count = len(fallacy_report.get('detected_fallacies', []))
        fallacy_penalty = min(5.0, fallacy_count * 1.5)
        
        # AI analysis assessment
        logical_positive_indicators = [
            'logical', 'consistent', 'coherent', 'well-reasoned', 'sound argument',
            'valid reasoning', 'follows logically', 'evidence supports'
        ]
        
        logical_negative_indicators = [
            'inconsistent', 'contradictory', 'illogical', 'flawed reasoning',
            'weak argument', 'non sequitur', 'unsupported leap'
        ]
        
        analysis_lower = logical_analysis.lower()
        positive_count = sum(1 for indicator in logical_positive_indicators if indicator in analysis_lower)
        negative_count = sum(1 for indicator in logical_negative_indicators if indicator in analysis_lower)
        
        ai_score = 5.0 + (positive_count * 0.5) - (negative_count * 0.8)
        
        # Combined score
        combined_score = max(0, min(10, (ai_score - fallacy_penalty)))
        
        return combined_score

    def _estimate_evidence_strength(self, evidence_analysis: str) -> float:
        """Estimate evidence strength from AI analysis"""
        
        strength_indicators = [
            'strong evidence', 'compelling evidence', 'robust evidence',
            'well-documented', 'thoroughly supported', 'comprehensive data',
            'multiple sources', 'corroborating evidence', 'primary sources'
        ]
        
        weakness_indicators = [
            'weak evidence', 'limited evidence', 'insufficient evidence',
            'poorly supported', 'unsubstantiated', 'lack of evidence',
            'anecdotal', 'unreliable sources', 'missing documentation'
        ]
        
        analysis_lower = evidence_analysis.lower()
        strength_count = sum(1 for indicator in strength_indicators if indicator in analysis_lower)
        weakness_count = sum(1 for indicator in weakness_indicators if indicator in analysis_lower)
        
        score = 5.0 + (strength_count * 0.8) - (weakness_count * 0.9)
        
        return max(0, min(10, score))

    def _create_evidence_summary(self, extracted_claims: List[Dict[str, Any]],
                               quality_assessment: Dict, evidence_scores: Dict) -> str:
        """Create formatted evidence summary for other agents"""
        
        if not extracted_claims:
            return "No claims available for evidence evaluation."
        
        summary_lines = [
            f"EVIDENCE EVALUATION SUMMARY",
            f"Overall Evidence Score: {evidence_scores['overall_evidence_score']:.1f}/10 ({evidence_scores['quality_level']})",
            ""
        ]
        
        # Add quality component scores
        summary_lines.extend([
            f"Quality Breakdown:",
            f" • Source Quality: {evidence_scores['source_quality_score']:.1f}/10",
            f" • Logical Consistency: {evidence_scores['logical_consistency_score']:.1f}/10",
            f" • Evidence Completeness: {evidence_scores['evidence_completeness_score']:.1f}/10",
            ""
        ])
        
        # Add reliability factors
        if evidence_scores['reliability_factors']:
            summary_lines.append("Reliability Factors:")
            for factor in evidence_scores['reliability_factors']:
                summary_lines.append(f" ✓ {factor}")
            summary_lines.append("")
        
        # Add concerns
        if evidence_scores['evidence_concerns']:
            summary_lines.append("Evidence Concerns:")
            for concern in evidence_scores['evidence_concerns']:
                summary_lines.append(f" ⚠ {concern}")
            summary_lines.append("")
        
        # Add claims assessment
        high_verifiability = [c for c in extracted_claims if c.get('verifiability_score', 0) >= 7]
        summary_lines.extend([
            f"Claims Assessment:",
            f" • Total Claims: {len(extracted_claims)}",
            f" • High Verifiability: {len(high_verifiability)}",
            f" • Quality Assessment Score: {quality_assessment.get('overall_quality_score', 0):.1f}/10"
        ])
        
        return "\n".join(summary_lines)

    def _respect_rate_limits(self):
        """Rate limiting using config values"""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
        
        self.last_request_time = time.time()

    def _update_evaluation_metrics(self, response_time: float, evidence_score: float, error: bool = False):
        """Update evaluation-specific metrics with config awareness"""
        self.evaluation_metrics['total_evaluations'] += 1
        
        if not error:
            # Update average response time
            total = self.evaluation_metrics['total_evaluations']
            current_avg = self.evaluation_metrics['average_response_time']
            self.evaluation_metrics['average_response_time'] = (
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
            'evidence_threshold': self.evidence_threshold,
            'enable_detailed_analysis': self.enable_detailed_analysis,
            'enable_fallacy_detection': self.enable_fallacy_detection,
            'enable_gap_analysis': self.enable_gap_analysis,
            'evidence_types_count': len(self.evidence_types),
            'source_quality_tiers_count': len(self.source_quality_tiers),
            'scoring_weights': self.scoring_weights,
            'quality_thresholds': {
                'high': self.high_quality_threshold,
                'medium': self.medium_quality_threshold,
                'poor': self.poor_quality_threshold
            },
            'rate_limit_seconds': self.rate_limit,
            'config_version': '2.0_integrated'
        }
        
        # Get component metrics
        component_metrics = {
            'quality_criteria_stats': self.quality_criteria.get_criteria_statistics(),
            'fallacy_detector_stats': self.fallacy_detector.get_detector_statistics(),
            'api_calls_made': self.evaluation_metrics['gemini_api_calls']
        }
        
        return {
            **base_metrics,
            'evaluation_specific_metrics': self.evaluation_metrics,
            'config_metrics': config_metrics,
            'component_info': component_metrics,
            'agent_type': 'evidence_evaluator',
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
            'evidence_threshold': self.evidence_threshold,
            'enable_detailed_analysis': self.enable_detailed_analysis,
            'enable_fallacy_detection': self.enable_fallacy_detection,
            'enable_gap_analysis': self.enable_gap_analysis,
            'evidence_types': self.evidence_types,
            'source_quality_tiers': self.source_quality_tiers,
            'scoring_weights': self.scoring_weights,
            'fallacy_types_count': self.fallacy_types_count,
            'reasoning_quality_threshold': self.reasoning_quality_threshold,
            'logical_health_threshold': self.logical_health_threshold,
            'quality_thresholds': {
                'high': self.high_quality_threshold,
                'medium': self.medium_quality_threshold,
                'poor': self.poor_quality_threshold
            },
            'rate_limit_seconds': self.rate_limit,
            'max_retries': self.max_retries,
            'config_source': 'config_files',
            'prompt_source': 'centralized_prompts_config'
        }

# Testing functionality with config integration
if __name__ == "__main__":
    """Test the modular evidence evaluator agent with config integration"""
    print("🧪 Testing Modular Evidence Evaluator Agent with Config Integration")
    print("=" * 75)
    
    try:
        # Initialize agent (will load from config files)
        agent = EvidenceEvaluatorAgent()
        print(f"✅ Agent initialized with config: {agent}")
        
        # Show config summary
        config_summary = agent.get_config_summary()
        print(f"\n⚙️ Configuration Summary:")
        for key, value in config_summary.items():
            if isinstance(value, (list, dict)):
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {len(value)} entries" if value else f"  {key}: empty")
            else:
                print(f"  {key}: {value}")
        
        # Test evidence evaluation
        test_article = """
        According to a study published in the New England Journal of Medicine,
        researchers at Harvard University found that the new treatment showed
        85% effectiveness in clinical trials with 2,400 participants.
        Dr. Sarah Johnson, the lead researcher, stated that the results were
        statistically significant with a p-value of 0.001. However, the study
        was funded by the pharmaceutical company that developed the treatment.
        """
        
        test_claims = [
            {
                'text': 'Study published in New England Journal of Medicine showed 85% effectiveness.',
                'claim_type': 'Research',
                'priority': 1,
                'verifiability_score': 8,
                'source': 'Harvard University researchers'
            },
            {
                'text': 'Clinical trial included 2,400 participants with p-value of 0.001.',
                'claim_type': 'Statistical',
                'priority': 1,
                'verifiability_score': 9,
                'source': 'Clinical trial data'
            },
            {
                'text': 'Study was funded by pharmaceutical company that developed treatment.',
                'claim_type': 'Attribution',
                'priority': 2,
                'verifiability_score': 6,
                'source': 'Study disclosure'
            }
        ]
        
        test_context = {
            'overall_context_score': 4.2,
            'risk_level': 'MEDIUM',
            'bias_counts': {'commercial_bias': 2}
        }
        
        test_input = {
            "text": test_article,
            "extracted_claims": test_claims,
            "context_analysis": test_context,
            "include_detailed_analysis": True
        }
        
        print(f"\n🔍 Testing evidence evaluation...")
        print(f"Article preview: {test_article[:100]}...")
        print(f"Claims to evaluate: {len(test_claims)}")
        print(f"Context score: {test_context['overall_context_score']:.1f}/10")
        
        result = agent.process(test_input)
        
        if result['success']:
            evaluation_data = result['result']
            print(f"✅ Evaluation completed successfully")
            print(f"  Overall evidence score: {evaluation_data['evidence_scores']['overall_evidence_score']:.1f}/10")
            print(f"  Quality level: {evaluation_data['evidence_scores']['quality_level']}")
            print(f"  Source quality: {evaluation_data['evidence_scores']['source_quality_score']:.1f}/10")
            print(f"  Logical consistency: {evaluation_data['evidence_scores']['logical_consistency_score']:.1f}/10")
            print(f"  Evidence completeness: {evaluation_data['evidence_scores']['evidence_completeness_score']:.1f}/10")
            print(f"  Response time: {evaluation_data['metadata']['response_time_seconds']}s")
            print(f"  Config version: {evaluation_data['metadata']['config_version']}")
            print(f"  Verification links: {evaluation_data['metadata']['verification_links_found']}")
            
            # Show analysis types generated
            analyses_generated = []
            if evaluation_data.get('evidence_analysis'):
                analyses_generated.append('Evidence Analysis')
            if evaluation_data.get('source_quality_analysis'):
                analyses_generated.append('Source Quality')
            if evaluation_data.get('logical_consistency_analysis'):
                analyses_generated.append('Logical Consistency')
            if evaluation_data.get('evidence_gaps_analysis'):
                analyses_generated.append('Evidence Gaps')
            
            print(f"  Analyses generated: {', '.join(analyses_generated)}")
            
            # Show reliability factors and concerns
            reliability_factors = evaluation_data['evidence_scores']['reliability_factors']
            evidence_concerns = evaluation_data['evidence_scores']['evidence_concerns']
            
            print(f"  Reliability factors: {len(reliability_factors)}")
            print(f"  Evidence concerns: {len(evidence_concerns)}")
            
        else:
            print(f"❌ Evaluation failed: {result['error']['message']}")
        
        # Show comprehensive metrics with config info
        print(f"\n📊 Comprehensive metrics with config info:")
        metrics = agent.get_comprehensive_metrics()
        print(f"Agent type: {metrics['agent_type']}")
        print(f"Config integrated: {metrics['config_integrated']}")
        print(f"Prompt source: {metrics['prompt_source']}")
        print(f"High quality evidence found: {metrics['evaluation_specific_metrics']['high_quality_evidence_found']}")
        print(f"Poor quality evidence found: {metrics['evaluation_specific_metrics']['poor_quality_evidence_found']}")
        print(f"Verification links parsed: {metrics['evaluation_specific_metrics']['verification_links_parsed']}")
        
        print(f"\n✅ Modular evidence evaluator agent with config integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure your GEMINI_API_KEY is set in your environment variables")
        import traceback
        traceback.print_exc()
