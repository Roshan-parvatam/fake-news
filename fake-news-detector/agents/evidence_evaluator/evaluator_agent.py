# agents/evidence_evaluator/evaluator_agent.py

"""
Enhanced Evidence Evaluator Agent with LLM-Powered Verification

This agent evaluates evidence quality and reliability using intelligent LLMs 
to directly provide verification links based on their credibility analysis.

Features:
- ✅ LLM DIRECTLY PROVIDES VERIFICATION LINKS
- ✅ NO WEB SCRAPING COMPLEXITY
- ✅ LEVERAGES LLM'S EXISTING KNOWLEDGE FOR CREDIBILITY
- ✅ SAFETY FILTER HANDLING
- ✅ BUG FIXES FOR KeyError CRASHES
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
from .criteria import EvidenceQualityCriteria
from .fallacy_detection import LogicalFallacyDetector

# ✅ IMPORT CONFIGURATION FILES
from config import get_model_config, get_prompt_template, get_settings
from utils.helpers import sanitize_text

class EvidenceEvaluatorAgent(BaseAgent):
    """
    📊 INTELLIGENT EVIDENCE EVALUATOR AGENT WITH LLM-POWERED VERIFICATION
    
    ✅ KEY FEATURES:
    - LLM directly provides verification sources based on credibility analysis
    - No external web scraping dependencies
    - Uses the same "mental model" as credibility scoring
    - Handles safety filters gracefully
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intelligent evidence evaluator agent"""
        
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
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 3072)
        
        # ✅ EVALUATION SETTINGS FROM CONFIG
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.evidence_threshold = self.config.get('evidence_threshold', 6.0)
        self.enable_fallacy_detection = self.config.get('enable_fallacy_detection', True)
        self.enable_gap_analysis = self.config.get('enable_gap_analysis', True)
        
        # ✅ NEW: LLM VERIFICATION SETTINGS (Simplified)
        self.enable_llm_verification = self.config.get('enable_llm_verification', True)
        self.max_verification_links = self.config.get('max_verification_links', 6)
        self.link_quality_threshold = self.config.get('link_quality_threshold', 0.6)
        
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
            'source_quality': 0.35,
            'logical_consistency': 0.3,
            'evidence_completeness': 0.25,
            'verification_links_quality': 0.1
        })
        
        # ✅ QUALITY THRESHOLDS FROM CONFIG
        self.high_quality_threshold = self.config.get('high_quality_threshold', 7.0)
        self.medium_quality_threshold = self.config.get('medium_quality_threshold', 5.0)
        self.poor_quality_threshold = self.config.get('poor_quality_threshold', 3.0)
        
        # ✅ GET API KEY FROM SYSTEM SETTINGS
        self.api_key = system_settings.gemini_api_key
        
        # ✅ LOAD PROMPTS FROM CONFIG
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
        
        # ✅ SIMPLIFIED PERFORMANCE TRACKING
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
            
            # ✅ LLM VERIFICATION METRICS
            'llm_verification_attempts': 0,
            'verification_links_generated': 0,
            'high_quality_links_found': 0,
            'llm_verification_success_rate': 0.0
        }
        
        # Rate limiting tracking
        self.last_request_time = None
        
        self.logger.info(f"✅ LLM-Powered Evidence Evaluator Agent initialized")
        self.logger.info(f"🤖 Model: {self.model_name}, Temperature: {self.temperature}")
        self.logger.info(f"🧠 LLM Verification: {'Enabled' if self.enable_llm_verification else 'Disabled'}")

    def _initialize_gemini_api(self):
        """🔐 INITIALIZE GEMINI API WITH ENHANCED SAFETY SETTINGS"""
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
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ])
            
            # Create model instance
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info("🔐 Gemini API initialized for LLM-powered evidence evaluation")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Gemini API: {str(e)}")
            raise

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """🎯 MAIN PROCESSING METHOD WITH LLM VERIFICATION"""
        
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
                context_score > 7.0 or
                len(extracted_claims) < 2 or
                self.enable_detailed_analysis
            )
            
            # ✅ PERFORM LLM-POWERED EVIDENCE EVALUATION
            evaluation_result = self.evaluate_evidence_with_llm_verification(
                article_text=article_text,
                extracted_claims=extracted_claims,
                context_analysis=context_analysis,
                include_detailed_analysis=force_detailed
            )
            
            # ✅ SAFE EXTRACTION OF EVIDENCE SCORE
            evidence_scores = evaluation_result.get('evidence_scores', {})
            evidence_score = evidence_scores.get('overall_evidence_score', 5.0)
            
            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(evidence_score / 10.0)
            self.evaluation_metrics['successful_evaluations'] += 1
            
            # ✅ UPDATE LLM VERIFICATION METRICS
            verification_links = evaluation_result.get('verification_links', [])
            self.evaluation_metrics['verification_links_generated'] += len(verification_links)
            self.evaluation_metrics['high_quality_links_found'] += len([
                l for l in verification_links if l.get('quality_score', 0) >= 0.8
            ])
            
            # Format output for LangGraph
            return self.format_output(
                result=evaluation_result,
                confidence=evidence_score / 10.0,
                metadata={
                    'response_time': evaluation_result['metadata']['response_time_seconds'],
                    'model_used': self.model_name,
                    'config_version': '5.0_llm_direct',
                    'agent_version': '5.0_llm_powered',
                    'detailed_analysis_triggered': force_detailed,
                    'evidence_threshold_used': self.evidence_threshold,
                    'scoring_weights_used': self.scoring_weights,
                    'verification_links_generated': len(verification_links),
                    'llm_verification_enabled': self.enable_llm_verification
                }
            )
            
        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            self.logger.error(f"❌ Evidence evaluation error: {str(e)}")
            return self.format_error_output(e, input_data)

    def evaluate_evidence_with_llm_verification(self,
                                               article_text: str,
                                               extracted_claims: List[Dict[str, Any]],
                                               context_analysis: Dict[str, Any],
                                               include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        📊 MAIN EVIDENCE EVALUATION WITH LLM-POWERED VERIFICATION
        
        ✅ KEY FEATURES:
        - LLM directly provides verification sources
        - No web scraping complexity
        - Leverages LLM's credibility assessment knowledge
        """
        
        self._respect_rate_limits()
        start_time = time.time()
        
        try:
            self.logger.info("Starting LLM-powered evidence evaluation...")
            
            # Step 1: Clean article text
            article_text = sanitize_text(article_text)
            max_text_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_text_length:
                article_text = article_text[:max_text_length] + "..."
            
            # Step 2: Run systematic evidence quality assessment
            quality_assessment = self.quality_criteria.assess_evidence_quality(
                article_text, extracted_claims
            )
            
            # ✅ STEP 3: LLM-POWERED VERIFICATION LINK GENERATION
            if self.enable_llm_verification:
                verification_analysis = self._generate_llm_verification_links(
                    article_text, extracted_claims
                )
            else:
                verification_analysis = self._generate_basic_verification_fallback(extracted_claims)
            
            # Step 4: Generate analyses with safety handling
            source_quality_analysis = self._generate_source_quality_analysis_safe(
                article_text, extracted_claims
            )
            
            logical_consistency_analysis = self._generate_logical_consistency_analysis_safe(
                article_text, extracted_claims
            )
            
            # Step 5: Optional evidence gaps analysis
            evidence_gaps_analysis = None
            if (self.enable_gap_analysis and
                (include_detailed_analysis or
                 quality_assessment.get('overall_quality_score', 5.0) < self.evidence_threshold)):
                
                evidence_gaps_analysis = self._generate_evidence_gaps_analysis_safe(
                    article_text, extracted_claims
                )
                self.evaluation_metrics['evidence_gap_analyses_generated'] += 1
            
            # Step 6: Run logical fallacy detection if enabled
            fallacy_report = {}
            if self.enable_fallacy_detection:
                try:
                    fallacy_report = self.fallacy_detector.detect_fallacies(article_text)
                    self.evaluation_metrics['fallacies_detected'] += len(fallacy_report.get('detected_fallacies', []))
                except Exception as e:
                    self.logger.warning(f"Fallacy detection failed: {str(e)}")
                    fallacy_report = {'detected_fallacies': []}
            
            # ✅ STEP 7: CALCULATE EVIDENCE SCORES WITH LLM LINK QUALITY
            evidence_scores = self._calculate_evidence_scores_with_llm_links(
                quality_assessment, verification_analysis, source_quality_analysis,
                logical_consistency_analysis, fallacy_report
            )
            
            # Step 8: Package results
            response_time = time.time() - start_time
            
            result = {
                'evidence_analysis': verification_analysis.get('analysis_text', 'Analysis not available'),
                'verification_links': verification_analysis.get('verification_links', []),
                'source_quality_analysis': source_quality_analysis,
                'logical_consistency_analysis': logical_consistency_analysis,
                'evidence_gaps_analysis': evidence_gaps_analysis,
                'quality_assessment': quality_assessment,
                'fallacy_report': fallacy_report,
                'evidence_scores': evidence_scores,
                'evidence_summary': self._create_evidence_summary_with_llm_links(
                    extracted_claims, quality_assessment, evidence_scores, verification_analysis
                ),
                'llm_verification_used': verification_analysis.get('llm_verification_used', False),
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
                    
                    # ✅ LLM VERIFICATION METADATA
                    'verification_links_found': len(verification_analysis.get('verification_links', [])),
                    'high_quality_links_count': verification_analysis.get('high_quality_links_count', 0),
                    'llm_verification_enabled': self.enable_llm_verification,
                    'config_version': '5.0_llm_direct',
                    'agent_version': '5.0_llm_powered'
                }
            }
            
            # Step 9: Update performance metrics
            self._update_evaluation_metrics(response_time, evidence_scores.get('overall_evidence_score', 5.0))
            
            link_count = len(verification_analysis.get('verification_links', []))
            high_quality_count = verification_analysis.get('high_quality_links_count', 0)
            
            self.logger.info(f"Successfully completed LLM-powered evidence evaluation in {response_time:.2f} seconds")
            self.logger.info(f"📊 Overall evidence score: {evidence_scores.get('overall_evidence_score', 5.0):.1f}/10 ({evidence_scores.get('quality_level', 'UNKNOWN')})")
            self.logger.info(f"🔗 Verification links: {link_count} total ({high_quality_count} high-quality)")
            self.logger.info(f"🧠 LLM verification used: {verification_analysis.get('llm_verification_used', False)}")
            
            return result
            
        except Exception as e:
            self._update_evaluation_metrics(time.time() - start_time, 0, error=True)
            self.logger.error(f"Error in LLM-powered evidence evaluation: {str(e)}")
            raise

    def _generate_llm_verification_links(self, article_text: str, extracted_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ✅ SIMPLE: Ask LLM directly for verification links it would use
        Since LLM is already scoring credibility, it knows what sources to check
        """
        try:
            self.logger.info("🧠 Asking LLM directly for verification sources...")
            
            # Prepare claims summary
            claims_text = []
            for i, claim in enumerate(extracted_claims, 1):
                claims_text.append(f"{i}. {claim.get('text', '')[:150]}")
            
            # ✅ DIRECT PROMPT: Ask LLM for verification links
            prompt = f"""
            You are analyzing this article for credibility and need to provide specific verification sources.

            ARTICLE: {article_text[:1000]}

            KEY CLAIMS TO VERIFY:
            {chr(10).join(claims_text)}

            Since you're evaluating this content's credibility, provide the SPECIFIC SOURCES you would check to verify these claims. 

            For each verification source, provide in this EXACT format:

            VERIFICATION SOURCE 1:
            CLAIM: [specific claim this verifies]
            INSTITUTION: [name of authoritative institution/database]
            URL: [specific URL to check this claim - be as specific as possible]
            WHY_RELIABLE: [why this source is trustworthy for this claim]
            VERIFICATION_TYPE: [primary_source/expert_opinion/database/research_paper]

            VERIFICATION SOURCE 2:
            [continue for 3-5 sources...]

            Requirements:
            - Provide REAL, SPECIFIC URLs that actually exist
            - Focus on authoritative sources (government, academic, medical institutions)
            - Match each source to a specific claim
            - Prioritize primary sources over secondary reporting
            - Include the exact page/section that would verify the claim

            The sources you recommend should be the same ones you'd mentally "check" when assessing this content's credibility.
            """
            
            self.evaluation_metrics['llm_verification_attempts'] += 1
            response = self.model.generate_content(prompt)
            
            if self._is_valid_llm_response(response):
                analysis_text = response.candidates[0].content.parts[0].text
                verification_links = self._parse_llm_verification_sources(analysis_text)
                
                high_quality_count = len([l for l in verification_links if l.get('quality_score', 0) >= 0.8])
                
                return {
                    'analysis_text': f"LLM-recommended verification sources based on credibility analysis:\n\n{analysis_text}",
                    'verification_links': verification_links,
                    'high_quality_links_count': high_quality_count,
                    'total_links_generated': len(verification_links),
                    'link_generation_success': len(verification_links) > 0,
                    'llm_verification_used': True
                }
            else:
                return self._generate_basic_verification_fallback(extracted_claims)
                
        except Exception as e:
            self.logger.error(f"LLM verification failed: {str(e)}")
            return self._generate_basic_verification_fallback(extracted_claims)

    def _parse_llm_verification_sources(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Parse LLM-provided verification sources"""
        
        verification_links = []
        
        # Split by verification source sections
        sections = re.split(r'VERIFICATION SOURCE \d+:', analysis_text)
        
        for section in sections[1:]:  # Skip first empty section
            try:
                # Extract fields using regex
                claim_match = re.search(r'CLAIM:\s*(.+?)(?=\n|INSTITUTION)', section, re.IGNORECASE)
                institution_match = re.search(r'INSTITUTION:\s*(.+?)(?=\n|URL)', section, re.IGNORECASE)
                url_match = re.search(r'URL:\s*(.+?)(?=\n|WHY_RELIABLE)', section, re.IGNORECASE)
                why_reliable_match = re.search(r'WHY_RELIABLE:\s*(.+?)(?=\n|VERIFICATION_TYPE)', section, re.IGNORECASE)
                verification_type_match = re.search(r'VERIFICATION_TYPE:\s*(.+?)(?=\n|$)', section, re.IGNORECASE)
                
                if claim_match and institution_match and url_match:
                    claim = claim_match.group(1).strip()
                    institution = institution_match.group(1).strip()
                    url = url_match.group(1).strip()
                    explanation = why_reliable_match.group(1).strip() if why_reliable_match else 'LLM-recommended source'
                    source_type = verification_type_match.group(1).strip() if verification_type_match else 'unknown'
                    
                    # Clean up URL
                    if not url.startswith(('http://', 'https://')):
                        if url.startswith('www.'):
                            url = f"https://{url}"
                        elif '.' in url and not url.startswith('/'):
                            url = f"https://{url}"
                    
                    # Calculate quality score based on source type and institution
                    quality_score = self._calculate_llm_source_quality(institution, source_type, url)
                    
                    verification_links.append({
                        'claim': claim,
                        'url': url,
                        'institution': institution,
                        'explanation': explanation,
                        'quality_score': quality_score,
                        'type': 'llm_recommended',
                        'source_type': self._map_verification_type_to_category(source_type)
                    })
                    
            except Exception as e:
                self.logger.debug(f"Failed to parse verification section: {str(e)}")
                continue
        
        return verification_links[:self.max_verification_links]

    def _calculate_llm_source_quality(self, institution: str, source_type: str, url: str) -> float:
        """Calculate quality score for LLM-recommended source"""
        
        score = 0.6  # Base score for LLM recommendation
        
        institution_lower = institution.lower()
        url_lower = url.lower()
        source_type_lower = source_type.lower()
        
        # High-quality institutions
        if any(inst in institution_lower for inst in [
            'cdc', 'who', 'nih', 'nejm', 'harvard', 'stanford', 'mit',
            'pubmed', 'nature', 'science', 'fda', 'government'
        ]):
            score += 0.3
        
        # High-quality domains
        if any(domain in url_lower for domain in [
            '.gov', '.edu', 'nejm.org', 'nature.com', 'pubmed', 'who.int'
        ]):
            score += 0.2
        
        # Source type bonus
        if source_type_lower in ['primary_source', 'research_paper']:
            score += 0.15
        elif source_type_lower in ['expert_opinion', 'database']:
            score += 0.1
        
        return min(1.0, max(0.3, score))

    def _map_verification_type_to_category(self, verification_type: str) -> str:
        """Map LLM verification type to standard category"""
        
        type_mapping = {
            'primary_source': 'primary',
            'expert_opinion': 'expert', 
            'database': 'database',
            'research_paper': 'academic',
            'government': 'government',
            'institutional': 'institutional'
        }
        
        return type_mapping.get(verification_type.lower(), 'general')

    def _generate_basic_verification_fallback(self, extracted_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Basic verification fallback when LLM approach fails"""
        
        fallback_links = []
        
        for claim in extracted_claims[:self.max_verification_links]:
            claim_text = claim.get('text', 'Unknown claim')
            
            fallback_links.append({
                'claim': claim_text,
                'url': 'https://www.snopes.com/',
                'institution': 'Snopes',
                'explanation': 'General fact-checking resource',
                'quality_score': 0.6,
                'type': 'fallback',
                'source_type': 'fact_checker'
            })
        
        return {
            'analysis_text': 'Basic verification sources provided as fallback.',
            'verification_links': fallback_links,
            'high_quality_links_count': 0,
            'total_links_generated': len(fallback_links),
            'link_generation_success': len(fallback_links) > 0,
            'llm_verification_used': False
        }

    def _is_valid_llm_response(self, response) -> bool:
        """Check if LLM response is valid and not blocked by safety filters"""
        return (response and 
                response.candidates and 
                len(response.candidates) > 0 and
                response.candidates[0].finish_reason != 2 and
                response.candidates[0].content and
                response.candidates[0].content.parts)

    # ✅ KEEP EXISTING METHODS FOR COMPATIBILITY

    def _generate_source_quality_analysis_safe(self, article_text: str, extracted_claims: List[Dict[str, Any]]) -> str:
        """Generate source quality analysis with safety handling"""
        try:
            source_list = []
            for claim in extracted_claims:
                source = claim.get('source', 'Not specified')
                if source != 'Not specified':
                    source_list.append(f"Source: {source}")
            
            sources_text = "\n".join(source_list[:10]) if source_list else "No specific sources identified"
            
            prompt = f"""
            Please analyze source quality for this content:
            
            CONTENT: {article_text[:800]}
            
            SOURCES: {sources_text}
            
            Provide analysis on:
            1. Source credibility assessment
            2. Source diversity  
            3. Verification potential
            4. Overall quality rating (1-10)
            """
            
            response = self.model.generate_content(prompt)
            
            if not response.candidates or response.candidates[0].finish_reason == 2:
                return self._generate_safe_source_analysis_fallback(len(source_list))
            if not response.candidates[0].content or not response.candidates[0].content.parts:
                return self._generate_safe_source_analysis_fallback(len(source_list))
            
            self.evaluation_metrics['gemini_api_calls'] += 1
            return response.candidates[0].content.parts[0].text
            
        except Exception as e:
            self.logger.error(f"Error in safe source quality analysis: {str(e)}")
            return self._generate_safe_source_analysis_fallback(len(extracted_claims))

    def _generate_safe_source_analysis_fallback(self, sources_count: int) -> str:
        """Generate safe fallback source analysis"""
        return f"""
        SOURCE QUALITY ANALYSIS (AUTOMATED ASSESSMENT)
        
        Sources Identified: {sources_count}
        
        Quality Assessment:
        • Source verification required through institutional channels
        • Cross-referencing with established authorities recommended  
        • Independent verification strongly advised
        
        Overall Source Quality: Requires human verification
        
        Recommended Approach:
        1. Verify sources through official channels
        2. Check source credentials and expertise
        3. Cross-reference with multiple independent sources
        4. Look for peer review or editorial oversight
        """

    def _generate_logical_consistency_analysis_safe(self, article_text: str, extracted_claims: List[Dict[str, Any]]) -> str:
        """Generate logical consistency analysis with safety handling"""
        try:
            key_claims = []
            for claim in extracted_claims[:6]:
                claim_text = claim.get('text', 'Unknown claim')[:100]
                claim_type = claim.get('claim_type', 'Other')
                key_claims.append(f"{claim_type}: {claim_text}")
            
            claims_text = "\n".join(key_claims) if key_claims else "No claims available"
            
            prompt = f"""
            Please analyze logical consistency:
            
            CLAIMS: {claims_text}
            
            CONTENT: {article_text[:800]}
            
            Assess:
            1. Internal logical consistency
            2. Evidence support for claims
            3. Reasoning quality
            4. Consistency rating (1-10)
            """
            
            response = self.model.generate_content(prompt)
            
            if not response.candidates or response.candidates[0].finish_reason == 2:
                return self._generate_safe_logical_analysis_fallback(len(key_claims))
            if not response.candidates[0].content or not response.candidates[0].content.parts:
                return self._generate_safe_logical_analysis_fallback(len(key_claims))
            
            self.evaluation_metrics['gemini_api_calls'] += 1
            return response.candidates[0].content.parts[0].text
            
        except Exception as e:
            self.logger.error(f"Error in safe logical consistency analysis: {str(e)}")
            return self._generate_safe_logical_analysis_fallback(len(extracted_claims))

    def _generate_safe_logical_analysis_fallback(self, claims_count: int) -> str:
        """Generate safe fallback logical analysis"""
        return f"""
        LOGICAL CONSISTENCY ANALYSIS (AUTOMATED ASSESSMENT)
        
        Claims Analyzed: {claims_count}
        
        Consistency Assessment:
        • Logical structure requires human review
        • Evidence support needs verification
        • Reasoning quality requires expert analysis
        
        Overall Logical Consistency: Requires human assessment
        
        Recommended Review:
        1. Check claim-to-claim consistency
        2. Verify evidence-to-conclusion logic
        3. Look for logical fallacies
        4. Assess reasoning quality
        """

    def _generate_evidence_gaps_analysis_safe(self, article_text: str, extracted_claims: List[Dict[str, Any]]) -> str:
        """Generate evidence gaps analysis with safety handling"""
        try:
            prompt = f"""
            Identify evidence gaps in this content:
            
            CONTENT: {article_text[:1000]}
            
            CLAIMS: {len(extracted_claims)} claims identified
            
            Please identify:
            1. Missing supporting evidence
            2. Unverified assertions
            3. Evidence quality gaps
            4. Verification needs
            """
            
            response = self.model.generate_content(prompt)
            
            if not response.candidates or response.candidates[0].finish_reason == 2:
                return "Evidence gaps analysis not available due to content restrictions."
            if not response.candidates[0].content or not response.candidates[0].content.parts:
                return "Evidence gaps analysis not available due to content restrictions."
            
            self.evaluation_metrics['gemini_api_calls'] += 1
            return response.candidates[0].content.parts[0].text
            
        except Exception as e:
            self.logger.error(f"Error in safe evidence gaps analysis: {str(e)}")
            return "Evidence gaps analysis unavailable."

    def _calculate_evidence_scores_with_llm_links(self, quality_assessment: Dict,
                                                 verification_analysis: Dict,
                                                 source_quality_analysis: str,
                                                 logical_consistency_analysis: str,
                                                 fallacy_report: Dict) -> Dict[str, Any]:
        """Calculate scores with LLM link quality awareness"""
        
        # ✅ SAFE EXTRACTION OF BASE SCORES WITH DEFAULTS
        source_quality_score = quality_assessment.get('source_quality_score', 5.0)
        completeness_score = quality_assessment.get('completeness_score', 5.0)
        
        # ✅ LLM LINK QUALITY CALCULATION
        verification_links = verification_analysis.get('verification_links', [])
        high_quality_count = verification_analysis.get('high_quality_links_count', 0)
        
        link_quality_score = 3.0  # Base score
        
        if len(verification_links) >= 3:
            link_quality_score += 2.0  # Bonus for multiple links
        if high_quality_count >= 2:
            link_quality_score += 2.5  # Bonus for high-quality links
        if verification_analysis.get('llm_verification_used', False):
            link_quality_score += 1.5  # Bonus for LLM verification
        if len(verification_links) == 0:
            link_quality_score -= 1.5  # Penalty for no links
        
        # Average quality score of individual links
        if verification_links:
            try:
                avg_link_quality = sum(link.get('quality_score', 0.5) for link in verification_links) / len(verification_links)
                link_quality_score = (link_quality_score + avg_link_quality * 7) / 2
            except:
                pass
        
        link_quality_score = max(0, min(10, link_quality_score))
        
        # Logical consistency score
        logical_score = 7.0 - len(fallacy_report.get('detected_fallacies', []))
        logical_score = max(1.0, min(10.0, logical_score))
        
        # ✅ SAFE CALCULATION OF OVERALL SCORE
        try:
            overall_score = (
                (source_quality_score * self.scoring_weights.get('source_quality', 0.35)) +
                (logical_score * self.scoring_weights.get('logical_consistency', 0.3)) +
                (completeness_score * self.scoring_weights.get('evidence_completeness', 0.25)) +
                (link_quality_score * self.scoring_weights.get('verification_links_quality', 0.1))
            )
        except Exception as e:
            self.logger.warning(f"Error in score calculation, using fallback: {str(e)}")
            overall_score = (source_quality_score + logical_score + completeness_score + link_quality_score) / 4
        
        overall_score = max(0, min(10, overall_score))
        
        # Quality level assessment
        if overall_score >= self.high_quality_threshold:
            quality_level = "HIGH QUALITY"
        elif overall_score >= self.medium_quality_threshold:
            quality_level = "MEDIUM QUALITY"
        elif overall_score >= self.poor_quality_threshold:
            quality_level = "POOR QUALITY"
        else:
            quality_level = "VERY POOR QUALITY"
        
        # Enhanced reliability factors
        reliability_factors = []
        if source_quality_score >= 7.0:
            reliability_factors.append("Strong source quality")
        if logical_score >= 7.0:
            reliability_factors.append("Logical consistency")
        if completeness_score >= 7.0:
            reliability_factors.append("Complete evidence")
        if link_quality_score >= 7.0:
            reliability_factors.append("High-quality verification links")
        if verification_analysis.get('llm_verification_used', False):
            reliability_factors.append("LLM verification analysis")
        if len(verification_links) > 0:
            reliability_factors.append("Verification sources available")
        
        # Enhanced evidence concerns
        evidence_concerns = []
        if source_quality_score < 4.0:
            evidence_concerns.append(f"Poor source quality ({source_quality_score:.1f}/10)")
        if logical_score < 4.0:
            evidence_concerns.append(f"Logical inconsistencies ({logical_score:.1f}/10)")
        if completeness_score < 4.0:
            evidence_concerns.append(f"Incomplete evidence ({completeness_score:.1f}/10)")
        if link_quality_score < 4.0:
            evidence_concerns.append(f"Poor verification link quality ({link_quality_score:.1f}/10)")
        if len(verification_links) == 0:
            evidence_concerns.append("No verification sources available")
        
        return {
            'source_quality_score': round(source_quality_score, 2),
            'logical_consistency_score': round(logical_score, 2),
            'evidence_completeness_score': round(completeness_score, 2),
            'verification_links_quality_score': round(link_quality_score, 2),
            'overall_evidence_score': round(overall_score, 2),
            'quality_level': quality_level,
            'reliability_factors': reliability_factors,
            'evidence_concerns': evidence_concerns,
            'verification_links_count': len(verification_links),
            'high_quality_links_count': high_quality_count,
            'llm_verification_analysis': {
                'llm_verification_used': verification_analysis.get('llm_verification_used', False),
                'verification_sources_provided': len(verification_links) > 0,
            },
            'scoring_method': 'llm_weighted_with_direct_links',
            'weights_used': self.scoring_weights,
            'thresholds_used': {
                'high_quality': self.high_quality_threshold,
                'medium_quality': self.medium_quality_threshold,
                'poor_quality': self.poor_quality_threshold
            }
        }

    def _create_evidence_summary_with_llm_links(self, extracted_claims: List[Dict[str, Any]],
                                               quality_assessment: Dict, evidence_scores: Dict,
                                               verification_analysis: Dict) -> str:
        """Create evidence summary with LLM link awareness"""
        
        if not extracted_claims:
            return "No claims available for evidence evaluation."
        
        verification_links = verification_analysis.get('verification_links', [])
        high_quality_count = verification_analysis.get('high_quality_links_count', 0)
        llm_verification_used = verification_analysis.get('llm_verification_used', False)
        
        summary_lines = [
            f"EVIDENCE EVALUATION SUMMARY",
            f"Overall Evidence Score: {evidence_scores.get('overall_evidence_score', 5.0):.1f}/10 ({evidence_scores.get('quality_level', 'UNKNOWN')})",
            f"Verification Links: {len(verification_links)} total ({high_quality_count} high-quality)",
            ""
        ]
        
        # ✅ ADD LLM VERIFICATION SUMMARY
        if llm_verification_used:
            summary_lines.extend([
                "🧠 LLM Verification Analysis:",
                f" • Direct LLM-powered verification sources provided",
                f" • Sources aligned with credibility assessment methodology",
                f" • No external dependencies required",
                ""
            ])
        
        # Add quality component scores
        summary_lines.extend([
            f"Quality Breakdown:",
            f" • Source Quality: {evidence_scores.get('source_quality_score', 5.0):.1f}/10",
            f" • Logical Consistency: {evidence_scores.get('logical_consistency_score', 5.0):.1f}/10",
            f" • Evidence Completeness: {evidence_scores.get('evidence_completeness_score', 5.0):.1f}/10",
            f" • Verification Links Quality: {evidence_scores.get('verification_links_quality_score', 5.0):.1f}/10",
            ""
        ])
        
        # Add link quality assessment
        if high_quality_count >= 3:
            summary_lines.append("✓ Excellent verification link quality - multiple authoritative sources provided")
        elif high_quality_count >= 1:
            summary_lines.append("⚠ Good verification link quality - some authoritative sources provided")
        elif len(verification_links) > 0:
            summary_lines.append("⚠ Basic verification sources available")
        else:
            summary_lines.append("❌ No verification links available")
        
        summary_lines.append("")
        
        # Add reliability factors
        reliability_factors = evidence_scores.get('reliability_factors', [])
        if reliability_factors:
            summary_lines.append("Reliability Factors:")
            for factor in reliability_factors:
                summary_lines.append(f" ✓ {factor}")
            summary_lines.append("")
        
        # Add concerns
        evidence_concerns = evidence_scores.get('evidence_concerns', [])
        if evidence_concerns:
            summary_lines.append("Evidence Concerns:")
            for concern in evidence_concerns:
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
        """Update evaluation-specific metrics"""
        self.evaluation_metrics['total_evaluations'] += 1
        if not error:
            total = self.evaluation_metrics['total_evaluations']
            current_avg = self.evaluation_metrics['average_response_time']
            self.evaluation_metrics['average_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics with LLM verification information"""
        
        base_metrics = self.get_performance_metrics()
        
        config_metrics = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'evidence_threshold': self.evidence_threshold,
            'enable_detailed_analysis': self.enable_detailed_analysis,
            'enable_fallacy_detection': self.enable_fallacy_detection,
            'enable_gap_analysis': self.enable_gap_analysis,
            'enable_llm_verification': self.enable_llm_verification,
            'max_verification_links': self.max_verification_links,
            'link_quality_threshold': self.link_quality_threshold,
            'evidence_types_count': len(self.evidence_types),
            'source_quality_tiers_count': len(self.source_quality_tiers),
            'scoring_weights': self.scoring_weights,
            'quality_thresholds': {
                'high': self.high_quality_threshold,
                'medium': self.medium_quality_threshold,
                'poor': self.poor_quality_threshold
            },
            'rate_limit_seconds': self.rate_limit,
            'config_version': '5.0_llm_direct'
        }
        
        return {
            **base_metrics,
            'evaluation_specific_metrics': self.evaluation_metrics,
            'config_metrics': config_metrics,
            'agent_type': 'evidence_evaluator',
            'modular_architecture': True,
            'config_integrated': True,
            'llm_verification_enabled': True,
            'web_search_enabled': False,
            'llm_powered': True,
            'prompt_source': 'centralized_config'
        }

# Testing functionality with LLM verification
if __name__ == "__main__":
    """Test the LLM-powered evidence evaluator agent"""
    print("🧪 Testing LLM-Powered Evidence Evaluator Agent")
    print("=" * 75)
    
    try:
        # Initialize agent
        agent = EvidenceEvaluatorAgent()
        print(f"✅ Agent initialized with LLM verification")
        
        # Test with COVID vaccine example
        test_article = """
        According to a study published in the New England Journal of Medicine,
        researchers found that the COVID-19 vaccine is 95% effective in preventing
        severe illness. Dr. Sarah Johnson from Harvard Medical School confirmed
        these findings in a recent interview.
        """
        
        test_claims = [{
            'text': 'COVID-19 vaccine is 95% effective in preventing severe illness',
            'claim_type': 'Medical',
            'priority': 1,
            'verifiability_score': 9,
            'source': 'New England Journal of Medicine study'
        }]
        
        test_context = {
            'overall_context_score': 2.5,
            'risk_level': 'LOW'
        }
        
        test_input = {
            "text": test_article,
            "extracted_claims": test_claims,
            "context_analysis": test_context,
            "include_detailed_analysis": True
        }
        
        print(f"\n🔍 Testing LLM verification with COVID vaccine effectiveness claim...")
        result = agent.process(test_input)
        
        if result['success']:
            evaluation_data = result['result']
            print(f"✅ Evaluation completed successfully")
            print(f" Overall evidence score: {evaluation_data['evidence_scores']['overall_evidence_score']:.1f}/10")
            print(f" LLM verification used: {evaluation_data.get('llm_verification_used', False)}")
            print(f" Verification links provided: {len(evaluation_data.get('verification_links', []))}")
            print(f" High-quality links: {evaluation_data['evidence_scores'].get('high_quality_links_count', 0)}")
            
            # Show sample verification links
            links = evaluation_data.get('verification_links', [])
            if links:
                print(f" Sample verification sources:")
                for i, link in enumerate(links[:3], 1):
                    print(f" {i}. {link.get('institution', 'Unknown')}")
                    print(f"    URL: {link.get('url', 'No URL')}")
                    print(f"    Quality: {link.get('quality_score', 0):.2f}")
        else:
            print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
        
        print(f"\n✅ LLM-powered evidence evaluator agent test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
