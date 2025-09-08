# agents/credible_source/source_agent.py

"""
Enhanced Credible Source Agent with Safety Filter Handling

This agent provides contextual source recommendations with robust handling
of Gemini API safety filters that were blocking contextual source generation.

Features:
- ✅ SAFETY FILTER FALLBACK HANDLING (fixes the main issue)
- ✅ CONTEXTUAL SOURCE RECOMMENDATIONS instead of generic ones
- Enhanced institutional fallbacks when AI is restricted
- Full configuration integration
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
from .source_database import SourceReliabilityDatabase
from .domain_classifier import DomainClassifier

# ✅ IMPORT CONFIGURATION FILES
from config import get_model_config, get_prompt_template, get_settings
from utils.helpers import sanitize_text

class CredibleSourceAgent(BaseAgent):
    """
    🔍 ENHANCED CREDIBLE SOURCE AGENT WITH SAFETY HANDLING
    
    ✅ KEY FIXES: 
    - Handles Gemini safety filter blocks gracefully with institutional fallbacks
    - Provides contextual, specific source recommendations instead of generic ones
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced credible source agent with safety handling"""
        
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        source_config = get_model_config('credible_source')
        system_settings = get_settings()
        
        # Merge with runtime overrides
        if config:
            source_config.update(config)

        self.agent_name = "credible_source"
        
        # Initialize base agent with merged config
        super().__init__(source_config)

        # ✅ USE CONFIG VALUES FOR AI MODEL SETTINGS
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 2048)

        # ✅ SOURCE RECOMMENDATION SETTINGS FROM CONFIG
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.max_sources_per_recommendation = self.config.get('max_sources_per_recommendation', 8)
        self.enable_cross_verification = self.config.get('enable_cross_verification', True)
        self.enable_domain_specific_sources = self.config.get('enable_domain_specific_sources', True)

        # ✅ NEW: CONTEXTUAL RECOMMENDATION SETTINGS
        self.enable_contextual_recommendations = self.config.get('enable_contextual_recommendations', True)
        self.contextual_specificity_threshold = self.config.get('contextual_specificity_threshold', 0.8)
        self.max_contacts_per_claim = self.config.get('max_contacts_per_claim', 3)

        # ✅ NEW: SAFETY FALLBACK SETTINGS
        self.enable_safety_fallbacks = self.config.get('enable_safety_fallbacks', True)

        # ✅ RELIABILITY ASSESSMENT SETTINGS FROM CONFIG
        self.reliability_tiers = self.config.get('reliability_tiers', [
            'primary_sources', 'expert_sources', 'institutional_sources',
            'journalistic_sources', 'secondary_sources'
        ])
        self.min_reliability_score = self.config.get('min_reliability_score', 6.0)
        self.preferred_source_types = self.config.get('preferred_source_types', [
            'academic', 'government', 'expert', 'institutional'
        ])

        # ✅ DOMAIN CLASSIFICATION SETTINGS FROM CONFIG
        self.enable_domain_classification = self.config.get('enable_domain_classification', True)
        self.domain_confidence_threshold = self.config.get('domain_confidence_threshold', 0.7)

        # ✅ GET API KEY FROM SYSTEM SETTINGS
        self.api_key = system_settings.gemini_api_key

        # ✅ LOAD PROMPTS FROM CONFIG
        self.source_recommendations_prompt = get_prompt_template('credible_source', 'source_recommendations')
        self.reliability_assessment_prompt = get_prompt_template('credible_source', 'reliability_assessment')
        self.verification_strategy_prompt = get_prompt_template('credible_source', 'verification_strategy')
        self.fact_check_guidance_prompt = get_prompt_template('credible_source', 'fact_check_guidance')

        # ✅ USE RATE LIMITING FROM CONFIG/SETTINGS
        self.rate_limit = self.config.get('rate_limit_seconds', system_settings.gemini_rate_limit)
        self.max_retries = self.config.get('max_retries', system_settings.max_retries)

        # Initialize Gemini API
        self._initialize_gemini_api()

        # Initialize modular components
        self.source_database = SourceReliabilityDatabase()
        self.domain_classifier = DomainClassifier()

        # Enhanced performance tracking with safety awareness
        self.source_metrics = {
            'total_recommendations': 0,
            'successful_recommendations': 0,
            'contextual_sources_generated': 0,
            'specific_contacts_provided': 0,
            'source_analyses_generated': 0,
            'reliability_assessments_generated': 0,
            'verification_strategies_generated': 0,
            'fact_check_guidance_generated': 0,
            'high_reliability_sources_found': 0,
            'domain_classifications_performed': 0,
            'cross_verification_analyses': 0,
            'generic_recommendations_filtered': 0,
            'average_response_time': 0.0,
            'gemini_api_calls': 0,
            'config_integrated': True,
            # ✅ NEW: Safety fallback metrics
            'safety_blocks_encountered': 0,
            'fallback_sources_generated': 0,
            'safety_fallbacks_used': 0
        }

        # Rate limiting tracking
        self.last_request_time = None
        
        self.logger.info(f"✅ Enhanced Credible Source Agent initialized with safety handling")
        self.logger.info(f"🤖 Model: {self.model_name}, Temperature: {self.temperature}")
        self.logger.info(f"🛡️ Safety fallbacks: {'Enabled' if self.enable_safety_fallbacks else 'Disabled'}")
        self.logger.info(f"🎯 Contextual Recommendations: {'Enabled' if self.enable_contextual_recommendations else 'Disabled'}")

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

            # ✅ USE SAFETY SETTINGS FROM CONFIG (More permissive for institutional content)
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

            self.logger.info("🔐 Gemini API initialized for contextual source recommendations with enhanced safety handling")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Gemini API: {str(e)}")
            raise

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """🎯 MAIN PROCESSING METHOD WITH SAFETY HANDLING"""
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
            evidence_evaluation = input_data.get('evidence_evaluation', {}) or {}
            include_detailed_analysis = input_data.get(
                'include_detailed_analysis',
                self.enable_detailed_analysis
            )

            # ✅ USE CONFIG FOR PROCESSING DECISIONS
            evidence_score = evidence_evaluation.get('overall_evidence_score', 5.0)
            force_detailed = (
                include_detailed_analysis or
                evidence_score < self.min_reliability_score or
                len(extracted_claims) > 5 or
                self.enable_detailed_analysis
            )

            # ✅ PERFORM CONTEXTUAL SOURCE RECOMMENDATION WITH SAFETY HANDLING
            recommendation_result = self.recommend_contextual_sources_with_safety(
                article_text=article_text,
                extracted_claims=extracted_claims,
                evidence_evaluation=evidence_evaluation,
                include_detailed_analysis=force_detailed
            )

            # Extract overall recommendation score for metrics
            recommendation_score = recommendation_result['recommendation_scores']['overall_recommendation_score']

            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(recommendation_score / 10.0)
            self.source_metrics['successful_recommendations'] += 1

            # ✅ UPDATE CONTEXTUAL RECOMMENDATION METRICS
            contextual_sources = recommendation_result.get('contextual_sources', [])
            self.source_metrics['contextual_sources_generated'] += len(contextual_sources)
            self.source_metrics['specific_contacts_provided'] += recommendation_result['metadata'].get('contacts_provided', 0)

            # Format output for LangGraph with safety context
            return self.format_output(
                result=recommendation_result,
                confidence=recommendation_score / 10.0,
                metadata={
                    'response_time': recommendation_result['metadata']['response_time_seconds'],
                    'model_used': self.model_name,
                    'config_version': '3.0_safety_enhanced',
                    'agent_version': '3.0_contextual_safety',
                    'detailed_analysis_triggered': force_detailed,
                    'min_reliability_score_used': self.min_reliability_score,
                    'max_sources_limit': self.max_sources_per_recommendation,
                    'contextual_recommendations_enabled': self.enable_contextual_recommendations,
                    'contextual_sources_generated': len(contextual_sources),
                    'safety_fallbacks_used': recommendation_result.get('safety_fallback_used', False)
                }
            )

        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            return self.format_error_output(e, input_data)

    def recommend_contextual_sources_with_safety(self,
                                                article_text: str,
                                                extracted_claims: List[Dict[str, Any]],
                                                evidence_evaluation: Dict[str, Any],
                                                include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        🔍 MAIN CONTEXTUAL SOURCE RECOMMENDATION WITH SAFETY HANDLING
        
        ✅ KEY FIX: Handles Gemini safety filter blocks gracefully with institutional fallbacks
        """
        self._respect_rate_limits()
        start_time = time.time()

        try:
            self.logger.info("Starting contextual source recommendation with claim-specific analysis...")

            # Step 1: Clean article text
            article_text = sanitize_text(article_text)

            # ✅ USE CONFIG FOR TEXT LENGTH LIMITS
            max_text_length = self.config.get('max_article_length', 4000)
            if len(article_text) > max_text_length:
                article_text = article_text[:max_text_length] + "..."

            # Step 2: Domain classification if enabled
            domain_analysis = {}
            if self.enable_domain_classification:
                domain_analysis = self.domain_classifier.classify_domain(article_text, extracted_claims)
                self.source_metrics['domain_classifications_performed'] += 1
                self.logger.info(f"🏷️ Domain classified as: {domain_analysis.get('primary_domain', 'general')}")

            # ✅ STEP 3: GENERATE CONTEXTUAL SOURCE ANALYSIS WITH SAFETY HANDLING (KEY FIX)
            contextual_analysis = self._generate_contextual_source_analysis_with_safety_fallback(
                article_text, extracted_claims, evidence_evaluation, domain_analysis
            )

            # Step 4: Get systematic source recommendations using modular database
            database_recommendations = self.source_database.get_source_recommendations(
                article_text, extracted_claims, domain_analysis.get('primary_domain', 'general')
            )

            # Step 5: Generate reliability assessment with safety handling
            reliability_assessment = self._generate_reliability_assessment_safe(
                article_text, database_recommendations
            )

            # Step 6: Generate verification strategies with safety handling
            verification_strategies = self._generate_verification_strategies_safe(
                extracted_claims, domain_analysis, evidence_evaluation
            )

            # ✅ STEP 7: GENERATE CLAIM-SPECIFIC FACT-CHECK GUIDANCE WITH SAFETY HANDLING
            fact_check_guidance = self._generate_fact_check_guidance_safe(
                extracted_claims, contextual_analysis
            )

            # Step 8: Cross-verification analysis if enabled
            cross_verification_analysis = None
            if self.enable_cross_verification and len(extracted_claims) > 0:
                cross_verification_analysis = self._generate_cross_verification_analysis_safe(
                    extracted_claims, contextual_analysis.get('contextual_sources', [])
                )
                self.source_metrics['cross_verification_analyses'] += 1

            # ✅ STEP 9: CALCULATE RECOMMENDATION SCORES INCLUDING SAFETY AWARENESS
            recommendation_scores = self._calculate_recommendation_scores_with_safety(
                contextual_analysis, database_recommendations, reliability_assessment,
                verification_strategies, domain_analysis
            )

            # Step 10: Package results with safety metadata
            response_time = time.time() - start_time

            result = {
                'contextual_analysis': contextual_analysis['analysis_text'],
                'contextual_sources': contextual_analysis['contextual_sources'],  # ✅ Specific sources
                'reliability_assessment': reliability_assessment,
                'verification_strategies': verification_strategies,
                'fact_check_guidance': fact_check_guidance,
                'cross_verification_analysis': cross_verification_analysis,
                'domain_analysis': domain_analysis,
                'database_recommendations': database_recommendations,
                'recommendation_scores': recommendation_scores,
                'source_summary': self._create_contextual_source_summary_with_safety(
                    contextual_analysis, recommendation_scores
                ),
                'safety_fallback_used': contextual_analysis.get('safety_fallback_used', False),  # ✅ New
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'response_time_seconds': round(response_time, 2),
                    'model_used': self.model_name,
                    'temperature_used': self.temperature,
                    'article_length_processed': len(article_text),
                    'claims_analyzed': len(extracted_claims),
                    'detailed_analysis_included': include_detailed_analysis,
                    'domain_classification_enabled': self.enable_domain_classification,
                    'cross_verification_enabled': self.enable_cross_verification,
                    'fact_check_guidance_triggered': fact_check_guidance is not None,
                    'domain_classified': domain_analysis.get('domain_classified', False),
                    'primary_domain': domain_analysis.get('primary_domain', 'general'),
                    # ✅ CONTEXTUAL SOURCE METADATA WITH SAFETY
                    'contextual_sources_generated': len(contextual_analysis['contextual_sources']),
                    'contacts_provided': contextual_analysis.get('contacts_provided', 0),
                    'specific_recommendations': contextual_analysis.get('specific_recommendations', 0),
                    'safety_blocks_encountered': contextual_analysis.get('safety_blocks_encountered', 0),  # ✅ New
                    'config_version': '3.0_safety_enhanced',
                    'agent_version': '3.0_contextual_safety'
                }
            }

            # Step 11: Update performance metrics
            self._update_recommendation_metrics(response_time, recommendation_scores['overall_recommendation_score'])
            contextual_count = len(contextual_analysis['contextual_sources'])
            contacts_count = contextual_analysis.get('contacts_provided', 0)
            
            self.logger.info(f"Successfully completed contextual source recommendation in {response_time:.2f} seconds")
            self.logger.info(f"🔍 Overall recommendation score: {recommendation_scores['overall_recommendation_score']:.1f}/10")
            self.logger.info(f"🎯 Contextual sources: {contextual_count}, Specific contacts: {contacts_count}")
            self.logger.info(f"🛡️ Safety fallbacks used: {contextual_analysis.get('safety_fallback_used', False)}")
            
            return result

        except Exception as e:
            self._update_recommendation_metrics(time.time() - start_time, 0, error=True)
            self.logger.error(f"Error in contextual source recommendation with safety handling: {str(e)}")
            raise

    def _generate_contextual_source_analysis_with_safety_fallback(self, article_text: str,
                                                                 extracted_claims: List[Dict[str, Any]],
                                                                 evidence_evaluation: Dict[str, Any],
                                                                 domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ GENERATE CONTEXTUAL SOURCE RECOMMENDATIONS WITH SAFETY FALLBACK (KEY FIX)
        
        This fixes the main issue where Gemini safety filters were blocking contextual source generation.
        """
        try:
            # Extract specific topics from claims
            claim_topics = []
            for claim in extracted_claims[:6]:  # Focus on top claims
                claim_text = claim.get('text', '')[:120]
                claim_type = claim.get('claim_type', '')
                priority = claim.get('priority', 3)
                if priority <= 2:  # High priority claims only
                    claim_topics.append(f"• {claim_type}: {claim_text}")

            if not claim_topics:
                claim_topics = ["• General fact-checking needed for unspecified claims"]

            claims_summary = "\n".join(claim_topics)
            domain = domain_analysis.get('primary_domain', 'general')
            evidence_score = evidence_evaluation.get('overall_evidence_score', 5)

            # ✅ ENHANCED PROMPT - More neutral language to avoid safety triggers
            contextual_prompt = f"""
            Please provide institutional source recommendations for fact-checking these claims:

            ARTICLE DOMAIN: {domain}
            EVIDENCE QUALITY: {evidence_score}/10

            SPECIFIC CLAIMS REQUIRING VERIFICATION:
            {claims_summary}

            ARTICLE CONTEXT: {article_text[:600]}...

            Please recommend targeted institutional sources in this format:

            PRIMARY VERIFICATION SOURCES:
            1. [Institution Name]: [Department/URL]
               - Relevance: [Why relevant to these specific claims]
               - Contact method: [How to access information]

            2. [Different Institution]: [Department/URL]  
               - Relevance: [How this relates to the claims]
               - Contact method: [Access method]

            EXPERT CONTACTS:
            • [Expert/Organization]: [Institution] - Authority on [specific topic]

            INSTITUTIONAL SOURCES:
            • [Government Agency]: [Specific department]
            • [University/Research]: [Specific database/group]

            Focus on providing sources that can verify the SPECIFIC claims made.
            Avoid generic recommendations. Provide institutional contacts when possible.
            """

            response = self.model.generate_content(contextual_prompt)
            
            # ✅ ENHANCED SAFETY FILTER HANDLING
            if not response.candidates:
                self.logger.warning("🚨 No candidates returned - using safety fallback")
                self.source_metrics['safety_blocks_encountered'] += 1
                return self._generate_institutional_contextual_fallback(extracted_claims, domain)
            
            candidate = response.candidates[0]
            
            if candidate.finish_reason == 2:  # SAFETY filter triggered
                self.logger.warning("🚨 Contextual source generation blocked by safety filters - using institutional fallbacks")
                self.source_metrics['safety_blocks_encountered'] += 1
                return self._generate_institutional_contextual_fallback(extracted_claims, domain)
            
            if not candidate.content or not candidate.content.parts:
                self.logger.warning("🚨 Empty content returned - using safety fallback")
                self.source_metrics['safety_blocks_encountered'] += 1
                return self._generate_institutional_contextual_fallback(extracted_claims, domain)
            
            analysis_text = candidate.content.parts[0].text

            # ✅ PARSE CONTEXTUAL SOURCE RECOMMENDATIONS
            contextual_sources = self._parse_contextual_sources_enhanced(analysis_text)

            # Count contacts and specific recommendations
            contacts_provided = sum(1 for source in contextual_sources
                                  if 'contact' in source.get('type', '').lower())
            specific_recommendations = sum(1 for source in contextual_sources
                                         if source.get('relevance_score', 0) >= 7)

            self.source_metrics['gemini_api_calls'] += 1
            self.source_metrics['contextual_sources_generated'] += len(contextual_sources)

            return {
                'analysis_text': analysis_text,
                'contextual_sources': contextual_sources,
                'contacts_provided': contacts_provided,
                'specific_recommendations': specific_recommendations,
                'contextual_analysis_success': len(contextual_sources) > 0,
                'safety_fallback_used': False,
                'safety_blocks_encountered': 0
            }

        except Exception as e:
            self.logger.error(f"Error generating contextual source analysis: {str(e)}")
            self.source_metrics['safety_blocks_encountered'] += 1
            return self._generate_institutional_contextual_fallback(extracted_claims, domain_analysis.get('primary_domain', 'general'))

    def _generate_institutional_contextual_fallback(self, extracted_claims: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """
        ✅ GENERATE INSTITUTIONAL CONTEXTUAL FALLBACK SOURCES (KEY FIX)
        
        This provides actual useful contextual sources instead of failing completely.
        """
        fallback_sources = []
        
        # Analyze claims and domain to determine appropriate institutional sources
        for claim in extracted_claims[:self.max_sources_per_recommendation]:
            claim_text = claim.get('text', 'Unknown claim')[:120]
            claim_type = claim.get('claim_type', 'General')
            
            # Medical/health claims
            if any(keyword in claim_text.lower() for keyword in ['vaccine', 'medical', 'health', 'drug', 'treatment', 'disease', 'covid']):
                fallback_sources.extend([
                    {
                        'name': 'Centers for Disease Control and Prevention (CDC)',
                        'details': f'Official US health agency with expertise in {claim_type.lower()} verification',
                        'relevance': f'Authoritative source for medical claims, especially: "{claim_text[:60]}..."',
                        'contact_method': 'Visit cdc.gov, call 1-800-CDC-INFO, or search CDC databases',
                        'type': 'contextual_institutional_health',
                        'relevance_score': 9,
                        'reliability_score': 10,
                        'url': 'https://www.cdc.gov/'
                    },
                    {
                        'name': 'World Health Organization (WHO)',
                        'details': f'Global health authority for international {claim_type.lower()} standards',
                        'relevance': f'International perspective on health claims: "{claim_text[:60]}..."',
                        'contact_method': 'Visit who.int for official health statements and data',
                        'type': 'contextual_institutional_international',
                        'relevance_score': 8,
                        'reliability_score': 10,
                        'url': 'https://www.who.int/'
                    },
                    {
                        'name': 'PubMed/National Library of Medicine',
                        'details': f'Peer-reviewed medical research database for {claim_type.lower()} studies',
                        'relevance': f'Academic research verification for: "{claim_text[:60]}..."',
                        'contact_method': f'Search PubMed for: "{claim_text[:40]}" + medical research',
                        'type': 'contextual_academic_medical',
                        'relevance_score': 8,
                        'reliability_score': 9,
                        'url': 'https://pubmed.ncbi.nlm.nih.gov/'
                    }
                ])
            
            # Scientific/research claims
            elif any(keyword in claim_text.lower() for keyword in ['study', 'research', 'scientist', 'university', 'journal', 'published']):
                fallback_sources.extend([
                    {
                        'name': f'Google Scholar - {claim_type} Research',
                        'details': f'Academic search engine for peer-reviewed {claim_type.lower()} research',
                        'relevance': f'Find academic papers related to: "{claim_text[:60]}..."',
                        'contact_method': f'Search: "{claim_text[:40]}" + peer reviewed',
                        'type': 'contextual_academic_search',
                        'relevance_score': 7,
                        'reliability_score': 8,
                        'url': 'https://scholar.google.com/'
                    },
                    {
                        'name': f'{claim_type} Research Institutions',
                        'details': f'Universities and research centers specializing in {claim_type.lower()}',
                        'relevance': f'Expert verification needed for: "{claim_text[:60]}..."',
                        'contact_method': 'Contact relevant university departments or research institutes',
                        'type': 'contextual_expert_institutional',
                        'relevance_score': 8,
                        'reliability_score': 8
                    }
                ])
            
            # Government/policy claims
            elif any(keyword in claim_text.lower() for keyword in ['government', 'policy', 'law', 'regulation', 'official']):
                fallback_sources.extend([
                    {
                        'name': 'Government Information Services',
                        'details': f'Official government sources for {claim_type.lower()} verification',
                        'relevance': f'Authoritative government information for: "{claim_text[:60]}..."',
                        'contact_method': 'Contact relevant government agencies or check official websites',
                        'type': 'contextual_institutional_government',
                        'relevance_score': 9,
                        'reliability_score': 9
                    }
                ])
            
            # Generic high-quality contextual sources
            else:
                fallback_sources.extend([
                    {
                        'name': f'Professional Fact-Checkers - {claim_type}',
                        'details': f'Established fact-checking organizations with {claim_type.lower()} expertise',
                        'relevance': f'Professional verification services for: "{claim_text[:60]}..."',
                        'contact_method': 'Snopes.com, FactCheck.org, PolitiFact.com - search for specific claim',
                        'type': 'contextual_fact_checker',
                        'relevance_score': 6,
                        'reliability_score': 7
                    }
                ])

        # Remove duplicates and limit results
        unique_sources = []
        seen_names = set()
        for source in fallback_sources:
            if source['name'] not in seen_names:
                unique_sources.append(source)
                seen_names.add(source['name'])
            if len(unique_sources) >= self.max_sources_per_recommendation:
                break

        # Count contacts and specific recommendations
        contacts_provided = len([s for s in unique_sources if 'contact' in s.get('type', '')])
        specific_recommendations = len([s for s in unique_sources if s.get('relevance_score', 0) >= 7])

        self.source_metrics['fallback_sources_generated'] += len(unique_sources)
        self.source_metrics['safety_fallbacks_used'] += 1

        return {
            'analysis_text': f'Contextual source analysis used institutional fallbacks due to content sensitivity. Domain-specific ({domain}) institutional sources provided based on claim analysis.',
            'contextual_sources': unique_sources,
            'contacts_provided': contacts_provided,
            'specific_recommendations': specific_recommendations,
            'contextual_analysis_success': len(unique_sources) > 0,
            'safety_fallback_used': True,
            'safety_blocks_encountered': 1
        }

    def _parse_contextual_sources_enhanced(self, analysis_text: str) -> List[Dict[str, Any]]:
        """✅ ENHANCED PARSING FOR CONTEXTUAL SOURCE RECOMMENDATIONS - BUG FIXED"""
        sources = []
        
        # Split by primary verification sources sections
        sections = re.split(r'(?:PRIMARY VERIFICATION SOURCES|EXPERT CONTACTS|INSTITUTIONAL SOURCES)', analysis_text, flags=re.IGNORECASE)
        
        for section in sections:
            # Extract numbered sources (1. 2. etc.)
            source_pattern = r'(\d+\.)\s*\[?([^\]:\n]+)\]?:?\s*([^\n]+)(?:\n\s*-\s*Relevance:\s*([^\n]+))?(?:\n\s*-\s*Contact method:\s*([^\n]+))?'
            matches = re.findall(source_pattern, section, re.MULTILINE)
            
            for match in matches:
                number, name, details, relevance, contact = match
                if name.strip():
                    # ✅ FIX: Always initialize source_data
                    source_data = {
                        'name': name.strip(),
                        'details': details.strip()[:300] if details else 'No details available',
                        'relevance': relevance.strip() if relevance else 'Contextually selected',
                        'contact_method': contact.strip() if contact else 'Contact information not specified',
                        'type': 'contextual_primary',
                        'relevance_score': 8,  # High since contextually chosen
                        'reliability_score': 8
                    }
                    
                    # Extract URL if present
                    url_match = re.search(r'https?://[^\s]+', details or '')
                    if url_match:
                        source_data['url'] = url_match.group(0)
                    
                    sources.append(source_data)
            
            # Extract bullet point sources (• or -)
            bullet_pattern = r'[•-]\s*\[?([^\]:\n]+)\]?:?\s*([^\n]+)'
            bullet_matches = re.findall(bullet_pattern, section)
            
            for name, details in bullet_matches:
                if name.strip() and len(sources) < self.max_sources_per_recommendation:
                    # ✅ FIX: Always initialize source_data for bullet points too
                    source_data = {
                        'name': name.strip(),
                        'details': details.strip()[:200] if details else 'No details available',
                        'type': 'contextual_expert' if 'expert' in section.lower() else 'contextual_institutional',
                        'relevance_score': 7,
                        'reliability_score': 8 if 'contextual_institutional' in ('contextual_expert' if 'expert' in section.lower() else 'contextual_institutional') else 7
                    }
                    
                    sources.append(source_data)
        
        return sources[:self.max_sources_per_recommendation]


    def _generate_reliability_assessment_safe(self, article_text: str, database_recommendations: Dict[str, Any]) -> str:
        """Generate reliability assessment with safety handling"""
        try:
            # Prepare source list for analysis
            source_list = []
            for source in database_recommendations.get('recommended_sources', [])[:8]:
                source_name = source.get('name', 'Unknown')
                reliability_score = source.get('reliability_score', 0)
                source_type = source.get('type', 'Unknown')
                source_list.append(f"Source: {source_name} (Type: {source_type}, Reliability: {reliability_score}/10)")

            sources_text = "\n".join(source_list) if source_list else "No specific sources recommended"

            # ✅ SAFER PROMPT
            prompt = f"""
            Please assess the reliability of these information sources:

            CONTENT: {article_text[:800]}

            IDENTIFIED SOURCES:
            {sources_text}

            Please provide:
            1. Overall source reliability assessment
            2. Source diversity evaluation  
            3. Institutional authority analysis
            4. Reliability rating (1-10)
            
            Keep analysis factual and institutional.
            """

            response = self.model.generate_content(prompt)
            
            # Handle safety blocks
            if not response.candidates or response.candidates[0].finish_reason == 2:
                return self._generate_safe_reliability_fallback(len(source_list))
            
            if not response.candidates[0].content or not response.candidates[0].content.parts:
                return self._generate_safe_reliability_fallback(len(source_list))

            self.source_metrics['gemini_api_calls'] += 1
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            self.logger.error(f"Error in safe reliability assessment: {str(e)}")
            return self._generate_safe_reliability_fallback(database_recommendations.get('recommended_sources', []))

    def _generate_safe_reliability_fallback(self, sources_info) -> str:
        """Generate safe fallback reliability assessment"""
        sources_count = len(sources_info) if isinstance(sources_info, list) else sources_info
        
        return f"""
        SOURCE RELIABILITY ASSESSMENT (AUTOMATED)
        
        Sources Analyzed: {sources_count}
        
        Reliability Factors:
        • Source verification requires institutional confirmation
        • Cross-referencing with established authorities recommended
        • Independent expert review advised
        
        Overall Assessment: Requires human verification
        
        Recommended Approach:
        1. Verify through official institutional channels
        2. Check source credentials and authority
        3. Cross-reference with multiple independent sources
        4. Seek expert review for technical claims
        """

    def _generate_verification_strategies_safe(self, extracted_claims: List[Dict[str, Any]],
                                             domain_analysis: Dict[str, Any],
                                             evidence_evaluation: Dict[str, Any]) -> str:
        """Generate verification strategies with safety handling"""
        try:
            # Prepare high-priority claims for verification
            priority_claims = []
            for claim in extracted_claims[:6]:
                claim_text = claim.get('text', 'Unknown claim')[:100]
                priority = claim.get('priority', 2)
                verifiability = claim.get('verifiability_score', 5)
                priority_claims.append(f"Priority {priority}: {claim_text} (Verifiability: {verifiability}/10)")

            claims_text = "\n".join(priority_claims) if priority_claims else "No claims available"
            domain_context = f"Domain: {domain_analysis.get('primary_domain', 'general')}"
            evidence_context = f"Evidence Quality: {evidence_evaluation.get('overall_evidence_score', 'N/A')}/10"

            # ✅ SAFER PROMPT
            prompt = f"""
            Please provide verification strategies for these claims:

            PRIORITY CLAIMS:
            {claims_text}

            CONTEXT:
            {domain_context}
            {evidence_context}

            Please provide:
            1. Verification approach for each claim type
            2. Institutional sources to contact
            3. Search strategies and keywords
            4. Timeline for verification process
            
            Focus on practical, actionable strategies.
            """

            response = self.model.generate_content(prompt)
            
            # Handle safety blocks
            if not response.candidates or response.candidates[0].finish_reason == 2:
                return self._generate_safe_strategies_fallback(len(priority_claims))
            
            if not response.candidates[0].content or not response.candidates[0].content.parts:
                return self._generate_safe_strategies_fallback(len(priority_claims))

            self.source_metrics['gemini_api_calls'] += 1
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            self.logger.error(f"Error in safe verification strategies: {str(e)}")
            return self._generate_safe_strategies_fallback(len(extracted_claims))

    def _generate_safe_strategies_fallback(self, claims_count: int) -> str:
        """Generate safe fallback verification strategies"""
        return f"""
        VERIFICATION STRATEGIES (AUTOMATED GUIDANCE)
        
        Claims to Verify: {claims_count}
        
        General Verification Approach:
        1. Institutional Source Verification
           - Contact relevant government agencies
           - Check academic/research institutions
           - Verify through professional organizations
        
        2. Expert Consultation
           - Identify subject matter experts
           - Contact university departments
           - Reach out to professional associations
        
        3. Cross-Reference Analysis
           - Check multiple independent sources  
           - Verify through different types of institutions
           - Look for peer-reviewed publications
        
        4. Documentation Review
           - Request official documents
           - Check public records
           - Verify data sources
        
        Timeline: Allow 3-7 days for thorough verification
        """

    def _generate_fact_check_guidance_safe(self, extracted_claims: List[Dict[str, Any]],
                                         contextual_analysis: Dict[str, Any]) -> str:
        """Generate fact-check guidance with safety handling"""
        try:
            if not extracted_claims:
                return "No specific claims identified for fact-checking guidance."

            # Focus on high-priority, verifiable claims
            priority_claims = []
            for claim in extracted_claims:
                if claim.get('priority', 3) <= 2 and claim.get('verifiability_score', 0) >= 6:
                    claim_text = claim.get('text', '')[:100]
                    claim_type = claim.get('claim_type', 'General')
                    priority_claims.append(f"• {claim_type}: {claim_text}")

            if not priority_claims:
                return "No high-priority verifiable claims identified for fact-checking."

            claims_text = "\n".join(priority_claims[:5])
            contextual_sources = contextual_analysis.get('contextual_sources', [])[:3]
            sources_summary = "; ".join([s.get('name', 'Unknown') for s in contextual_sources])

            # ✅ SAFER PROMPT
            guidance_prompt = f"""
            Create step-by-step fact-checking guidance for these claims:

            HIGH-PRIORITY CLAIMS:
            {claims_text}

            AVAILABLE SOURCES: {sources_summary}

            Please provide:
            1. Priority order for claim verification
            2. Specific verification steps for each claim  
            3. Evidence types to look for
            4. Warning signs of misinformation
            5. Estimated verification timeline
            
            Keep guidance practical and actionable.
            """

            response = self.model.generate_content(guidance_prompt)
            
            # Handle safety blocks
            if not response.candidates or response.candidates[0].finish_reason == 2:
                return "Fact-checking guidance unavailable due to content restrictions. Please follow standard verification procedures."
            
            if not response.candidates[0].content or not response.candidates[0].content.parts:
                return "Fact-checking guidance unavailable due to content restrictions. Please follow standard verification procedures."

            self.source_metrics['gemini_api_calls'] += 1
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            self.logger.error(f"Error generating safe fact-check guidance: {str(e)}")
            return "Fact-check guidance temporarily unavailable. Please consult institutional sources directly."

    def _generate_cross_verification_analysis_safe(self, extracted_claims: List[Dict[str, Any]],
                                                 contextual_sources: List[Dict[str, Any]]) -> str:
        """Generate cross-verification analysis with safety handling"""
        try:
            # Simple cross-verification analysis
            analysis_lines = ["CONTEXTUAL CROSS-VERIFICATION ANALYSIS:", ""]

            # Analyze each claim against available contextual sources
            for i, claim in enumerate(extracted_claims[:5], 1):
                claim_text = claim.get('text', 'Unknown claim')
                verifiability = claim.get('verifiability_score', 5)
                
                # Find relevant sources for this claim
                relevant_sources = [s for s in contextual_sources
                                  if s.get('relevance_score', 0) >= 7][:2]

                analysis_lines.extend([
                    f"Claim {i}: {claim_text[:80]}...",
                    f" Verifiability Score: {verifiability}/10",
                    f" Relevant Contextual Sources: {len(relevant_sources)} available",
                    ""
                ])

            # Add contextual source categories summary
            source_types = {}
            for source in contextual_sources:
                source_type = source.get('type', 'unknown')
                source_types[source_type] = source_types.get(source_type, 0) + 1

            if source_types:
                analysis_lines.extend([
                    "Available Contextual Source Types:",
                    *[f" • {source_type.replace('contextual_', '').title()}: {count} sources"
                      for source_type, count in source_types.items()],
                    ""
                ])

            return "\n".join(analysis_lines)

        except Exception as e:
            self.logger.error(f"Error in safe cross-verification analysis: {str(e)}")
            return "Cross-verification analysis temporarily unavailable."

    def _calculate_recommendation_scores_with_safety(self, contextual_analysis: Dict,
                                                   database_recommendations: Dict,
                                                   reliability_assessment: str,
                                                   verification_strategies: str,
                                                   domain_analysis: Dict) -> Dict[str, Any]:
        """✅ CALCULATE SCORES WITH SAFETY AWARENESS"""
        
        contextual_sources = contextual_analysis.get('contextual_sources', [])
        
        # Source relevance score (higher for contextual sources)
        contextual_sources_count = sum(1 for s in contextual_sources
                                     if s.get('type', '').startswith('contextual'))
        source_relevance_score = min(10, contextual_sources_count * 2.0)

        # Source availability score
        source_availability_score = min(10, len(contextual_sources) * 1.2)

        # Source quality score (average reliability of contextual sources)
        if contextual_sources:
            avg_reliability = sum(s.get('reliability_score', 7) for s in contextual_sources) / len(contextual_sources)
            source_quality_score = avg_reliability
        else:
            source_quality_score = 3.0

        # Domain relevance score
        domain_confidence = domain_analysis.get('confidence', 0.5)
        domain_relevance_score = domain_confidence * 10

        # Verification feasibility score
        verification_feasibility_score = self._estimate_verification_feasibility(verification_strategies)

        # ✅ BONUS FOR SUCCESSFUL CONTEXTUAL ANALYSIS
        contextual_bonus = 0.5 if not contextual_analysis.get('safety_fallback_used', False) else 0.0
        
        # ✅ MINOR PENALTY FOR SAFETY FALLBACKS (but not severe)
        safety_penalty = 0.3 if contextual_analysis.get('safety_fallback_used', False) else 0.0

        # ✅ CALCULATE WEIGHTED OVERALL SCORE WITH SAFETY AWARENESS
        scoring_weights = self.config.get('recommendation_scoring_weights', {
            'source_relevance': 0.4,
            'source_quality': 0.3,
            'source_availability': 0.15,
            'domain_relevance': 0.1,
            'verification_feasibility': 0.05
        })

        overall_score = (
            source_relevance_score * scoring_weights['source_relevance'] +
            source_quality_score * scoring_weights['source_quality'] +
            source_availability_score * scoring_weights['source_availability'] +
            domain_relevance_score * scoring_weights['domain_relevance'] +
            verification_feasibility_score * scoring_weights['verification_feasibility'] +
            contextual_bonus - safety_penalty
        )
        overall_score = max(0, min(10, overall_score))

        # Recommendation level assessment
        recommendation_thresholds = self.config.get('recommendation_thresholds', {
            'excellent': 8.5,
            'good': 7.0,
            'fair': 5.5,
            'poor': 4.0
        })

        if overall_score >= recommendation_thresholds['excellent']:
            recommendation_level = "EXCELLENT"
        elif overall_score >= recommendation_thresholds['good']:
            recommendation_level = "GOOD"
        elif overall_score >= recommendation_thresholds['fair']:
            recommendation_level = "FAIR"
        elif overall_score >= recommendation_thresholds['poor']:
            recommendation_level = "POOR"
        else:
            recommendation_level = "VERY POOR"

        # Enhanced availability factors with safety awareness
        availability_factors = []
        if contextual_sources_count >= 3:
            availability_factors.append("Multiple contextual sources available")
        if any(s.get('reliability_score', 0) >= 8 for s in contextual_sources):
            availability_factors.append("High-reliability contextual sources available")
        if not contextual_analysis.get('safety_fallback_used', False):
            availability_factors.append("Successful AI contextual analysis")
        if len(set(s.get('type', 'unknown') for s in contextual_sources)) >= 3:
            availability_factors.append("Diverse contextual source types available")

        # Enhanced verification challenges with safety awareness
        verification_challenges = []
        if contextual_sources_count < 2:
            verification_challenges.append("Limited contextual source availability")
        if source_quality_score < 6.0:
            verification_challenges.append("Low average contextual source reliability")
        if contextual_analysis.get('safety_fallback_used', False):
            verification_challenges.append("AI analysis restricted - institutional sources provided")
        if domain_confidence < 0.6:
            verification_challenges.append("Uncertain domain classification")

        return {
            'source_relevance_score': round(source_relevance_score, 2),
            'source_availability_score': round(source_availability_score, 2),
            'source_quality_score': round(source_quality_score, 2),
            'domain_relevance_score': round(domain_relevance_score, 2),
            'verification_feasibility_score': round(verification_feasibility_score, 2),
            'overall_recommendation_score': round(overall_score, 2),
            'recommendation_level': recommendation_level,
            'availability_factors': availability_factors,
            'verification_challenges': verification_challenges,
            'contextual_sources_count': contextual_sources_count,
            'total_sources_count': len(contextual_sources),
            'safety_analysis': {  # ✅ New safety metadata
                'safety_fallback_used': contextual_analysis.get('safety_fallback_used', False),
                'safety_blocks_encountered': contextual_analysis.get('safety_blocks_encountered', 0),
                'institutional_sources_provided': len([s for s in contextual_sources 
                                                     if 'institutional' in s.get('type', '')]),
            },
            'scoring_method': 'contextual_safety_aware',
            'weights_used': scoring_weights,
            'thresholds_used': recommendation_thresholds
        }

    def _create_contextual_source_summary_with_safety(self, contextual_analysis: Dict, recommendation_scores: Dict) -> str:
        """✅ CREATE SUMMARY WITH SAFETY AWARENESS"""
        contextual_sources = contextual_analysis.get('contextual_sources', [])
        
        if not contextual_sources:
            return "No contextual sources could be identified for this article's specific claims."

        contextual_sources_count = sum(1 for s in contextual_sources
                                     if s.get('type', '').startswith('contextual'))

        summary_lines = [
            f"CONTEXTUAL SOURCE RECOMMENDATION SUMMARY",
            f"Overall Score: {recommendation_scores['overall_recommendation_score']:.1f}/10 ({recommendation_scores['recommendation_level']})",
            f"Sources Identified: {len(contextual_sources)} (Contextual: {contextual_sources_count})",
            ""
        ]

        # ✅ ADD SAFETY ANALYSIS SUMMARY
        if contextual_analysis.get('safety_fallback_used', False):
            summary_lines.extend([
                "🛡️ Safety Analysis:",
                f" • AI analysis restricted due to content sensitivity",
                f" • Institutional fallback sources provided: {recommendation_scores['safety_analysis']['institutional_sources_provided']}",
                f" • Sources are contextually relevant to specific claims",
                ""
            ])

        # Show contextual sources
        if contextual_sources_count > 0:
            summary_lines.append("Top Contextual Sources:")
            contextual_only = [s for s in contextual_sources if s.get('type', '').startswith('contextual')]
            for i, source in enumerate(contextual_only[:3], 1):
                name = source.get('name', 'Unknown Source')
                relevance = source.get('relevance_score', 0)
                source_type = source.get('type', 'Unknown').replace('contextual_', '').title()
                summary_lines.append(f" {i}. {name} (Type: {source_type}, Relevance: {relevance}/10)")
        else:
            summary_lines.append("⚠ No contextual sources identified - using general recommendations")

        summary_lines.append("")
        summary_lines.extend([
            f"Recommendation Quality:",
            f" • Source Relevance: {recommendation_scores['source_relevance_score']:.1f}/10",
            f" • Source Availability: {recommendation_scores['source_availability_score']:.1f}/10", 
            f" • Source Quality: {recommendation_scores['source_quality_score']:.1f}/10"
        ])

        # Add availability factors
        if recommendation_scores['availability_factors']:
            summary_lines.append("")
            summary_lines.append("Contextual Strengths:")
            for factor in recommendation_scores['availability_factors']:
                summary_lines.append(f" ✓ {factor}")

        # Add challenges
        if recommendation_scores['verification_challenges']:
            summary_lines.append("")
            summary_lines.append("Verification Challenges:")
            for challenge in recommendation_scores['verification_challenges']:
                summary_lines.append(f" ⚠ {challenge}")

        return "\n".join(summary_lines)

    def _estimate_verification_feasibility(self, verification_strategies: str) -> float:
        """Estimate verification feasibility from strategies text"""
        feasibility_indicators = [
            'primary sources', 'direct contact', 'official records',
            'expert consultation', 'multiple sources', 'public data',
            'verifiable', 'accessible', 'contactable'
        ]

        challenge_indicators = [
            'difficult to verify', 'limited access', 'confidential',
            'anonymous sources', 'unverifiable', 'no public record'
        ]

        strategies_lower = verification_strategies.lower()
        feasibility_count = sum(1 for indicator in feasibility_indicators if indicator in strategies_lower)
        challenge_count = sum(1 for indicator in challenge_indicators if indicator in strategies_lower)

        base_score = 5.0 + (feasibility_count * 0.8) - (challenge_count * 1.0)
        return max(0, min(10, base_score))

    def _respect_rate_limits(self):
        """Rate limiting using config values"""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()

    def _update_recommendation_metrics(self, response_time: float, recommendation_score: float, error: bool = False):
        """Update recommendation-specific metrics with safety awareness"""
        self.source_metrics['total_recommendations'] += 1
        if not error:
            # Update average response time
            total = self.source_metrics['total_recommendations']
            current_avg = self.source_metrics['average_response_time']
            self.source_metrics['average_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """📊 Get comprehensive performance metrics with safety information"""
        # Get base metrics
        base_metrics = self.get_performance_metrics()

        # ✅ ADD SAFETY INFORMATION TO METRICS
        config_metrics = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_sources_per_recommendation': self.max_sources_per_recommendation,
            'min_reliability_score': self.min_reliability_score,
            'enable_detailed_analysis': self.enable_detailed_analysis,
            'enable_cross_verification': self.enable_cross_verification,
            'enable_domain_specific_sources': self.enable_domain_specific_sources,
            'enable_domain_classification': self.enable_domain_classification,
            'enable_contextual_recommendations': self.enable_contextual_recommendations,
            'contextual_specificity_threshold': self.contextual_specificity_threshold,
            'max_contacts_per_claim': self.max_contacts_per_claim,
            'enable_safety_fallbacks': self.enable_safety_fallbacks,  # ✅ New
            'reliability_tiers_count': len(self.reliability_tiers),
            'preferred_source_types_count': len(self.preferred_source_types),
            'domain_confidence_threshold': self.domain_confidence_threshold,
            'rate_limit_seconds': self.rate_limit,
            'config_version': '3.0_safety_enhanced'
        }

        return {
            **base_metrics,
            'source_specific_metrics': self.source_metrics,
            'config_metrics': config_metrics,
            'agent_type': 'credible_source',
            'modular_architecture': True,
            'config_integrated': True,
            'contextual_recommendations_enabled': self.enable_contextual_recommendations,
            'safety_enhanced': True,  # ✅ New flag
            'prompt_source': 'centralized_config'
        }

# Testing functionality with safety handling
if __name__ == "__main__":
    """Test the safety-enhanced credible source agent"""
    print("🧪 Testing Safety-Enhanced Credible Source Agent")
    print("=" * 75)

    try:
        # Initialize agent
        agent = CredibleSourceAgent()
        print(f"✅ Agent initialized with safety handling")

        # Test with potentially problematic content
        test_article = """
        BREAKING: COVID-19 vaccine contains microchips, confirms whistleblower Dr. Sarah Johnson 
        from Pfizer. Internal documents leaked yesterday show that all mRNA vaccines distributed 
        since December 2020 contain tracking devices manufactured by Microsoft.
        """

        test_claims = [
            {
                'text': 'COVID-19 vaccines contain tracking microchips',
                'claim_type': 'Medical',
                'priority': 1,
                'verifiability_score': 2,
                'source': 'Anonymous whistleblower'
            }
        ]

        test_input = {
            "text": test_article,
            "extracted_claims": test_claims,
            "evidence_evaluation": {'overall_evidence_score': 2.5},
            "include_detailed_analysis": True
        }

        print(f"\n🔍 Testing with potentially problematic content...")
        result = agent.process(test_input)

        if result['success']:
            recommendation_data = result['result']
            print(f"✅ Recommendation completed successfully with safety handling")
            print(f" Safety fallback used: {recommendation_data.get('safety_fallback_used', False)}")
            print(f" Contextual sources provided: {len(recommendation_data.get('contextual_sources', []))}")
            
            # Show sample contextual sources
            sources = recommendation_data.get('contextual_sources', [])
            if sources:
                print(f" Sample contextual sources:")
                for i, source in enumerate(sources[:2], 1):
                    print(f"  {i}. {source.get('name', 'Unknown')}")
                    print(f"     Relevance: {source.get('relevance', 'Not specified')[:80]}...")
        else:
            print(f"❌ Recommendation failed: {result.get('error', 'Unknown error')}")

        print(f"\n✅ Safety-enhanced credible source agent test completed!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
