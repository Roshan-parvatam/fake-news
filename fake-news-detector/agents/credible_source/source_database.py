# agents/credible_source/source_database.py

"""
Source Reliability Database - Production Ready

Comprehensive database of credible information sources with reliability scoring,
domain-specific categorization, intelligent recommendation algorithms, and
production-level performance tracking and error handling.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .exceptions import (
    SourceDatabaseError,
    InputValidationError,
    raise_source_database_error,
    raise_input_validation_error
)


class SourceReliabilityDatabase:
    """
    Production-ready database of credible information sources.
    
    Manages comprehensive source databases with reliability scoring,
    domain-specific categorization, and intelligent recommendation algorithms
    enhanced for production use with performance tracking and error handling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize source reliability database with production configuration.

        Args:
            config: Optional configuration for source recommendation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.SourceReliabilityDatabase")
        
        try:
            # Initialize source databases with error handling
            self.primary_sources = self._initialize_primary_sources()
            self.expert_sources = self._initialize_expert_sources()
            self.institutional_sources = self._initialize_institutional_sources()
            self.journalistic_sources = self._initialize_journalistic_sources()
            self.fact_check_sources = self._initialize_fact_check_sources()

            # Configurable recommendation parameters
            self.min_reliability_threshold = self.config.get('min_reliability_threshold', 6.0)
            self.max_sources_per_recommendation = self.config.get('max_sources_per_recommendation', 12)
            self.domain_weight_multiplier = self.config.get('domain_weight_multiplier', 2.0)
            self.reliability_weight = self.config.get('reliability_weight', 1.5)
            
            # Performance and usage tracking
            self.recommendation_count = 0
            self.total_processing_time = 0.0
            self.sources_recommended = 0
            self.cache_hits = 0
            self.domain_usage_stats = defaultdict(int)
            self.error_count = 0

            # Calculate total sources for logging
            total_sources = sum([
                len(self.primary_sources),
                len(self.expert_sources),
                len(self.institutional_sources),
                len(self.journalistic_sources),
                len(self.fact_check_sources)
            ])

            self.logger.info(
                f"Source reliability database initialized successfully",
                extra={
                    'total_sources': total_sources,
                    'source_breakdown': {
                        'primary': len(self.primary_sources),
                        'expert': len(self.expert_sources),
                        'institutional': len(self.institutional_sources),
                        'journalistic': len(self.journalistic_sources),
                        'fact_check': len(self.fact_check_sources)
                    },
                    'min_reliability_threshold': self.min_reliability_threshold,
                    'max_sources_per_recommendation': self.max_sources_per_recommendation
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize source reliability database: {str(e)}")
            raise SourceDatabaseError(
                f"Database initialization failed: {str(e)}",
                operation="initialization"
            )

    def _initialize_primary_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize government and official primary sources with enhanced metadata."""
        return {
            'cdc': {
                'name': 'Centers for Disease Control and Prevention',
                'type': 'government',
                'domain': 'health',
                'reliability_score': 9.5,
                'url': 'https://www.cdc.gov',
                'contact_info': 'Public inquiries: 1-800-CDC-INFO (1-800-232-4636)',
                'verification_methods': ['official_statements', 'data_requests', 'press_releases', 'morbidity_reports'],
                'expertise_areas': ['public health', 'disease prevention', 'epidemiology', 'vaccines', 'health_surveillance'],
                'response_time': 'fast',
                'data_access': 'public',
                'languages': ['english', 'spanish'],
                'established': 1946
            },
            'fda': {
                'name': 'U.S. Food and Drug Administration',
                'type': 'government',
                'domain': 'health',
                'reliability_score': 9.4,
                'url': 'https://www.fda.gov',
                'contact_info': 'Media relations: fdaoma@fda.hhs.gov',
                'verification_methods': ['approval_databases', 'official_announcements', 'regulatory_documents', 'inspection_reports'],
                'expertise_areas': ['drug_approval', 'medical_devices', 'food_safety', 'tobacco_regulation', 'clinical_trials'],
                'response_time': 'moderate',
                'data_access': 'public',
                'languages': ['english'],
                'established': 1906
            },
            'census_bureau': {
                'name': 'U.S. Census Bureau',
                'type': 'government',
                'domain': 'economics',
                'reliability_score': 9.8,
                'url': 'https://www.census.gov',
                'contact_info': 'Public Information Office: pio@census.gov',
                'verification_methods': ['official_data', 'public_records', 'statistical_releases', 'survey_data'],
                'expertise_areas': ['population_data', 'economic_statistics', 'housing_data', 'business_statistics', 'demographic_analysis'],
                'response_time': 'fast',
                'data_access': 'public',
                'languages': ['english', 'spanish'],
                'established': 1790
            },
            'bls': {
                'name': 'Bureau of Labor Statistics',
                'type': 'government',
                'domain': 'economics',
                'reliability_score': 9.7,
                'url': 'https://www.bls.gov',
                'contact_info': 'Media relations: blspress@bls.gov',
                'verification_methods': ['official_statistics', 'labor_reports', 'economic_indicators', 'survey_methodology'],
                'expertise_areas': ['employment', 'inflation', 'wages', 'productivity', 'workplace_safety', 'consumer_prices'],
                'response_time': 'fast',
                'data_access': 'public',
                'languages': ['english'],
                'established': 1884
            },
            'epa': {
                'name': 'Environmental Protection Agency',
                'type': 'government',
                'domain': 'environment',
                'reliability_score': 9.2,
                'url': 'https://www.epa.gov',
                'contact_info': 'Public affairs: press@epa.gov',
                'verification_methods': ['environmental_data', 'regulatory_documents', 'scientific_studies', 'monitoring_reports'],
                'expertise_areas': ['air_quality', 'water_quality', 'chemical_safety', 'climate_change', 'environmental_monitoring'],
                'response_time': 'moderate',
                'data_access': 'public',
                'languages': ['english', 'spanish'],
                'established': 1970
            }
        }

    def _initialize_expert_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize expert and professional organization sources."""
        return {
            'who': {
                'name': 'World Health Organization',
                'type': 'international_organization',
                'domain': 'health',
                'reliability_score': 9.0,
                'url': 'https://www.who.int',
                'contact_info': 'Media centre: mediainquiries@who.int',
                'verification_methods': ['official_statements', 'expert_committees', 'global_health_data', 'technical_reports'],
                'expertise_areas': ['global_health', 'disease_outbreaks', 'health_policy', 'health_emergencies', 'international_health_regulations'],
                'response_time': 'moderate',
                'data_access': 'public',
                'languages': ['english', 'french', 'spanish', 'arabic', 'chinese', 'russian'],
                'established': 1948
            },
            'ama': {
                'name': 'American Medical Association',
                'type': 'professional_organization',
                'domain': 'health',
                'reliability_score': 8.5,
                'url': 'https://www.ama-assn.org',
                'contact_info': 'Media relations: media@ama-assn.org',
                'verification_methods': ['position_statements', 'expert_panels', 'medical_guidelines', 'peer_review'],
                'expertise_areas': ['medical_practice', 'healthcare_policy', 'medical_ethics', 'physician_advocacy', 'medical_education'],
                'response_time': 'moderate',
                'data_access': 'members_public',
                'languages': ['english'],
                'established': 1847
            },
            'aaas': {
                'name': 'American Association for the Advancement of Science',
                'type': 'professional_organization',
                'domain': 'science',
                'reliability_score': 8.8,
                'url': 'https://www.aaas.org',
                'contact_info': 'Communications: media@aaas.org',
                'verification_methods': ['expert_directory', 'policy_statements', 'scientific_reports', 'peer_review_process'],
                'expertise_areas': ['scientific_research', 'science_policy', 'STEM_education', 'research_ethics', 'science_communication'],
                'response_time': 'moderate',
                'data_access': 'public_members',
                'languages': ['english'],
                'established': 1848
            },
            'ieee': {
                'name': 'Institute of Electrical and Electronics Engineers',
                'type': 'professional_organization',
                'domain': 'technology',
                'reliability_score': 8.7,
                'url': 'https://www.ieee.org',
                'contact_info': 'Media relations: media@ieee.org',
                'verification_methods': ['standards_documents', 'expert_networks', 'technical_publications', 'conference_proceedings'],
                'expertise_areas': ['electrical_engineering', 'computer_science', 'telecommunications', 'robotics', 'technology_standards'],
                'response_time': 'moderate',
                'data_access': 'members_public',
                'languages': ['english'],
                'established': 1963
            }
        }

    def _initialize_institutional_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize academic and research institutional sources."""
        return {
            'harvard_med': {
                'name': 'Harvard Medical School',
                'type': 'academic',
                'domain': 'health',
                'reliability_score': 9.2,
                'url': 'https://hms.harvard.edu',
                'contact_info': 'Media relations: media@hms.harvard.edu',
                'verification_methods': ['published_research', 'expert_contacts', 'medical_studies', 'clinical_trials'],
                'expertise_areas': ['medical_research', 'clinical_medicine', 'biomedical_science', 'public_health', 'medical_education'],
                'response_time': 'moderate',
                'data_access': 'academic_public',
                'languages': ['english'],
                'established': 1782
            },
            'mit': {
                'name': 'Massachusetts Institute of Technology',
                'type': 'academic',
                'domain': 'technology',
                'reliability_score': 9.3,
                'url': 'https://www.mit.edu',
                'contact_info': 'News office: newsoffice@mit.edu',
                'verification_methods': ['research_publications', 'expert_interviews', 'technical_reports', 'laboratory_data'],
                'expertise_areas': ['engineering', 'computer_science', 'artificial_intelligence', 'biotechnology', 'energy_research'],
                'response_time': 'fast',
                'data_access': 'academic_public',
                'languages': ['english'],
                'established': 1861
            },
            'stanford': {
                'name': 'Stanford University',
                'type': 'academic',
                'domain': 'general',
                'reliability_score': 9.1,
                'url': 'https://www.stanford.edu',
                'contact_info': 'Media relations: media@stanford.edu',
                'verification_methods': ['faculty_experts', 'research_papers', 'academic_studies', 'departmental_contacts'],
                'expertise_areas': ['medicine', 'engineering', 'computer_science', 'business', 'policy', 'interdisciplinary_research'],
                'response_time': 'fast',
                'data_access': 'academic_public',
                'languages': ['english'],
                'established': 1885
            },
            'mayo_clinic': {
                'name': 'Mayo Clinic',
                'type': 'medical_institution',
                'domain': 'health',
                'reliability_score': 9.0,
                'url': 'https://www.mayoclinic.org',
                'contact_info': 'Public affairs: publicaffairs@mayo.edu',
                'verification_methods': ['medical_experts', 'clinical_data', 'patient_care_guidelines', 'medical_research'],
                'expertise_areas': ['clinical_medicine', 'medical_research', 'patient_care', 'medical_education', 'specialty_care'],
                'response_time': 'moderate',
                'data_access': 'public_professional',
                'languages': ['english'],
                'established': 1889
            },
            'brookings': {
                'name': 'Brookings Institution',
                'type': 'think_tank',
                'domain': 'politics',
                'reliability_score': 8.2,
                'url': 'https://www.brookings.edu',
                'contact_info': 'Communications: communications@brookings.edu',
                'verification_methods': ['research_reports', 'expert_analysis', 'policy_papers', 'data_analysis'],
                'expertise_areas': ['economic_policy', 'foreign_policy', 'governance', 'metropolitan_policy', 'technology_policy'],
                'response_time': 'fast',
                'data_access': 'public',
                'languages': ['english'],
                'established': 1916
            }
        }

    def _initialize_journalistic_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize high-quality news organizations."""
        return {
            'reuters': {
                'name': 'Reuters',
                'type': 'news_agency',
                'domain': 'general',
                'reliability_score': 8.8,
                'url': 'https://www.reuters.com',
                'contact_info': 'Newsroom: reuters.newsroom@reuters.com',
                'verification_methods': ['fact_checking_team', 'source_verification', 'editorial_standards', 'multiple_sourcing'],
                'expertise_areas': ['breaking_news', 'financial_markets', 'international_affairs', 'politics', 'business_reporting'],
                'response_time': 'very_fast',
                'data_access': 'public_subscription',
                'languages': ['english', 'multiple'],
                'established': 1851
            },
            'ap_news': {
                'name': 'Associated Press',
                'type': 'news_agency',
                'domain': 'general',
                'reliability_score': 8.9,
                'url': 'https://apnews.com',
                'contact_info': 'Media relations: apcorporatecomm@ap.org',
                'verification_methods': ['fact_checking', 'source_corroboration', 'editorial_review', 'wire_verification'],
                'expertise_areas': ['news_reporting', 'fact_checking', 'wire_services', 'photography', 'breaking_news'],
                'response_time': 'very_fast',
                'data_access': 'public_subscription',
                'languages': ['english', 'multiple'],
                'established': 1846
            },
            'bbc': {
                'name': 'BBC News',
                'type': 'broadcaster',
                'domain': 'general',
                'reliability_score': 8.5,
                'url': 'https://www.bbc.com/news',
                'contact_info': 'Press office: press.office@bbc.co.uk',
                'verification_methods': ['editorial_standards', 'fact_checking', 'multiple_sourcing', 'impartiality_guidelines'],
                'expertise_areas': ['international_news', 'UK_affairs', 'analysis', 'investigative_journalism', 'world_service'],
                'response_time': 'fast',
                'data_access': 'public',
                'languages': ['english', 'multiple'],
                'established': 1922
            },
            'npr': {
                'name': 'National Public Radio',
                'type': 'broadcaster',
                'domain': 'general',
                'reliability_score': 8.3,
                'url': 'https://www.npr.org',
                'contact_info': 'Media relations: press@npr.org',
                'verification_methods': ['editorial_guidelines', 'source_verification', 'public_radio_standards', 'transparency_reporting'],
                'expertise_areas': ['news_analysis', 'cultural_coverage', 'science_reporting', 'politics', 'investigative_reporting'],
                'response_time': 'fast',
                'data_access': 'public',
                'languages': ['english'],
                'established': 1970
            }
        }

    def _initialize_fact_check_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dedicated fact-checking organizations."""
        return {
            'snopes': {
                'name': 'Snopes',
                'type': 'fact_checker',
                'domain': 'general',
                'reliability_score': 8.0,
                'url': 'https://www.snopes.com',
                'contact_info': 'Editorial team: editors@snopes.com',
                'verification_methods': ['fact_verification', 'source_investigation', 'claim_analysis', 'evidence_documentation'],
                'expertise_areas': ['urban_legends', 'viral_claims', 'political_fact_checking', 'misinformation', 'internet_rumors'],
                'response_time': 'moderate',
                'data_access': 'public_subscription',
                'languages': ['english'],
                'established': 1995
            },
            'politifact': {
                'name': 'PolitiFact',
                'type': 'fact_checker',
                'domain': 'politics',
                'reliability_score': 8.2,
                'url': 'https://www.politifact.com',
                'contact_info': 'Editorial office: editor@politifact.com',
                'verification_methods': ['truth_o_meter', 'source_checking', 'political_verification', 'database_research'],
                'expertise_areas': ['political_claims', 'campaign_promises', 'policy_analysis', 'election_coverage', 'government_accountability'],
                'response_time': 'moderate',
                'data_access': 'public',
                'languages': ['english'],
                'established': 2007
            },
            'factcheck_org': {
                'name': 'FactCheck.org',
                'type': 'fact_checker',
                'domain': 'politics',
                'reliability_score': 8.4,
                'url': 'https://www.factcheck.org',
                'contact_info': 'Annenberg Public Policy Center: factcheck@annenberg.upenn.edu',
                'verification_methods': ['political_fact_checking', 'source_analysis', 'claim_verification', 'academic_backing'],
                'expertise_areas': ['political_accuracy', 'campaign_advertising', 'policy_claims', 'political_rhetoric', 'government_statements'],
                'response_time': 'moderate',
                'data_access': 'public',
                'languages': ['english'],
                'established': 2003
            },
            'ap_fact_check': {
                'name': 'AP Fact Check',
                'type': 'fact_checker',
                'domain': 'general',
                'reliability_score': 8.6,
                'url': 'https://apnews.com/hub/ap-fact-check',
                'contact_info': 'AP News fact-check team: factcheck@ap.org',
                'verification_methods': ['comprehensive_fact_checking', 'source_verification', 'claim_analysis', 'editorial_review'],
                'expertise_areas': ['news_verification', 'claim_checking', 'misinformation_debunking', 'source_validation', 'breaking_news_verification'],
                'response_time': 'fast',
                'data_access': 'public',
                'languages': ['english', 'multiple'],
                'established': 2016
            }
        }

    def get_source_recommendations(self,
                                 article_text: str,
                                 extracted_claims: List[Dict[str, Any]],
                                 domain: str = 'general',
                                 session_id: str = None) -> Dict[str, Any]:
        """
        Generate source recommendations with comprehensive analysis and error handling.

        Args:
            article_text: Article content to analyze
            extracted_claims: Claims extracted from the article
            domain: Primary domain classification
            session_id: Optional session ID for tracking

        Returns:
            Dictionary containing recommended sources and analysis metadata
        """
        start_time = time.time()
        
        self.logger.info(
            f"Starting source recommendation generation",
            extra={
                'session_id': session_id,
                'domain': domain,
                'article_length': len(article_text) if article_text else 0,
                'claims_count': len(extracted_claims) if extracted_claims else 0
            }
        )

        try:
            # Input validation
            if not isinstance(article_text, str):
                raise_input_validation_error(
                    "article_text",
                    f"Must be string, got {type(article_text).__name__}",
                    article_text,
                    session_id
                )

            if not article_text.strip():
                raise_input_validation_error(
                    "article_text",
                    "Cannot be empty",
                    article_text,
                    session_id
                )

            if not isinstance(extracted_claims, list):
                self.logger.warning(
                    f"Invalid claims format: {type(extracted_claims).__name__}, using empty list",
                    extra={'session_id': session_id}
                )
                extracted_claims = []

            # Combine all source databases
            all_sources = self._combine_all_sources()

            # Apply configuration filters
            min_reliability = self.min_reliability_threshold
            max_sources = self.max_sources_per_recommendation

            self.logger.debug(
                f"Filtering {len(all_sources)} sources",
                extra={
                    'session_id': session_id,
                    'min_reliability': min_reliability,
                    'max_sources': max_sources,
                    'target_domain': domain
                }
            )

            # Filter and score sources with error handling
            relevant_sources = []
            
            for source_key, source_info in all_sources.items():
                try:
                    # Domain relevance filter
                    source_domain = source_info.get('domain', 'general')
                    if domain != 'general' and source_domain not in [domain, 'general']:
                        continue

                    # Reliability threshold filter
                    if source_info.get('reliability_score', 0) < min_reliability:
                        continue

                    # Calculate relevance score
                    relevance_score = self._calculate_source_relevance(
                        article_text, extracted_claims, source_info, session_id
                    )

                    # Create recommendation entry
                    source_recommendation = {
                        **source_info,
                        'source_key': source_key,
                        'relevance_score': relevance_score,
                        'domain_match': source_domain == domain or source_domain == 'general',
                        'combined_score': self._calculate_combined_score(
                            source_info.get('reliability_score', 0), 
                            relevance_score
                        )
                    }

                    relevant_sources.append(source_recommendation)

                except Exception as source_error:
                    self.logger.warning(
                        f"Error processing source {source_key}: {str(source_error)}",
                        extra={'session_id': session_id, 'source_key': source_key}
                    )
                    continue

            # Sort by combined score and limit results
            relevant_sources.sort(key=lambda x: x['combined_score'], reverse=True)
            recommended_sources = relevant_sources[:max_sources]

            # Categorize sources by type
            source_categories = self._categorize_sources(recommended_sources)

            # Calculate recommendation quality metrics
            quality_metrics = self._calculate_quality_metrics(recommended_sources, all_sources)

            # Update performance tracking
            processing_time = time.time() - start_time
            self.recommendation_count += 1
            self.total_processing_time += processing_time
            self.sources_recommended += len(recommended_sources)
            self.domain_usage_stats[domain] += 1

            self.logger.info(
                f"Source recommendation completed successfully",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time * 1000, 2),
                    'sources_recommended': len(recommended_sources),
                    'domain': domain,
                    'quality_score': quality_metrics.get('average_reliability', 0)
                }
            )

            return {
                'recommended_sources': recommended_sources,
                'source_categories': source_categories,
                'quality_metrics': quality_metrics,
                'domain_analyzed': domain,
                'total_sources_available': len(all_sources),
                'sources_recommended': len(recommended_sources),
                'min_reliability_used': min_reliability,
                'processing_time_ms': round(processing_time * 1000, 2),
                'session_id': session_id,
                'recommendation_metadata': {
                    'filtering_applied': True,
                    'scoring_method': 'relevance_reliability_combined',
                    'domain_weighting': self.domain_weight_multiplier,
                    'reliability_weighting': self.reliability_weight
                }
            }

        except (InputValidationError, SourceDatabaseError):
            # Re-raise validation and database errors
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            self.error_count += 1
            
            self.logger.error(
                f"Source recommendation failed: {str(e)}",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time * 1000, 2),
                    'error_type': type(e).__name__
                }
            )
            
            raise SourceDatabaseError(
                f"Source recommendation generation failed: {str(e)}",
                operation="recommendation_generation",
                session_id=session_id
            )

    def _combine_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """Combine all source databases into a single dictionary."""
        all_sources = {}
        all_sources.update(self.primary_sources)
        all_sources.update(self.expert_sources)
        all_sources.update(self.institutional_sources)
        all_sources.update(self.journalistic_sources)
        all_sources.update(self.fact_check_sources)
        return all_sources

    def _calculate_source_relevance(self,
                                  article_text: str,
                                  extracted_claims: List[Dict[str, Any]],
                                  source_info: Dict[str, Any],
                                  session_id: str = None) -> float:
        """Calculate source relevance with enhanced scoring algorithm."""
        try:
            relevance_score = 5.0  # Base score
            article_lower = article_text.lower()
            
            source_type = source_info.get('type', '')
            source_domain = source_info.get('domain', 'general')
            expertise_areas = source_info.get('expertise_areas', [])

            # Domain matching bonus
            if source_domain != 'general' and source_domain in article_lower:
                relevance_score += 2.0 * self.domain_weight_multiplier

            # Expertise area matching with weighted scoring
            expertise_matches = 0
            for area in expertise_areas:
                area_keywords = area.replace('_', ' ').split()
                for keyword in area_keywords:
                    if keyword in article_lower and len(keyword) > 3:
                        relevance_score += 1.2
                        expertise_matches += 1
                        break  # Only count each area once

            # Source type relevance weighting based on credibility and response time
            type_weights = {
                'government': 2.2,
                'academic': 1.8,
                'international_organization': 1.6,
                'professional_organization': 1.4,
                'medical_institution': 1.9,
                'think_tank': 1.2,
                'news_agency': 1.0,
                'broadcaster': 0.9,
                'fact_checker': 1.3
            }
            
            relevance_score += type_weights.get(source_type, 0.5)

            # Response time bonus (faster response = higher relevance)
            response_time = source_info.get('response_time', 'moderate')
            response_bonuses = {
                'very_fast': 1.0,
                'fast': 0.7,
                'moderate': 0.3,
                'slow': 0.0
            }
            relevance_score += response_bonuses.get(response_time, 0)

            # Claim type matching analysis with priority weighting
            claim_relevance = 0
            for claim in extracted_claims[:8]:  # Limit for performance
                try:
                    claim_type = claim.get('claim_type', '').lower()
                    claim_text = claim.get('text', '').lower()
                    priority = claim.get('priority', 3)
                    
                    # Priority weight: higher priority = more influence
                    priority_weight = max(0.5, 2.0 - (priority * 0.3))
                    
                    # Match claim types to source expertise
                    if claim_type in ['research', 'statistical'] and source_type in ['academic', 'government']:
                        claim_relevance += 1.2 * priority_weight
                    elif claim_type == 'medical' and source_type in ['medical_institution', 'government']:
                        claim_relevance += 1.5 * priority_weight
                    elif claim_type == 'political' and source_type in ['government', 'think_tank']:
                        claim_relevance += 1.0 * priority_weight
                    elif claim_type == 'factual' and source_type == 'fact_checker':
                        claim_relevance += 0.8 * priority_weight
                    
                    # Match specific expertise areas to claim content
                    for area in expertise_areas:
                        area_clean = area.replace('_', ' ')
                        if area_clean in claim_text:
                            claim_relevance += 0.7 * priority_weight
                            break
                            
                except Exception as claim_error:
                    self.logger.debug(
                        f"Error processing claim in relevance calculation: {str(claim_error)}",
                        extra={'session_id': session_id}
                    )
                    continue

            relevance_score += min(claim_relevance, 3.0)  # Cap claim bonus

            # Established institution bonus (older = more established)
            established_year = source_info.get('established')
            if established_year and isinstance(established_year, int):
                age = 2025 - established_year
                if age > 50:
                    relevance_score += 0.8
                elif age > 20:
                    relevance_score += 0.4

            return min(10.0, max(0.0, relevance_score))
            
        except Exception as e:
            self.logger.warning(
                f"Error calculating source relevance: {str(e)}",
                extra={'session_id': session_id}
            )
            return 5.0  # Default relevance score

    def _calculate_combined_score(self, reliability_score: float, relevance_score: float) -> float:
        """Calculate combined score from reliability and relevance."""
        # Weighted combination with configurable weights
        combined_score = (
            reliability_score * self.reliability_weight +
            relevance_score * (2.0 - self.reliability_weight)
        ) / 2.0
        
        return round(combined_score, 2)

    def _categorize_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize sources by type and domain with detailed breakdown."""
        categories = {
            'by_type': defaultdict(int),
            'by_domain': defaultdict(int),
            'by_response_time': defaultdict(int)
        }

        for source in sources:
            source_type = source.get('type', 'unknown')
            source_domain = source.get('domain', 'general')
            response_time = source.get('response_time', 'unknown')
            
            categories['by_type'][source_type] += 1
            categories['by_domain'][source_domain] += 1
            categories['by_response_time'][response_time] += 1

        # Convert defaultdicts to regular dicts for JSON serialization
        return {
            'by_type': dict(categories['by_type']),
            'by_domain': dict(categories['by_domain']),
            'by_response_time': dict(categories['by_response_time']),
            'total_categories': len(categories['by_type'])
        }

    def _calculate_quality_metrics(self,
                                 recommended_sources: List[Dict[str, Any]],
                                 all_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive recommendation quality metrics."""
        try:
            if not recommended_sources:
                return {
                    'average_reliability': 0.0,
                    'average_relevance': 0.0,
                    'source_diversity': 0.0,
                    'coverage_ratio': 0.0,
                    'quality_distribution': {},
                    'response_time_distribution': {}
                }

            # Calculate average scores
            avg_reliability = sum(s.get('reliability_score', 0) for s in recommended_sources) / len(recommended_sources)
            avg_relevance = sum(s.get('relevance_score', 0) for s in recommended_sources) / len(recommended_sources)
            avg_combined = sum(s.get('combined_score', 0) for s in recommended_sources) / len(recommended_sources)

            # Calculate source diversity (unique types)
            unique_types = len(set(s.get('type', 'unknown') for s in recommended_sources))
            max_possible_types = len(set(s.get('type', 'unknown') for s in all_sources.values()))
            source_diversity = unique_types / max_possible_types if max_possible_types > 0 else 0

            # Calculate coverage ratio (recommended vs available)
            coverage_ratio = len(recommended_sources) / len(all_sources) if all_sources else 0

            # Quality distribution analysis
            quality_ranges = {'high': 0, 'medium': 0, 'low': 0}
            for source in recommended_sources:
                reliability = source.get('reliability_score', 0)
                if reliability >= 8.5:
                    quality_ranges['high'] += 1
                elif reliability >= 7.0:
                    quality_ranges['medium'] += 1
                else:
                    quality_ranges['low'] += 1

            # Response time distribution
            response_times = defaultdict(int)
            for source in recommended_sources:
                response_time = source.get('response_time', 'unknown')
                response_times[response_time] += 1

            return {
                'average_reliability': round(avg_reliability, 2),
                'average_relevance': round(avg_relevance, 2),
                'average_combined_score': round(avg_combined, 2),
                'source_diversity': round(source_diversity, 2),
                'coverage_ratio': round(coverage_ratio, 3),
                'quality_distribution': dict(quality_ranges),
                'response_time_distribution': dict(response_times),
                'unique_source_types': unique_types,
                'total_source_types_available': max_possible_types,
                'recommendation_strength': 'high' if avg_combined >= 8.0 else 'medium' if avg_combined >= 6.5 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {str(e)}")
            return {
                'average_reliability': 0.0,
                'average_relevance': 0.0,
                'source_diversity': 0.0,
                'coverage_ratio': 0.0,
                'error': str(e)
            }

    def get_sources_by_domain(self, 
                            domain: str, 
                            min_reliability: float = 7.0,
                            session_id: str = None) -> List[Dict[str, Any]]:
        """
        Get all sources for a specific domain above reliability threshold.

        Args:
            domain: Domain to filter by
            min_reliability: Minimum reliability score threshold
            session_id: Optional session ID for tracking

        Returns:
            List of sources matching domain and reliability criteria
        """
        try:
            all_sources = self._combine_all_sources()
            domain_sources = []

            for source_key, source_info in all_sources.items():
                source_domain = source_info.get('domain', 'general')
                reliability = source_info.get('reliability_score', 0)

                if ((domain == 'general' or source_domain in [domain, 'general']) and 
                    reliability >= min_reliability):
                    
                    source_data = source_info.copy()
                    source_data['source_key'] = source_key
                    domain_sources.append(source_data)

            # Sort by reliability score
            domain_sources.sort(key=lambda x: x.get('reliability_score', 0), reverse=True)

            self.logger.info(
                f"Retrieved {len(domain_sources)} sources for domain: {domain}",
                extra={
                    'session_id': session_id,
                    'domain': domain,
                    'min_reliability': min_reliability,
                    'sources_found': len(domain_sources)
                }
            )

            return domain_sources
            
        except Exception as e:
            self.logger.error(
                f"Error retrieving sources by domain: {str(e)}",
                extra={'session_id': session_id, 'domain': domain}
            )
            raise SourceDatabaseError(
                f"Failed to retrieve sources for domain {domain}: {str(e)}",
                operation="domain_filter",
                session_id=session_id
            )

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics and performance metrics."""
        try:
            total_sources = sum([
                len(self.primary_sources),
                len(self.expert_sources),
                len(self.institutional_sources),
                len(self.journalistic_sources),
                len(self.fact_check_sources)
            ])

            source_breakdown = {
                'primary_sources': len(self.primary_sources),
                'expert_sources': len(self.expert_sources),
                'institutional_sources': len(self.institutional_sources),
                'journalistic_sources': len(self.journalistic_sources),
                'fact_check_sources': len(self.fact_check_sources)
            }

            # Performance statistics
            avg_processing_time = (
                self.total_processing_time / self.recommendation_count
                if self.recommendation_count > 0 else 0
            )

            avg_sources_per_recommendation = (
                self.sources_recommended / self.recommendation_count
                if self.recommendation_count > 0 else 0
            )

            # Domain usage statistics
            most_requested_domain = 'none'
            if self.domain_usage_stats:
                most_requested_domain = max(
                    self.domain_usage_stats, 
                    key=self.domain_usage_stats.get
                )

            return {
                'database_info': {
                    'total_sources': total_sources,
                    'source_breakdown': source_breakdown,
                    'domains_supported': ['health', 'science', 'technology', 'politics', 'economics', 'environment', 'general'],
                    'source_types_available': ['government', 'academic', 'institutional', 'journalistic', 'fact_checker']
                },
                'performance_metrics': {
                    'recommendations_completed': self.recommendation_count,
                    'sources_recommended_total': self.sources_recommended,
                    'average_processing_time_ms': round(avg_processing_time * 1000, 2),
                    'average_sources_per_recommendation': round(avg_sources_per_recommendation, 2),
                    'cache_hits': self.cache_hits,
                    'error_count': self.error_count,
                    'error_rate_percent': round((self.error_count / max(self.recommendation_count, 1)) * 100, 2)
                },
                'usage_statistics': {
                    'domain_usage_stats': dict(self.domain_usage_stats),
                    'most_requested_domain': most_requested_domain,
                    'total_domain_requests': sum(self.domain_usage_stats.values())
                },
                'configuration': {
                    'min_reliability_threshold': self.min_reliability_threshold,
                    'max_sources_per_recommendation': self.max_sources_per_recommendation,
                    'domain_weight_multiplier': self.domain_weight_multiplier,
                    'reliability_weight': self.reliability_weight
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating database statistics: {str(e)}")
            return {
                'database_info': {'total_sources': 0, 'error': str(e)},
                'performance_metrics': {'error': str(e)},
                'usage_statistics': {'error': str(e)},
                'configuration': {'error': str(e)}
            }

    def validate_database_integrity(self) -> Dict[str, Any]:
        """Validate database integrity and completeness with detailed reporting."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }

        try:
            all_databases = [
                ('primary_sources', self.primary_sources),
                ('expert_sources', self.expert_sources),
                ('institutional_sources', self.institutional_sources),
                ('journalistic_sources', self.journalistic_sources),
                ('fact_check_sources', self.fact_check_sources)
            ]

            total_sources_checked = 0
            valid_sources = 0

            for db_name, db_data in all_databases:
                if not db_data:
                    validation_results['issues'].append(f"Empty database: {db_name}")
                    validation_results['valid'] = False
                    continue

                for source_key, source_info in db_data.items():
                    total_sources_checked += 1
                    
                    # Check required fields
                    required_fields = ['name', 'type', 'domain', 'reliability_score', 'url']
                    missing_fields = []
                    
                    for field in required_fields:
                        if field not in source_info:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        validation_results['issues'].append(
                            f"Source {source_key} missing fields: {missing_fields}"
                        )
                        validation_results['valid'] = False
                        continue

                    # Validate reliability score
                    reliability = source_info.get('reliability_score', 0)
                    if not isinstance(reliability, (int, float)) or not (0 <= reliability <= 10):
                        validation_results['issues'].append(
                            f"Invalid reliability score for {source_key}: {reliability}"
                        )
                        validation_results['valid'] = False
                        continue

                    # Check URL format
                    url = source_info.get('url', '')
                    if url and not (url.startswith('http://') or url.startswith('https://')):
                        validation_results['warnings'].append(
                            f"Invalid URL format for {source_key}: {url}"
                        )

                    # Check expertise areas
                    expertise_areas = source_info.get('expertise_areas', [])
                    if not isinstance(expertise_areas, list) or len(expertise_areas) == 0:
                        validation_results['warnings'].append(
                            f"No expertise areas defined for {source_key}"
                        )

                    valid_sources += 1

            # Generate validation statistics
            validation_results['statistics'] = {
                'total_sources_checked': total_sources_checked,
                'valid_sources': valid_sources,
                'validation_rate_percent': round((valid_sources / max(total_sources_checked, 1)) * 100, 2),
                'databases_checked': len(all_databases),
                'issues_found': len(validation_results['issues']),
                'warnings_found': len(validation_results['warnings'])
            }

            self.logger.info(
                f"Database integrity validation completed: {'PASSED' if validation_results['valid'] else 'FAILED'}",
                extra={
                    'total_sources': total_sources_checked,
                    'valid_sources': valid_sources,
                    'issues_count': len(validation_results['issues']),
                    'warnings_count': len(validation_results['warnings'])
                }
            )

        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Database integrity validation failed: {str(e)}")

        return validation_results


# Testing functionality
if __name__ == "__main__":
    """Test source database functionality with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test configuration
    test_config = {
        'min_reliability_threshold': 8.0,
        'max_sources_per_recommendation': 8,
        'domain_weight_multiplier': 2.5,
        'reliability_weight': 1.8
    }
    
    print("=== SOURCE DATABASE TEST ===")
    
    try:
        database = SourceReliabilityDatabase(test_config)
        test_session_id = "source_db_test_456"
        
        # Test health domain source recommendations
        test_article = """
        A new medical study published in Nature Medicine shows significant
        improvements in treatment outcomes. The research was conducted by
        Harvard Medical School in collaboration with the CDC. The FDA is
        reviewing the clinical trial data for potential approval.
        """
        
        test_claims = [
            {
                'text': 'Medical study published in Nature Medicine',
                'claim_type': 'Research',
                'priority': 1,
                'verifiability_score': 8.5
            },
            {
                'text': 'Harvard Medical School collaboration with CDC',
                'claim_type': 'Attribution',
                'priority': 2,
                'verifiability_score': 9.0
            },
            {
                'text': 'FDA reviewing clinical trial data',
                'claim_type': 'Medical',
                'priority': 1,
                'verifiability_score': 8.0
            }
        ]
        
        print("\n--- Health Domain Recommendation Test ---")
        recommendations = database.get_source_recommendations(
            test_article, test_claims, 'health', test_session_id
        )
        
        print(f"Sources recommended: {recommendations['sources_recommended']}")
        print(f"Domain analyzed: {recommendations['domain_analyzed']}")
        print(f"Processing time: {recommendations['processing_time_ms']:.1f}ms")
        print(f"Quality score: {recommendations['quality_metrics']['average_reliability']:.1f}/10")
        
        print(f"\nTop recommended sources:")
        for i, source in enumerate(recommendations['recommended_sources'][:5], 1):
            print(f"{i}. {source['name']}")
            print(f"   Type: {source['type']} | Reliability: {source['reliability_score']}/10")
            print(f"   Relevance: {source['relevance_score']:.1f}/10 | Combined: {source['combined_score']:.1f}/10")
            print(f"   Response time: {source.get('response_time', 'unknown')}")
        
        print(f"\nSource categories:")
        categories = recommendations['source_categories']
        print(f"  By type: {categories['by_type']}")
        print(f"  By domain: {categories['by_domain']}")
        print(f"  By response time: {categories['by_response_time']}")
        
        print(f"\nQuality metrics:")
        metrics = recommendations['quality_metrics']
        print(f"  Average reliability: {metrics['average_reliability']:.1f}/10")
        print(f"  Average relevance: {metrics['average_relevance']:.1f}/10")
        print(f"  Source diversity: {metrics['source_diversity']:.2f}")
        print(f"  Quality distribution: {metrics['quality_distribution']}")
        print(f"  Recommendation strength: {metrics['recommendation_strength']}")

        # Test domain-specific source retrieval
        print(f"\n--- Domain-Specific Source Retrieval Test ---")
        health_sources = database.get_sources_by_domain('health', min_reliability=8.5, session_id=test_session_id)
        print(f"High-reliability health sources: {len(health_sources)}")
        
        for source in health_sources[:3]:
            print(f"  • {source['name']} ({source['reliability_score']}/10)")

        # Test database statistics
        stats = database.get_database_statistics()
        print(f"\n--- Database Statistics ---")
        print(f"Total sources: {stats['database_info']['total_sources']}")
        print(f"Source breakdown: {stats['database_info']['source_breakdown']}")
        print(f"Recommendations completed: {stats['performance_metrics']['recommendations_completed']}")
        print(f"Average processing time: {stats['performance_metrics']['average_processing_time_ms']:.1f}ms")
        print(f"Average sources per recommendation: {stats['performance_metrics']['average_sources_per_recommendation']:.1f}")
        print(f"Most requested domain: {stats['usage_statistics']['most_requested_domain']}")
        print(f"Error rate: {stats['performance_metrics']['error_rate_percent']:.2f}%")

        # Test database validation
        validation = database.validate_database_integrity()
        print(f"\n--- Database Validation ---")
        print(f"Database valid: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
        print(f"Sources checked: {validation['statistics']['total_sources_checked']}")
        print(f"Valid sources: {validation['statistics']['valid_sources']}")
        print(f"Validation rate: {validation['statistics']['validation_rate_percent']:.1f}%")
        
        if validation['issues']:
            print(f"Issues found: {validation['issues'][:2]}")  # Show first 2 issues
        if validation['warnings']:
            print(f"Warnings: {validation['warnings'][:2]}")  # Show first 2 warnings

        print("\n✅ Source database tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
