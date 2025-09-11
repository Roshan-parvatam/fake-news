# agents/llm_explanation/source_database.py

"""
Source Reliability Database - Production Ready

Comprehensive institutional source reliability database for assessing news source
credibility with enhanced bias analysis, factual reporting assessment, pattern-based
detection, and advanced search capabilities. Provides structured source evaluation
with detailed institutional authority ratings for professional fact-checking environments.
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from .exceptions import SourceAssessmentError


@dataclass
class SourceInfo:
    """Structured container for source information with enhanced metadata."""
    name: str
    reliability_tier: str
    source_type: str
    bias_level: str
    factual_reporting: str
    description: str
    matched_source: Optional[str] = None
    confidence: float = 1.0
    last_updated: str = None
    verification_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'name': self.name,
            'reliability_tier': self.reliability_tier,
            'source_type': self.source_type,
            'bias_level': self.bias_level,
            'factual_reporting': self.factual_reporting,
            'description': self.description,
            'matched_source': self.matched_source,
            'confidence': self.confidence,
            'last_updated': self.last_updated,
            'verification_notes': self.verification_notes
        }


class SourceReliabilityDatabase:
    """
    Comprehensive institutional source reliability database with enhanced assessment capabilities.

    Features:
    - Multi-tiered reliability classification system with 8 comprehensive tiers
    - Advanced bias detection and assessment across political spectrum  
    - Factual reporting quality evaluation with institutional standards
    - Pattern-based dynamic detection for unknown sources
    - Comprehensive search and filtering capabilities
    - Performance tracking and quality metrics
    - Regular database updates and maintenance protocols
    """

    def __init__(self, custom_sources: Optional[Dict[str, Dict]] = None):
        """
        Initialize comprehensive source reliability database.

        Args:
            custom_sources: Optional additional sources to include in database
        """
        self.logger = logging.getLogger(f"{__name__}.SourceReliabilityDatabase")
        
        # Initialize core database
        self.database = self._build_comprehensive_database()
        self.tier_descriptions = self._build_tier_descriptions()
        
        # Add custom sources if provided
        if custom_sources:
            self._integrate_custom_sources(custom_sources)
        
        # Performance tracking
        self.assessment_count = 0
        self.cache_hits = 0
        self.total_assessment_time = 0.0
        self.assessment_cache = {}
        
        # Database statistics
        self.database_version = "3.0.0"
        self.last_updated = datetime.now().isoformat()
        
        # Initialize database validation
        self._validate_database_integrity()
        
        self.logger.info(f"Source Reliability Database initialized - Version: {self.database_version}")
        self.logger.info(f"Total sources: {self._count_total_sources()}")

    def _build_comprehensive_database(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Build comprehensive source database with institutional authority ratings.

        Returns:
            Multi-tier database with detailed source classifications and metadata
        """
        return {
            # Tier 1: Exceptional Reliability - Wire Services and International Broadcasters
            'EXCEPTIONAL': {
                # Major wire services with global reach
                'reuters': {
                    'type': 'international_wire_service',
                    'bias': 'minimal',
                    'factual': 'exceptional',
                    'authority': 'global_standard',
                    'specialties': ['breaking_news', 'financial_markets', 'international_affairs']
                },
                'associated press': {
                    'type': 'international_wire_service',
                    'bias': 'minimal',
                    'factual': 'exceptional',
                    'authority': 'global_standard',
                    'specialties': ['breaking_news', 'elections', 'disaster_reporting']
                },
                'ap news': {
                    'type': 'international_wire_service',
                    'bias': 'minimal',
                    'factual': 'exceptional',
                    'authority': 'global_standard',
                    'specialties': ['breaking_news', 'elections', 'disaster_reporting']
                },
                'agence france-presse': {
                    'type': 'international_wire_service',
                    'bias': 'minimal',
                    'factual': 'exceptional',
                    'authority': 'global_standard',
                    'specialties': ['international_news', 'european_affairs', 'culture']
                },
                'afp': {
                    'type': 'international_wire_service',
                    'bias': 'minimal',
                    'factual': 'exceptional',
                    'authority': 'global_standard',
                    'specialties': ['international_news', 'european_affairs', 'culture']
                },
                # International public broadcasters
                'bbc news': {
                    'type': 'international_public_broadcaster',
                    'bias': 'minimal',
                    'factual': 'exceptional',
                    'authority': 'international_standard',
                    'specialties': ['international_news', 'investigative_journalism', 'analysis']
                },
                'bbc': {
                    'type': 'international_public_broadcaster',
                    'bias': 'minimal',
                    'factual': 'exceptional',
                    'authority': 'international_standard',
                    'specialties': ['international_news', 'investigative_journalism', 'analysis']
                }
            },

            # Tier 2: High Reliability - Major National Publications
            'HIGH_RELIABILITY': {
                # Major newspapers with strong editorial standards
                'wall street journal': {
                    'type': 'major_newspaper',
                    'bias': 'slight_right',
                    'factual': 'high',
                    'authority': 'national_standard',
                    'specialties': ['business', 'finance', 'economics', 'politics']
                },
                'new york times': {
                    'type': 'major_newspaper',
                    'bias': 'slight_left',
                    'factual': 'high',
                    'authority': 'national_standard',
                    'specialties': ['national_politics', 'international_affairs', 'culture', 'investigation']
                },
                'washington post': {
                    'type': 'major_newspaper',
                    'bias': 'slight_left',
                    'factual': 'high',
                    'authority': 'national_standard',
                    'specialties': ['politics', 'government', 'investigation', 'democracy']
                },
                'financial times': {
                    'type': 'international_newspaper',
                    'bias': 'minimal',
                    'factual': 'high',
                    'authority': 'global_business_standard',
                    'specialties': ['global_finance', 'markets', 'economics', 'business']
                },
                'usa today': {
                    'type': 'national_newspaper',
                    'bias': 'slight_left',
                    'factual': 'high',
                    'authority': 'national_standard',
                    'specialties': ['general_news', 'sports', 'lifestyle', 'breaking_news']
                },
                'the guardian': {
                    'type': 'international_newspaper',
                    'bias': 'left',
                    'factual': 'high',
                    'authority': 'international_standard',
                    'specialties': ['progressive_politics', 'environment', 'social_justice', 'investigation']
                },
                'guardian': {
                    'type': 'international_newspaper',
                    'bias': 'left',
                    'factual': 'high',
                    'authority': 'international_standard',
                    'specialties': ['progressive_politics', 'environment', 'social_justice', 'investigation']
                },
                # Major magazines with strong editorial standards
                'the economist': {
                    'type': 'international_magazine',
                    'bias': 'slight_right',
                    'factual': 'high',
                    'authority': 'global_analysis_standard',
                    'specialties': ['global_economics', 'policy_analysis', 'international_relations']
                },
                'economist': {
                    'type': 'international_magazine',
                    'bias': 'slight_right',
                    'factual': 'high',
                    'authority': 'global_analysis_standard',
                    'specialties': ['global_economics', 'policy_analysis', 'international_relations']
                }
            },

            # Tier 3: Government and Official Institutional Sources
            'INSTITUTIONAL_AUTHORITY': {
                # US Government agencies
                '.gov': {
                    'type': 'us_government',
                    'bias': 'institutional',
                    'factual': 'authoritative',
                    'authority': 'official_government',
                    'specialties': ['official_policy', 'government_data', 'regulatory_information']
                },
                'cdc.gov': {
                    'type': 'health_authority',
                    'bias': 'scientific_institutional',
                    'factual': 'authoritative',
                    'authority': 'medical_authority',
                    'specialties': ['public_health', 'disease_control', 'health_guidelines']
                },
                'nih.gov': {
                    'type': 'medical_research_authority',
                    'bias': 'scientific_institutional',
                    'factual': 'authoritative',
                    'authority': 'medical_research_standard',
                    'specialties': ['medical_research', 'health_policy', 'clinical_trials']
                },
                'fda.gov': {
                    'type': 'regulatory_authority',
                    'bias': 'regulatory_institutional',
                    'factual': 'authoritative',
                    'authority': 'regulatory_standard',
                    'specialties': ['drug_safety', 'medical_devices', 'food_safety']
                },
                'nasa.gov': {
                    'type': 'scientific_authority',
                    'bias': 'scientific_institutional',
                    'factual': 'authoritative',
                    'authority': 'space_science_standard',
                    'specialties': ['space_science', 'astronomy', 'climate_research']
                },
                # International organizations
                'who.int': {
                    'type': 'international_health_organization',
                    'bias': 'international_institutional',
                    'factual': 'authoritative',
                    'authority': 'global_health_standard',
                    'specialties': ['global_health', 'pandemic_response', 'health_policy']
                },
                'un.org': {
                    'type': 'international_organization',
                    'bias': 'international_institutional',
                    'factual': 'high',
                    'authority': 'international_diplomatic',
                    'specialties': ['international_relations', 'humanitarian_affairs', 'peacekeeping']
                }
            },

            # Tier 4: Academic and Scientific Institutions
            'ACADEMIC_AUTHORITY': {
                # Academic domains and journals
                '.edu': {
                    'type': 'academic_institution',
                    'bias': 'academic_institutional',
                    'factual': 'authoritative',
                    'authority': 'academic_standard',
                    'specialties': ['research', 'education', 'scholarly_analysis']
                },
                'nature.com': {
                    'type': 'peer_reviewed_journal',
                    'bias': 'scientific_institutional',
                    'factual': 'exceptional',
                    'authority': 'scientific_standard',
                    'specialties': ['scientific_research', 'peer_review', 'breakthrough_discoveries']
                },
                'science.org': {
                    'type': 'peer_reviewed_journal',
                    'bias': 'scientific_institutional',
                    'factual': 'exceptional',
                    'authority': 'scientific_standard',
                    'specialties': ['multidisciplinary_science', 'research_publication']
                },
                'nejm.org': {
                    'type': 'medical_journal',
                    'bias': 'medical_institutional',
                    'factual': 'exceptional',
                    'authority': 'medical_research_standard',
                    'specialties': ['clinical_medicine', 'medical_research', 'healthcare_policy']
                },
                'pubmed.ncbi.nlm.nih.gov': {
                    'type': 'medical_database',
                    'bias': 'scientific_institutional',
                    'factual': 'authoritative',
                    'authority': 'medical_literature_standard',
                    'specialties': ['medical_literature', 'research_database']
                }
            },

            # Tier 5: Fact-Checking Organizations
            'FACT_CHECKING_AUTHORITY': {
                'snopes': {
                    'type': 'fact_checking_organization',
                    'bias': 'fact_checking_neutral',
                    'factual': 'high',
                    'authority': 'fact_checking_standard',
                    'specialties': ['rumor_verification', 'urban_legends', 'viral_claims']
                },
                'factcheck.org': {
                    'type': 'fact_checking_organization',
                    'bias': 'fact_checking_neutral',
                    'factual': 'high',
                    'authority': 'political_fact_checking',
                    'specialties': ['political_claims', 'policy_analysis', 'election_verification']
                },
                'politifact': {
                    'type': 'fact_checking_organization',
                    'bias': 'slight_left',
                    'factual': 'high',
                    'authority': 'political_fact_checking',
                    'specialties': ['political_fact_checking', 'truth_rating', 'politician_statements']
                },
                'ap fact check': {
                    'type': 'wire_service_fact_checking',
                    'bias': 'minimal',
                    'factual': 'exceptional',
                    'authority': 'news_verification_standard',
                    'specialties': ['breaking_news_verification', 'misinformation_detection']
                }
            },

            # Tier 6: Mainstream Media with Editorial Perspective
            'MAINSTREAM_EDITORIAL': {
                # Cable news networks
                'cnn': {
                    'type': 'cable_news_network',
                    'bias': 'left',
                    'factual': 'medium_high',
                    'authority': 'mainstream_media',
                    'specialties': ['breaking_news', 'politics', 'international_affairs']
                },
                'fox news': {
                    'type': 'cable_news_network',
                    'bias': 'right',
                    'factual': 'medium',
                    'authority': 'mainstream_media',
                    'specialties': ['conservative_politics', 'opinion', 'breaking_news']
                },
                'msnbc': {
                    'type': 'cable_news_network',
                    'bias': 'left',
                    'factual': 'medium',
                    'authority': 'mainstream_media',
                    'specialties': ['progressive_politics', 'opinion', 'analysis']
                },
                # Broadcast networks
                'abc news': {
                    'type': 'broadcast_network',
                    'bias': 'slight_left',
                    'factual': 'high',
                    'authority': 'broadcast_standard',
                    'specialties': ['national_news', 'investigative_journalism', 'breaking_news']
                },
                'cbs news': {
                    'type': 'broadcast_network',
                    'bias': 'slight_left',
                    'factual': 'high',
                    'authority': 'broadcast_standard',
                    'specialties': ['national_news', 'investigative_journalism', '60_minutes']
                },
                'nbc news': {
                    'type': 'broadcast_network',
                    'bias': 'slight_left',
                    'factual': 'high',
                    'authority': 'broadcast_standard',
                    'specialties': ['national_news', 'politics', 'breaking_news']
                },
                # Business and financial news
                'bloomberg': {
                    'type': 'financial_news_service',
                    'bias': 'slight_right',
                    'factual': 'high',
                    'authority': 'financial_news_standard',
                    'specialties': ['financial_markets', 'business_news', 'economic_analysis']
                },
                'cnbc': {
                    'type': 'business_news_network',
                    'bias': 'slight_right',
                    'factual': 'high',
                    'authority': 'business_news_standard',
                    'specialties': ['stock_market', 'business_news', 'financial_analysis']
                }
            },

            # Tier 7: Lower Reliability - Tabloids and Partisan Sources
            'LIMITED_RELIABILITY': {
                # Tabloid media
                'daily mail': {
                    'type': 'tabloid',
                    'bias': 'right',
                    'factual': 'medium_low',
                    'authority': 'tabloid_standard',
                    'specialties': ['celebrity_news', 'sensationalism', 'opinion']
                },
                'new york post': {
                    'type': 'tabloid',
                    'bias': 'right',
                    'factual': 'medium',
                    'authority': 'tabloid_standard',
                    'specialties': ['local_news', 'sports', 'conservative_opinion']
                },
                # Partisan sources
                'breitbart': {
                    'type': 'partisan_news',
                    'bias': 'far_right',
                    'factual': 'low',
                    'authority': 'partisan_media',
                    'specialties': ['conservative_activism', 'anti_establishment', 'immigration']
                },
                'daily wire': {
                    'type': 'partisan_news',
                    'bias': 'right',
                    'factual': 'medium_low',
                    'authority': 'partisan_media',
                    'specialties': ['conservative_politics', 'cultural_commentary']
                },
                'huffpost': {
                    'type': 'partisan_news',
                    'bias': 'left',
                    'factual': 'medium',
                    'authority': 'partisan_media',
                    'specialties': ['progressive_politics', 'social_justice', 'opinion']
                }
            },

            # Tier 8: Questionable Sources - Conspiracy and Misinformation
            'QUESTIONABLE': {
                'infowars': {
                    'type': 'conspiracy_site',
                    'bias': 'far_right',
                    'factual': 'very_low',
                    'authority': 'conspiracy_media',
                    'specialties': ['conspiracy_theories', 'anti_government', 'alternative_health']
                },
                'natural news': {
                    'type': 'pseudoscience_site',
                    'bias': 'right',
                    'factual': 'very_low',
                    'authority': 'pseudoscience_media',
                    'specialties': ['alternative_medicine', 'anti_vaccine', 'health_misinformation']
                },
                'zerohedge': {
                    'type': 'financial_conspiracy',
                    'bias': 'right',
                    'factual': 'low',
                    'authority': 'conspiracy_financial',
                    'specialties': ['economic_conspiracy', 'market_doom', 'anti_establishment']
                },
                # State-sponsored media with propaganda concerns
                'rt.com': {
                    'type': 'state_sponsored_media',
                    'bias': 'pro_russian',
                    'factual': 'low',
                    'authority': 'state_propaganda',
                    'specialties': ['russian_perspective', 'anti_western', 'geopolitics']
                },
                'russia today': {
                    'type': 'state_sponsored_media',
                    'bias': 'pro_russian',
                    'factual': 'low',
                    'authority': 'state_propaganda',
                    'specialties': ['russian_perspective', 'anti_western', 'geopolitics']
                },
                'sputnik news': {
                    'type': 'state_sponsored_media',
                    'bias': 'pro_russian',
                    'factual': 'low',
                    'authority': 'state_propaganda',
                    'specialties': ['russian_perspective', 'anti_western', 'international_affairs']
                }
            },

            # Pattern-based detection for dynamic assessment
            'PATTERN_DETECTION': {
                # Social media and user-generated content platforms
                'facebook': {
                    'reliability': 'USER_GENERATED',
                    'reason': 'social_media_platform',
                    'warning': 'User-generated content with variable accuracy',
                    'verification_needed': True
                },
                'twitter': {
                    'reliability': 'USER_GENERATED',
                    'reason': 'social_media_platform',
                    'warning': 'Real-time information with variable verification',
                    'verification_needed': True
                },
                'x.com': {
                    'reliability': 'USER_GENERATED',
                    'reason': 'social_media_platform',
                    'warning': 'Real-time information with variable verification',
                    'verification_needed': True
                },
                'youtube': {
                    'reliability': 'USER_GENERATED',
                    'reason': 'video_platform',
                    'warning': 'Video content with variable editorial standards',
                    'verification_needed': True
                },
                'tiktok': {
                    'reliability': 'USER_GENERATED',
                    'reason': 'short_form_video',
                    'warning': 'Entertainment-focused platform with minimal fact-checking',
                    'verification_needed': True
                },
                # Blog and personal website patterns
                'wordpress': {
                    'reliability': 'PERSONAL_BLOG',
                    'reason': 'blog_platform',
                    'warning': 'Personal blog platform with no editorial oversight',
                    'verification_needed': True
                },
                'blogspot': {
                    'reliability': 'PERSONAL_BLOG',
                    'reason': 'blog_platform',
                    'warning': 'Personal blog platform with minimal quality control',
                    'verification_needed': True
                },
                'medium.com': {
                    'reliability': 'USER_GENERATED',
                    'reason': 'publishing_platform',
                    'warning': 'User publishing platform with variable editorial standards',
                    'verification_needed': True
                },
                # Anonymous and unattributed sources
                'anonymous': {
                    'reliability': 'UNVERIFIABLE',
                    'reason': 'no_attribution',
                    'warning': 'Anonymous sources require independent verification',
                    'verification_needed': True
                },
                'unknown': {
                    'reliability': 'UNVERIFIABLE',
                    'reason': 'unidentified_source',
                    'warning': 'Unidentified sources require verification',
                    'verification_needed': True
                },
                # Forwarded and viral content
                'forwarded': {
                    'reliability': 'UNVERIFIABLE',
                    'reason': 'forwarded_content',
                    'warning': 'Forwarded content often lacks source verification',
                    'verification_needed': True
                },
                'viral': {
                    'reliability': 'UNVERIFIABLE',
                    'reason': 'viral_content',
                    'warning': 'Viral content may be manipulated or taken out of context',
                    'verification_needed': True
                }
            }
        }

    def _build_tier_descriptions(self) -> Dict[str, str]:
        """Build comprehensive tier descriptions for user understanding."""
        return {
            'EXCEPTIONAL': 'Exceptional reliability - Global wire services and international public broadcasters with the highest editorial standards and minimal bias',
            'HIGH_RELIABILITY': 'High reliability - Major national publications and magazines with strong editorial standards and professional journalism practices',
            'INSTITUTIONAL_AUTHORITY': 'Institutional authority - Official government agencies and international organizations with authoritative information in their domains',
            'ACADEMIC_AUTHORITY': 'Academic authority - Peer-reviewed journals, academic institutions, and scientific publications with rigorous quality control',
            'FACT_CHECKING_AUTHORITY': 'Fact-checking authority - Specialized verification organizations focused on accuracy and misinformation detection',
            'MAINSTREAM_EDITORIAL': 'Mainstream with editorial perspective - Established media organizations with clear editorial viewpoints but professional standards',
            'LIMITED_RELIABILITY': 'Limited reliability - Sources with significant bias, sensationalism, or reduced editorial standards requiring careful verification',
            'QUESTIONABLE': 'Questionable reliability - Sources associated with conspiracy theories, pseudoscience, or propaganda requiring extreme caution',
            'PATTERN_DETECTION': 'Pattern-based assessment - Dynamic evaluation of sources based on platform type, attribution, and content characteristics'
        }

    def assess_source_reliability(self, source: str, session_id: str = None) -> str:
        """
        Assess source reliability with enhanced matching and caching.

        Args:
            source: Source name or URL to assess
            session_id: Optional session ID for tracking

        Returns:
            Reliability level: EXCEPTIONAL, HIGH_RELIABILITY, INSTITUTIONAL_AUTHORITY, 
                              ACADEMIC_AUTHORITY, FACT_CHECKING_AUTHORITY, MAINSTREAM_EDITORIAL,
                              LIMITED_RELIABILITY, QUESTIONABLE, or UNKNOWN
        """
        start_time = time.time()
        self.assessment_count += 1

        try:
            if not source or not isinstance(source, str):
                self.logger.warning("Invalid source input for assessment", extra={'session_id': session_id})
                return 'UNKNOWN'

            # Check cache first
            source_normalized = source.lower().strip()
            if source_normalized in self.assessment_cache:
                self.cache_hits += 1
                return self.assessment_cache[source_normalized]

            self.logger.debug(f"Assessing source reliability: {source}", extra={'session_id': session_id})

            # Check against comprehensive database
            for tier, sources in self.database.items():
                if tier == 'PATTERN_DETECTION':
                    continue

                for known_source, source_info in sources.items():
                    if self._source_matches(known_source, source_normalized):
                        # Cache the result
                        self.assessment_cache[source_normalized] = tier
                        
                        assessment_time = time.time() - start_time
                        self.total_assessment_time += assessment_time
                        
                        self.logger.info(
                            f"Source matched in tier {tier}",
                            extra={'session_id': session_id, 'matched_source': known_source}
                        )
                        
                        return tier

            # Check pattern-based indicators
            pattern_result = self._assess_pattern_based(source_normalized, session_id)
            if pattern_result != 'UNKNOWN':
                self.assessment_cache[source_normalized] = pattern_result
                return pattern_result

            # Domain-based fallback assessment
            domain_result = self._assess_domain_based(source_normalized, session_id)
            self.assessment_cache[source_normalized] = domain_result

            assessment_time = time.time() - start_time
            self.total_assessment_time += assessment_time

            return domain_result

        except Exception as e:
            self.logger.error(f"Source assessment failed: {str(e)}", extra={'session_id': session_id})
            raise SourceAssessmentError(f"Assessment failed: {str(e)}", source)

    def get_source_details(self, source: str, session_id: str = None) -> SourceInfo:
        """
        Get comprehensive source information with enhanced metadata.

        Args:
            source: Source name or URL to analyze
            session_id: Optional session ID for tracking

        Returns:
            SourceInfo object with detailed assessment
        """
        try:
            if not source or not isinstance(source, str):
                return self._create_unknown_source_info(source)

            source_normalized = source.lower().strip()
            
            self.logger.debug(f"Getting source details: {source}", extra={'session_id': session_id})

            # Search comprehensive database
            for tier, sources in self.database.items():
                if tier == 'PATTERN_DETECTION':
                    continue

                for known_source, source_info in sources.items():
                    if self._source_matches(known_source, source_normalized):
                        return SourceInfo(
                            name=source,
                            reliability_tier=tier,
                            source_type=source_info.get('type', 'unknown'),
                            bias_level=source_info.get('bias', 'unknown'),
                            factual_reporting=source_info.get('factual', 'unknown'),
                            description=self.tier_descriptions.get(tier, 'Unknown tier'),
                            matched_source=known_source,
                            confidence=1.0,
                            last_updated=self.last_updated,
                            verification_notes=f"Matched against {known_source} in {tier} tier"
                        )

            # Check pattern detection
            pattern_info = self._get_pattern_detection_info(source_normalized, session_id)
            if pattern_info:
                return pattern_info

            # Return unknown source info
            return self._create_unknown_source_info(source)

        except Exception as e:
            self.logger.error(f"Failed to get source details: {str(e)}", extra={'session_id': session_id})
            return self._create_unknown_source_info(source, error_msg=str(e))

    def get_reliability_summary(self, source: str, session_id: str = None) -> Dict[str, Any]:
        """
        Get comprehensive reliability summary for user presentation.

        Args:
            source: Source to analyze
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with user-friendly reliability assessment and recommendations
        """
        try:
            source_info = self.get_source_details(source, session_id)
            reliability_tier = source_info.reliability_tier
            
            # Generate user-friendly descriptions
            reliability_descriptions = {
                'EXCEPTIONAL': 'This is an exceptionally reliable source with the highest editorial standards and minimal bias.',
                'HIGH_RELIABILITY': 'This is a highly reliable source with strong editorial standards, though some bias may be present.',
                'INSTITUTIONAL_AUTHORITY': 'This is an authoritative institutional source with official standing in its domain.',
                'ACADEMIC_AUTHORITY': 'This is a peer-reviewed academic source with rigorous scientific standards.',
                'FACT_CHECKING_AUTHORITY': 'This is a specialized fact-checking organization focused on verification and accuracy.',
                'MAINSTREAM_EDITORIAL': 'This is an established media source with clear editorial perspective but professional standards.',
                'LIMITED_RELIABILITY': 'This source has limited reliability due to bias, sensationalism, or reduced editorial standards.',
                'QUESTIONABLE': 'This source has questionable reliability and is associated with conspiracy theories or misinformation.',
                'UNKNOWN': 'Source reliability cannot be determined from available information.'
            }

            # Generate verification recommendations
            verification_recommendations = {
                'EXCEPTIONAL': 'Cross-reference with additional high-quality sources for complete context and perspective.',
                'HIGH_RELIABILITY': 'Generally trustworthy but consider checking alternative perspectives on controversial topics.',
                'INSTITUTIONAL_AUTHORITY': 'Authoritative for official information but consider independent analysis for policy implications.',
                'ACADEMIC_AUTHORITY': 'Highly reliable for scientific information but ensure relevance and recency of research.',
                'FACT_CHECKING_AUTHORITY': 'Reliable for verification purposes but consider multiple fact-checkers for comprehensive assessment.',
                'MAINSTREAM_EDITORIAL': 'Verify key claims through multiple independent sources and consider potential bias.',
                'LIMITED_RELIABILITY': 'Approach with caution and verify all claims through highly reliable sources.',
                'QUESTIONABLE': 'Exercise extreme skepticism and seek verification from authoritative sources before accepting any claims.',
                'UNKNOWN': 'Research source credibility thoroughly and verify all information through established, reliable sources.'
            }

            # Generate bias warnings
            bias_warning = self._generate_bias_warning(source_info.bias_level)

            return {
                'source': source,
                'reliability_level': reliability_tier,
                'reliability_description': reliability_descriptions.get(reliability_tier, 'Unknown reliability level'),
                'source_details': source_info.to_dict(),
                'verification_recommendation': verification_recommendations.get(reliability_tier, 'Verify independently'),
                'bias_warning': bias_warning,
                'confidence_level': source_info.confidence,
                'assessment_metadata': {
                    'database_version': self.database_version,
                    'assessment_timestamp': datetime.now().isoformat(),
                    'session_id': session_id
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate reliability summary: {str(e)}", extra={'session_id': session_id})
            return self._create_error_summary(source, str(e))

    def search_sources(self, search_term: str, limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search sources by term with enhanced filtering and ranking.

        Args:
            search_term: Term to search for in source names
            limit: Maximum number of results per tier

        Returns:
            Dictionary of matching sources organized by tier
        """
        try:
            if not search_term or not isinstance(search_term, str):
                return {}

            search_term_normalized = search_term.lower().strip()
            results = defaultdict(list)

            for tier, sources in self.database.items():
                if tier == 'PATTERN_DETECTION':
                    continue

                tier_matches = []
                for source_name, source_info in sources.items():
                    if search_term_normalized in source_name:
                        match_info = {
                            'source_name': source_name,
                            'source_type': source_info.get('type', 'unknown'),
                            'bias_level': source_info.get('bias', 'unknown'),
                            'factual_reporting': source_info.get('factual', 'unknown'),
                            'specialties': source_info.get('specialties', []),
                            'authority': source_info.get('authority', 'unknown'),
                            'tier_description': self.tier_descriptions.get(tier, 'Unknown tier')
                        }
                        tier_matches.append(match_info)

                # Sort by relevance and limit results
                tier_matches.sort(key=lambda x: len(x['source_name']))  # Shorter names first
                results[tier] = tier_matches[:limit]

            return dict(results)

        except Exception as e:
            self.logger.error(f"Source search failed: {str(e)}")
            return {}

    def get_sources_by_tier(self, tier: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all sources in a specific reliability tier.

        Args:
            tier: Reliability tier to retrieve

        Returns:
            Dictionary of sources in the specified tier
        """
        return self.database.get(tier, {})

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics and metrics."""
        try:
            tier_counts = {}
            total_sources = 0
            
            for tier, sources in self.database.items():
                count = len(sources)
                tier_counts[tier] = count
                if tier != 'PATTERN_DETECTION':
                    total_sources += count

            # Performance metrics
            avg_assessment_time = (self.total_assessment_time / self.assessment_count) if self.assessment_count > 0 else 0
            cache_hit_rate = (self.cache_hits / self.assessment_count * 100) if self.assessment_count > 0 else 0

            return {
                'database_composition': {
                    'tier_counts': tier_counts,
                    'total_sources': total_sources,
                    'total_tiers': len([t for t in self.database.keys() if t != 'PATTERN_DETECTION']),
                    'pattern_indicators': tier_counts.get('PATTERN_DETECTION', 0)
                },
                'performance_metrics': {
                    'total_assessments': self.assessment_count,
                    'cache_hits': self.cache_hits,
                    'cache_hit_rate_percent': round(cache_hit_rate, 2),
                    'average_assessment_time_ms': round(avg_assessment_time * 1000, 2),
                    'total_assessment_time': self.total_assessment_time
                },
                'database_metadata': {
                    'version': self.database_version,
                    'last_updated': self.last_updated,
                    'cache_size': len(self.assessment_cache)
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get database statistics: {str(e)}")
            return {'error': f"Statistics generation failed: {str(e)}"}

    def _source_matches(self, known_source: str, source_input: str) -> bool:
        """Enhanced source matching with multiple strategies."""
        # Exact match
        if known_source == source_input:
            return True

        # Substring match
        if known_source in source_input or source_input in known_source:
            return True

        # Domain extraction for URL matching
        if '.' in source_input:
            # Extract domain from URL
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', source_input)
            if domain_match:
                domain = domain_match.group(1).lower()
                if known_source in domain or domain in known_source:
                    return True

        return False

    def _assess_pattern_based(self, source: str, session_id: str = None) -> str:
        """Assess source using pattern-based detection."""
        pattern_detection = self.database.get('PATTERN_DETECTION', {})
        
        for pattern, details in pattern_detection.items():
            if pattern in source:
                self.logger.info(
                    f"Pattern-based assessment: {pattern}",
                    extra={'session_id': session_id, 'reliability': details.get('reliability')}
                )
                return details.get('reliability', 'UNKNOWN')

        return 'UNKNOWN'

    def _assess_domain_based(self, source: str, session_id: str = None) -> str:
        """Domain-based fallback assessment."""
        # Government domains
        if source.endswith('.gov') or '.gov/' in source:
            return 'INSTITUTIONAL_AUTHORITY'
        
        # Educational domains
        if source.endswith('.edu') or '.edu/' in source:
            return 'ACADEMIC_AUTHORITY'
        
        # Non-profit organizations
        if source.endswith('.org') or '.org/' in source:
            return 'MAINSTREAM_EDITORIAL'  # Conservative assessment for .org
        
        # Commercial domains - unknown by default
        return 'UNKNOWN'

    def _get_pattern_detection_info(self, source: str, session_id: str = None) -> Optional[SourceInfo]:
        """Get source info from pattern detection."""
        pattern_detection = self.database.get('PATTERN_DETECTION', {})
        
        for pattern, details in pattern_detection.items():
            if pattern in source:
                return SourceInfo(
                    name=source,
                    reliability_tier=details.get('reliability', 'UNKNOWN'),
                    source_type='pattern_detected',
                    bias_level='unknown',
                    factual_reporting='unknown',
                    description=f"Pattern detected: {details.get('reason', 'unknown')}",
                    matched_source=pattern,
                    confidence=0.7,  # Lower confidence for pattern matching
                    last_updated=self.last_updated,
                    verification_notes=details.get('warning', 'Pattern-based assessment')
                )
        
        return None

    def _create_unknown_source_info(self, source: str, error_msg: str = None) -> SourceInfo:
        """Create SourceInfo for unknown sources."""
        return SourceInfo(
            name=source,
            reliability_tier='UNKNOWN',
            source_type='unclassified',
            bias_level='unknown',
            factual_reporting='unknown',
            description='Source not found in reliability database',
            matched_source=None,
            confidence=0.0,
            last_updated=self.last_updated,
            verification_notes=error_msg or 'Source not in database'
        )

    def _generate_bias_warning(self, bias_level: str) -> Optional[str]:
        """Generate appropriate bias warning based on bias level."""
        bias_warnings = {
            'far_right': 'Warning: This source exhibits extreme right-wing bias that significantly affects reporting accuracy and perspective.',
            'far_left': 'Warning: This source exhibits extreme left-wing bias that significantly affects reporting accuracy and perspective.',
            'right': 'Note: This source has a noticeable right-leaning editorial perspective that may influence story selection and framing.',
            'left': 'Note: This source has a noticeable left-leaning editorial perspective that may influence story selection and framing.',
            'slight_right': 'This source has a slight right-leaning perspective in editorial content but maintains professional reporting standards.',
            'slight_left': 'This source has a slight left-leaning perspective in editorial content but maintains professional reporting standards.',
            'pro_russian': 'Warning: This source represents Russian state interests and may contain propaganda or biased international reporting.',
            'minimal': None,  # No warning needed
            'institutional': 'This source represents an institutional perspective that may favor established positions and policies.',
            'scientific_institutional': 'This source follows scientific institutional standards with evidence-based reporting.',
            'fact_checking_neutral': None  # No warning needed for neutral fact-checkers
        }
        
        return bias_warnings.get(bias_level)

    def _create_error_summary(self, source: str, error_msg: str) -> Dict[str, Any]:
        """Create error summary when assessment fails."""
        return {
            'source': source,
            'reliability_level': 'ERROR',
            'reliability_description': f'Assessment failed: {error_msg}',
            'source_details': {'error': error_msg},
            'verification_recommendation': 'Unable to assess - verify through multiple reliable sources',
            'bias_warning': None,
            'confidence_level': 0.0,
            'assessment_metadata': {
                'database_version': self.database_version,
                'error': error_msg,
                'assessment_timestamp': datetime.now().isoformat()
            }
        }

    def _integrate_custom_sources(self, custom_sources: Dict[str, Dict]) -> None:
        """Integrate custom sources into database."""
        try:
            for tier, sources in custom_sources.items():
                if tier not in self.database:
                    self.database[tier] = {}
                
                for source_name, source_info in sources.items():
                    self.database[tier][source_name.lower()] = source_info
                    
            self.logger.info(f"Integrated custom sources into database")
        
        except Exception as e:
            self.logger.warning(f"Failed to integrate custom sources: {str(e)}")

    def _validate_database_integrity(self) -> bool:
        """Validate database structure and content integrity."""
        try:
            required_fields = ['type', 'bias', 'factual']
            validation_errors = []

            for tier, sources in self.database.items():
                if tier == 'PATTERN_DETECTION':
                    continue
                    
                for source_name, source_info in sources.items():
                    if not isinstance(source_info, dict):
                        validation_errors.append(f"Invalid source info for {source_name}")
                        continue
                    
                    for field in required_fields:
                        if field not in source_info:
                            validation_errors.append(f"Missing {field} for {source_name}")

            if validation_errors:
                self.logger.warning(f"Database validation found {len(validation_errors)} issues")
                return False
            
            self.logger.info("Database integrity validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Database validation failed: {str(e)}")
            return False

    def _count_total_sources(self) -> int:
        """Count total sources in database excluding patterns."""
        return sum(len(sources) for tier, sources in self.database.items() 
                  if tier != 'PATTERN_DETECTION')


# Testing functionality
if __name__ == "__main__":
    """Test source reliability database functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== SOURCE RELIABILITY DATABASE TEST ===")
    
    # Initialize database
    db = SourceReliabilityDatabase()
    
    # Test comprehensive source assessment
    test_sources = [
        "Reuters.com",
        "CNN.com", 
        "Infowars.com",
        "Harvard.edu",
        "CDC.gov",
        "Facebook.com",
        "UnknownBlog.net",
        "Nature.com",
        "Fox News",
        "BBC News"
    ]

    print("--- Source Reliability Assessment Test ---")
    for source in test_sources:
        reliability = db.assess_source_reliability(source, f"test_session_{hash(source) % 1000}")
        details = db.get_source_details(source)
        summary = db.get_reliability_summary(source)
        
        print(f"\n🔍 Source: {source}")
        print(f"   📊 Reliability: {reliability}")
        print(f"   🏷️  Type: {details.source_type}")
        print(f"   ⚖️  Bias: {details.bias_level}")
        print(f"   📈 Factual: {details.factual_reporting}")
        print(f"   🎯 Confidence: {details.confidence}")
        
        if summary['bias_warning']:
            print(f"   ⚠️  Warning: {summary['bias_warning'][:60]}...")

    # Test search functionality
    print(f"\n--- Search Functionality Test ---")
    search_results = db.search_sources("news", limit=3)
    print(f"🔍 Sources containing 'news': {len(search_results)} tiers found")
    
    for tier, sources in search_results.items():
        print(f"   {tier}: {len(sources)} sources")

    # Test database statistics
    print(f"\n--- Database Statistics Test ---")
    stats = db.get_database_statistics()
    composition = stats['database_composition']
    performance = stats['performance_metrics']
    
    print(f"📊 Total sources: {composition['total_sources']}")
    print(f"📊 Total tiers: {composition['total_tiers']}")
    print(f"📊 Pattern indicators: {composition['pattern_indicators']}")
    print(f"⚡ Total assessments: {performance['total_assessments']}")
    print(f"⚡ Cache hit rate: {performance['cache_hit_rate_percent']:.1f}%")
    print(f"⚡ Average assessment time: {performance['average_assessment_time_ms']:.2f}ms")

    # Test tier-specific queries
    print(f"\n--- Tier-Specific Queries Test ---")
    exceptional_sources = db.get_sources_by_tier('EXCEPTIONAL')
    questionable_sources = db.get_sources_by_tier('QUESTIONABLE')
    
    print(f"✅ Exceptional sources: {len(exceptional_sources)}")
    print(f"❌ Questionable sources: {len(questionable_sources)}")

    # Test reliability summary formatting
    print(f"\n--- Reliability Summary Test ---")
    test_summary = db.get_reliability_summary("Reuters", "test_summary_001")
    
    print(f"📋 Summary for Reuters:")
    print(f"   Level: {test_summary['reliability_level']}")
    print(f"   Description: {test_summary['reliability_description'][:80]}...")
    print(f"   Recommendation: {test_summary['verification_recommendation'][:60]}...")

    # Test custom sources integration
    print(f"\n--- Custom Sources Integration Test ---")
    custom_sources = {
        'HIGH_RELIABILITY': {
            'test_news_outlet': {
                'type': 'test_publication',
                'bias': 'minimal',
                'factual': 'high',
                'authority': 'test_standard'
            }
        }
    }
    
    custom_db = SourceReliabilityDatabase(custom_sources)
    custom_assessment = custom_db.assess_source_reliability("test_news_outlet")
    print(f"✅ Custom source assessment: {custom_assessment}")

    print(f"\n🎯 Source reliability database tests completed successfully!")
