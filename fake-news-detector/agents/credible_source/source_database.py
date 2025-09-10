# agents/credible_source/source_database.py

"""
Source Reliability Database

Comprehensive database of credible information sources with reliability scoring,
domain-specific categorization, and intelligent recommendation algorithms.
Enhanced for production use with clean architecture and performance tracking.
"""

import time
import logging
from typing import Dict, List, Any, Optional


class SourceReliabilityDatabase:
    """
    Manages a comprehensive database of credible information sources.
    
    Provides source recommendations based on domain relevance, reliability scores,
    and content analysis with configurable filtering and scoring algorithms.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize source reliability database.
        
        Args:
            config: Optional configuration for source recommendation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize source databases
        self.primary_sources = self._initialize_primary_sources()
        self.expert_sources = self._initialize_expert_sources()
        self.institutional_sources = self._initialize_institutional_sources()
        self.journalistic_sources = self._initialize_journalistic_sources()
        self.fact_check_sources = self._initialize_fact_check_sources()
        
        # Performance tracking
        self.recommendation_count = 0
        self.total_processing_time = 0.0
        self.sources_recommended = 0
        
        total_sources = sum([
            len(self.primary_sources),
            len(self.expert_sources), 
            len(self.institutional_sources),
            len(self.journalistic_sources),
            len(self.fact_check_sources)
        ])
        
        self.logger.info(f"Source database initialized with {total_sources} credible sources")

    def _initialize_primary_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize government and official primary sources."""
        return {
            'cdc': {
                'name': 'Centers for Disease Control and Prevention',
                'type': 'government',
                'domain': 'health',
                'reliability_score': 9.5,
                'url': 'https://www.cdc.gov',
                'contact_info': 'Public inquiries: 1-800-CDC-INFO',
                'verification_methods': ['official_statements', 'data_requests', 'press_releases'],
                'expertise_areas': ['public health', 'disease prevention', 'epidemiology', 'vaccines']
            },
            'fda': {
                'name': 'U.S. Food and Drug Administration',
                'type': 'government',
                'domain': 'health',
                'reliability_score': 9.4,
                'url': 'https://www.fda.gov',
                'contact_info': 'Media relations: press@fda.hhs.gov',
                'verification_methods': ['approval_databases', 'official_announcements', 'regulatory_documents'],
                'expertise_areas': ['drug_approval', 'medical_devices', 'food_safety', 'tobacco_regulation']
            },
            'census_bureau': {
                'name': 'U.S. Census Bureau',
                'type': 'government',
                'domain': 'demographics',
                'reliability_score': 9.8,
                'url': 'https://www.census.gov',
                'contact_info': 'Public Information Office: pio@census.gov',
                'verification_methods': ['official_data', 'public_records', 'statistical_releases'],
                'expertise_areas': ['population_data', 'economic_statistics', 'housing_data', 'business_statistics']
            },
            'bls': {
                'name': 'Bureau of Labor Statistics',
                'type': 'government',
                'domain': 'economics',
                'reliability_score': 9.7,
                'url': 'https://www.bls.gov',
                'contact_info': 'Media relations: blspress@bls.gov',
                'verification_methods': ['official_statistics', 'labor_reports', 'economic_indicators'],
                'expertise_areas': ['employment', 'inflation', 'wages', 'productivity', 'workplace_safety']
            },
            'epa': {
                'name': 'Environmental Protection Agency',
                'type': 'government',
                'domain': 'environment',
                'reliability_score': 9.2,
                'url': 'https://www.epa.gov',
                'contact_info': 'Public affairs: press@epa.gov',
                'verification_methods': ['environmental_data', 'regulatory_documents', 'scientific_studies'],
                'expertise_areas': ['air_quality', 'water_quality', 'chemical_safety', 'climate_change']
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
                'verification_methods': ['official_statements', 'expert_committees', 'global_health_data'],
                'expertise_areas': ['global_health', 'disease_outbreaks', 'health_policy', 'health_emergencies']
            },
            'ama': {
                'name': 'American Medical Association',
                'type': 'professional_organization',
                'domain': 'health',
                'reliability_score': 8.5,
                'url': 'https://www.ama-assn.org',
                'contact_info': 'Media relations: media@ama-assn.org',
                'verification_methods': ['position_statements', 'expert_panels', 'medical_guidelines'],
                'expertise_areas': ['medical_practice', 'healthcare_policy', 'medical_ethics', 'physician_advocacy']
            },
            'aaas': {
                'name': 'American Association for the Advancement of Science',
                'type': 'professional_organization',
                'domain': 'science',
                'reliability_score': 8.8,
                'url': 'https://www.aaas.org',
                'contact_info': 'Communications: media@aaas.org',
                'verification_methods': ['expert_directory', 'policy_statements', 'scientific_reports'],
                'expertise_areas': ['scientific_research', 'science_policy', 'STEM_education', 'research_ethics']
            },
            'ieee': {
                'name': 'Institute of Electrical and Electronics Engineers',
                'type': 'professional_organization',
                'domain': 'technology',
                'reliability_score': 8.7,
                'url': 'https://www.ieee.org',
                'contact_info': 'Media relations: media@ieee.org',
                'verification_methods': ['standards_documents', 'expert_networks', 'technical_publications'],
                'expertise_areas': ['electrical_engineering', 'computer_science', 'telecommunications', 'robotics']
            },
            'apa': {
                'name': 'American Psychological Association',
                'type': 'professional_organization',
                'domain': 'psychology',
                'reliability_score': 8.6,
                'url': 'https://www.apa.org',
                'contact_info': 'Public affairs: media@apa.org',
                'verification_methods': ['research_publications', 'clinical_guidelines', 'expert_opinions'],
                'expertise_areas': ['psychology', 'mental_health', 'behavioral_science', 'clinical_practice']
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
                'verification_methods': ['published_research', 'expert_contacts', 'medical_studies'],
                'expertise_areas': ['medical_research', 'clinical_medicine', 'biomedical_science', 'public_health']
            },
            'mit': {
                'name': 'Massachusetts Institute of Technology',
                'type': 'academic',
                'domain': 'technology',
                'reliability_score': 9.3,
                'url': 'https://www.mit.edu',
                'contact_info': 'News office: newsoffice@mit.edu',
                'verification_methods': ['research_publications', 'expert_interviews', 'technical_reports'],
                'expertise_areas': ['engineering', 'computer_science', 'artificial_intelligence', 'biotechnology']
            },
            'stanford': {
                'name': 'Stanford University',
                'type': 'academic',
                'domain': 'general',
                'reliability_score': 9.1,
                'url': 'https://www.stanford.edu',
                'contact_info': 'Media relations: media@stanford.edu',
                'verification_methods': ['faculty_experts', 'research_papers', 'academic_studies'],
                'expertise_areas': ['medicine', 'engineering', 'computer_science', 'business', 'policy']
            },
            'mayo_clinic': {
                'name': 'Mayo Clinic',
                'type': 'medical_institution',
                'domain': 'health',
                'reliability_score': 9.0,
                'url': 'https://www.mayoclinic.org',
                'contact_info': 'Public affairs: publicaffairs@mayo.edu',
                'verification_methods': ['medical_experts', 'clinical_data', 'patient_care_guidelines'],
                'expertise_areas': ['clinical_medicine', 'medical_research', 'patient_care', 'medical_education']
            },
            'brookings': {
                'name': 'Brookings Institution',
                'type': 'think_tank',
                'domain': 'policy',
                'reliability_score': 8.2,
                'url': 'https://www.brookings.edu',
                'contact_info': 'Communications: communications@brookings.edu',
                'verification_methods': ['research_reports', 'expert_analysis', 'policy_papers'],
                'expertise_areas': ['economic_policy', 'foreign_policy', 'governance', 'metropolitan_policy']
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
                'verification_methods': ['fact_checking_team', 'source_verification', 'editorial_standards'],
                'expertise_areas': ['breaking_news', 'financial_markets', 'international_affairs', 'politics']
            },
            'ap_news': {
                'name': 'Associated Press',
                'type': 'news_agency',
                'domain': 'general',
                'reliability_score': 8.9,
                'url': 'https://apnews.com',
                'contact_info': 'Media relations: apcorporatecomm@ap.org',
                'verification_methods': ['fact_checking', 'source_corroboration', 'editorial_review'],
                'expertise_areas': ['news_reporting', 'fact_checking', 'wire_services', 'photography']
            },
            'bbc': {
                'name': 'BBC News',
                'type': 'broadcaster',
                'domain': 'general',
                'reliability_score': 8.5,
                'url': 'https://www.bbc.com/news',
                'contact_info': 'Press office: press.office@bbc.co.uk',
                'verification_methods': ['editorial_standards', 'fact_checking', 'multiple_sourcing'],
                'expertise_areas': ['international_news', 'UK_affairs', 'analysis', 'investigative_journalism']
            },
            'npr': {
                'name': 'National Public Radio',
                'type': 'broadcaster',
                'domain': 'general',
                'reliability_score': 8.3,
                'url': 'https://www.npr.org',
                'contact_info': 'Media relations: press@npr.org',
                'verification_methods': ['editorial_guidelines', 'source_verification', 'public_radio_standards'],
                'expertise_areas': ['news_analysis', 'cultural_coverage', 'science_reporting', 'politics']
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
                'verification_methods': ['fact_verification', 'source_investigation', 'claim_analysis'],
                'expertise_areas': ['urban_legends', 'viral_claims', 'political_fact_checking', 'misinformation']
            },
            'politifact': {
                'name': 'PolitiFact',
                'type': 'fact_checker',
                'domain': 'politics',
                'reliability_score': 8.2,
                'url': 'https://www.politifact.com',
                'contact_info': 'Editorial office: editor@politifact.com',
                'verification_methods': ['truth_o_meter', 'source_checking', 'political_verification'],
                'expertise_areas': ['political_claims', 'campaign_promises', 'policy_analysis', 'election_coverage']
            },
            'factcheck_org': {
                'name': 'FactCheck.org',
                'type': 'fact_checker',
                'domain': 'politics',
                'reliability_score': 8.4,
                'url': 'https://www.factcheck.org',
                'contact_info': 'Annenberg Public Policy Center',
                'verification_methods': ['political_fact_checking', 'source_analysis', 'claim_verification'],
                'expertise_areas': ['political_accuracy', 'campaign_advertising', 'policy_claims', 'political_rhetoric']
            },
            'ap_fact_check': {
                'name': 'AP Fact Check',
                'type': 'fact_checker',
                'domain': 'general',
                'reliability_score': 8.6,
                'url': 'https://apnews.com/hub/ap-fact-check',
                'contact_info': 'AP News fact-check team',
                'verification_methods': ['comprehensive_fact_checking', 'source_verification', 'claim_analysis'],
                'expertise_areas': ['news_verification', 'claim_checking', 'misinformation_debunking', 'source_validation']
            }
        }

    def get_source_recommendations(self, 
                                 article_text: str, 
                                 extracted_claims: List[Dict[str, Any]], 
                                 domain: str = 'general') -> Dict[str, Any]:
        """
        Generate source recommendations based on article content and domain.
        
        Args:
            article_text: Article content to analyze
            extracted_claims: Claims extracted from the article
            domain: Primary domain classification
            
        Returns:
            Dictionary containing recommended sources and analysis metadata
        """
        start_time = time.time()
        
        # Combine all source databases
        all_sources = {
            **self.primary_sources,
            **self.expert_sources,
            **self.institutional_sources,
            **self.journalistic_sources,
            **self.fact_check_sources
        }
        
        # Apply configuration filters
        min_reliability = self.config.get('min_reliability_threshold', 7.0)
        max_sources = self.config.get('max_sources_per_domain', 12)
        
        # Filter and score sources
        relevant_sources = []
        for source_key, source_info in all_sources.items():
            # Domain relevance filter
            source_domain = source_info.get('domain', 'general')
            if domain != 'general' and source_domain not in [domain, 'general']:
                continue
            
            # Reliability threshold filter
            if source_info.get('reliability_score', 0) < min_reliability:
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_source_relevance(
                article_text, extracted_claims, source_info
            )
            
            # Create recommendation entry
            source_recommendation = {
                **source_info,
                'source_key': source_key,
                'relevance_score': relevance_score,
                'domain_match': source_domain == domain or source_domain == 'general',
                'combined_score': (source_info.get('reliability_score', 0) + relevance_score) / 2
            }
            
            relevant_sources.append(source_recommendation)
        
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
        
        return {
            'recommended_sources': recommended_sources,
            'source_categories': source_categories,
            'quality_metrics': quality_metrics,
            'domain_analyzed': domain,
            'total_sources_available': len(all_sources),
            'sources_recommended': len(recommended_sources),
            'min_reliability_used': min_reliability,
            'processing_time_ms': round(processing_time * 1000, 2)
        }

    def _calculate_source_relevance(self, 
                                  article_text: str, 
                                  extracted_claims: List[Dict[str, Any]], 
                                  source_info: Dict[str, Any]) -> float:
        """Calculate source relevance based on content analysis."""
        relevance_score = 5.0  # Base score
        
        article_lower = article_text.lower()
        source_type = source_info.get('type', '')
        source_domain = source_info.get('domain', 'general')
        expertise_areas = source_info.get('expertise_areas', [])
        
        # Domain matching bonus
        if source_domain != 'general' and source_domain in article_lower:
            relevance_score += 1.5
        
        # Expertise area matching
        for area in expertise_areas:
            if area.replace('_', ' ') in article_lower:
                relevance_score += 1.0
                break
        
        # Source type relevance weighting
        type_weights = {
            'government': 1.8,
            'academic': 1.5,
            'international_organization': 1.4,
            'professional_organization': 1.2,
            'medical_institution': 1.6,
            'think_tank': 1.0,
            'news_agency': 0.9,
            'broadcaster': 0.8,
            'fact_checker': 1.1
        }
        
        relevance_score += type_weights.get(source_type, 0)
        
        # Claim type matching analysis
        for claim in extracted_claims[:5]:  # Limit for performance
            claim_type = claim.get('claim_type', '').lower()
            claim_text = claim.get('text', '').lower()
            
            # Match claim types to source expertise
            if claim_type in ['research', 'statistical'] and source_type in ['academic', 'government']:
                relevance_score += 0.8
            elif claim_type == 'medical' and source_type in ['medical_institution', 'government']:
                relevance_score += 1.0
            elif claim_type == 'political' and source_type in ['government', 'think_tank']:
                relevance_score += 0.8
            elif claim_type == 'factual' and source_type == 'fact_checker':
                relevance_score += 0.6
            
            # Match specific expertise areas to claim content
            for area in expertise_areas:
                if area.replace('_', ' ') in claim_text:
                    relevance_score += 0.5
                    break
        
        return min(10.0, relevance_score)

    def _categorize_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize sources by type and domain."""
        categories = {}
        
        for source in sources:
            source_type = source.get('type', 'unknown')
            categories[source_type] = categories.get(source_type, 0) + 1
        
        return categories

    def _calculate_quality_metrics(self, 
                                 recommended_sources: List[Dict[str, Any]], 
                                 all_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate recommendation quality metrics."""
        if not recommended_sources:
            return {
                'average_reliability': 0.0,
                'average_relevance': 0.0,
                'source_diversity': 0.0,
                'coverage_ratio': 0.0
            }
        
        # Calculate average scores
        avg_reliability = sum(s.get('reliability_score', 0) for s in recommended_sources) / len(recommended_sources)
        avg_relevance = sum(s.get('relevance_score', 0) for s in recommended_sources) / len(recommended_sources)
        
        # Calculate source diversity (unique types)
        unique_types = len(set(s.get('type', 'unknown') for s in recommended_sources))
        max_possible_types = len(set(s.get('type', 'unknown') for s in all_sources.values()))
        source_diversity = unique_types / max_possible_types if max_possible_types > 0 else 0
        
        # Calculate coverage ratio (recommended vs available)
        coverage_ratio = len(recommended_sources) / len(all_sources) if all_sources else 0
        
        return {
            'average_reliability': round(avg_reliability, 2),
            'average_relevance': round(avg_relevance, 2),
            'source_diversity': round(source_diversity, 2),
            'coverage_ratio': round(coverage_ratio, 3),
            'unique_source_types': unique_types,
            'total_source_types_available': max_possible_types
        }

    def get_sources_by_domain(self, domain: str, min_reliability: float = 7.0) -> List[Dict[str, Any]]:
        """Get all sources for a specific domain above reliability threshold."""
        all_sources = {
            **self.primary_sources,
            **self.expert_sources,
            **self.institutional_sources,
            **self.journalistic_sources,
            **self.fact_check_sources
        }
        
        domain_sources = []
        for source_key, source_info in all_sources.items():
            source_domain = source_info.get('domain', 'general')
            reliability = source_info.get('reliability_score', 0)
            
            if (domain == 'general' or source_domain in [domain, 'general']) and reliability >= min_reliability:
                source_data = source_info.copy()
                source_data['source_key'] = source_key
                domain_sources.append(source_data)
        
        # Sort by reliability score
        domain_sources.sort(key=lambda x: x.get('reliability_score', 0), reverse=True)
        return domain_sources

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
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
        
        return {
            'total_sources': total_sources,
            'source_breakdown': source_breakdown,
            'recommendations_completed': self.recommendation_count,
            'sources_recommended_total': self.sources_recommended,
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'average_sources_per_recommendation': round(avg_sources_per_recommendation, 2),
            'domains_supported': ['health', 'science', 'technology', 'politics', 'economics', 'environment', 'general'],
            'source_types_available': ['government', 'academic', 'institutional', 'journalistic', 'fact_checker']
        }

    def validate_database_integrity(self) -> Dict[str, Any]:
        """Validate database integrity and completeness."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        all_databases = [
            ('primary_sources', self.primary_sources),
            ('expert_sources', self.expert_sources),
            ('institutional_sources', self.institutional_sources),
            ('journalistic_sources', self.journalistic_sources),
            ('fact_check_sources', self.fact_check_sources)
        ]
        
        for db_name, db_data in all_databases:
            if not db_data:
                validation_results['issues'].append(f"Empty database: {db_name}")
                validation_results['valid'] = False
            
            for source_key, source_info in db_data.items():
                # Check required fields
                required_fields = ['name', 'type', 'domain', 'reliability_score', 'url']
                for field in required_fields:
                    if field not in source_info:
                        validation_results['issues'].append(f"Missing {field} in {source_key}")
                        validation_results['valid'] = False
                
                # Validate reliability score
                reliability = source_info.get('reliability_score', 0)
                if not isinstance(reliability, (int, float)) or reliability < 0 or reliability > 10:
                    validation_results['issues'].append(f"Invalid reliability score for {source_key}: {reliability}")
                    validation_results['valid'] = False
                
                # Check URL format
                url = source_info.get('url', '')
                if url and not (url.startswith('http://') or url.startswith('https://')):
                    validation_results['warnings'].append(f"Invalid URL format for {source_key}: {url}")
        
        return validation_results


# Testing functionality
if __name__ == "__main__":
    """Test source database functionality."""
    
    # Initialize database with test configuration
    test_config = {
        'min_reliability_threshold': 8.0,
        'max_sources_per_domain': 10
    }
    
    database = SourceReliabilityDatabase(test_config)
    
    # Test source recommendations
    test_article = """
    A new medical study published in Nature Medicine shows significant 
    improvements in treatment outcomes. The research was conducted by 
    Harvard Medical School in collaboration with the CDC.
    """
    
    test_claims = [
        {
            'text': 'Medical study published in Nature Medicine',
            'claim_type': 'Research',
            'priority': 1
        },
        {
            'text': 'Harvard Medical School collaboration with CDC',
            'claim_type': 'Attribution',
            'priority': 2
        }
    ]
    
    print("=== SOURCE RECOMMENDATION TEST ===")
    recommendations = database.get_source_recommendations(test_article, test_claims, 'health')
    
    print(f"Sources recommended: {recommendations['sources_recommended']}")
    print(f"Domain analyzed: {recommendations['domain_analyzed']}")
    print(f"Processing time: {recommendations['processing_time_ms']:.1f}ms")
    
    print("\nTop recommended sources:")
    for i, source in enumerate(recommendations['recommended_sources'][:5], 1):
        print(f"{i}. {source['name']}")
        print(f"   Reliability: {source['reliability_score']}/10")
        print(f"   Relevance: {source['relevance_score']:.1f}/10")
        print(f"   Type: {source['type']}")
    
    print(f"\nSource categories: {recommendations['source_categories']}")
    print(f"Quality metrics: {recommendations['quality_metrics']}")
    
    # Test database statistics
    stats = database.get_database_statistics()
    print(f"\n=== DATABASE STATISTICS ===")
    print(f"Total sources: {stats['total_sources']}")
    print(f"Source breakdown: {stats['source_breakdown']}")
    print(f"Domains supported: {len(stats['domains_supported'])}")
    
    # Test database validation
    validation = database.validate_database_integrity()
    print(f"\n=== DATABASE VALIDATION ===")
    print(f"Database valid: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
    if validation['issues']:
        print(f"Issues found: {validation['issues']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    print("\n=== SOURCE DATABASE TESTING COMPLETED ===")
