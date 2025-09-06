# agents/credible_source/source_database.py
"""
Source Reliability Database for Credible Source Agent - Config Enhanced

Enhanced source database with better performance tracking and configuration awareness.
"""

from typing import Dict, List, Any
import logging
import time

class SourceReliabilityDatabase:
    """
    📊 ENHANCED SOURCE RELIABILITY DATABASE WITH CONFIG AWARENESS
    
    This class manages a database of credible sources with reliability scores
    and provides source recommendations with enhanced performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the source reliability database with optional config
        
        Args:
            config: Optional configuration for source recommendations
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
        self.database_stats = {
            'total_recommendations': 0,
            'total_sources_recommended': 0,
            'recommendation_time_total': 0.0,
            'config_applied': bool(config)
        }
        
        total_sources = (len(self.primary_sources) + len(self.expert_sources) + 
                        len(self.institutional_sources) + len(self.journalistic_sources) + 
                        len(self.fact_check_sources))
        
        self.logger.info(f"✅ SourceReliabilityDatabase initialized with {total_sources} sources")
    
    def _initialize_primary_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        📊 PRIMARY SOURCES DATABASE - Government, official, academic
        """
        return {
            # Government Sources
            'cdc': {
                'name': 'Centers for Disease Control and Prevention',
                'type': 'government',
                'domain': 'health',
                'reliability_score': 9.5,
                'url': 'https://www.cdc.gov',
                'contact_info': 'Public inquiries available',
                'verification_methods': ['official_statements', 'data_requests', 'press_releases']
            },
            'fda': {
                'name': 'U.S. Food and Drug Administration',
                'type': 'government',
                'domain': 'health',
                'reliability_score': 9.4,
                'url': 'https://www.fda.gov',
                'contact_info': 'Media relations available',
                'verification_methods': ['approval_databases', 'official_announcements']
            },
            'census_bureau': {
                'name': 'U.S. Census Bureau',
                'type': 'government',
                'domain': 'demographics',
                'reliability_score': 9.8,
                'url': 'https://www.census.gov',
                'contact_info': 'Public information office',
                'verification_methods': ['official_data', 'public_records']
            },
            'bls': {
                'name': 'Bureau of Labor Statistics',
                'type': 'government',
                'domain': 'economics',
                'reliability_score': 9.7,
                'url': 'https://www.bls.gov',
                'contact_info': 'Media relations',
                'verification_methods': ['official_statistics', 'reports']
            },
            
            # Academic Sources
            'harvard_med': {
                'name': 'Harvard Medical School',
                'type': 'academic',
                'domain': 'health',
                'reliability_score': 9.2,
                'url': 'https://hms.harvard.edu',
                'contact_info': 'Media relations office',
                'verification_methods': ['published_research', 'expert_contacts']
            },
            'mit': {
                'name': 'Massachusetts Institute of Technology',
                'type': 'academic',
                'domain': 'technology',
                'reliability_score': 9.3,
                'url': 'https://www.mit.edu',
                'contact_info': 'News office',
                'verification_methods': ['research_publications', 'expert_interviews']
            },
            'stanford': {
                'name': 'Stanford University',
                'type': 'academic',
                'domain': 'general',
                'reliability_score': 9.1,
                'url': 'https://www.stanford.edu',
                'contact_info': 'Media relations',
                'verification_methods': ['faculty_experts', 'research_papers']
            }
        }
    
    def _initialize_expert_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        📊 EXPERT SOURCES DATABASE - Subject matter experts
        """
        return {
            'who': {
                'name': 'World Health Organization',
                'type': 'international_organization',
                'domain': 'health',
                'reliability_score': 9.0,
                'url': 'https://www.who.int',
                'contact_info': 'Media centre',
                'verification_methods': ['official_statements', 'expert_committees']
            },
            'ama': {
                'name': 'American Medical Association',
                'type': 'professional_organization',
                'domain': 'health',
                'reliability_score': 8.5,
                'url': 'https://www.ama-assn.org',
                'contact_info': 'Media relations',
                'verification_methods': ['position_statements', 'expert_panels']
            },
            'aaas': {
                'name': 'American Association for the Advancement of Science',
                'type': 'professional_organization',
                'domain': 'science',
                'reliability_score': 8.8,
                'url': 'https://www.aaas.org',
                'contact_info': 'Communications office',
                'verification_methods': ['expert_directory', 'policy_statements']
            },
            'ieee': {
                'name': 'Institute of Electrical and Electronics Engineers',
                'type': 'professional_organization',
                'domain': 'technology',
                'reliability_score': 8.7,
                'url': 'https://www.ieee.org',
                'contact_info': 'Media relations',
                'verification_methods': ['standards_documents', 'expert_networks']
            }
        }
    
    def _initialize_institutional_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        📊 INSTITUTIONAL SOURCES DATABASE - Think tanks, research institutions
        """
        return {
            'brookings': {
                'name': 'Brookings Institution',
                'type': 'think_tank',
                'domain': 'policy',
                'reliability_score': 8.2,
                'url': 'https://www.brookings.edu',
                'contact_info': 'Communications office',
                'verification_methods': ['research_reports', 'expert_analysis']
            },
            'cfr': {
                'name': 'Council on Foreign Relations',
                'type': 'think_tank',
                'domain': 'international',
                'reliability_score': 8.4,
                'url': 'https://www.cfr.org',
                'contact_info': 'Media relations',
                'verification_methods': ['policy_briefs', 'expert_interviews']
            },
            'mayo_clinic': {
                'name': 'Mayo Clinic',
                'type': 'medical_institution',
                'domain': 'health',
                'reliability_score': 9.0,
                'url': 'https://www.mayoclinic.org',
                'contact_info': 'Public affairs',
                'verification_methods': ['medical_experts', 'clinical_data']
            },
            'cleveland_clinic': {
                'name': 'Cleveland Clinic',
                'type': 'medical_institution',
                'domain': 'health',
                'reliability_score': 8.8,
                'url': 'https://my.clevelandclinic.org',
                'contact_info': 'Media relations',
                'verification_methods': ['physician_experts', 'research_publications']
            }
        }
    
    def _initialize_journalistic_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        📊 JOURNALISTIC SOURCES DATABASE - High-quality news organizations
        """
        return {
            'reuters': {
                'name': 'Reuters',
                'type': 'news_agency',
                'domain': 'general',
                'reliability_score': 8.8,
                'url': 'https://www.reuters.com',
                'contact_info': 'Newsroom contacts',
                'verification_methods': ['fact_checking_team', 'source_verification']
            },
            'ap_news': {
                'name': 'Associated Press',
                'type': 'news_agency',
                'domain': 'general',
                'reliability_score': 8.9,
                'url': 'https://apnews.com',
                'contact_info': 'Media relations',
                'verification_methods': ['fact_checking', 'source_corroboration']
            },
            'bbc': {
                'name': 'BBC News',
                'type': 'broadcaster',
                'domain': 'general',
                'reliability_score': 8.5,
                'url': 'https://www.bbc.com/news',
                'contact_info': 'Press office',
                'verification_methods': ['editorial_standards', 'fact_checking']
            },
            'npr': {
                'name': 'National Public Radio',
                'type': 'broadcaster',
                'domain': 'general',
                'reliability_score': 8.3,
                'url': 'https://www.npr.org',
                'contact_info': 'Media relations',
                'verification_methods': ['editorial_guidelines', 'source_verification']
            }
        }
    
    def _initialize_fact_check_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        📊 FACT-CHECK SOURCES DATABASE - Dedicated fact-checking organizations
        """
        return {
            'snopes': {
                'name': 'Snopes',
                'type': 'fact_checker',
                'domain': 'general',
                'reliability_score': 8.0,
                'url': 'https://www.snopes.com',
                'contact_info': 'Editorial team',
                'verification_methods': ['fact_verification', 'source_investigation']
            },
            'politifact': {
                'name': 'PolitiFact',
                'type': 'fact_checker',
                'domain': 'politics',
                'reliability_score': 8.2,
                'url': 'https://www.politifact.com',
                'contact_info': 'Editorial office',
                'verification_methods': ['truth_o_meter', 'source_checking']
            },
            'factcheck_org': {
                'name': 'FactCheck.org',
                'type': 'fact_checker',
                'domain': 'politics',
                'reliability_score': 8.4,
                'url': 'https://www.factcheck.org',
                'contact_info': 'Annenberg Public Policy Center',
                'verification_methods': ['political_fact_checking', 'source_analysis']
            },
            'ap_fact_check': {
                'name': 'AP Fact Check',
                'type': 'fact_checker',
                'domain': 'general',
                'reliability_score': 8.6,
                'url': 'https://apnews.com/hub/ap-fact-check',
                'contact_info': 'AP News',
                'verification_methods': ['comprehensive_fact_checking', 'source_verification']
            }
        }
    
    def get_source_recommendations(self, article_text: str, extracted_claims: List[Dict[str, Any]], 
                                 domain: str = 'general') -> Dict[str, Any]:
        """
        📊 GET SOURCE RECOMMENDATIONS WITH CONFIG AWARENESS
        
        Provide source recommendations based on article content and domain with performance tracking.
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
        
        # Filter sources by domain relevance and reliability
        min_reliability = self.config.get('min_reliability_threshold', 7.0) if self.config else 7.0
        max_sources = self.config.get('max_sources_per_domain', 15) if self.config else 15
        
        relevant_sources = []
        
        for source_key, source_info in all_sources.items():
            # Check domain relevance
            source_domain = source_info.get('domain', 'general')
            if domain != 'general' and source_domain not in [domain, 'general']:
                continue
            
            # Check reliability threshold
            if source_info.get('reliability_score', 0) < min_reliability:
                continue
            
            # Add relevance score based on content analysis
            relevance_score = self._calculate_source_relevance(
                article_text, extracted_claims, source_info
            )
            
            source_recommendation = {
                **source_info,
                'source_key': source_key,
                'relevance_score': relevance_score,
                'domain_match': source_domain == domain or source_domain == 'general'
            }
            
            relevant_sources.append(source_recommendation)
        
        # Sort by combined score (reliability + relevance)
        relevant_sources.sort(
            key=lambda x: (x['reliability_score'] + x['relevance_score']) / 2,
            reverse=True
        )
        
        # Limit to max sources
        recommended_sources = relevant_sources[:max_sources]
        
        # Categorize sources
        source_categories = self._categorize_sources(recommended_sources)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.database_stats['total_recommendations'] += 1
        self.database_stats['total_sources_recommended'] += len(recommended_sources)
        self.database_stats['recommendation_time_total'] += processing_time
        
        return {
            'recommended_sources': recommended_sources,
            'source_categories': source_categories,
            'domain_analyzed': domain,
            'total_sources_available': len(all_sources),
            'sources_recommended': len(recommended_sources),
            'min_reliability_used': min_reliability,
            'processing_time_ms': round(processing_time * 1000, 2),
            'config_applied': bool(self.config)
        }
    
    def _calculate_source_relevance(self, article_text: str, extracted_claims: List[Dict[str, Any]], 
                                   source_info: Dict[str, Any]) -> float:
        """Calculate source relevance based on content analysis"""
        relevance_score = 5.0  # Base score
        
        article_lower = article_text.lower()
        source_type = source_info.get('type', '')
        source_domain = source_info.get('domain', 'general')
        
        # Domain matching bonus
        if source_domain in article_lower:
            relevance_score += 1.0
        
        # Source type relevance
        type_bonuses = {
            'government': 1.5,
            'academic': 1.2,
            'professional_organization': 1.0,
            'medical_institution': 1.3,
            'fact_checker': 0.8
        }
        relevance_score += type_bonuses.get(source_type, 0)
        
        # Claim type matching
        for claim in extracted_claims[:5]:  # Limit for performance
            claim_type = claim.get('claim_type', '').lower()
            if claim_type == 'research' and source_type in ['academic', 'medical_institution']:
                relevance_score += 0.5
            elif claim_type == 'statistical' and source_type == 'government':
                relevance_score += 0.5
        
        return min(10.0, relevance_score)
    
    def _categorize_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize sources by type"""
        categories = {}
        
        for source in sources:
            source_type = source.get('type', 'unknown')
            categories[source_type] = categories.get(source_type, 0) + 1
        
        return categories
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        total_sources = (len(self.primary_sources) + len(self.expert_sources) + 
                        len(self.institutional_sources) + len(self.journalistic_sources) + 
                        len(self.fact_check_sources))
        
        source_breakdown = {
            'primary_sources': len(self.primary_sources),
            'expert_sources': len(self.expert_sources),
            'institutional_sources': len(self.institutional_sources),
            'journalistic_sources': len(self.journalistic_sources),
            'fact_check_sources': len(self.fact_check_sources)
        }
        
        # Performance stats
        performance_stats = self.database_stats.copy()
        if performance_stats['total_recommendations'] > 0:
            performance_stats['average_recommendation_time_ms'] = round(
                (performance_stats['recommendation_time_total'] / performance_stats['total_recommendations']) * 1000, 2
            )
            performance_stats['average_sources_per_recommendation'] = round(
                performance_stats['total_sources_recommended'] / performance_stats['total_recommendations'], 2
            )
        
        return {
            'total_sources': total_sources,
            'source_breakdown': source_breakdown,
            'performance_stats': performance_stats
        }

# Testing
if __name__ == "__main__":
    """Test source database with config"""
    test_config = {
        'min_reliability_threshold': 8.0,
        'max_sources_per_domain': 10
    }
    
    database = SourceReliabilityDatabase(test_config)
    
    test_text = """
    A new medical study published in Nature shows significant health benefits.
    The research was conducted by Harvard Medical School with CDC collaboration.
    """
    
    test_claims = [
        {'claim_type': 'Research', 'text': 'Study published in Nature'},
        {'claim_type': 'Attribution', 'text': 'Research by Harvard Medical School'}
    ]
    
    recommendations = database.get_source_recommendations(test_text, test_claims, 'health')
    
    print(f"Source recommendations:")
    print(f"Total sources recommended: {recommendations['sources_recommended']}")
    print(f"Domain analyzed: {recommendations['domain_analyzed']}")
    print(f"Source categories: {recommendations['source_categories']}")
    
    for i, source in enumerate(recommendations['recommended_sources'][:3], 1):
        print(f"  {i}. {source['name']} (Reliability: {source['reliability_score']}/10)")
    
    stats = database.get_database_statistics()
    print(f"Database contains {stats['total_sources']} total sources")
