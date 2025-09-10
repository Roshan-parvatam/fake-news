# agents/llm_explanation/source_database.py

"""
Source Reliability Database for LLM Explanation Agent

Comprehensive source reliability database for assessing news source credibility.
Provides structured source evaluation with bias analysis, factual reporting
assessment, and pattern-based detection capabilities.
"""

from typing import Dict, Any, Optional, List


class SourceReliabilityDatabase:
    """
    Comprehensive database for evaluating news source reliability and credibility.
    
    Manages source classifications, bias assessments, and factual reporting
    quality ratings to support explanation generation and source evaluation.
    """

    def __init__(self):
        """Initialize the source reliability database."""
        self.database = self._build_source_database()
        self.tier_descriptions = self._build_tier_descriptions()
        
    def _build_source_database(self) -> Dict[str, Dict]:
        """
        Build comprehensive source database organized by reliability tiers.
        
        Returns:
            Dictionary mapping tier names to source collections
        """
        return {
            # Tier 1: Highest Reliability - Wire Services and Public Broadcasters
            'HIGH_PLUS': {
                'reuters': {'type': 'wire_service', 'bias': 'minimal', 'factual': 'very_high'},
                'associated press': {'type': 'wire_service', 'bias': 'minimal', 'factual': 'very_high'},
                'ap news': {'type': 'wire_service', 'bias': 'minimal', 'factual': 'very_high'},
                'bloomberg': {'type': 'wire_service', 'bias': 'minimal', 'factual': 'very_high'},
                'agence france-presse': {'type': 'wire_service', 'bias': 'minimal', 'factual': 'very_high'},
                'afp': {'type': 'wire_service', 'bias': 'minimal', 'factual': 'very_high'},
                'bbc news': {'type': 'public_broadcaster', 'bias': 'minimal', 'factual': 'very_high'},
                'bbc': {'type': 'public_broadcaster', 'bias': 'minimal', 'factual': 'very_high'},
                'npr': {'type': 'public_broadcaster', 'bias': 'slight_left', 'factual': 'very_high'},
                'pbs': {'type': 'public_broadcaster', 'bias': 'minimal', 'factual': 'very_high'},
                'pbs newshour': {'type': 'public_broadcaster', 'bias': 'minimal', 'factual': 'very_high'},
            },
            
            # Tier 2: High Reliability - Established Newspapers and Magazines
            'HIGH': {
                'wall street journal': {'type': 'newspaper', 'bias': 'slight_right', 'factual': 'high'},
                'wsj': {'type': 'newspaper', 'bias': 'slight_right', 'factual': 'high'},
                'new york times': {'type': 'newspaper', 'bias': 'slight_left', 'factual': 'high'},
                'nytimes': {'type': 'newspaper', 'bias': 'slight_left', 'factual': 'high'},
                'washington post': {'type': 'newspaper', 'bias': 'slight_left', 'factual': 'high'},
                'financial times': {'type': 'newspaper', 'bias': 'minimal', 'factual': 'high'},
                'usa today': {'type': 'newspaper', 'bias': 'slight_left', 'factual': 'high'},
                'los angeles times': {'type': 'newspaper', 'bias': 'left', 'factual': 'high'},
                'chicago tribune': {'type': 'newspaper', 'bias': 'slight_right', 'factual': 'high'},
                'boston globe': {'type': 'newspaper', 'bias': 'left', 'factual': 'high'},
                'guardian': {'type': 'newspaper', 'bias': 'left', 'factual': 'high'},
                'the guardian': {'type': 'newspaper', 'bias': 'left', 'factual': 'high'},
                'times of london': {'type': 'newspaper', 'bias': 'slight_right', 'factual': 'high'},
                'economist': {'type': 'magazine', 'bias': 'slight_right', 'factual': 'high'},
                'atlantic': {'type': 'magazine', 'bias': 'left', 'factual': 'high'},
                'new yorker': {'type': 'magazine', 'bias': 'left', 'factual': 'high'},
                'time magazine': {'type': 'magazine', 'bias': 'slight_left', 'factual': 'high'},
                'newsweek': {'type': 'magazine', 'bias': 'slight_left', 'factual': 'high'},
                'foreign affairs': {'type': 'magazine', 'bias': 'minimal', 'factual': 'high'},
            },
            
            # Tier 3: Fact-Checking Organizations
            'FACT_CHECKERS': {
                'snopes': {'type': 'fact_checker', 'bias': 'minimal', 'factual': 'very_high'},
                'factcheck.org': {'type': 'fact_checker', 'bias': 'minimal', 'factual': 'very_high'},
                'politifact': {'type': 'fact_checker', 'bias': 'slight_left', 'factual': 'very_high'},
                'ap fact check': {'type': 'fact_checker', 'bias': 'minimal', 'factual': 'very_high'},
                'reuters fact check': {'type': 'fact_checker', 'bias': 'minimal', 'factual': 'very_high'},
                'afp fact check': {'type': 'fact_checker', 'bias': 'minimal', 'factual': 'very_high'},
                'full fact': {'type': 'fact_checker', 'bias': 'minimal', 'factual': 'very_high'},
                'washington post fact checker': {'type': 'fact_checker', 'bias': 'slight_left', 'factual': 'high'},
            },
            
            # Tier 4: Government and Official Sources
            'GOVERNMENT': {
                '.gov': {'type': 'government', 'bias': 'institutional', 'factual': 'very_high'},
                'cdc.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'nih.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'fda.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'nasa.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'noaa.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'usgs.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'census.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'bls.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'sec.gov': {'type': 'government', 'bias': 'minimal', 'factual': 'very_high'},
                'who.int': {'type': 'international_org', 'bias': 'minimal', 'factual': 'very_high'},
                'un.org': {'type': 'international_org', 'bias': 'minimal', 'factual': 'high'},
                'worldbank.org': {'type': 'international_org', 'bias': 'minimal', 'factual': 'high'},
            },
            
            # Tier 5: Academic and Research Institutions
            'ACADEMIC': {
                '.edu': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'harvard.edu': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'mit.edu': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'stanford.edu': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'nature.com': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'science.org': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'cell.com': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'nejm.org': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'lancet.com': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'pubmed.ncbi.nlm.nih.gov': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'jstor.org': {'type': 'academic', 'bias': 'minimal', 'factual': 'very_high'},
                'arxiv.org': {'type': 'academic', 'bias': 'minimal', 'factual': 'high'},
            },
            
            # Tier 6: Moderate Reliability - Mainstream Media with Noticeable Bias
            'MEDIUM_HIGH': {
                'cnn': {'type': 'cable_news', 'bias': 'left', 'factual': 'medium_high'},
                'msnbc': {'type': 'cable_news', 'bias': 'left', 'factual': 'medium'},
                'fox news': {'type': 'cable_news', 'bias': 'right', 'factual': 'medium'},
                'abc news': {'type': 'broadcast_news', 'bias': 'slight_left', 'factual': 'high'},
                'cbs news': {'type': 'broadcast_news', 'bias': 'slight_left', 'factual': 'high'},
                'nbc news': {'type': 'broadcast_news', 'bias': 'slight_left', 'factual': 'high'},
                'cnbc': {'type': 'business_news', 'bias': 'slight_right', 'factual': 'high'},
                'forbes': {'type': 'business_news', 'bias': 'slight_right', 'factual': 'medium_high'},
                'business insider': {'type': 'business_news', 'bias': 'slight_left', 'factual': 'medium_high'},
                'huffpost': {'type': 'online_news', 'bias': 'left', 'factual': 'medium'},
                'buzzfeed news': {'type': 'online_news', 'bias': 'left', 'factual': 'medium_high'},
                'vox': {'type': 'online_news', 'bias': 'left', 'factual': 'medium_high'},
                'vice news': {'type': 'online_news', 'bias': 'left', 'factual': 'medium'},
            },
            
            # Tier 7: Lower Reliability - Tabloids and Partisan Sources
            'MEDIUM_LOW': {
                'daily mail': {'type': 'tabloid', 'bias': 'right', 'factual': 'low'},
                'new york post': {'type': 'tabloid', 'bias': 'right', 'factual': 'medium'},
                'breitbart': {'type': 'partisan', 'bias': 'right', 'factual': 'low'},
                'daily wire': {'type': 'partisan', 'bias': 'right', 'factual': 'medium'},
                'mother jones': {'type': 'partisan', 'bias': 'left', 'factual': 'medium'},
                'salon': {'type': 'partisan', 'bias': 'left', 'factual': 'medium'},
                'slate': {'type': 'online_magazine', 'bias': 'left', 'factual': 'medium'},
                'daily beast': {'type': 'online_news', 'bias': 'left', 'factual': 'medium'},
            },
            
            # Tier 8: Questionable Sources - Conspiracy, Pseudoscience, State Media
            'LOW': {
                'infowars': {'type': 'conspiracy', 'bias': 'extreme_right', 'factual': 'very_low'},
                'natural news': {'type': 'pseudoscience', 'bias': 'right', 'factual': 'very_low'},
                'prisonplanet': {'type': 'conspiracy', 'bias': 'extreme_right', 'factual': 'very_low'},
                'globalresearch': {'type': 'conspiracy', 'bias': 'left', 'factual': 'very_low'},
                'zerohedge': {'type': 'financial_blog', 'bias': 'right', 'factual': 'low'},
                'russia today': {'type': 'state_media', 'bias': 'extreme_right', 'factual': 'low'},
                'rt.com': {'type': 'state_media', 'bias': 'extreme_right', 'factual': 'low'},
                'sputnik news': {'type': 'state_media', 'bias': 'extreme_right', 'factual': 'low'},
                'xinhua': {'type': 'state_media', 'bias': 'extreme_left', 'factual': 'low'},
                'presstv': {'type': 'state_media', 'bias': 'extreme_left', 'factual': 'low'},
            },
            
            # Pattern-Based Indicators for Dynamic Assessment
            'PATTERNS': {
                'blog': {'reliability': 'LOW', 'reason': 'personal_blog'},
                'wordpress': {'reliability': 'LOW', 'reason': 'personal_blog'},
                'blogspot': {'reliability': 'LOW', 'reason': 'personal_blog'},
                'medium.com': {'reliability': 'MEDIUM_LOW', 'reason': 'user_generated'},
                'substack': {'reliability': 'MEDIUM_LOW', 'reason': 'newsletter_platform'},
                'facebook': {'reliability': 'LOW', 'reason': 'social_media'},
                'twitter': {'reliability': 'LOW', 'reason': 'social_media'},
                'instagram': {'reliability': 'LOW', 'reason': 'social_media'},
                'youtube': {'reliability': 'LOW', 'reason': 'video_platform'},
                'tiktok': {'reliability': 'LOW', 'reason': 'social_media'},
                'telegram': {'reliability': 'LOW', 'reason': 'messaging_app'},
                'whatsapp': {'reliability': 'LOW', 'reason': 'messaging_app'},
                'forwarded': {'reliability': 'LOW', 'reason': 'unverified_sharing'},
                'anonymous': {'reliability': 'LOW', 'reason': 'no_attribution'},
                'unknown': {'reliability': 'LOW', 'reason': 'unidentified_source'},
                'viral': {'reliability': 'LOW', 'reason': 'viral_content'},
                'breaking': {'reliability': 'MEDIUM_LOW', 'reason': 'unverified_breaking_news'},
            }
        }

    def _build_tier_descriptions(self) -> Dict[str, str]:
        """Build human-readable descriptions for reliability tiers."""
        return {
            'HIGH_PLUS': 'Exceptional reliability - Wire services and major public broadcasters with rigorous editorial standards',
            'HIGH': 'High reliability - Established newspapers and magazines with strong fact-checking processes',
            'FACT_CHECKERS': 'Specialized fact-checking organizations focused on verification and accuracy',
            'GOVERNMENT': 'Official government and international organization sources with institutional credibility',
            'ACADEMIC': 'Peer-reviewed academic and research institutions with scholarly standards',
            'MEDIUM_HIGH': 'Generally reliable sources but may exhibit noticeable editorial bias',
            'MEDIUM_LOW': 'Mixed reliability with significant bias, sensationalism, or quality issues',
            'LOW': 'Questionable reliability - conspiracy theories, pseudoscience, or propaganda sources'
        }

    def assess_source_reliability(self, source: str) -> str:
        """
        Determine reliability level of a news source.
        
        Args:
            source: Source name or URL to assess
            
        Returns:
            Reliability level: 'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'
        """
        if not source or not isinstance(source, str):
            return 'UNKNOWN'
        
        source_lower = source.lower().strip()
        
        # Check against comprehensive database
        for tier, sources in self.database.items():
            if tier == 'PATTERNS':
                continue
                
            for known_source in sources.keys():
                if known_source in source_lower:
                    # Map detailed tiers to simplified categories
                    reliability_mapping = {
                        'HIGH_PLUS': 'HIGH',
                        'HIGH': 'HIGH',
                        'FACT_CHECKERS': 'HIGH',
                        'GOVERNMENT': 'HIGH',
                        'ACADEMIC': 'HIGH',
                        'MEDIUM_HIGH': 'MEDIUM',
                        'MEDIUM_LOW': 'MEDIUM',
                        'LOW': 'LOW'
                    }
                    return reliability_mapping.get(tier, 'MEDIUM')
        
        # Check pattern-based indicators
        for pattern, info in self.database['PATTERNS'].items():
            if pattern in source_lower:
                return info['reliability']
        
        # Domain-based fallback assessment
        if source_lower.endswith('.gov'):
            return 'HIGH'
        elif source_lower.endswith('.edu'):
            return 'HIGH'
        elif source_lower.endswith('.org'):
            return 'MEDIUM'
        elif any(domain in source_lower for domain in ['.com', '.net', '.info']):
            return 'MEDIUM'
        
        return 'UNKNOWN'

    def get_source_details(self, source: str) -> Dict[str, Any]:
        """
        Get comprehensive source information including bias and factual reporting.
        
        Args:
            source: Source name or URL to analyze
            
        Returns:
            Dictionary with detailed source assessment
        """
        if not source or not isinstance(source, str):
            return self._get_unknown_source_details()
        
        source_lower = source.lower().strip()
        
        # Search database for exact matches
        for tier, sources in self.database.items():
            if tier == 'PATTERNS':
                continue
                
            for known_source, info in sources.items():
                if known_source in source_lower:
                    return {
                        'reliability_tier': tier,
                        'source_type': info['type'],
                        'bias_level': info['bias'],
                        'factual_reporting': info['factual'],
                        'matched_source': known_source,
                        'tier_description': self.tier_descriptions.get(tier, 'Unknown tier'),
                        'assessment_confidence': 'high'
                    }
        
        # Check for pattern matches
        for pattern, info in self.database['PATTERNS'].items():
            if pattern in source_lower:
                return {
                    'reliability_tier': info['reliability'],
                    'source_type': 'pattern_detected',
                    'bias_level': 'unknown',
                    'factual_reporting': 'unknown',
                    'matched_source': pattern,
                    'tier_description': f"Detected pattern: {info['reason']}",
                    'assessment_confidence': 'medium'
                }
        
        return self._get_unknown_source_details()

    def _get_unknown_source_details(self) -> Dict[str, Any]:
        """Return default details for unknown sources."""
        return {
            'reliability_tier': 'UNKNOWN',
            'source_type': 'unclassified',
            'bias_level': 'unknown',
            'factual_reporting': 'unknown',
            'matched_source': None,
            'tier_description': 'Source not found in reliability database',
            'assessment_confidence': 'low'
        }

    def get_reliability_summary(self, source: str) -> Dict[str, Any]:
        """
        Get comprehensive reliability summary for explanation generation.
        
        Args:
            source: Source to analyze
            
        Returns:
            Summary dictionary with reliability assessment and recommendations
        """
        details = self.get_source_details(source)
        reliability = self.assess_source_reliability(source)
        
        # Generate user-friendly summary
        reliability_descriptions = {
            'HIGH': 'This is a highly reliable source with strong editorial standards and fact-checking processes.',
            'MEDIUM': 'This source has mixed reliability - verify claims independently and consider potential bias.',
            'LOW': 'This source has questionable reliability - approach with significant skepticism and verify through other sources.',
            'UNKNOWN': 'Source reliability cannot be determined - exercise caution and verify information independently.'
        }
        
        # Generate verification recommendations
        verification_recommendations = {
            'HIGH': 'Cross-reference with other high-quality sources for complete context.',
            'MEDIUM': 'Verify key claims through multiple independent sources before sharing.',
            'LOW': 'Treat claims with extreme skepticism and seek verification from credible sources.',
            'UNKNOWN': 'Research source credibility and verify all claims through established sources.'
        }
        
        return {
            'source': source,
            'reliability_level': reliability,
            'reliability_description': reliability_descriptions.get(reliability, 'Unknown reliability level'),
            'source_details': details,
            'verification_recommendation': verification_recommendations.get(reliability, 'Verify independently'),
            'bias_warning': self._generate_bias_warning(details.get('bias_level', 'unknown')),
            'confidence_level': details.get('assessment_confidence', 'low')
        }

    def _generate_bias_warning(self, bias_level: str) -> Optional[str]:
        """Generate appropriate bias warning based on bias level."""
        bias_warnings = {
            'extreme_left': 'Warning: This source exhibits extreme left-wing bias that may significantly distort reporting.',
            'extreme_right': 'Warning: This source exhibits extreme right-wing bias that may significantly distort reporting.',
            'left': 'Note: This source has a noticeable left-leaning editorial perspective.',
            'right': 'Note: This source has a noticeable right-leaning editorial perspective.',
            'slight_left': 'This source has a slight left-leaning perspective in editorial content.',
            'slight_right': 'This source has a slight right-leaning perspective in editorial content.',
            'minimal': None,  # No warning needed for minimal bias
            'institutional': 'This source represents an institutional perspective that may favor established positions.',
        }
        return bias_warnings.get(bias_level)

    def search_sources(self, search_term: str) -> Dict[str, Any]:
        """
        Search for sources matching a specific term.
        
        Args:
            search_term: Term to search for in source names
            
        Returns:
            Dictionary of matching sources with their reliability details
        """
        if not search_term or not isinstance(search_term, str):
            return {}
        
        search_term = search_term.lower().strip()
        matches = {}
        
        for tier, sources in self.database.items():
            if tier == 'PATTERNS':
                continue
                
            for source_name, info in sources.items():
                if search_term in source_name:
                    matches[source_name] = {
                        'reliability_tier': tier,
                        'source_info': info,
                        'tier_description': self.tier_descriptions.get(tier, 'Unknown tier')
                    }
        
        return matches

    def get_tier_sources(self, tier: str) -> Dict[str, Dict]:
        """
        Get all sources in a specific reliability tier.
        
        Args:
            tier: Reliability tier to retrieve
            
        Returns:
            Dictionary of sources in the specified tier
        """
        return self.database.get(tier, {})

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get statistics about the source database."""
        stats = {}
        total_sources = 0
        
        for tier, sources in self.database.items():
            if tier == 'PATTERNS':
                stats[tier] = len(sources)
            else:
                count = len(sources)
                stats[tier] = count
                total_sources += count
        
        return {
            'tier_counts': stats,
            'total_sources': total_sources,
            'total_tiers': len([t for t in self.database.keys() if t != 'PATTERNS']),
            'total_patterns': stats.get('PATTERNS', 0),
            'database_version': '2.0'
        }

    def add_source(self, source_name: str, tier: str, source_info: Dict[str, str]) -> bool:
        """
        Add a new source to the database.
        
        Args:
            source_name: Name of source to add
            tier: Reliability tier for the source
            source_info: Dictionary with type, bias, factual information
            
        Returns:
            True if added successfully, False otherwise
        """
        if tier not in self.database or tier == 'PATTERNS':
            return False
        
        required_fields = {'type', 'bias', 'factual'}
        if not all(field in source_info for field in required_fields):
            return False
        
        self.database[tier][source_name.lower().strip()] = source_info
        return True

    def validate_source_entry(self, source_info: Dict[str, str]) -> tuple[bool, List[str]]:
        """
        Validate source information for database entry.
        
        Args:
            source_info: Source information to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required_fields = {'type', 'bias', 'factual'}
        
        # Check required fields
        missing_fields = required_fields - set(source_info.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Validate field values
        valid_types = {
            'wire_service', 'public_broadcaster', 'newspaper', 'magazine', 
            'fact_checker', 'government', 'international_org', 'academic',
            'cable_news', 'broadcast_news', 'business_news', 'online_news',
            'tabloid', 'partisan', 'conspiracy', 'pseudoscience', 'state_media',
            'online_magazine', 'financial_blog'
        }
        
        valid_bias = {
            'minimal', 'slight_left', 'slight_right', 'left', 'right',
            'extreme_left', 'extreme_right', 'institutional'
        }
        
        valid_factual = {'very_high', 'high', 'medium_high', 'medium', 'low', 'very_low'}
        
        if source_info.get('type') not in valid_types:
            errors.append(f"Invalid source type: {source_info.get('type')}")
        
        if source_info.get('bias') not in valid_bias:
            errors.append(f"Invalid bias level: {source_info.get('bias')}")
        
        if source_info.get('factual') not in valid_factual:
            errors.append(f"Invalid factual reporting level: {source_info.get('factual')}")
        
        return len(errors) == 0, errors


# Testing functionality
if __name__ == "__main__":
    """Test source reliability database functionality."""
    
    # Initialize database
    db = SourceReliabilityDatabase()
    
    print("=== SOURCE RELIABILITY DATABASE TEST ===")
    
    # Test source assessment
    test_sources = [
        "reuters.com",
        "cnn.com", 
        "infowars.com",
        "harvard.edu",
        "facebook.com",
        "unknown-blog.net"
    ]
    
    for source in test_sources:
        reliability = db.assess_source_reliability(source)
        details = db.get_source_details(source)
        summary = db.get_reliability_summary(source)
        
        print(f"\nSource: {source}")
        print(f"  Reliability: {reliability}")
        print(f"  Type: {details['source_type']}")
        print(f"  Bias: {details['bias_level']}")
        print(f"  Factual: {details['factual_reporting']}")
        if summary['bias_warning']:
            print(f"  Warning: {summary['bias_warning']}")
    
    # Test database statistics
    stats = db.get_database_statistics()
    print(f"\n=== DATABASE STATISTICS ===")
    print(f"Total sources: {stats['total_sources']}")
    print(f"Total tiers: {stats['total_tiers']}")
    print(f"Pattern indicators: {stats['total_patterns']}")
    
    # Test search functionality
    search_results = db.search_sources("news")
    print(f"\n=== SEARCH TEST ===")
    print(f"Sources containing 'news': {len(search_results)}")
    
    print("\n=== SOURCE DATABASE TESTING COMPLETED ===")
