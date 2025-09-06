# agents/llm_explanation/source_database.py
"""
Source Reliability Database for LLM Explanation Agent

This module contains the comprehensive source reliability database that helps
assess the credibility of news sources. Separating this allows for easier
updates and maintenance of source information.

Features:
- Comprehensive source reliability tiers
- Bias and factual reporting assessments
- Pattern-based source detection
- Dynamic source lookup capabilities
"""

from typing import Dict, Any, Optional

class SourceReliabilityDatabase:
    """
    📊 COMPREHENSIVE SOURCE RELIABILITY DATABASE
    
    This class manages a comprehensive database of news sources and their
    reliability characteristics. It provides methods to assess source
    credibility, bias levels, and factual reporting quality.
    """
    
    def __init__(self):
        """Initialize the source database"""
        self.database = self._build_source_database()
        self.tier_descriptions = self._build_tier_descriptions()
    
    def _build_source_database(self) -> Dict[str, Dict]:
        """
        🏗️ BUILD COMPREHENSIVE SOURCE DATABASE
        
        Creates the complete database of news sources organized by reliability tiers.
        """
        return {
            # 🥇 TIER 1: HIGHEST RELIABILITY
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
            
            # 🥈 TIER 2: HIGH RELIABILITY
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
            
            # 🔍 TIER 3: FACT-CHECKERS
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
            
            # 🏛️ TIER 4: GOVERNMENT & OFFICIAL SOURCES
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
            
            # 🎓 TIER 5: ACADEMIC & RESEARCH
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
            
            # 🥉 TIER 6: MODERATE RELIABILITY
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
            
            # ⚠️ TIER 7: LOWER RELIABILITY
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
            
            # 🚨 TIER 8: QUESTIONABLE SOURCES
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
            
            # 🔍 PATTERN-BASED INDICATORS
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
        """Build human-readable descriptions for each reliability tier"""
        return {
            'HIGH_PLUS': 'Exceptional reliability - Wire services and major public broadcasters',
            'HIGH': 'High reliability - Established newspapers and magazines with strong editorial standards',
            'FACT_CHECKERS': 'Specialized fact-checking organizations with verification focus',
            'GOVERNMENT': 'Official government and international organization sources',
            'ACADEMIC': 'Peer-reviewed academic and research institutions',
            'MEDIUM_HIGH': 'Generally reliable but may have noticeable bias',
            'MEDIUM_LOW': 'Mixed reliability with significant bias or sensationalism',
            'LOW': 'Questionable reliability - conspiracy, pseudoscience, or state propaganda'
        }
    
    def assess_source_reliability(self, source: str) -> str:
        """
        📊 ASSESS SOURCE RELIABILITY
        
        Determines the reliability level of a news source.
        
        Args:
            source: Name or URL of the news source
            
        Returns:
            String: 'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'
        """
        source_lower = source.lower().strip()
        
        # Check against comprehensive database
        for tier, sources in self.database.items():
            if tier == 'PATTERNS':
                continue
            
            for known_source, info in sources.items():
                if known_source in source_lower:
                    # Map detailed tiers to simple HIGH/MEDIUM/LOW
                    reliability_map = {
                        'HIGH_PLUS': 'HIGH',
                        'HIGH': 'HIGH',
                        'FACT_CHECKERS': 'HIGH',
                        'GOVERNMENT': 'HIGH',
                        'ACADEMIC': 'HIGH',
                        'MEDIUM_HIGH': 'MEDIUM',
                        'MEDIUM_LOW': 'MEDIUM',
                        'LOW': 'LOW'
                    }
                    return reliability_map.get(tier, 'MEDIUM')
        
        # Check pattern-based indicators
        for pattern, info in self.database['PATTERNS'].items():
            if pattern in source_lower:
                return info['reliability']
        
        # Domain-based assessment
        if source_lower.endswith('.gov'):
            return 'HIGH'
        elif source_lower.endswith('.edu'):
            return 'HIGH'
        elif source_lower.endswith('.org'):
            return 'MEDIUM'
        elif any(indicator in source_lower for indicator in ['.com', '.net', '.info']):
            return 'MEDIUM'
        
        return 'UNKNOWN'
    
    def get_source_details(self, source: str) -> Dict[str, Any]:
        """
        📋 GET DETAILED SOURCE INFORMATION
        
        Provides comprehensive information about a source.
        
        Args:
            source: Name or URL of the news source
            
        Returns:
            Dictionary with detailed source information
        """
        source_lower = source.lower().strip()
        
        # Search through database for a match
        for tier, sources in self.database.items():
            if tier == 'PATTERNS':
                continue
            
            for known_source, info in sources.items():
                if known_source in source_lower:
                    return {
                        'reliability': tier,
                        'source_type': info['type'],
                        'bias_level': info['bias'],
                        'factual_reporting': info['factual'],
                        'matched_source': known_source,
                        'tier_description': self.tier_descriptions.get(tier, 'Unknown tier')
                    }
        
        # If no match found, return unknown values
        return {
            'reliability': 'UNKNOWN',
            'source_type': 'unclassified',
            'bias_level': 'unknown',
            'factual_reporting': 'unknown',
            'matched_source': None,
            'tier_description': 'Source not found in database'
        }
    
    def add_source(self, source_name: str, tier: str, source_info: Dict[str, str]):
        """
        ➕ ADD NEW SOURCE TO DATABASE
        
        Allows dynamic addition of new sources to the database.
        
        Args:
            source_name: Name of the source to add
            tier: Reliability tier to add it to
            source_info: Dictionary with type, bias, factual information
        """
        if tier not in self.database:
            raise ValueError(f"Unknown tier: {tier}")
        
        self.database[tier][source_name.lower()] = source_info
    
    def get_tier_description(self, tier: str) -> str:
        """Get human-readable description for a tier"""
        return self.tier_descriptions.get(tier, 'Unknown tier')
    
    def list_sources_by_tier(self, tier: str) -> Dict[str, Dict]:
        """Get all sources in a specific tier"""
        return self.database.get(tier, {})
    
    def search_sources(self, search_term: str) -> Dict[str, Any]:
        """
        🔍 SEARCH FOR SOURCES
        
        Search for sources containing a specific term.
        
        Args:
            search_term: Term to search for
            
        Returns:
            Dictionary of matching sources with their details
        """
        search_term = search_term.lower()
        matches = {}
        
        for tier, sources in self.database.items():
            if tier == 'PATTERNS':
                continue
            
            for source_name, info in sources.items():
                if search_term in source_name:
                    matches[source_name] = {
                        'tier': tier,
                        'info': info,
                        'description': self.tier_descriptions.get(tier, 'Unknown')
                    }
        
        return matches
