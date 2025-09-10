# agents/credible_source/domain_classifier.py

"""
Domain Classifier

Production-ready domain classification for credible source agent.
Classifies article content into subject domains to enable targeted
source recommendations with enhanced performance tracking.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional


class DomainClassifier:
    """
    Classifies articles and claims into subject domains.
    
    Provides domain-specific classification using keyword analysis,
    pattern matching, and claim type inference for targeted source
    recommendations.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize domain classifier.
        
        Args:
            config: Optional configuration for classification parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize classification systems
        self.domain_keywords = self._initialize_domain_keywords()
        self.domain_patterns = self._initialize_domain_patterns()
        self.claim_type_mappings = self._initialize_claim_type_mappings()
        
        # Performance tracking
        self.classification_count = 0
        self.successful_classifications = 0
        self.total_processing_time = 0.0
        
        self.logger.info(f"Domain classifier initialized with {len(self.domain_keywords)} domains")

    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keyword databases."""
        return {
            'health': [
                'medical', 'health', 'disease', 'treatment', 'medicine', 'doctor',
                'hospital', 'clinic', 'patient', 'symptoms', 'diagnosis', 'therapy',
                'vaccine', 'drug', 'pharmaceutical', 'clinical', 'healthcare', 
                'wellness', 'epidemic', 'pandemic', 'virus', 'bacteria', 'infection',
                'immunity', 'mental health', 'public health', 'medical research'
            ],
            'politics': [
                'government', 'political', 'election', 'vote', 'congress', 'senate',
                'president', 'politician', 'policy', 'law', 'legislation', 'democrat',
                'republican', 'party', 'campaign', 'debate', 'administration',
                'white house', 'capitol', 'federal', 'state government', 'local government',
                'supreme court', 'justice', 'ruling', 'constitutional', 'democracy'
            ],
            'science': [
                'science', 'scientific', 'research', 'study', 'experiment', 'data',
                'analysis', 'laboratory', 'peer-reviewed', 'journal', 'publication',
                'hypothesis', 'theory', 'evidence', 'discovery', 'innovation',
                'physics', 'chemistry', 'biology', 'mathematics', 'statistics',
                'methodology', 'academic', 'university research', 'scientific method'
            ],
            'technology': [
                'technology', 'tech', 'computer', 'software', 'hardware', 'internet',
                'digital', 'online', 'app', 'platform', 'algorithm', 'artificial intelligence',
                'machine learning', 'ai', 'cybersecurity', 'data', 'cloud computing',
                'mobile', 'smartphone', 'programming', 'coding', 'blockchain',
                'automation', 'robotics', 'virtual reality', 'augmented reality'
            ],
            'economics': [
                'economic', 'economy', 'financial', 'money', 'market', 'stock',
                'investment', 'business', 'company', 'corporation', 'trade',
                'inflation', 'recession', 'gdp', 'unemployment', 'employment',
                'banking', 'finance', 'wall street', 'federal reserve', 'interest rate',
                'fiscal policy', 'monetary policy', 'budget', 'debt', 'deficit'
            ],
            'environment': [
                'environment', 'environmental', 'climate', 'weather', 'global warming',
                'climate change', 'carbon', 'emissions', 'renewable energy',
                'pollution', 'conservation', 'wildlife', 'ecosystem', 'sustainability',
                'green energy', 'solar', 'wind power', 'fossil fuel', 'biodiversity',
                'deforestation', 'ocean', 'atmosphere', 'greenhouse gas'
            ],
            'education': [
                'education', 'educational', 'school', 'university', 'college',
                'student', 'teacher', 'professor', 'learning', 'curriculum',
                'academic', 'scholarship', 'graduation', 'degree', 'classroom',
                'training', 'skill development', 'pedagogy', 'literacy'
            ],
            'international': [
                'international', 'global', 'world', 'foreign', 'country', 'nation',
                'diplomatic', 'embassy', 'trade', 'conflict', 'peace',
                'united nations', 'nato', 'alliance', 'treaty', 'sanctions',
                'foreign policy', 'geopolitics', 'multinational', 'cross-border'
            ]
        }

    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize domain-specific pattern matching rules."""
        return {
            'health': [
                r'published\s+in.*medical\s+journal',
                r'clinical\s+trial\s+results',
                r'fda\s+approve',
                r'health\s+officials\s+say',
                r'medical\s+experts\s+warn',
                r'public\s+health\s+emergency',
                r'disease\s+outbreak',
                r'vaccine\s+effectiveness'
            ],
            'politics': [
                r'according\s+to\s+government\s+sources',
                r'congressional\s+hearing',
                r'white\s+house\s+statement',
                r'supreme\s+court\s+ruling',
                r'election\s+results',
                r'policy\s+announcement',
                r'legislative\s+session',
                r'political\s+campaign'
            ],
            'science': [
                r'peer-reviewed\s+study',
                r'published\s+in\s+nature',
                r'research\s+team\s+at',
                r'scientific\s+breakthrough',
                r'study\s+finds',
                r'research\s+shows',
                r'laboratory\s+analysis',
                r'experimental\s+results'
            ],
            'technology': [
                r'tech\s+company\s+announces',
                r'new\s+app\s+launch',
                r'cybersecurity\s+breach',
                r'ai\s+technology',
                r'software\s+update',
                r'digital\s+transformation',
                r'cloud\s+computing',
                r'machine\s+learning\s+model'
            ],
            'economics': [
                r'stock\s+market\s+analysis',
                r'economic\s+indicators',
                r'federal\s+reserve\s+decision',
                r'quarterly\s+earnings',
                r'market\s+volatility',
                r'inflation\s+rate',
                r'unemployment\s+data',
                r'economic\s+growth'
            ],
            'environment': [
                r'climate\s+change\s+report',
                r'environmental\s+impact',
                r'carbon\s+emissions',
                r'renewable\s+energy',
                r'pollution\s+levels',
                r'conservation\s+efforts',
                r'ecosystem\s+study',
                r'global\s+warming'
            ],
            'education': [
                r'education\s+policy',
                r'student\s+performance',
                r'curriculum\s+changes',
                r'university\s+research',
                r'academic\s+achievement',
                r'educational\s+funding',
                r'school\s+district',
                r'higher\s+education'
            ],
            'international': [
                r'international\s+relations',
                r'foreign\s+policy',
                r'diplomatic\s+talks',
                r'global\s+summit',
                r'trade\s+agreement',
                r'international\s+conflict',
                r'cross-border',
                r'multinational\s+cooperation'
            ]
        }

    def _initialize_claim_type_mappings(self) -> Dict[str, str]:
        """Initialize claim type to domain mappings."""
        return {
            'Research': 'science',
            'Statistical': 'science',
            'Medical': 'health',
            'Political': 'politics',
            'Economic': 'economics',
            'Environmental': 'environment',
            'Technology': 'technology',
            'Educational': 'education',
            'International': 'international',
            'Attribution': 'general',
            'Factual': 'general'
        }

    def classify_domain(self, article_text: str, extracted_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify article domain using comprehensive analysis.
        
        Args:
            article_text: Article content to classify
            extracted_claims: Claims extracted from the article
            
        Returns:
            Dictionary containing classification results and metadata
        """
        start_time = time.time()
        
        # Initialize domain scores
        domain_scores = {domain: 0 for domain in self.domain_keywords.keys()}
        article_lower = article_text.lower()
        
        # Keyword-based scoring
        for domain, keywords in self.domain_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword in article_lower)
            domain_scores[domain] += keyword_matches
        
        # Pattern-based scoring (weighted higher)
        for domain, patterns in self.domain_patterns.items():
            pattern_score = 0
            for pattern in patterns:
                matches = re.findall(pattern, article_lower, re.IGNORECASE)
                pattern_score += len(matches) * 2  # Patterns weighted higher than keywords
            domain_scores[domain] += pattern_score
        
        # Claim type influence (weighted highest)
        for claim in extracted_claims:
            claim_type = claim.get('claim_type', '')
            mapped_domain = self.claim_type_mappings.get(claim_type)
            if mapped_domain and mapped_domain in domain_scores:
                priority = claim.get('priority', 3)
                # Higher priority claims have more influence
                weight = 4 - min(priority, 3)  # Priority 1 = weight 3, priority 3+ = weight 1
                domain_scores[mapped_domain] += weight
        
        # Calculate confidence and determine primary domain
        total_score = sum(domain_scores.values())
        
        if total_score == 0:
            primary_domain = 'general'
            confidence = 0.0
            domain_classified = False
        else:
            primary_domain = max(domain_scores, key=domain_scores.get)
            primary_score = domain_scores[primary_domain]
            confidence = primary_score / total_score
            
            # Use configurable confidence threshold
            confidence_threshold = self.config.get('confidence_threshold', 0.3)
            domain_classified = confidence >= confidence_threshold
            
            if not domain_classified:
                primary_domain = 'general'
        
        # Get secondary domains
        sorted_domains = sorted(
            [(domain, score) for domain, score in domain_scores.items() 
             if domain != primary_domain and score > 0],
            key=lambda x: x[1], 
            reverse=True
        )
        secondary_domains = [domain for domain, score in sorted_domains[:2]]
        
        # Calculate processing time and update metrics
        processing_time = time.time() - start_time
        self.classification_count += 1
        if domain_classified:
            self.successful_classifications += 1
        self.total_processing_time += processing_time
        
        # Calculate domain strength (how decisive the classification is)
        domain_strength = self._calculate_domain_strength(domain_scores, primary_domain)
        
        return {
            'primary_domain': primary_domain,
            'confidence': round(confidence, 3),
            'domain_classified': domain_classified,
            'domain_scores': domain_scores,
            'secondary_domains': secondary_domains,
            'domain_strength': domain_strength,
            'total_indicators': total_score,
            'classification_time_ms': round(processing_time * 1000, 2),
            'confidence_threshold_used': confidence_threshold,
            'claims_analyzed': len(extracted_claims)
        }

    def _calculate_domain_strength(self, domain_scores: Dict[str, int], primary_domain: str) -> str:
        """
        Calculate how decisive the domain classification is.
        
        Args:
            domain_scores: Dictionary of domain scores
            primary_domain: The classified primary domain
            
        Returns:
            Strength indicator (strong, moderate, weak)
        """
        if primary_domain == 'general':
            return 'weak'
        
        primary_score = domain_scores[primary_domain]
        other_scores = [score for domain, score in domain_scores.items() 
                       if domain != primary_domain]
        
        if not other_scores or primary_score == 0:
            return 'weak'
        
        max_other_score = max(other_scores)
        
        # Calculate ratio between primary and next highest
        if max_other_score == 0:
            return 'strong'
        
        ratio = primary_score / max_other_score
        
        if ratio >= 2.0:
            return 'strong'
        elif ratio >= 1.5:
            return 'moderate'
        else:
            return 'weak'

    def get_domain_keywords(self, domain: str) -> List[str]:
        """
        Get keywords for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of keywords for the domain
        """
        return self.domain_keywords.get(domain, [])

    def get_domain_patterns(self, domain: str) -> List[str]:
        """
        Get patterns for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of regex patterns for the domain
        """
        return self.domain_patterns.get(domain, [])

    def get_supported_domains(self) -> List[str]:
        """Get list of all supported domains."""
        return list(self.domain_keywords.keys())

    def analyze_domain_distribution(self, domain_scores: Dict[str, int]) -> Dict[str, Any]:
        """
        Analyze the distribution of domain indicators.
        
        Args:
            domain_scores: Dictionary of domain scores
            
        Returns:
            Analysis of domain score distribution
        """
        total_score = sum(domain_scores.values())
        
        if total_score == 0:
            return {
                'distribution': 'no_indicators',
                'dominant_domains': [],
                'score_spread': 0.0,
                'classification_difficulty': 'impossible'
            }
        
        # Calculate percentages
        domain_percentages = {
            domain: (score / total_score) * 100 
            for domain, score in domain_scores.items()
        }
        
        # Find dominant domains (>20% of total score)
        dominant_domains = [
            domain for domain, percentage in domain_percentages.items() 
            if percentage >= 20.0
        ]
        
        # Calculate score spread (standard deviation)
        scores = list(domain_scores.values())
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        score_spread = variance ** 0.5
        
        # Determine classification difficulty
        if len(dominant_domains) == 1 and domain_percentages[dominant_domains[0]] >= 50:
            difficulty = 'easy'
        elif len(dominant_domains) <= 2:
            difficulty = 'moderate'
        else:
            difficulty = 'difficult'
        
        return {
            'distribution': 'multi_domain' if len(dominant_domains) > 1 else 'single_domain',
            'dominant_domains': dominant_domains,
            'domain_percentages': {k: round(v, 1) for k, v in domain_percentages.items() if v > 0},
            'score_spread': round(score_spread, 2),
            'classification_difficulty': difficulty
        }

    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics."""
        success_rate = (
            (self.successful_classifications / self.classification_count * 100) 
            if self.classification_count > 0 else 0
        )
        
        avg_processing_time = (
            (self.total_processing_time / self.classification_count) 
            if self.classification_count > 0 else 0
        )
        
        return {
            'total_classifications': self.classification_count,
            'successful_classifications': self.successful_classifications,
            'success_rate_percent': round(success_rate, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'domains_supported': len(self.domain_keywords),
            'total_keywords': sum(len(keywords) for keywords in self.domain_keywords.values()),
            'total_patterns': sum(len(patterns) for patterns in self.domain_patterns.values()),
            'claim_mappings_available': len(self.claim_type_mappings),
            'configuration_applied': bool(self.config)
        }

    def validate_classification_data(self) -> Dict[str, Any]:
        """Validate classification database integrity."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for empty keyword lists
        for domain, keywords in self.domain_keywords.items():
            if not keywords:
                validation_results['issues'].append(f"Empty keyword list for domain: {domain}")
                validation_results['valid'] = False
        
        # Check for empty pattern lists
        for domain, patterns in self.domain_patterns.items():
            if not patterns:
                validation_results['warnings'].append(f"Empty pattern list for domain: {domain}")
        
        # Check for orphaned claim mappings
        supported_domains = set(self.domain_keywords.keys())
        for claim_type, mapped_domain in self.claim_type_mappings.items():
            if mapped_domain != 'general' and mapped_domain not in supported_domains:
                validation_results['issues'].append(f"Claim type '{claim_type}' maps to unsupported domain: {mapped_domain}")
                validation_results['valid'] = False
        
        # Check for very short keywords (might be too generic)
        short_keywords = []
        for domain, keywords in self.domain_keywords.items():
            short_keywords.extend([kw for kw in keywords if len(kw) < 4])
        
        if short_keywords:
            validation_results['warnings'].append(f"Very short keywords detected: {short_keywords[:5]}")
        
        return validation_results


# Testing functionality
if __name__ == "__main__":
    """Test domain classifier functionality."""
    
    # Initialize classifier with test configuration
    test_config = {
        'confidence_threshold': 0.4
    }
    
    classifier = DomainClassifier(test_config)
    
    # Test classification
    test_article = """
    A groundbreaking study published in Nature Medicine by researchers at Harvard 
    Medical School reveals new insights into cancer treatment. The clinical trial 
    involved 500 patients and showed a 75% improvement in survival rates. The FDA 
    is reviewing the data for potential drug approval.
    """
    
    test_claims = [
        {
            'text': 'Study published in Nature Medicine shows cancer treatment breakthrough',
            'claim_type': 'Research',
            'priority': 1
        },
        {
            'text': 'Clinical trial showed 75% improvement in survival rates',
            'claim_type': 'Statistical',
            'priority': 1
        },
        {
            'text': 'FDA reviewing data for drug approval',
            'claim_type': 'Medical',
            'priority': 2
        }
    ]
    
    print("=== DOMAIN CLASSIFICATION TEST ===")
    result = classifier.classify_domain(test_article, test_claims)
    
    print(f"Primary domain: {result['primary_domain']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Domain classified: {result['domain_classified']}")
    print(f"Domain strength: {result['domain_strength']}")
    print(f"Secondary domains: {result['secondary_domains']}")
    print(f"Total indicators: {result['total_indicators']}")
    print(f"Processing time: {result['classification_time_ms']:.1f}ms")
    
    print(f"\nDomain scores:")
    for domain, score in result['domain_scores'].items():
        if score > 0:
            print(f"  {domain}: {score}")
    
    # Test domain distribution analysis
    distribution_analysis = classifier.analyze_domain_distribution(result['domain_scores'])
    print(f"\n=== DOMAIN DISTRIBUTION ANALYSIS ===")
    print(f"Distribution type: {distribution_analysis['distribution']}")
    print(f"Dominant domains: {distribution_analysis['dominant_domains']}")
    print(f"Classification difficulty: {distribution_analysis['classification_difficulty']}")
    print(f"Domain percentages: {distribution_analysis['domain_percentages']}")
    
    # Test statistics
    stats = classifier.get_classification_statistics()
    print(f"\n=== CLASSIFICATION STATISTICS ===")
    print(f"Classifications completed: {stats['total_classifications']}")
    print(f"Success rate: {stats['success_rate_percent']:.1f}%")
    print(f"Average processing time: {stats['average_processing_time_ms']:.1f}ms")
    print(f"Domains supported: {stats['domains_supported']}")
    print(f"Total keywords: {stats['total_keywords']}")
    print(f"Total patterns: {stats['total_patterns']}")
    
    # Test validation
    validation = classifier.validate_classification_data()
    print(f"\n=== DATABASE VALIDATION ===")
    print(f"Validation result: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    print("\n=== DOMAIN CLASSIFICATION TESTING COMPLETED ===")
