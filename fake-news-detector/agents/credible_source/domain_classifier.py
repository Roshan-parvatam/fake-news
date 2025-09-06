# agents/credible_source/domain_classifier.py
"""
Domain Classification for Credible Source Agent - Config Enhanced

Enhanced domain classification with better performance tracking and configuration awareness.
"""

from typing import Dict, List, Any
import re
import logging
import time

class DomainClassifier:
    """
    🏷️ ENHANCED DOMAIN CLASSIFIER WITH CONFIG AWARENESS
    
    This class classifies articles into domains to provide domain-specific
    source recommendations with enhanced performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the domain classifier with optional config
        
        Args:
            config: Optional configuration for domain classification
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize domain classification systems
        self.domain_keywords = self._initialize_domain_keywords()
        self.domain_patterns = self._initialize_domain_patterns()
        self.claim_type_mappings = self._initialize_claim_type_mappings()
        
        # Performance tracking
        self.classifier_stats = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'classification_time_total': 0.0,
            'config_applied': bool(config)
        }
        
        self.logger.info(f"✅ DomainClassifier initialized with {len(self.domain_keywords)} domain categories")
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """
        🏷️ DOMAIN KEYWORDS DATABASE - Enhanced categorization
        """
        return {
            'health': [
                'medical', 'health', 'disease', 'treatment', 'medicine', 'doctor',
                'hospital', 'clinic', 'patient', 'symptoms', 'diagnosis', 'therapy',
                'vaccine', 'drug', 'pharmaceutical', 'clinical', 'study', 'research',
                'fda', 'cdc', 'who', 'healthcare', 'wellness', 'epidemic', 'pandemic',
                'virus', 'bacteria', 'infection', 'immunity', 'mental health'
            ],
            'politics': [
                'government', 'political', 'election', 'vote', 'congress', 'senate',
                'president', 'politician', 'policy', 'law', 'legislation', 'democrat',
                'republican', 'party', 'campaign', 'debate', 'administration',
                'white house', 'capitol', 'federal', 'state', 'local government',
                'supreme court', 'justice', 'ruling'
            ],
            'science': [
                'science', 'scientific', 'research', 'study', 'experiment', 'data',
                'analysis', 'laboratory', 'peer-reviewed', 'journal', 'publication',
                'hypothesis', 'theory', 'evidence', 'discovery', 'innovation',
                'technology', 'engineering', 'physics', 'chemistry', 'biology',
                'mathematics', 'statistics', 'methodology'
            ],
            'technology': [
                'technology', 'tech', 'computer', 'software', 'hardware', 'internet',
                'digital', 'online', 'app', 'platform', 'algorithm', 'artificial intelligence',
                'machine learning', 'ai', 'cybersecurity', 'data', 'cloud', 'mobile',
                'smartphone', 'innovation', 'startup', 'silicon valley'
            ],
            'economics': [
                'economic', 'economy', 'financial', 'money', 'market', 'stock',
                'investment', 'business', 'company', 'corporation', 'trade',
                'inflation', 'recession', 'gdp', 'unemployment', 'job', 'employment',
                'banking', 'finance', 'wall street', 'federal reserve', 'interest rate'
            ],
            'environment': [
                'environment', 'environmental', 'climate', 'weather', 'global warming',
                'climate change', 'carbon', 'emissions', 'renewable', 'energy',
                'pollution', 'conservation', 'wildlife', 'ecosystem', 'sustainability',
                'green', 'solar', 'wind', 'fossil fuel', 'epa'
            ],
            'education': [
                'education', 'educational', 'school', 'university', 'college',
                'student', 'teacher', 'professor', 'learning', 'curriculum',
                'academic', 'scholarship', 'graduation', 'degree', 'classroom',
                'training', 'skill', 'knowledge'
            ],
            'international': [
                'international', 'global', 'world', 'foreign', 'country', 'nation',
                'diplomatic', 'embassy', 'trade', 'war', 'conflict', 'peace',
                'united nations', 'nato', 'alliance', 'treaty', 'sanctions'
            ]
        }
    
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """
        🏷️ DOMAIN PATTERN DATABASE - Advanced pattern matching
        """
        return {
            'health': [
                r'published\s+in.*medical\s+journal',
                r'clinical\s+trial\s+results',
                r'fda\s+approve',
                r'health\s+officials\s+say',
                r'medical\s+experts\s+warn'
            ],
            'politics': [
                r'according\s+to\s+government\s+sources',
                r'congressional\s+hearing',
                r'white\s+house\s+statement',
                r'supreme\s+court\s+ruling',
                r'election\s+results'
            ],
            'science': [
                r'peer-reviewed\s+study',
                r'published\s+in\s+nature',
                r'research\s+team\s+at',
                r'scientific\s+breakthrough',
                r'study\s+finds'
            ],
            'technology': [
                r'tech\s+company\s+announces',
                r'new\s+app\s+launch',
                r'cybersecurity\s+breach',
                r'ai\s+technology',
                r'software\s+update'
            ],
            'economics': [
                r'stock\s+market\s+analysis',
                r'economic\s+indicators',
                r'federal\s+reserve\s+decision',
                r'quarterly\s+earnings',
                r'market\s+volatility'
            ]
        }
    
    def _initialize_claim_type_mappings(self) -> Dict[str, str]:
        """
        🏷️ CLAIM TYPE TO DOMAIN MAPPINGS
        """
        return {
            'Research': 'science',
            'Statistical': 'science',
            'Medical': 'health',
            'Political': 'politics',
            'Economic': 'economics',
            'Environmental': 'environment',
            'Technology': 'technology'
        }
    
    def classify_domain(self, article_text: str, extracted_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        🏷️ CLASSIFY ARTICLE DOMAIN WITH CONFIG AWARENESS
        
        Classify article domain using keywords, patterns, and claims with performance tracking.
        """
        start_time = time.time()
        
        # Initialize domain scores
        domain_scores = {domain: 0 for domain in self.domain_keywords.keys()}
        
        # Analyze text content
        article_lower = article_text.lower()
        
        # Score based on keywords
        for domain, keywords in self.domain_keywords.items():
            keyword_score = sum(1 for keyword in keywords if keyword in article_lower)
            domain_scores[domain] += keyword_score
        
        # Score based on patterns
        for domain, patterns in self.domain_patterns.items():
            pattern_score = 0
            for pattern in patterns:
                matches = re.findall(pattern, article_lower, re.IGNORECASE)
                pattern_score += len(matches) * 2  # Patterns weighted higher
            domain_scores[domain] += pattern_score
        
        # Score based on extracted claims
        for claim in extracted_claims:
            claim_type = claim.get('claim_type', '')
            mapped_domain = self.claim_type_mappings.get(claim_type)
            if mapped_domain and mapped_domain in domain_scores:
                domain_scores[mapped_domain] += 1
        
        # Calculate confidence and determine primary domain
        total_score = sum(domain_scores.values())
        
        if total_score == 0:
            primary_domain = 'general'
            confidence = 0.1
            domain_classified = False
        else:
            primary_domain = max(domain_scores, key=domain_scores.get)
            primary_score = domain_scores[primary_domain]
            confidence = primary_score / total_score if total_score > 0 else 0
            
            # Use config threshold
            confidence_threshold = self.config.get('confidence_threshold', 0.3) if self.config else 0.3
            domain_classified = confidence >= confidence_threshold
            
            if not domain_classified:
                primary_domain = 'general'
        
        # Get secondary domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_domains = [domain for domain, score in sorted_domains[1:3] if score > 0]
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.classifier_stats['total_classifications'] += 1
        if domain_classified:
            self.classifier_stats['successful_classifications'] += 1
        self.classifier_stats['classification_time_total'] += processing_time
        
        return {
            'primary_domain': primary_domain,
            'confidence': round(confidence, 3),
            'domain_classified': domain_classified,
            'domain_scores': domain_scores,
            'secondary_domains': secondary_domains,
            'total_indicators': total_score,
            'classification_time_ms': round(processing_time * 1000, 2),
            'confidence_threshold_used': confidence_threshold if domain_classified else None,
            'config_applied': bool(self.config)
        }
    
    def get_domain_specific_keywords(self, domain: str) -> List[str]:
        """Get keywords for a specific domain"""
        return self.domain_keywords.get(domain, [])
    
    def get_classifier_statistics(self) -> Dict[str, Any]:
        """Get comprehensive classifier statistics"""
        base_stats = {
            'total_domains': len(self.domain_keywords),
            'total_keywords': sum(len(keywords) for keywords in self.domain_keywords.values()),
            'total_patterns': sum(len(patterns) for patterns in self.domain_patterns.values()),
            'claim_type_mappings': len(self.claim_type_mappings)
        }
        
        # Performance stats
        performance_stats = self.classifier_stats.copy()
        if performance_stats['total_classifications'] > 0:
            performance_stats['success_rate'] = round(
                (performance_stats['successful_classifications'] / performance_stats['total_classifications']) * 100, 2
            )
            performance_stats['average_classification_time_ms'] = round(
                (performance_stats['classification_time_total'] / performance_stats['total_classifications']) * 1000, 2
            )
        
        return {**base_stats, 'performance_stats': performance_stats}

# Testing
if __name__ == "__main__":
    """Test domain classifier with config"""
    test_config = {
        'confidence_threshold': 0.4
    }
    
    classifier = DomainClassifier(test_config)
    
    test_text = """
    A new medical study published in Nature Medicine shows that the experimental
    treatment reduces symptoms by 75%. The FDA is reviewing the clinical trial
    data from Harvard Medical School researchers who conducted the study.
    """
    
    test_claims = [
        {'claim_type': 'Research', 'text': 'Study published in Nature Medicine'},
        {'claim_type': 'Statistical', 'text': '75% symptom reduction'},
        {'claim_type': 'Attribution', 'text': 'Harvard Medical School researchers'}
    ]
    
    classification = classifier.classify_domain(test_text, test_claims)
    
    print(f"Domain classification results:")
    print(f"Primary domain: {classification['primary_domain']}")
    print(f"Confidence: {classification['confidence']:.3f}")
    print(f"Domain classified: {classification['domain_classified']}")
    print(f"Total indicators: {classification['total_indicators']}")
    
    print(f"\nDomain scores:")
    for domain, score in classification['domain_scores'].items():
        if score > 0:
            print(f"  {domain}: {score}")
    
    stats = classifier.get_classifier_statistics()
    print(f"\nClassifier has {stats['total_keywords']} keywords across {stats['total_domains']} domains")
