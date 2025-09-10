# agents/claim_extractor/patterns.py

"""
Claim Pattern Database

Production-ready pattern definitions and analysis for claim extraction agent.
Enhanced with configurable detection algorithms, comprehensive pattern matching,
and performance tracking without emoji spam.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional


class ClaimPatternDatabase:
    """
    Pattern database for identifying claim-rich content and potential claims.
    
    Provides configurable pattern matching for statistical claims, attributions,
    research findings, and other verifiable statements with performance tracking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize claim pattern database.
        
        Args:
            config: Optional configuration for pattern matching parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize pattern databases
        self.claim_patterns = self._initialize_claim_patterns()
        self.statistical_indicators = self._initialize_statistical_indicators()
        self.quote_indicators = self._initialize_quote_indicators()
        self.verification_indicators = self._initialize_verification_indicators()
        
        # Performance tracking
        self.analysis_count = 0
        self.total_processing_time = 0.0
        self.total_patterns_found = 0
        
        total_patterns = sum(len(patterns) for patterns in self.claim_patterns.values())
        self.logger.info(f"Claim pattern database initialized with {total_patterns} patterns across {len(self.claim_patterns)} categories")

    def _initialize_claim_patterns(self) -> Dict[str, List[str]]:
        """Initialize comprehensive claim pattern categories."""
        return {
            'statistical': [
                r'\d+\.?\d*\s*%',  # Percentages
                r'\d+\.?\d*\s*percent',  # Percent spelled out
                r'\d+\.?\d*\s*(million|billion|thousand|trillion)',  # Large numbers
                r'\$\d+\.?\d*[kmbt]?',  # Dollar amounts
                r'\d+\.?\d*\s*(times|fold)',  # Multipliers
                r'increased.*by.*\d+',
                r'decreased.*by.*\d+',
                r'rose.*to.*\d+',
                r'fell.*to.*\d+',
                r'dropped.*by.*\d+',
                r'climbed.*to.*\d+',
                r'\d+\.?\d*\s*(points?|basis\s+points?)',
                r'(up|down)\s+\d+\.?\d*\s*%',
                r'growth.*of.*\d+',
                r'decline.*of.*\d+',
                r'surge.*of.*\d+',
                r'jump.*of.*\d+',
                r'\d+\s*(year|month|day|week)s?\s+(ago|later)',
                r'within\s+\d+\s*(days?|weeks?|months?|years?)',
                r'more\s+than\s+\d+',
                r'less\s+than\s+\d+',
                r'at\s+least\s+\d+',
                r'approximately\s+\d+',
                r'roughly\s+\d+',
                r'about\s+\d+'
            ],
            'event': [
                r'announced\s+(?:on|that|yesterday|today|this\s+week)',
                r'occurred\s+(?:on|at|in|yesterday|today)',
                r'will\s+(?:take\s+place|happen|occur|begin)',
                r'meeting\s+(?:held|scheduled|planned|took\s+place)',
                r'launched\s+(?:on|in|at|yesterday|today)',
                r'signed\s+(?:on|in|at|yesterday|today)',
                r'passed\s+(?:the|a)\s+(?:law|bill|resolution)',
                r'voted\s+(?:to|on|for|against|yesterday)',
                r'appointed\s+(?:as|to|on|yesterday)',
                r'resigned\s+(?:from|as|on|yesterday)',
                r'arrested\s+(?:on|for|yesterday|today)',
                r'died\s+(?:on|at|from|yesterday|today)',
                r'began\s+(?:on|in|at|yesterday|today)',
                r'ended\s+(?:on|in|at|yesterday|today)',
                r'scheduled\s+for',
                r'took\s+place\s+(?:on|in|at|yesterday)',
                r'confirmed\s+(?:on|yesterday|today)',
                r'denied\s+(?:on|yesterday|today)',
                r'rejected\s+(?:on|yesterday|today)',
                r'approved\s+(?:on|yesterday|today)'
            ],
            'research': [
                r'study\s+(?:shows|found|reveals|indicates|suggests)',
                r'research\s+(?:indicates|suggests|shows|found|reveals)',
                r'scientists\s+(?:discovered|found|revealed|announced)',
                r'published\s+in\s+(?:journal|nature|science|nejm)',
                r'peer[-\s]?reviewed\s+(?:study|research|paper)',
                r'clinical\s+trial\s+(?:shows|found|revealed)',
                r'experiment\s+(?:showed|demonstrated|found|revealed)',
                r'survey\s+(?:found|showed|revealed|indicates)',
                r'poll\s+(?:shows|indicates|found|reveals)',
                r'analysis\s+(?:reveals|shows|indicates|found)',
                r'data\s+(?:shows|indicates|suggests|reveals)',
                r'findings\s+(?:suggest|show|indicate|reveal)',
                r'meta[-\s]?analysis\s+(?:found|showed|revealed)',
                r'longitudinal\s+study\s+(?:found|shows)',
                r'randomized.*trial\s+(?:found|shows)',
                r'controlled\s+study\s+(?:found|shows)',
                r'researchers\s+(?:found|discovered|reported)',
                r'investigation\s+(?:found|revealed|showed)',
                r'report\s+(?:found|shows|indicates)',
                r'evidence\s+(?:shows|suggests|indicates)'
            ],
            'attribution': [
                r'said\s+(?:that|in|during|on|yesterday)',
                r'stated\s+(?:that|in|during|on|yesterday)',
                r'according\s+to\s+(?:the|a|sources?)',
                r'spokesperson\s+(?:said|announced|confirmed)',
                r'CEO\s+(?:announced|said|stated|confirmed)',
                r'president\s+(?:said|announced|declared|stated)',
                r'minister\s+(?:stated|announced|said|confirmed)',
                r'official\s+(?:said|confirmed|announced|stated)',
                r'expert\s+(?:said|explained|noted|warned)',
                r'witness\s+(?:reported|said|testified|claimed)',
                r'source\s+(?:told|said|revealed|confirmed)',
                r'interview\s+(?:with|on|during)',
                r'testified\s+(?:that|in|before)',
                r'confirmed\s+(?:that|in|to)',
                r'denied\s+(?:that|the|any)',
                r'claimed\s+(?:that|in|to)',
                r'insisted\s+(?:that|on)',
                r'maintained\s+(?:that|his|her)',
                r'argued\s+(?:that|for|against)',
                r'contended\s+(?:that|with)'
            ],
            'policy': [
                r'law\s+(?:requires|mandates|prohibits|allows)',
                r'regulation\s+(?:states|requires|prohibits|mandates)',
                r'policy\s+(?:allows|prevents|ensures|requires)',
                r'court\s+(?:ruled|decided|ordered|found)',
                r'judge\s+(?:ruled|ordered|sentenced|decided)',
                r'verdict\s+(?:was|found|reached)',
                r'settlement\s+(?:reached|agreed|announced)',
                r'constitution\s+(?:guarantees|protects|ensures)',
                r'amendment\s+(?:protects|ensures|guarantees)',
                r'supreme\s+court\s+(?:ruled|decided|found)',
                r'legislation\s+(?:passed|failed|introduced)',
                r'bill\s+(?:signed|vetoed|passed|introduced)',
                r'executive\s+order\s+(?:signed|issued)',
                r'regulatory\s+(?:approval|decision|ruling)',
                r'legal\s+(?:ruling|decision|requirement)',
                r'administrative\s+(?:decision|ruling|order)'
            ],
            'causal': [
                r'caused\s+by\s+(?:the|a)',
                r'because\s+of\s+(?:the|a)',
                r'due\s+to\s+(?:the|a)',
                r'result\s+of\s+(?:the|a)',
                r'leads\s+to\s+(?:increased|decreased)',
                r'triggers\s+(?:a|the)',
                r'responsible\s+for\s+(?:the|a)',
                r'blame.*for\s+(?:the|a)',
                r'linked\s+to\s+(?:the|a)',
                r'correlation.*between',
                r'stems\s+from\s+(?:the|a)',
                r'attributed\s+to\s+(?:the|a)',
                r'resulted\s+in\s+(?:the|a)',
                r'contributed\s+to\s+(?:the|a)',
                r'factor\s+in\s+(?:the|a)',
                r'consequence\s+of\s+(?:the|a)'
            ]
        }

    def _initialize_statistical_indicators(self) -> List[str]:
        """Initialize statistical and numerical indicators."""
        return [
            'percent', 'percentage', '%', 'rate', 'ratio', 'proportion',
            'increase', 'decrease', 'growth', 'decline', 'rise', 'fall',
            'surge', 'jump', 'drop', 'plunge', 'spike', 'dip',
            'average', 'median', 'mean', 'total', 'sum', 'count',
            'million', 'billion', 'thousand', 'trillion',
            'times', 'fold', 'double', 'triple', 'half', 'quarter',
            'third', 'majority', 'minority', 'most', 'least',
            'highest', 'lowest', 'maximum', 'minimum', 'peak',
            'sample size', 'margin of error', 'confidence interval',
            'statistically significant', 'p-value', 'standard deviation',
            'correlation coefficient', 'regression analysis',
            'baseline', 'benchmark', 'threshold', 'ceiling', 'floor',
            'approximately', 'roughly', 'about', 'nearly', 'close to',
            'more than', 'less than', 'at least', 'up to', 'as much as'
        ]

    def _initialize_quote_indicators(self) -> List[str]:
        """Initialize quote and attribution indicators."""
        return [
            'said', 'stated', 'announced', 'declared', 'claimed',
            'according to', 'spokesperson', 'representative',
            'CEO', 'president', 'minister', 'official', 'expert',
            'researcher', 'scientist', 'professor', 'doctor',
            'analyst', 'economist', 'historian', 'lawyer',
            'judge', 'senator', 'congressman', 'mayor',
            'governor', 'secretary', 'director', 'chief',
            'confirmed', 'denied', 'admitted', 'revealed',
            'disclosed', 'testified', 'argued', 'contended',
            'maintained', 'asserted', 'emphasized', 'stressed',
            'insisted', 'warned', 'cautioned', 'noted',
            'explained', 'clarified', 'added', 'continued',
            'concluded', 'summarized', 'responded', 'replied'
        ]

    def _initialize_verification_indicators(self) -> Dict[str, List[str]]:
        """Initialize verification level indicators."""
        return {
            'high_verifiability': [
                'official document', 'public record', 'government data',
                'published study', 'peer-reviewed', 'statistical office',
                'census data', 'financial filing', 'court record',
                'parliamentary record', 'press release', 'official statement',
                'regulatory filing', 'academic paper', 'scientific journal',
                'government website', 'official database', 'legal document',
                'verified data', 'confirmed report', 'authenticated source',
                'documented evidence', 'recorded statement', 'official transcript'
            ],
            'medium_verifiability': [
                'news report', 'interview', 'survey', 'poll',
                'analyst report', 'expert opinion', 'industry data',
                'company statement', 'spokesperson said', 'according to',
                'conference presentation', 'white paper', 'research report',
                'market analysis', 'trade publication', 'professional journal',
                'credible source', 'reliable report', 'established media',
                'recognized authority', 'industry expert', 'professional analysis'
            ],
            'low_verifiability': [
                'anonymous source', 'unnamed official', 'insider information',
                'rumor', 'speculation', 'alleged', 'reportedly',
                'sources say', 'it is believed', 'word is',
                'unconfirmed reports', 'social media post', 'blog post',
                'forum discussion', 'hearsay', 'gossip', 'leaked information',
                'unverified claim', 'questionable source', 'disputed report',
                'unsubstantiated allegation', 'unreliable source', 'doubtful claim'
            ]
        }

    def analyze_claim_patterns(self, article_text: str) -> Dict[str, Any]:
        """
        Analyze article text for claim patterns and indicators.
        
        Args:
            article_text: Article content to analyze
            
        Returns:
            Dictionary containing pattern analysis results
        """
        start_time = time.time()
        text_lower = article_text.lower()
        
        # Count claim type patterns
        claim_counts = {}
        patterns_found = []
        
        for claim_type, patterns in self.claim_patterns.items():
            count = 0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                count += len(matches)
                if matches:
                    # Limit matches per pattern for performance
                    matched_patterns.extend([(claim_type, match) for match in matches[:2]])
            
            claim_counts[claim_type] = count
            patterns_found.extend(matched_patterns)
        
        # Count indicator types
        statistical_count = sum(1 for indicator in self.statistical_indicators if indicator in text_lower)
        quote_count = sum(1 for indicator in self.quote_indicators if indicator in text_lower)
        
        # Count verifiability indicators
        verifiability_scores = {}
        for level, indicators in self.verification_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            verifiability_scores[level] = count
        
        # Calculate claim richness
        total_indicators = sum(claim_counts.values()) + statistical_count + quote_count
        richness_multiplier = self.config.get('richness_multiplier', 0.5)
        claim_richness_score = min(10, total_indicators * richness_multiplier)
        
        # Determine dominant claim type
        dominant_claim_type = max(claim_counts, key=claim_counts.get) if claim_counts else 'none'
        
        # Calculate processing time and update metrics
        processing_time = time.time() - start_time
        self.analysis_count += 1
        self.total_processing_time += processing_time
        self.total_patterns_found += len(patterns_found)
        
        # Determine if article is claim-rich
        rich_threshold = self.config.get('rich_threshold', 5)
        is_claim_rich = total_indicators > rich_threshold
        
        return {
            'claim_type_counts': claim_counts,
            'statistical_indicators': statistical_count,
            'quote_indicators': quote_count,
            'verifiability_scores': verifiability_scores,
            'patterns_found': patterns_found[:20],  # Limit for performance
            'total_claim_indicators': total_indicators,
            'claim_richness_score': round(claim_richness_score, 2),
            'likely_claim_rich': is_claim_rich,
            'dominant_claim_type': dominant_claim_type,
            'processing_time_ms': round(processing_time * 1000, 2)
        }

    def extract_potential_claims(self, article_text: str, max_claims: int = 10) -> List[str]:
        """
        Extract potential claim sentences from article text.
        
        Args:
            article_text: Article content to process
            max_claims: Maximum number of claims to extract
            
        Returns:
            List of potential claim sentences
        """
        sentences = article_text.split('.')
        
        # Get configuration weights
        statistical_weight = self.config.get('statistical_weight', 2)
        quote_weight = self.config.get('quote_weight', 1)
        
        potential_claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short sentences
            if len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            claim_score = 0
            
            # Check for claim patterns
            for claim_type, patterns in self.claim_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sentence_lower):
                        claim_score += 1
                        break  # Only count one pattern per type per sentence
            
            # Boost score for statistical indicators
            if any(indicator in sentence_lower for indicator in self.statistical_indicators[:15]):
                claim_score += statistical_weight
            
            # Boost score for quote indicators
            if any(indicator in sentence_lower for indicator in self.quote_indicators[:15]):
                claim_score += quote_weight
            
            # Add to potential claims if it has indicators
            if claim_score > 0:
                potential_claims.append({
                    'text': sentence + '.',
                    'score': claim_score,
                    'length': len(sentence)
                })
        
        # Sort by score (descending) and length (descending as tiebreaker)
        potential_claims.sort(key=lambda x: (-x['score'], -x['length']))
        
        # Return top claims
        return [claim['text'] for claim in potential_claims[:max_claims]]

    def get_claim_density_analysis(self, article_text: str) -> Dict[str, Any]:
        """
        Analyze claim density throughout the article.
        
        Args:
            article_text: Article content to analyze
            
        Returns:
            Dictionary with claim density analysis
        """
        # Split into paragraphs for density analysis
        paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return {'claim_density': 0.0, 'high_density_paragraphs': 0}
        
        paragraph_scores = []
        high_density_count = 0
        density_threshold = self.config.get('high_density_threshold', 3)
        
        for paragraph in paragraphs:
            potential_claims = self.extract_potential_claims(paragraph, max_claims=50)
            claim_count = len(potential_claims)
            
            # Calculate density as claims per 100 words
            word_count = len(paragraph.split())
            density = (claim_count / max(word_count, 1)) * 100
            
            paragraph_scores.append(density)
            
            if claim_count >= density_threshold:
                high_density_count += 1
        
        overall_density = sum(paragraph_scores) / len(paragraph_scores) if paragraph_scores else 0
        
        return {
            'claim_density': round(overall_density, 2),
            'high_density_paragraphs': high_density_count,
            'total_paragraphs': len(paragraphs),
            'density_distribution': paragraph_scores,
            'max_density': max(paragraph_scores) if paragraph_scores else 0,
            'min_density': min(paragraph_scores) if paragraph_scores else 0
        }

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern database statistics."""
        # Calculate database composition
        total_claim_patterns = sum(len(patterns) for patterns in self.claim_patterns.values())
        total_verifiability_indicators = sum(len(indicators) for indicators in self.verification_indicators.values())
        
        # Calculate performance statistics
        avg_processing_time = (
            (self.total_processing_time / self.analysis_count) 
            if self.analysis_count > 0 else 0
        )
        
        avg_patterns_per_analysis = (
            (self.total_patterns_found / self.analysis_count) 
            if self.analysis_count > 0 else 0
        )
        
        return {
            'database_composition': {
                'claim_pattern_types': len(self.claim_patterns),
                'total_claim_patterns': total_claim_patterns,
                'statistical_indicators': len(self.statistical_indicators),
                'quote_indicators': len(self.quote_indicators),
                'verifiability_levels': len(self.verification_indicators),
                'total_verifiability_indicators': total_verifiability_indicators
            },
            'performance_statistics': {
                'total_analyses': self.analysis_count,
                'total_patterns_found': self.total_patterns_found,
                'average_processing_time_ms': round(avg_processing_time * 1000, 2),
                'average_patterns_per_analysis': round(avg_patterns_per_analysis, 2)
            },
            'configuration_applied': bool(self.config)
        }

    def validate_pattern_database(self) -> Dict[str, Any]:
        """Validate pattern database integrity and completeness."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for empty pattern categories
        for category, patterns in self.claim_patterns.items():
            if not patterns:
                validation_results['issues'].append(f"Empty pattern category: {category}")
                validation_results['valid'] = False
        
        # Check for duplicate patterns across categories
        all_patterns = []
        for patterns in self.claim_patterns.values():
            all_patterns.extend(patterns)
        
        duplicates = set([x for x in all_patterns if all_patterns.count(x) > 1])
        if duplicates:
            validation_results['warnings'].append(f"Duplicate patterns found: {list(duplicates)[:5]}")
        
        # Check pattern syntax
        for category, patterns in self.claim_patterns.items():
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    validation_results['issues'].append(f"Invalid regex in {category}: {pattern} - {str(e)}")
                    validation_results['valid'] = False
        
        # Check for very short indicators (might be too generic)
        short_indicators = []
        for indicators in [self.statistical_indicators, self.quote_indicators]:
            short_indicators.extend([ind for ind in indicators if len(ind) < 3])
        
        if short_indicators:
            validation_results['warnings'].append(f"Very short indicators: {short_indicators[:5]}")
        
        return validation_results


# Testing functionality
if __name__ == "__main__":
    """Test pattern database functionality."""
    
    # Initialize with test configuration
    test_config = {
        'richness_multiplier': 0.6,
        'rich_threshold': 4,
        'statistical_weight': 2,
        'quote_weight': 1,
        'high_density_threshold': 2
    }
    
    pattern_db = ClaimPatternDatabase(test_config)
    
    # Test article with various claim types
    test_article = """
    A new study published in Nature Medicine found that 85% of patients showed improvement.
    Dr. Sarah Johnson, lead researcher at Harvard, announced the results yesterday.
    The clinical trial involved 1,200 participants across 15 hospitals worldwide.
    According to government data, healthcare spending increased by $50 billion this year.
    The Supreme Court ruled that the new regulation violates constitutional rights.
    """
    
    print("=== PATTERN ANALYSIS TEST ===")
    analysis = pattern_db.analyze_claim_patterns(test_article)
    
    print(f"Total claim indicators: {analysis['total_claim_indicators']}")
    print(f"Claim richness score: {analysis['claim_richness_score']}/10")
    print(f"Likely claim-rich: {analysis['likely_claim_rich']}")
    print(f"Dominant claim type: {analysis['dominant_claim_type']}")
    print(f"Processing time: {analysis['processing_time_ms']:.1f}ms")
    
    print(f"\nClaim type counts: {analysis['claim_type_counts']}")
    print(f"Statistical indicators: {analysis['statistical_indicators']}")
    print(f"Quote indicators: {analysis['quote_indicators']}")
    print(f"Verifiability scores: {analysis['verifiability_scores']}")
    
    # Test potential claim extraction
    print("\n=== POTENTIAL CLAIMS EXTRACTION ===")
    potential_claims = pattern_db.extract_potential_claims(test_article, max_claims=5)
    
    for i, claim in enumerate(potential_claims, 1):
        print(f"{i}. {claim}")
    
    # Test claim density analysis
    print("\n=== CLAIM DENSITY ANALYSIS ===")
    density_analysis = pattern_db.get_claim_density_analysis(test_article)
    
    print(f"Overall claim density: {density_analysis['claim_density']:.2f}")
    print(f"High-density paragraphs: {density_analysis['high_density_paragraphs']}")
    print(f"Total paragraphs: {density_analysis['total_paragraphs']}")
    
    # Test database statistics
    print("\n=== DATABASE STATISTICS ===")
    stats = pattern_db.get_pattern_statistics()
    
    print(f"Pattern types: {stats['database_composition']['claim_pattern_types']}")
    print(f"Total patterns: {stats['database_composition']['total_claim_patterns']}")
    print(f"Analyses completed: {stats['performance_statistics']['total_analyses']}")
    print(f"Average processing time: {stats['performance_statistics']['average_processing_time_ms']:.1f}ms")
    
    # Test validation
    print("\n=== DATABASE VALIDATION ===")
    validation = pattern_db.validate_pattern_database()
    
    print(f"Database valid: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    print("\n=== PATTERN DATABASE TESTING COMPLETED ===")
