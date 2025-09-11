# agents/credible_source/domain_classifier.py

"""
Domain Classifier - Production Ready

Production-ready domain classification for credible source agent with enhanced
performance tracking, error handling, structured logging, and configurable
thresholds for reliable topic classification.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional

from .exceptions import (
    DomainClassificationError,
    InputValidationError,
    raise_input_validation_error
)


class DomainClassifier:
    """
    Production-ready domain classifier for news articles and claims.
    
    Classifies article content into subject domains to enable targeted
    source recommendations with enhanced performance tracking, error handling,
    and configurable classification parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize domain classifier with production configuration.

        Args:
            config: Optional configuration for classification parameters and thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.DomainClassifier")
        
        try:
            # Initialize classification systems with error handling
            self.domain_keywords = self._initialize_domain_keywords()
            self.domain_patterns = self._initialize_domain_patterns()
            self.claim_type_mappings = self._initialize_claim_type_mappings()
            
            # Configurable classification parameters
            self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
            self.pattern_weight_multiplier = self.config.get('pattern_weight_multiplier', 2.0)
            self.claim_priority_weight = self.config.get('claim_priority_weight', True)
            self.min_text_length = self.config.get('min_text_length', 20)
            
            # Performance tracking with detailed metrics
            self.classification_count = 0
            self.successful_classifications = 0
            self.total_processing_time = 0.0
            self.domain_distribution = {domain: 0 for domain in self.domain_keywords.keys()}
            self.error_count = 0
            
            self.logger.info(
                f"Domain classifier initialized with {len(self.domain_keywords)} domains",
                extra={
                    'domains_available': list(self.domain_keywords.keys()),
                    'confidence_threshold': self.confidence_threshold,
                    'total_keywords': sum(len(keywords) for keywords in self.domain_keywords.values()),
                    'total_patterns': sum(len(patterns) for patterns in self.domain_patterns.values())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize domain classifier: {str(e)}")
            raise DomainClassificationError(
                f"Domain classifier initialization failed: {str(e)}",
                classification_stage="initialization"
            )

    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize comprehensive domain-specific keyword databases."""
        return {
            'health': [
                'medical', 'health', 'disease', 'treatment', 'medicine', 'doctor',
                'hospital', 'clinic', 'patient', 'symptoms', 'diagnosis', 'therapy',
                'vaccine', 'drug', 'pharmaceutical', 'clinical', 'healthcare',
                'wellness', 'epidemic', 'pandemic', 'virus', 'bacteria', 'infection',
                'immunity', 'mental health', 'public health', 'medical research',
                'fda', 'cdc', 'who', 'clinical trial', 'side effects', 'dosage'
            ],
            'politics': [
                'government', 'political', 'election', 'vote', 'congress', 'senate',
                'president', 'politician', 'policy', 'law', 'legislation', 'democrat',
                'republican', 'party', 'campaign', 'debate', 'administration',
                'white house', 'capitol', 'federal', 'state government', 'local government',
                'supreme court', 'justice', 'ruling', 'constitutional', 'democracy',
                'mayor', 'governor', 'senator', 'representative', 'ballot'
            ],
            'science': [
                'science', 'scientific', 'research', 'study', 'experiment', 'data',
                'analysis', 'laboratory', 'peer-reviewed', 'journal', 'publication',
                'hypothesis', 'theory', 'evidence', 'discovery', 'innovation',
                'physics', 'chemistry', 'biology', 'mathematics', 'statistics',
                'methodology', 'academic', 'university research', 'scientific method',
                'peer review', 'citation', 'publication', 'findings', 'results'
            ],
            'technology': [
                'technology', 'tech', 'computer', 'software', 'hardware', 'internet',
                'digital', 'online', 'app', 'platform', 'algorithm', 'artificial intelligence',
                'machine learning', 'ai', 'cybersecurity', 'data', 'cloud computing',
                'mobile', 'smartphone', 'programming', 'coding', 'blockchain',
                'automation', 'robotics', 'virtual reality', 'augmented reality',
                'cryptocurrency', 'bitcoin', 'startup', 'silicon valley'
            ],
            'economics': [
                'economic', 'economy', 'financial', 'money', 'market', 'stock',
                'investment', 'business', 'company', 'corporation', 'trade',
                'inflation', 'recession', 'gdp', 'unemployment', 'employment',
                'banking', 'finance', 'wall street', 'federal reserve', 'interest rate',
                'fiscal policy', 'monetary policy', 'budget', 'debt', 'deficit',
                'revenue', 'profit', 'earnings', 'growth', 'startup'
            ],
            'environment': [
                'environment', 'environmental', 'climate', 'weather', 'global warming',
                'climate change', 'carbon', 'emissions', 'renewable energy',
                'pollution', 'conservation', 'wildlife', 'ecosystem', 'sustainability',
                'green energy', 'solar', 'wind power', 'fossil fuel', 'biodiversity',
                'deforestation', 'ocean', 'atmosphere', 'greenhouse gas',
                'recycling', 'waste', 'endangered species', 'habitat'
            ],
            'education': [
                'education', 'educational', 'school', 'university', 'college',
                'student', 'teacher', 'professor', 'learning', 'curriculum',
                'academic', 'scholarship', 'graduation', 'degree', 'classroom',
                'training', 'skill development', 'pedagogy', 'literacy',
                'test scores', 'enrollment', 'tuition', 'campus'
            ],
            'international': [
                'international', 'global', 'world', 'foreign', 'country', 'nation',
                'diplomatic', 'embassy', 'trade', 'conflict', 'peace',
                'united nations', 'nato', 'alliance', 'treaty', 'sanctions',
                'foreign policy', 'geopolitics', 'multinational', 'cross-border',
                'immigration', 'refugee', 'border', 'sovereignty'
            ]
        }

    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize domain-specific regex pattern matching rules."""
        return {
            'health': [
                r'published\s+in.*medical\s+journal',
                r'clinical\s+trial\s+results?',
                r'fda\s+approve[sd]?',
                r'health\s+officials\s+say',
                r'medical\s+experts?\s+warn',
                r'public\s+health\s+emergency',
                r'disease\s+outbreak',
                r'vaccine\s+effectiveness',
                r'side\s+effects?',
                r'cdc\s+reports?'
            ],
            'politics': [
                r'according\s+to\s+government\s+sources?',
                r'congressional\s+hearing',
                r'white\s+house\s+statement',
                r'supreme\s+court\s+ruling',
                r'election\s+results?',
                r'policy\s+announcement',
                r'legislative\s+session',
                r'political\s+campaign',
                r'bill\s+passed',
                r'executive\s+order'
            ],
            'science': [
                r'peer-reviewed\s+study',
                r'published\s+in\s+nature',
                r'research\s+team\s+at',
                r'scientific\s+breakthrough',
                r'study\s+finds',
                r'research\s+shows',
                r'laboratory\s+analysis',
                r'experimental\s+results?',
                r'journal\s+article',
                r'scientific\s+paper'
            ],
            'technology': [
                r'tech\s+company\s+announces?',
                r'new\s+app\s+launch',
                r'cybersecurity\s+breach',
                r'ai\s+technology',
                r'software\s+update',
                r'digital\s+transformation',
                r'cloud\s+computing',
                r'machine\s+learning\s+model',
                r'startup\s+funding',
                r'silicon\s+valley'
            ],
            'economics': [
                r'stock\s+market\s+analysis',
                r'economic\s+indicators?',
                r'federal\s+reserve\s+decision',
                r'quarterly\s+earnings',
                r'market\s+volatility',
                r'inflation\s+rate',
                r'unemployment\s+data',
                r'economic\s+growth',
                r'gdp\s+figures?',
                r'wall\s+street'
            ],
            'environment': [
                r'climate\s+change\s+report',
                r'environmental\s+impact',
                r'carbon\s+emissions',
                r'renewable\s+energy',
                r'pollution\s+levels?',
                r'conservation\s+efforts?',
                r'ecosystem\s+study',
                r'global\s+warming',
                r'greenhouse\s+gas',
                r'sustainability\s+initiative'
            ],
            'education': [
                r'education\s+policy',
                r'student\s+performance',
                r'curriculum\s+changes?',
                r'university\s+research',
                r'academic\s+achievement',
                r'educational\s+funding',
                r'school\s+district',
                r'higher\s+education',
                r'test\s+scores?',
                r'graduation\s+rates?'
            ],
            'international': [
                r'international\s+relations',
                r'foreign\s+policy',
                r'diplomatic\s+talks',
                r'global\s+summit',
                r'trade\s+agreement',
                r'international\s+conflict',
                r'cross-border',
                r'multinational\s+cooperation',
                r'united\s+nations',
                r'embassy\s+statement'
            ]
        }

    def _initialize_claim_type_mappings(self) -> Dict[str, str]:
        """Initialize claim type to domain mappings for targeted classification."""
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

    def classify_domain(self, 
                       article_text: str, 
                       extracted_claims: List[Dict[str, Any]], 
                       session_id: str = None) -> Dict[str, Any]:
        """
        Classify article domain using comprehensive analysis with error handling.

        Args:
            article_text: Article content to classify
            extracted_claims: Claims extracted from the article
            session_id: Optional session ID for tracking

        Returns:
            Dictionary containing classification results and metadata

        Raises:
            DomainClassificationError: If classification fails
            InputValidationError: If inputs are invalid
        """
        start_time = time.time()
        
        self.logger.debug(
            f"Starting domain classification",
            extra={
                'session_id': session_id,
                'article_length': len(article_text) if article_text else 0,
                'claims_count': len(extracted_claims) if extracted_claims else 0
            }
        )

        try:
            # Input validation with detailed error messages
            if not isinstance(article_text, str):
                raise_input_validation_error(
                    "article_text",
                    f"Must be string, got {type(article_text).__name__}",
                    article_text,
                    session_id
                )

            if not article_text or len(article_text.strip()) < self.min_text_length:
                raise_input_validation_error(
                    "article_text",
                    f"Text too short: {len(article_text.strip())} characters (minimum: {self.min_text_length})",
                    len(article_text.strip()) if article_text else 0,
                    session_id
                )

            if not isinstance(extracted_claims, list):
                self.logger.warning(
                    f"Invalid claims format: {type(extracted_claims).__name__}, using empty list",
                    extra={'session_id': session_id}
                )
                extracted_claims = []

            # Initialize domain scores
            domain_scores = {domain: 0 for domain in self.domain_keywords.keys()}
            article_lower = article_text.lower()

            # Keyword-based scoring with error handling
            keyword_matches = 0
            try:
                for domain, keywords in self.domain_keywords.items():
                    for keyword in keywords:
                        if keyword in article_lower:
                            domain_scores[domain] += 1
                            keyword_matches += 1
            except Exception as e:
                self.logger.warning(f"Error in keyword matching: {str(e)}", 
                                  extra={'session_id': session_id})

            # Pattern-based scoring with error handling (weighted higher)
            pattern_matches = 0
            try:
                for domain, patterns in self.domain_patterns.items():
                    for pattern in patterns:
                        try:
                            matches = re.findall(pattern, article_lower, re.IGNORECASE)
                            if matches:
                                match_count = len(matches)
                                domain_scores[domain] += match_count * self.pattern_weight_multiplier
                                pattern_matches += match_count
                        except re.error as regex_error:
                            self.logger.warning(
                                f"Regex error in domain {domain}: {str(regex_error)}",
                                extra={'session_id': session_id}
                            )
                            continue
            except Exception as e:
                self.logger.warning(f"Error in pattern matching: {str(e)}", 
                                  extra={'session_id': session_id})

            # Claim type influence with configurable weighting
            claim_matches = 0
            try:
                for claim in extracted_claims[:10]:  # Limit claims for performance
                    if isinstance(claim, dict):
                        claim_type = claim.get('claim_type', '')
                        mapped_domain = self.claim_type_mappings.get(claim_type)
                        
                        if mapped_domain and mapped_domain in domain_scores:
                            if self.claim_priority_weight:
                                priority = claim.get('priority', 3)
                                # Higher priority claims have more influence: priority 1 = weight 3, priority 3+ = weight 1
                                weight = max(1, 4 - min(priority, 3))
                            else:
                                weight = 2  # Fixed weight
                                
                            domain_scores[mapped_domain] += weight
                            claim_matches += 1
            except Exception as e:
                self.logger.warning(f"Error in claim type analysis: {str(e)}", 
                                  extra={'session_id': session_id})

            # Calculate confidence and determine primary domain
            total_score = sum(domain_scores.values())
            
            if total_score == 0:
                primary_domain = 'general'
                confidence = 0.0
                domain_classified = False
                self.logger.info(
                    "No domain indicators found, using general domain",
                    extra={'session_id': session_id}
                )
            else:
                primary_domain = max(domain_scores, key=domain_scores.get)
                primary_score = domain_scores[primary_domain]
                confidence = primary_score / total_score
                domain_classified = confidence >= self.confidence_threshold

                if not domain_classified:
                    self.logger.info(
                        f"Domain confidence too low: {confidence:.3f} < {self.confidence_threshold}, using general",
                        extra={'session_id': session_id}
                    )
                    primary_domain = 'general'

            # Get secondary domains (sorted by score, excluding primary)
            secondary_domains = []
            sorted_domains = sorted(
                [(domain, score) for domain, score in domain_scores.items()
                 if domain != primary_domain and score > 0],
                key=lambda x: x[1],
                reverse=True
            )
            secondary_domains = [domain for domain, score in sorted_domains[:3]]  # Top 3 secondary

            # Calculate domain strength (how decisive the classification is)
            domain_strength = self._calculate_domain_strength(domain_scores, primary_domain)

            # Calculate processing time and update metrics
            processing_time = time.time() - start_time
            self.classification_count += 1
            self.total_processing_time += processing_time
            
            if domain_classified:
                self.successful_classifications += 1
                self.domain_distribution[primary_domain] += 1

            self.logger.info(
                f"Domain classification completed: {primary_domain} (confidence: {confidence:.3f})",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time * 1000, 2),
                    'total_indicators': total_score,
                    'keyword_matches': keyword_matches,
                    'pattern_matches': pattern_matches,
                    'claim_matches': claim_matches,
                    'domain_classified': domain_classified
                }
            )

            return {
                'primary_domain': primary_domain,
                'confidence': round(confidence, 3),
                'domain_classified': domain_classified,
                'domain_scores': domain_scores,
                'secondary_domains': secondary_domains,
                'domain_strength': domain_strength,
                'total_indicators': total_score,
                'classification_time_ms': round(processing_time * 1000, 2),
                'confidence_threshold_used': self.confidence_threshold,
                'claims_analyzed': len(extracted_claims),
                'analysis_breakdown': {
                    'keyword_matches': keyword_matches,
                    'pattern_matches': pattern_matches,
                    'claim_matches': claim_matches
                },
                'session_id': session_id
            }

        except (InputValidationError, DomainClassificationError):
            # Re-raise validation and classification errors
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            self.error_count += 1
            
            self.logger.error(
                f"Domain classification failed: {str(e)}",
                extra={
                    'session_id': session_id,
                    'processing_time': round(processing_time * 1000, 2),
                    'error_type': type(e).__name__
                }
            )
            
            raise DomainClassificationError(
                f"Classification failed: {str(e)}",
                classification_stage="domain_analysis",
                session_id=session_id
            )

    def _calculate_domain_strength(self, 
                                 domain_scores: Dict[str, int], 
                                 primary_domain: str) -> str:
        """
        Calculate how decisive the domain classification is.

        Args:
            domain_scores: Dictionary of domain scores
            primary_domain: The classified primary domain

        Returns:
            Strength indicator (strong, moderate, weak)
        """
        try:
            if primary_domain == 'general':
                return 'weak'

            primary_score = domain_scores[primary_domain]
            other_scores = [score for domain, score in domain_scores.items()
                          if domain != primary_domain and score > 0]

            if not other_scores or primary_score == 0:
                return 'weak'

            max_other_score = max(other_scores)

            # Calculate ratio between primary and next highest
            if max_other_score == 0:
                return 'strong'

            ratio = primary_score / max_other_score

            if ratio >= 2.5:
                return 'strong'
            elif ratio >= 1.8:
                return 'moderate'
            else:
                return 'weak'
                
        except Exception as e:
            self.logger.warning(f"Error calculating domain strength: {str(e)}")
            return 'unknown'

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
        Get regex patterns for a specific domain.

        Args:
            domain: Domain name

        Returns:
            List of regex patterns for the domain
        """
        return self.domain_patterns.get(domain, [])

    def get_supported_domains(self) -> List[str]:
        """Get list of all supported domains for classification."""
        return list(self.domain_keywords.keys())

    def analyze_domain_distribution(self, domain_scores: Dict[str, int]) -> Dict[str, Any]:
        """
        Analyze the distribution of domain indicators with enhanced metrics.

        Args:
            domain_scores: Dictionary of domain scores

        Returns:
            Analysis of domain score distribution and classification difficulty
        """
        try:
            total_score = sum(domain_scores.values())
            
            if total_score == 0:
                return {
                    'distribution': 'no_indicators',
                    'dominant_domains': [],
                    'score_spread': 0.0,
                    'classification_difficulty': 'impossible',
                    'diversity_index': 0.0
                }

            # Calculate percentages
            domain_percentages = {
                domain: (score / total_score) * 100
                for domain, score in domain_scores.items()
                if score > 0
            }

            # Find dominant domains (>15% of total score)
            dominant_threshold = self.config.get('dominant_threshold', 15.0)
            dominant_domains = [
                domain for domain, percentage in domain_percentages.items()
                if percentage >= dominant_threshold
            ]

            # Calculate score spread (standard deviation)
            scores = list(domain_scores.values())
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            score_spread = variance ** 0.5

            # Calculate diversity index (how spread out the scores are)
            non_zero_scores = [score for score in scores if score > 0]
            diversity_index = len(non_zero_scores) / len(scores) if scores else 0

            # Determine classification difficulty
            max_percentage = max(domain_percentages.values()) if domain_percentages else 0
            
            if len(dominant_domains) == 1 and max_percentage >= 60:
                difficulty = 'easy'
            elif len(dominant_domains) <= 2 and max_percentage >= 40:
                difficulty = 'moderate'
            elif len(dominant_domains) <= 3:
                difficulty = 'difficult'
            else:
                difficulty = 'very_difficult'

            return {
                'distribution': 'multi_domain' if len(dominant_domains) > 1 else 'single_domain',
                'dominant_domains': dominant_domains,
                'domain_percentages': {k: round(v, 1) for k, v in domain_percentages.items()},
                'score_spread': round(score_spread, 2),
                'classification_difficulty': difficulty,
                'diversity_index': round(diversity_index, 2),
                'max_percentage': round(max_percentage, 1),
                'active_domains': len(non_zero_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing domain distribution: {str(e)}")
            return {
                'distribution': 'analysis_error',
                'error': str(e)
            }

    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics and performance metrics."""
        success_rate = (
            (self.successful_classifications / self.classification_count * 100)
            if self.classification_count > 0 else 0
        )

        avg_processing_time = (
            (self.total_processing_time / self.classification_count)
            if self.classification_count > 0 else 0
        )

        # Find most common domain
        most_common_domain = 'none'
        if self.domain_distribution:
            most_common_domain = max(
                self.domain_distribution, 
                key=self.domain_distribution.get
            )

        return {
            'total_classifications': self.classification_count,
            'successful_classifications': self.successful_classifications,
            'success_rate_percent': round(success_rate, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'error_count': self.error_count,
            'error_rate_percent': round((self.error_count / max(self.classification_count, 1)) * 100, 2),
            'domains_supported': len(self.domain_keywords),
            'total_keywords': sum(len(keywords) for keywords in self.domain_keywords.values()),
            'total_patterns': sum(len(patterns) for patterns in self.domain_patterns.values()),
            'claim_mappings_available': len(self.claim_type_mappings),
            'configuration_applied': bool(self.config),
            'domain_distribution': dict(self.domain_distribution),
            'most_common_domain': most_common_domain,
            'confidence_threshold': self.confidence_threshold
        }

    def validate_classification_data(self) -> Dict[str, Any]:
        """Validate classification database integrity and completeness."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }

        try:
            # Check for empty keyword lists
            for domain, keywords in self.domain_keywords.items():
                if not keywords:
                    validation_results['issues'].append(f"Empty keyword list for domain: {domain}")
                    validation_results['valid'] = False
                elif len(keywords) < 5:
                    validation_results['warnings'].append(f"Few keywords for domain {domain}: {len(keywords)}")

            # Check for empty pattern lists
            for domain, patterns in self.domain_patterns.items():
                if not patterns:
                    validation_results['warnings'].append(f"Empty pattern list for domain: {domain}")
                elif len(patterns) < 3:
                    validation_results['warnings'].append(f"Few patterns for domain {domain}: {len(patterns)}")

            # Check for orphaned claim mappings
            supported_domains = set(self.domain_keywords.keys()) | {'general'}
            for claim_type, mapped_domain in self.claim_type_mappings.items():
                if mapped_domain not in supported_domains:
                    validation_results['issues'].append(
                        f"Claim type '{claim_type}' maps to unsupported domain: {mapped_domain}"
                    )
                    validation_results['valid'] = False

            # Check for very short keywords (might be too generic)
            short_keywords = []
            for domain, keywords in self.domain_keywords.items():
                short_keywords.extend([kw for kw in keywords if len(kw) < 4])
            
            if short_keywords:
                validation_results['warnings'].append(
                    f"Very short keywords detected (may cause false positives): {short_keywords[:5]}"
                )

            # Check for pattern syntax validity
            invalid_patterns = []
            for domain, patterns in self.domain_patterns.items():
                for pattern in patterns:
                    try:
                        re.compile(pattern)
                    except re.error:
                        invalid_patterns.append(f"{domain}:{pattern}")
                        validation_results['valid'] = False
            
            if invalid_patterns:
                validation_results['issues'].append(f"Invalid regex patterns: {invalid_patterns}")

            # Generate statistics
            validation_results['statistics'] = {
                'total_domains': len(self.domain_keywords),
                'total_keywords': sum(len(keywords) for keywords in self.domain_keywords.values()),
                'total_patterns': sum(len(patterns) for patterns in self.domain_patterns.values()),
                'total_claim_mappings': len(self.claim_type_mappings),
                'avg_keywords_per_domain': round(
                    sum(len(keywords) for keywords in self.domain_keywords.values()) / len(self.domain_keywords), 1
                ),
                'avg_patterns_per_domain': round(
                    sum(len(patterns) for patterns in self.domain_patterns.values()) / len(self.domain_patterns), 1
                )
            }

            self.logger.info(f"Classification data validation completed: {'✓ VALID' if validation_results['valid'] else '✗ INVALID'}")

        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Classification data validation failed: {str(e)}")

        return validation_results


# Testing functionality
if __name__ == "__main__":
    """Test domain classifier functionality with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Test configuration
    test_config = {
        'confidence_threshold': 0.4,
        'pattern_weight_multiplier': 2.5,
        'claim_priority_weight': True
    }
    
    print("=== DOMAIN CLASSIFICATION TEST ===")
    
    try:
        classifier = DomainClassifier(test_config)
        test_session_id = "domain_test_123"
        
        # Test health domain classification
        health_article = """
        A groundbreaking study published in Nature Medicine by researchers at Harvard
        Medical School reveals new insights into cancer treatment. The clinical trial
        involved 500 patients and showed a 75% improvement in survival rates. The FDA
        is reviewing the data for potential drug approval. Health officials say this
        could revolutionize cancer therapy.
        """
        
        health_claims = [
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
        
        print("\n--- Health Domain Test ---")
        health_result = classifier.classify_domain(health_article, health_claims, test_session_id)
        
        print(f"Primary domain: {health_result['primary_domain']}")
        print(f"Confidence: {health_result['confidence']:.3f}")
        print(f"Domain classified: {health_result['domain_classified']}")
        print(f"Domain strength: {health_result['domain_strength']}")
        print(f"Secondary domains: {health_result['secondary_domains']}")
        print(f"Total indicators: {health_result['total_indicators']}")
        print(f"Processing time: {health_result['classification_time_ms']:.1f}ms")
        
        print(f"\nAnalysis breakdown:")
        breakdown = health_result['analysis_breakdown']
        print(f"  Keyword matches: {breakdown['keyword_matches']}")
        print(f"  Pattern matches: {breakdown['pattern_matches']}")
        print(f"  Claim matches: {breakdown['claim_matches']}")
        
        print(f"\nDomain scores (top 5):")
        sorted_scores = sorted(health_result['domain_scores'].items(), key=lambda x: x[1], reverse=True)
        for domain, score in sorted_scores[:5]:
            if score > 0:
                print(f"  {domain}: {score}")

        # Test domain distribution analysis
        distribution_analysis = classifier.analyze_domain_distribution(health_result['domain_scores'])
        print(f"\n--- Domain Distribution Analysis ---")
        print(f"Distribution type: {distribution_analysis['distribution']}")
        print(f"Dominant domains: {distribution_analysis['dominant_domains']}")
        print(f"Classification difficulty: {distribution_analysis['classification_difficulty']}")
        print(f"Domain percentages: {distribution_analysis['domain_percentages']}")
        print(f"Diversity index: {distribution_analysis['diversity_index']}")

        # Test classification statistics
        stats = classifier.get_classification_statistics()
        print(f"\n--- Classification Statistics ---")
        print(f"Classifications completed: {stats['total_classifications']}")
        print(f"Success rate: {stats['success_rate_percent']:.1f}%")
        print(f"Average processing time: {stats['average_processing_time_ms']:.1f}ms")
        print(f"Domains supported: {stats['domains_supported']}")
        print(f"Total keywords: {stats['total_keywords']}")
        print(f"Total patterns: {stats['total_patterns']}")
        print(f"Most common domain: {stats['most_common_domain']}")

        # Test database validation
        validation = classifier.validate_classification_data()
        print(f"\n--- Database Validation ---")
        print(f"Validation result: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
        print(f"Total domains: {validation['statistics']['total_domains']}")
        print(f"Average keywords per domain: {validation['statistics']['avg_keywords_per_domain']}")
        print(f"Average patterns per domain: {validation['statistics']['avg_patterns_per_domain']}")
        
        if validation['issues']:
            print(f"Issues found: {validation['issues'][:2]}")  # Show first 2 issues
        if validation['warnings']:
            print(f"Warnings: {validation['warnings'][:2]}")  # Show first 2 warnings

        # Test edge cases
        print(f"\n--- Edge Case Tests ---")
        
        # Empty text
        try:
            empty_result = classifier.classify_domain("", [], test_session_id)
            print("❌ Empty text should have failed")
        except InputValidationError:
            print("✅ Empty text properly rejected")
        
        # Short text
        try:
            short_result = classifier.classify_domain("Short text", [], test_session_id)
            print("❌ Short text should have failed")
        except InputValidationError:
            print("✅ Short text properly rejected")
        
        # General domain classification
        general_article = "This is a generic article that doesn't match any specific domain patterns or keywords."
        general_result = classifier.classify_domain(general_article, [], test_session_id)
        print(f"✅ General domain classification: {general_result['primary_domain']} (confidence: {general_result['confidence']:.3f})")

        print("\n✅ Domain classification tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
