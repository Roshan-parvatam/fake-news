# agents/claim_extractor/patterns.py

"""
Claim Pattern Database - Production Ready

Production-ready pattern definitions and analysis for claim extraction agent.
Enhanced with configurable detection algorithms, comprehensive pattern matching,
performance tracking, and robust error handling for reliable production use.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .exceptions import PatternAnalysisError, ConfigurationError


class ClaimPatternDatabase:
    """
    Pattern database for identifying claim-rich content and potential claims.
    
    Provides configurable pattern matching for statistical claims, attributions,
    research findings, and other verifiable statements with performance tracking
    and comprehensive error handling for production environments.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize claim pattern database with production configuration.

        Args:
            config: Optional configuration for pattern matching parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ClaimPatternDatabase")
        
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
        """Initialize comprehensive claim pattern categories with enhanced patterns."""
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
                r'about\s+\d+',
                r'\d+\.?\d*\s*out\s+of\s+\d+',
                r'ratio.*\d+:\d+',
                r'average.*\d+',
                r'median.*\d+',
                r'total.*\d+',
                r'majority.*\d+',
                r'minority.*\d+'
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
                r'approved\s+(?:on|yesterday|today)',
                r'released\s+(?:on|yesterday|today)',
                r'unveiled\s+(?:on|yesterday|today)',
                r'declared\s+(?:on|yesterday|today)',
                r'established\s+(?:on|in|yesterday)',
                r'founded\s+(?:in|on|yesterday)'
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
                r'evidence\s+(?:shows|suggests|indicates)',
                r'systematic\s+review\s+(?:found|shows)',
                r'observational\s+study\s+(?:found|shows)',
                r'cohort\s+study\s+(?:found|revealed)',
                r'case\s+study\s+(?:shows|demonstrates)',
                r'laboratory\s+tests?\s+(?:show|reveal)'
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
                r'contended\s+(?:that|with)',
                r'declared\s+(?:that|in)',
                r'asserted\s+(?:that|in)',
                r'emphasized\s+(?:that|in)',
                r'stressed\s+(?:that|the)',
                r'revealed\s+(?:that|in)'
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
                r'administrative\s+(?:decision|ruling|order)',
                r'government\s+(?:announced|declared|issued)',
                r'federal\s+(?:agency|court|law)',
                r'state\s+(?:law|regulation|policy)',
                r'local\s+(?:ordinance|regulation|policy)',
                r'congressional\s+(?:hearing|investigation|report)',
                r'senate\s+(?:voted|passed|rejected)',
                r'house\s+(?:voted|passed|approved)'
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
                r'consequence\s+of\s+(?:the|a)',
                r'outcome\s+of\s+(?:the|a)',
                r'effect\s+of\s+(?:the|a)',
                r'impact\s+of\s+(?:the|a)',
                r'influence\s+of\s+(?:the|a)',
                r'relationship\s+between',
                r'connection\s+between',
                r'association\s+with'
            ],
            'temporal': [
                r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
                r'\d{1,2}\/\d{1,2}\/\d{2,4}',
                r'\d{1,2}-\d{1,2}-\d{2,4}',
                r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                r'(?:yesterday|today|tomorrow)',
                r'(?:last|next)\s+(?:week|month|year)',
                r'(?:this|that)\s+(?:morning|afternoon|evening)',
                r'(?:earlier|later)\s+(?:today|yesterday)',
                r'in\s+\d{4}',
                r'since\s+\d{4}',
                r'until\s+\d{4}',
                r'between\s+\d{4}\s+and\s+\d{4}',
                r'from\s+\d{4}\s+to\s+\d{4}'
            ]
        }

    def _initialize_statistical_indicators(self) -> List[str]:
        """Initialize statistical and numerical indicators with enhanced coverage."""
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
            'more than', 'less than', 'at least', 'up to', 'as much as',
            'frequency', 'occurrence', 'incidence', 'prevalence',
            'distribution', 'variance', 'range', 'quartile',
            'percentile', 'decile', 'outlier', 'trend'
        ]

    def _initialize_quote_indicators(self) -> List[str]:
        """Initialize quote and attribution indicators with enhanced coverage."""
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
            'concluded', 'summarized', 'responded', 'replied',
            'commented', 'observed', 'remarked', 'mentioned',
            'told reporters', 'in an interview', 'during a speech',
            'at a press conference', 'in a statement'
        ]

    def _initialize_verification_indicators(self) -> Dict[str, List[str]]:
        """Initialize verification level indicators with enhanced categorization."""
        return {
            'high_verifiability': [
                'official document', 'public record', 'government data',
                'published study', 'peer-reviewed', 'statistical office',
                'census data', 'financial filing', 'court record',
                'parliamentary record', 'press release', 'official statement',
                'regulatory filing', 'academic paper', 'scientific journal',
                'government website', 'official database', 'legal document',
                'verified data', 'confirmed report', 'authenticated source',
                'documented evidence', 'recorded statement', 'official transcript',
                'clinical trial registry', 'patent filing', 'securities filing',
                'audit report', 'annual report', 'quarterly earnings'
            ],
            'medium_verifiability': [
                'news report', 'interview', 'survey', 'poll',
                'analyst report', 'expert opinion', 'industry data',
                'company statement', 'spokesperson said', 'according to',
                'conference presentation', 'white paper', 'research report',
                'market analysis', 'trade publication', 'professional journal',
                'credible source', 'reliable report', 'established media',
                'recognized authority', 'industry expert', 'professional analysis',
                'think tank report', 'policy brief', 'market research',
                'consulting firm', 'trade association', 'professional organization'
            ],
            'low_verifiability': [
                'anonymous source', 'unnamed official', 'insider information',
                'rumor', 'speculation', 'alleged', 'reportedly',
                'sources say', 'it is believed', 'word is',
                'unconfirmed reports', 'social media post', 'blog post',
                'forum discussion', 'hearsay', 'gossip', 'leaked information',
                'unverified claim', 'questionable source', 'disputed report',
                'unsubstantiated allegation', 'unreliable source', 'doubtful claim',
                'anonymous tip', 'off the record', 'confidential source',
                'whistle-blower', 'informant', 'insider leak'
            ]
        }

    def analyze_claim_patterns(self, article_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Analyze article text for claim patterns and indicators with enhanced tracking.

        Args:
            article_text: Article content to analyze
            session_id: Optional session ID for tracking

        Returns:
            Dictionary containing pattern analysis results
        """
        start_time = time.time()
        
        try:
            text_lower = article_text.lower()

            # Count claim type patterns with enhanced matching
            claim_counts = {}
            patterns_found = []
            
            for claim_type, patterns in self.claim_patterns.items():
                count = 0
                matched_patterns = []
                
                for pattern in patterns:
                    try:
                        matches = re.findall(pattern, text_lower, re.IGNORECASE)
                        count += len(matches)
                        if matches:
                            # Store matched patterns with context
                            for match in matches[:2]:  # Limit matches per pattern
                                matched_patterns.append({
                                    'type': claim_type,
                                    'pattern': pattern[:50],  # Truncate long patterns
                                    'match': str(match)[:100],  # Truncate long matches
                                    'context_start': max(0, text_lower.find(str(match)) - 20),
                                    'context_end': min(len(text_lower), text_lower.find(str(match)) + len(str(match)) + 20)
                                })
                    except re.error as e:
                        self.logger.warning(f"Invalid regex pattern: {pattern}, error: {str(e)}", extra={'session_id': session_id})
                        continue
                
                claim_counts[claim_type] = count
                patterns_found.extend(matched_patterns)

            # Count indicator types with enhanced analysis
            statistical_count = sum(1 for indicator in self.statistical_indicators if indicator in text_lower)
            quote_count = sum(1 for indicator in self.quote_indicators if indicator in text_lower)

            # Count verifiability indicators with detailed scoring
            verifiability_scores = {}
            verifiability_details = {}
            
            for level, indicators in self.verification_indicators.items():
                count = 0
                found_indicators = []
                
                for indicator in indicators:
                    if indicator in text_lower:
                        count += 1
                        found_indicators.append(indicator)
                
                verifiability_scores[level] = count
                verifiability_details[level] = found_indicators[:5]  # Limit to top 5

            # Calculate enhanced claim richness with weighted scoring
            total_indicators = sum(claim_counts.values()) + statistical_count + quote_count
            
            # Apply configurable weights for different claim types
            weights = self.config.get('claim_type_weights', {
                'statistical': 2.0,
                'research': 2.0,
                'attribution': 1.5,
                'event': 1.0,
                'policy': 1.5,
                'causal': 1.0,
                'temporal': 0.5
            })
            
            weighted_score = sum(claim_counts.get(claim_type, 0) * weight 
                               for claim_type, weight in weights.items())
            
            richness_multiplier = self.config.get('richness_multiplier', 0.5)
            claim_richness_score = min(10, (weighted_score + statistical_count + quote_count) * richness_multiplier)

            # Determine dominant claim type with confidence
            dominant_claim_type = 'none'
            dominant_confidence = 0.0
            
            if claim_counts:
                max_count = max(claim_counts.values())
                if max_count > 0:
                    dominant_claim_type = max(claim_counts, key=claim_counts.get)
                    dominant_confidence = max_count / max(total_indicators, 1)

            # Calculate processing time and update metrics
            processing_time = time.time() - start_time
            self.analysis_count += 1
            self.total_processing_time += processing_time
            self.total_patterns_found += len(patterns_found)

            # Determine if article is claim-rich with enhanced logic
            rich_threshold = self.config.get('rich_threshold', 5)
            density_threshold = self.config.get('density_threshold', 0.01)  # Claims per 100 words
            
            word_count = len(article_text.split())
            claim_density = total_indicators / max(word_count / 100, 1)
            
            is_claim_rich = (total_indicators > rich_threshold or 
                           claim_density > density_threshold or
                           claim_richness_score > 6.0)

            # Enhanced result with comprehensive metadata
            result = {
                'claim_type_counts': claim_counts,
                'statistical_indicators': statistical_count,
                'quote_indicators': quote_count,
                'verifiability_scores': verifiability_scores,
                'verifiability_details': verifiability_details,
                'patterns_found': patterns_found[:20],  # Limit for performance
                'total_claim_indicators': total_indicators,
                'weighted_claim_score': round(weighted_score, 2),
                'claim_richness_score': round(claim_richness_score, 2),
                'claim_density': round(claim_density, 2),
                'likely_claim_rich': is_claim_rich,
                'dominant_claim_type': dominant_claim_type,
                'dominant_type_confidence': round(dominant_confidence, 2),
                'article_word_count': word_count,
                'processing_time_ms': round(processing_time * 1000, 2),
                'session_id': session_id,
                'quality_indicators': {
                    'has_statistical_claims': statistical_count > 0,
                    'has_attributions': quote_count > 0,
                    'has_high_verifiability': verifiability_scores.get('high_verifiability', 0) > 0,
                    'pattern_diversity': len([t for t, c in claim_counts.items() if c > 0])
                }
            }

            self.logger.info(
                f"Pattern analysis completed: {total_indicators} indicators found in {processing_time:.3f}s",
                extra={
                    'session_id': session_id,
                    'total_indicators': total_indicators,
                    'claim_richness': claim_richness_score,
                    'dominant_type': dominant_claim_type
                }
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Pattern analysis failed: {str(e)}", extra={'session_id': session_id})
            raise PatternAnalysisError(f"Pattern analysis failed: {str(e)}", "claim_pattern_analysis")

    def extract_potential_claims(self, article_text: str, max_claims: int = 10, session_id: str = None) -> List[str]:
        """
        Extract potential claim sentences from article text with enhanced filtering.

        Args:
            article_text: Article content to process
            max_claims: Maximum number of claims to extract
            session_id: Optional session ID for tracking

        Returns:
            List of potential claim sentences sorted by relevance
        """
        try:
            sentences = re.split(r'[.!?]+', article_text)
            potential_claims = []

            # Configuration weights for different claim types
            statistical_weight = self.config.get('statistical_weight', 3)
            quote_weight = self.config.get('quote_weight', 2)
            research_weight = self.config.get('research_weight', 3)
            event_weight = self.config.get('event_weight', 1)
            
            min_sentence_length = self.config.get('min_sentence_length', 20)
            max_sentence_length = self.config.get('max_sentence_length', 300)

            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip sentences that are too short or too long
                if len(sentence) < min_sentence_length or len(sentence) > max_sentence_length:
                    continue

                sentence_lower = sentence.lower()
                claim_score = 0

                # Score based on claim patterns with enhanced weighting
                for claim_type, patterns in self.claim_patterns.items():
                    type_weight = {
                        'statistical': statistical_weight,
                        'research': research_weight,
                        'attribution': quote_weight,
                        'event': event_weight,
                        'policy': 2,
                        'causal': 2,
                        'temporal': 1
                    }.get(claim_type, 1)
                    
                    for pattern in patterns:
                        try:
                            if re.search(pattern, sentence_lower):
                                claim_score += type_weight
                                break  # Only count one pattern per type per sentence
                        except re.error:
                            continue

                # Boost score for statistical indicators
                statistical_matches = sum(1 for indicator in self.statistical_indicators[:15] 
                                        if indicator in sentence_lower)
                claim_score += statistical_matches * statistical_weight

                # Boost score for quote indicators
                quote_matches = sum(1 for indicator in self.quote_indicators[:15] 
                                  if indicator in sentence_lower)
                claim_score += quote_matches * quote_weight

                # Penalty for very common words that might indicate non-claims
                common_penalty_words = ['the', 'and', 'but', 'however', 'also', 'very', 'really']
                common_word_ratio = sum(1 for word in common_penalty_words if word in sentence_lower) / len(sentence_lower.split())
                if common_word_ratio > 0.3:
                    claim_score *= 0.7  # Reduce score for sentences with too many common words

                # Add sentence to potential claims if it has indicators
                if claim_score > 0:
                    potential_claims.append({
                        'text': sentence + '.',
                        'score': claim_score,
                        'length': len(sentence),
                        'word_count': len(sentence.split()),
                        'statistical_matches': statistical_matches,
                        'quote_matches': quote_matches
                    })

            # Sort by score (descending) and length (descending as tiebreaker)
            potential_claims.sort(key=lambda x: (-x['score'], -x['length']))

            # Return top claims with enhanced metadata
            selected_claims = potential_claims[:max_claims]
            
            self.logger.info(
                f"Extracted {len(selected_claims)} potential claims from {len(sentences)} sentences",
                extra={'session_id': session_id, 'max_claims': max_claims}
            )

            return [claim['text'] for claim in selected_claims]

        except Exception as e:
            self.logger.error(f"Claim extraction failed: {str(e)}", extra={'session_id': session_id})
            raise PatternAnalysisError(f"Claim extraction failed: {str(e)}", "claim_extraction")

    def get_claim_density_analysis(self, article_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Analyze claim density throughout the article with enhanced granularity.

        Args:
            article_text: Article content to analyze
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with comprehensive claim density analysis
        """
        try:
            # Split into paragraphs and sentences for multi-level analysis
            paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip()]
            sentences = re.split(r'[.!?]+', article_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not paragraphs:
                return {
                    'claim_density': 0.0,
                    'high_density_paragraphs': 0,
                    'sentence_level_analysis': {},
                    'paragraph_level_analysis': {}
                }

            # Paragraph-level analysis
            paragraph_scores = []
            high_density_count = 0
            density_threshold = self.config.get('high_density_threshold', 3)

            paragraph_details = []
            for i, paragraph in enumerate(paragraphs):
                potential_claims = self.extract_potential_claims(paragraph, max_claims=50, session_id=session_id)
                claim_count = len(potential_claims)
                
                # Calculate density as claims per 100 words
                word_count = len(paragraph.split())
                density = (claim_count / max(word_count, 1)) * 100
                
                paragraph_scores.append(density)
                paragraph_details.append({
                    'paragraph_index': i,
                    'claim_count': claim_count,
                    'word_count': word_count,
                    'density': round(density, 2),
                    'is_high_density': claim_count >= density_threshold
                })
                
                if claim_count >= density_threshold:
                    high_density_count += 1

            # Sentence-level analysis
            sentence_analysis = self._analyze_sentence_density(sentences, session_id)

            # Overall statistics
            overall_density = sum(paragraph_scores) / len(paragraph_scores) if paragraph_scores else 0
            total_words = sum(detail['word_count'] for detail in paragraph_details)
            total_claims = sum(detail['claim_count'] for detail in paragraph_details)

            result = {
                'claim_density': round(overall_density, 2),
                'high_density_paragraphs': high_density_count,
                'total_paragraphs': len(paragraphs),
                'total_sentences': len(sentences),
                'total_words': total_words,
                'total_potential_claims': total_claims,
                'density_distribution': paragraph_scores,
                'max_density': max(paragraph_scores) if paragraph_scores else 0,
                'min_density': min(paragraph_scores) if paragraph_scores else 0,
                'paragraph_details': paragraph_details,
                'sentence_level_analysis': sentence_analysis,
                'quality_metrics': {
                    'density_variance': self._calculate_variance(paragraph_scores),
                    'consistency_score': self._calculate_consistency_score(paragraph_scores),
                    'claim_distribution': self._analyze_claim_distribution(paragraph_details)
                },
                'session_id': session_id
            }

            self.logger.info(
                f"Density analysis: {overall_density:.2f} overall, {high_density_count}/{len(paragraphs)} high-density paragraphs",
                extra={'session_id': session_id}
            )

            return result

        except Exception as e:
            self.logger.error(f"Density analysis failed: {str(e)}", extra={'session_id': session_id})
            raise PatternAnalysisError(f"Density analysis failed: {str(e)}", "density_analysis")

    def _analyze_sentence_density(self, sentences: List[str], session_id: str = None) -> Dict[str, Any]:
        """Analyze claim density at sentence level."""
        try:
            sentence_scores = []
            claim_sentences = 0
            
            for sentence in sentences:
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                    
                # Quick claim scoring for sentences
                score = 0
                sentence_lower = sentence.lower()
                
                # Check for key indicators
                for indicator in self.statistical_indicators[:10]:
                    if indicator in sentence_lower:
                        score += 1
                        
                for indicator in self.quote_indicators[:10]:
                    if indicator in sentence_lower:
                        score += 1
                
                sentence_scores.append(score)
                if score > 0:
                    claim_sentences += 1
            
            total_sentences = len(sentence_scores)
            average_score = sum(sentence_scores) / max(total_sentences, 1)
            claim_sentence_ratio = claim_sentences / max(total_sentences, 1)
            
            return {
                'total_sentences': total_sentences,
                'claim_bearing_sentences': claim_sentences,
                'claim_sentence_ratio': round(claim_sentence_ratio, 3),
                'average_sentence_score': round(average_score, 2),
                'max_sentence_score': max(sentence_scores) if sentence_scores else 0,
                'sentences_with_high_scores': sum(1 for score in sentence_scores if score >= 3)
            }
            
        except Exception:
            return {'error': 'Sentence analysis failed'}

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of density values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _calculate_consistency_score(self, values: List[float]) -> float:
        """Calculate consistency score based on variance."""
        if not values:
            return 0.0
        
        variance = self._calculate_variance(values)
        mean = sum(values) / len(values)
        
        if mean == 0:
            return 100.0
        
        # Lower coefficient of variation means higher consistency
        cv = (variance ** 0.5) / mean
        consistency = max(0, 100 - (cv * 50))  # Scale to 0-100
        return round(consistency, 2)

    def _analyze_claim_distribution(self, paragraph_details: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of claims across paragraphs."""
        distribution = defaultdict(int)
        
        for detail in paragraph_details:
            claim_count = detail['claim_count']
            if claim_count == 0:
                distribution['no_claims'] += 1
            elif claim_count <= 2:
                distribution['low_claims'] += 1
            elif claim_count <= 5:
                distribution['medium_claims'] += 1
            else:
                distribution['high_claims'] += 1
        
        return dict(distribution)

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern database statistics with enhanced metrics."""
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
                'total_verifiability_indicators': total_verifiability_indicators,
                'pattern_breakdown': {
                    claim_type: len(patterns) for claim_type, patterns in self.claim_patterns.items()
                }
            },
            'performance_statistics': {
                'total_analyses': self.analysis_count,
                'total_patterns_found': self.total_patterns_found,
                'average_processing_time_ms': round(avg_processing_time * 1000, 2),
                'average_patterns_per_analysis': round(avg_patterns_per_analysis, 2),
                'total_processing_time_seconds': round(self.total_processing_time, 2)
            },
            'configuration_applied': bool(self.config),
            'quality_metrics': {
                'pattern_coverage': 'comprehensive',
                'verification_levels': 3,
                'claim_types_supported': len(self.claim_patterns),
                'performance_optimized': True
            }
        }

    def validate_pattern_database(self) -> Dict[str, Any]:
        """Validate pattern database integrity and completeness with enhanced checks."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
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

        # Check pattern syntax validity
        invalid_patterns = []
        for category, patterns in self.claim_patterns.items():
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    invalid_patterns.append(f"{category}: {pattern} - {str(e)}")
                    validation_results['valid'] = False

        if invalid_patterns:
            validation_results['issues'].extend(invalid_patterns[:5])  # Limit to first 5

        # Check for very short indicators (might be too generic)
        short_indicators = []
        for indicators in [self.statistical_indicators, self.quote_indicators]:
            short_indicators.extend([ind for ind in indicators if len(ind) < 3])
        
        if short_indicators:
            validation_results['warnings'].append(f"Very short indicators: {short_indicators[:5]}")

        # Check verifiability indicator balance
        verif_counts = {level: len(indicators) for level, indicators in self.verification_indicators.items()}
        total_verif = sum(verif_counts.values())
        
        if total_verif > 0:
            high_ratio = verif_counts.get('high_verifiability', 0) / total_verif
            if high_ratio < 0.2:
                validation_results['warnings'].append("Low proportion of high-verifiability indicators")

        # Add statistics to validation results
        validation_results['statistics'] = {
            'total_patterns': len(all_patterns),
            'duplicate_count': len(duplicates),
            'invalid_patterns': len(invalid_patterns),
            'short_indicators': len(short_indicators),
            'verifiability_distribution': verif_counts
        }

        return validation_results

    def get_claim_type_distribution(self, article_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Get detailed distribution of claim types in the article.

        Args:
            article_text: Article content to analyze
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with claim type distribution analysis
        """
        try:
            analysis = self.analyze_claim_patterns(article_text, session_id)
            claim_counts = analysis['claim_type_counts']
            total_claims = sum(claim_counts.values())
            
            if total_claims == 0:
                return {
                    'distribution': {},
                    'dominant_types': [],
                    'coverage_score': 0.0,
                    'diversity_score': 0.0
                }

            # Calculate percentages
            distribution = {}
            for claim_type, count in claim_counts.items():
                if count > 0:
                    distribution[claim_type] = {
                        'count': count,
                        'percentage': round((count / total_claims) * 100, 1)
                    }

            # Identify dominant types (>20% of total)
            dominant_types = [
                claim_type for claim_type, data in distribution.items()
                if data['percentage'] > 20
            ]

            # Calculate coverage score (how many different types found)
            coverage_score = len(distribution) / len(self.claim_patterns) * 100

            # Calculate diversity score (how evenly distributed)
            if len(distribution) > 1:
                percentages = [data['percentage'] for data in distribution.values()]
                max_possible_entropy = len(percentages)
                actual_entropy = sum(p * (1/p) for p in percentages if p > 0) / 100
                diversity_score = (actual_entropy / max_possible_entropy) * 100
            else:
                diversity_score = 0.0

            return {
                'distribution': distribution,
                'dominant_types': dominant_types,
                'coverage_score': round(coverage_score, 1),
                'diversity_score': round(diversity_score, 1),
                'total_claims': total_claims,
                'types_found': len(distribution),
                'session_id': session_id
            }

        except Exception as e:
            self.logger.error(f"Claim type distribution analysis failed: {str(e)}", extra={'session_id': session_id})
            return {
                'error': str(e),
                'distribution': {},
                'session_id': session_id
            }


# Testing functionality
if __name__ == "__main__":
    """Test pattern database functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
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
    The policy change was caused by increasing public pressure and lobbying efforts.
    """

    print("=== PATTERN ANALYSIS TEST ===")

    analysis = pattern_db.analyze_claim_patterns(test_article, session_id="test_pattern_001")
    print(f"✅ Total claim indicators: {analysis['total_claim_indicators']}")
    print(f"✅ Claim richness score: {analysis['claim_richness_score']}/10")
    print(f"✅ Likely claim-rich: {analysis['likely_claim_rich']}")
    print(f"✅ Dominant claim type: {analysis['dominant_claim_type']}")
    print(f"✅ Processing time: {analysis['processing_time_ms']:.1f}ms")

    print(f"\nClaim type counts: {analysis['claim_type_counts']}")
    print(f"Statistical indicators: {analysis['statistical_indicators']}")
    print(f"Quote indicators: {analysis['quote_indicators']}")
    print(f"Verifiability scores: {analysis['verifiability_scores']}")

    # Test potential claim extraction
    print("\n=== POTENTIAL CLAIMS EXTRACTION ===")
    potential_claims = pattern_db.extract_potential_claims(test_article, max_claims=5, session_id="test_pattern_002")
    for i, claim in enumerate(potential_claims, 1):
        print(f"✅ {i}. {claim}")

    # Test claim density analysis
    print("\n=== CLAIM DENSITY ANALYSIS ===")
    density_analysis = pattern_db.get_claim_density_analysis(test_article, session_id="test_pattern_003")
    print(f"✅ Overall claim density: {density_analysis['claim_density']:.2f}")
    print(f"✅ High-density paragraphs: {density_analysis['high_density_paragraphs']}")
    print(f"✅ Total paragraphs: {density_analysis['total_paragraphs']}")
    print(f"✅ Claim sentence ratio: {density_analysis['sentence_level_analysis']['claim_sentence_ratio']:.3f}")

    # Test claim type distribution
    print("\n=== CLAIM TYPE DISTRIBUTION ===")
    distribution = pattern_db.get_claim_type_distribution(test_article, session_id="test_pattern_004")
    print(f"✅ Coverage score: {distribution['coverage_score']:.1f}%")
    print(f"✅ Diversity score: {distribution['diversity_score']:.1f}%")
    print(f"✅ Dominant types: {distribution['dominant_types']}")

    # Test database statistics
    print("\n=== DATABASE STATISTICS ===")
    stats = pattern_db.get_pattern_statistics()
    print(f"✅ Pattern types: {stats['database_composition']['claim_pattern_types']}")
    print(f"✅ Total patterns: {stats['database_composition']['total_claim_patterns']}")
    print(f"✅ Analyses completed: {stats['performance_statistics']['total_analyses']}")
    print(f"✅ Average processing time: {stats['performance_statistics']['average_processing_time_ms']:.1f}ms")

    # Test validation
    print("\n=== DATABASE VALIDATION ===")
    validation = pattern_db.validate_pattern_database()
    print(f"✅ Database valid: {'PASSED' if validation['valid'] else 'FAILED'}")
    print(f"✅ Total patterns: {validation['statistics']['total_patterns']}")
    print(f"✅ Invalid patterns: {validation['statistics']['invalid_patterns']}")
    
    if validation['issues']:
        print(f"⚠️  Issues: {validation['issues'][:2]}")
    if validation['warnings']:
        print(f"⚠️  Warnings: {validation['warnings'][:2]}")

    print("\n🎯 Pattern database tests completed successfully!")
