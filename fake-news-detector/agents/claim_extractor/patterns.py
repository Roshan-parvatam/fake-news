# agents/claim_extractor/patterns.py
"""
Claim Pattern Database for Claim Extractor Agent - Config Enhanced

Enhanced claim detection patterns with better performance tracking
and configuration awareness.
"""

from typing import Dict, List, Any
import re
import logging

class ClaimPatternDatabase:
    """
    📊 ENHANCED CLAIM PATTERN DATABASE WITH CONFIG AWARENESS
    
    This class manages patterns and indicators for detecting different types of
    verifiable claims in news articles with enhanced performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the claim pattern database with optional config
        
        Args:
            config: Optional configuration for pattern matching
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize patterns
        self.claim_patterns = self._initialize_claim_patterns()
        self.statistical_indicators = self._initialize_statistical_indicators()
        self.quote_indicators = self._initialize_quote_indicators()
        self.verification_indicators = self._initialize_verification_indicators()
        
        # Performance tracking
        self.pattern_stats = {
            'total_analyses': 0,
            'total_patterns_found': 0,
            'analysis_time_total': 0.0,
            'config_applied': bool(config)
        }
        
        self.logger.info(f"✅ ClaimPatternDatabase initialized with {len(self.claim_patterns)} pattern types")
    
    def _initialize_claim_patterns(self) -> Dict[str, List[str]]:
        """
        📊 CLAIM PATTERN DATABASE - Enhanced with more patterns
        
        Creates patterns that help automatically identify different types of claims
        before sending to AI analysis.
        """
        return {
            # Statistical claim indicators  
            'statistical': [
                r'\d+\.?\d*\s*%',  # Percentages: "45%", "12.5%"
                r'\d+\.?\d*\s*(million|billion|thousand)',  # Large numbers
                r'\$\d+\.?\d*[kmb]?',  # Dollar amounts: "$500", "$1.2M"  
                r'\d+\.?\d*\s*(times|fold)',  # Multipliers: "3 times", "10-fold"
                r'increased.*by.*\d+',  # Change indicators
                r'decreased.*by.*\d+',
                r'rose.*to.*\d+', 
                r'fell.*to.*\d+',
                r'\d+\.?\d*\s*(points?|basis points?)',  # Point changes
                r'(up|down)\s+\d+\.?\d*\s*%',  # Percentage changes
                r'\d+\.?\d*\s*(percent|per cent)',  # Written percentages
                r'growth.*of.*\d+',  # Growth indicators
                r'decline.*of.*\d+',  # Decline indicators
                r'\d+\s*(year|month|day|week)s?\s+(ago|later)',  # Time indicators
                r'within\s+\d+\s*(days?|weeks?|months?)',  # Timeframe claims
            ],
            
            # Event claim indicators
            'event': [
                r'announced\s+(?:on|that|yesterday|today)',  # Announcements
                r'occurred\s+(?:on|at|in)',  # Events  
                r'will\s+(?:take place|happen|occur)',  # Future events
                r'meeting\s+(?:held|scheduled|planned)',  # Meetings
                r'launched\s+(?:on|in|at)',  # Launches
                r'signed\s+(?:on|in|at)',  # Signings
                r'passed\s+(?:the|a)\s+(?:law|bill)',  # Legislation
                r'voted\s+(?:to|on|for|against)',  # Voting
                r'appointed\s+(?:as|to)',  # Appointments
                r'resigned\s+(?:from|as)',  # Resignations
                r'arrested\s+(?:on|for)',  # Arrests
                r'died\s+(?:on|at|from)',  # Deaths
                r'began\s+(?:on|in|at)',  # Beginning events
                r'ended\s+(?:on|in|at)',  # Ending events
                r'scheduled\s+for',  # Scheduled events
                r'took\s+place\s+(?:on|in|at)'  # Past events
            ],
            
            # Research/study claim indicators  
            'research': [
                r'study\s+(?:shows|found|reveals)',  # Study findings
                r'research\s+(?:indicates|suggests)',  # Research results
                r'scientists\s+(?:discovered|found)',  # Scientific claims
                r'published\s+in\s+(?:journal|nature)',  # Publications
                r'peer.reviewed',  # Peer review
                r'clinical\s+trial',  # Clinical trials
                r'experiment\s+(?:showed|demonstrated)',  # Experiments
                r'survey\s+(?:found|showed|revealed)',  # Survey results
                r'poll\s+(?:shows|indicates)',  # Poll results
                r'analysis\s+(?:reveals|shows)',  # Analysis results
                r'data\s+(?:shows|indicates|suggests)',  # Data claims
                r'findings\s+(?:suggest|show|indicate)',  # General findings
                r'meta.analysis\s+(?:found|showed)',  # Meta-analysis
                r'longitudinal\s+study',  # Longitudinal studies
                r'randomized.*trial',  # Randomized trials
                r'controlled\s+study'  # Controlled studies
            ],
            
            # Quote/attribution claim indicators
            'attribution': [
                r'said\s+(?:that|in|during)',  # Direct quotes
                r'stated\s+(?:that|in|during)',  # Statements  
                r'according\s+to',  # Attributions
                r'spokesperson\s+(?:said|announced)',  # Official statements
                r'CEO\s+(?:announced|said|stated)',  # Executive statements
                r'president\s+(?:said|announced|declared)',  # Presidential statements
                r'minister\s+(?:stated|announced)',  # Ministerial statements
                r'official\s+(?:said|confirmed|announced)',  # Official confirmations
                r'expert\s+(?:said|explained|noted)',  # Expert opinions
                r'witness\s+(?:reported|said|testified)',  # Witness accounts
                r'source\s+(?:told|said|revealed)',  # Source attributions
                r'interview\s+(?:with|on|during)',  # Interview references
                r'testified\s+(?:that|in)',  # Testimonies
                r'confirmed\s+(?:that|in)',  # Confirmations
                r'denied\s+(?:that|the)',  # Denials
                r'claimed\s+(?:that|in)'  # Claims
            ],
            
            # Policy/legal claim indicators
            'policy': [
                r'law\s+(?:requires|mandates|prohibits)',  # Legal requirements
                r'regulation\s+(?:states|requires)',  # Regulatory claims
                r'policy\s+(?:allows|prevents|ensures)',  # Policy claims
                r'court\s+(?:ruled|decided|ordered)',  # Legal decisions
                r'judge\s+(?:ruled|ordered|sentenced)',  # Judicial decisions
                r'verdict\s+(?:was|found)',  # Court verdicts
                r'settlement\s+(?:reached|agreed)',  # Legal settlements
                r'constitution\s+(?:guarantees|protects)',  # Constitutional claims
                r'amendment\s+(?:protects|ensures)',  # Amendment claims
                r'Supreme Court\s+(?:ruled|decided)',  # Supreme Court decisions
                r'legislation\s+(?:passed|failed)',  # Legislative outcomes
                r'bill\s+(?:signed|vetoed)',  # Bill outcomes
                r'executive\s+order',  # Executive orders
                r'regulatory\s+approval'  # Regulatory decisions
            ],
            
            # Causal claim indicators
            'causal': [
                r'caused\s+by',  # Direct causation
                r'because\s+of',  # Causal explanation
                r'due\s+to',  # Attribution  
                r'result\s+of',  # Result indication
                r'leads\s+to',  # Forward causation
                r'triggers',  # Trigger indication
                r'responsible\s+for',  # Responsibility attribution
                r'blame.*for',  # Blame attribution
                r'linked\s+to',  # Association claims
                r'correlation.*between',  # Correlation claims
                r'stems\s+from',  # Origin claims
                r'attributed\s+to',  # Attribution claims
                r'resulted\s+in',  # Outcome claims
                r'contributed\s+to'  # Contributing factors
            ]
        }
    
    def _initialize_statistical_indicators(self) -> List[str]:
        """📈 STATISTICAL INDICATORS - Enhanced list"""
        return [
            'percent', 'percentage', '%', 'rate', 'ratio', 'proportion',
            'increase', 'decrease', 'growth', 'decline', 'rise', 'fall',
            'average', 'median', 'mean', 'total', 'sum', 'count',
            'million', 'billion', 'thousand', 'trillion',
            'times', 'fold', 'double', 'triple', 'half',
            'quarter', 'third', 'majority', 'minority',
            'most', 'least', 'highest', 'lowest',
            'sample size', 'margin of error', 'confidence interval',
            'statistically significant', 'p-value', 'standard deviation',
            'correlation coefficient', 'regression analysis',
            'baseline', 'benchmark', 'threshold', 'ceiling', 'floor'
        ]
    
    def _initialize_quote_indicators(self) -> List[str]:
        """💬 QUOTE INDICATORS - Enhanced list"""
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
            'maintained', 'asserted', 'emphasized', 'stressed'
        ]
    
    def _initialize_verification_indicators(self) -> Dict[str, List[str]]:
        """✅ VERIFICATION INDICATORS - Enhanced classification"""
        return {
            'high_verifiability': [
                'official document', 'public record', 'government data',
                'published study', 'peer-reviewed', 'statistical office',
                'census data', 'financial filing', 'court record',
                'parliamentary record', 'press release', 'official statement',
                'regulatory filing', 'academic paper', 'scientific journal',
                'government website', 'official database', 'legal document'
            ],
            'medium_verifiability': [
                'news report', 'interview', 'survey', 'poll',
                'analyst report', 'expert opinion', 'industry data',
                'company statement', 'spokesperson said', 'according to',
                'conference presentation', 'white paper', 'research report',
                'market analysis', 'trade publication', 'professional journal'
            ],
            'low_verifiability': [
                'anonymous source', 'unnamed official', 'insider information',
                'rumor', 'speculation', 'alleged', 'reportedly',
                'sources say', 'it is believed', 'word is',
                'unconfirmed reports', 'social media post', 'blog post',
                'forum discussion', 'hearsay', 'gossip', 'leaked information'
            ]
        }
    
    def analyze_claim_patterns(self, article_text: str) -> Dict[str, Any]:
        """
        🔍 ENHANCED PATTERN-BASED CLAIM DETECTION
        
        Performs comprehensive analysis using predefined patterns with performance tracking.
        """
        import time
        start_time = time.time()
        
        text_lower = article_text.lower()
        
        # Count different types of claim indicators
        claim_counts = {}
        patterns_found = []
        
        for claim_type, patterns in self.claim_patterns.items():
            count = 0
            type_patterns = []
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                count += len(matches)
                if matches:
                    type_patterns.extend([(claim_type, match) for match in matches[:2]])  # Limit per pattern
            
            claim_counts[claim_type] = count
            patterns_found.extend(type_patterns)
        
        # Count statistical indicators
        statistical_count = sum(1 for indicator in self.statistical_indicators
                               if indicator in text_lower)
        
        # Count quote indicators  
        quote_count = sum(1 for indicator in self.quote_indicators
                         if indicator in text_lower)
        
        # Assess verifiability indicators
        verifiability_scores = {}
        for level, indicators in self.verification_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            verifiability_scores[level] = count
        
        # Calculate claim richness score with config awareness
        total_indicators = sum(claim_counts.values()) + statistical_count + quote_count
        base_multiplier = self.config.get('richness_multiplier', 0.5) if self.config else 0.5
        claim_richness = min(10, total_indicators * base_multiplier)
        
        # Determine dominant claim type
        dominant_type = max(claim_counts, key=claim_counts.get) if claim_counts else 'none'
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.pattern_stats['total_analyses'] += 1
        self.pattern_stats['total_patterns_found'] += len(patterns_found)
        self.pattern_stats['analysis_time_total'] += processing_time
        
        result = {
            'claim_type_counts': claim_counts,
            'statistical_indicators': statistical_count,
            'quote_indicators': quote_count,
            'verifiability_scores': verifiability_scores,
            'patterns_found': patterns_found[:20],  # Limit to top 20
            'total_claim_indicators': total_indicators,
            'claim_richness_score': round(claim_richness, 2),
            'likely_claim_rich': total_indicators > self.config.get('rich_threshold', 5) if self.config else total_indicators > 5,
            'dominant_claim_type': dominant_type,
            'analysis_time_ms': round(processing_time * 1000, 2),
            'config_applied': bool(self.config)
        }
        
        return result
    
    def extract_potential_claims(self, article_text: str, max_claims: int = 10) -> List[str]:
        """
        ⚡ ENHANCED POTENTIAL CLAIMS EXTRACTION
        
        Fast extraction method with better scoring algorithm.
        """
        potential_claims = []
        sentences = article_text.split('.')
        
        # Get config scoring weights
        statistical_weight = self.config.get('statistical_weight', 2) if self.config else 2
        quote_weight = self.config.get('quote_weight', 1) if self.config else 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            sentence_lower = sentence.lower()
            claim_score = 0
            
            # Score based on different pattern types
            for claim_type, patterns in self.claim_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sentence_lower):
                        claim_score += 1
                        break  # Only count one match per claim type per sentence
            
            # Additional scoring with config weights
            if any(indicator in sentence_lower for indicator in self.statistical_indicators[:15]):
                claim_score += statistical_weight
            
            if any(indicator in sentence_lower for indicator in self.quote_indicators[:15]):
                claim_score += quote_weight
            
            # If sentence has claim indicators, add it
            if claim_score > 0:
                potential_claims.append({
                    'text': sentence.strip() + '.',
                    'score': claim_score,
                    'length': len(sentence)
                })
        
        # Sort by score (descending) and return top claims
        potential_claims.sort(key=lambda x: (-x['score'], -x['length']))
        return [claim['text'] for claim in potential_claims[:max_claims]]
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics about the pattern database"""
        base_stats = {
            'claim_pattern_types': len(self.claim_patterns),
            'total_claim_patterns': sum(len(patterns) for patterns in self.claim_patterns.values()),
            'statistical_indicators': len(self.statistical_indicators),
            'quote_indicators': len(self.quote_indicators),
            'verifiability_levels': len(self.verification_indicators),
            'total_verifiability_indicators': sum(len(indicators) for indicators in self.verification_indicators.values())
        }
        
        # Add performance stats
        performance_stats = self.pattern_stats.copy()
        if performance_stats['total_analyses'] > 0:
            performance_stats['average_analysis_time_ms'] = round(
                (performance_stats['analysis_time_total'] / performance_stats['total_analyses']) * 1000, 2
            )
            performance_stats['average_patterns_per_analysis'] = round(
                performance_stats['total_patterns_found'] / performance_stats['total_analyses'], 2
            )
        
        return {**base_stats, 'performance_stats': performance_stats}
