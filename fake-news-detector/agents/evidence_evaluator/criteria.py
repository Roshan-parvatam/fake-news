# agents/evidence_evaluator/criteria.py
"""
Evidence Quality Criteria for Evidence Evaluator Agent - Config Enhanced

Enhanced evidence quality assessment with better performance tracking
and configuration awareness.
"""

from typing import Dict, List, Any
import re
import logging
import time

class EvidenceQualityCriteria:
    """
    📊 ENHANCED EVIDENCE QUALITY CRITERIA WITH CONFIG AWARENESS
    
    This class provides systematic criteria for assessing evidence quality
    in news articles with enhanced performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the evidence quality criteria with optional config
        
        Args:
            config: Optional configuration for evidence assessment
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize quality criteria systems
        self.source_quality_indicators = self._initialize_source_quality_indicators()
        self.evidence_strength_indicators = self._initialize_evidence_strength_indicators()
        self.verification_indicators = self._initialize_verification_indicators()
        self.transparency_indicators = self._initialize_transparency_indicators()
        
        # Performance tracking
        self.criteria_stats = {
            'total_assessments': 0,
            'total_quality_indicators_found': 0,
            'analysis_time_total': 0.0,
            'config_applied': bool(config)
        }
        
        self.logger.info(f"✅ EvidenceQualityCriteria initialized with {len(self.source_quality_indicators)} quality categories")
    
    def _initialize_source_quality_indicators(self) -> Dict[str, Dict[str, Any]]:
        """
        📊 SOURCE QUALITY INDICATORS DATABASE - Enhanced assessment criteria
        """
        return {
            'primary_sources': {
                'indicators': [
                    'original document', 'official statement', 'direct quote',
                    'eyewitness account', 'first-hand experience', 'primary data',
                    'government record', 'court filing', 'official report'
                ],
                'patterns': [
                    r'according\s+to\s+the\s+original',
                    r'official\s+statement\s+from',
                    r'eyewitness\s+reports?',
                    r'direct\s+quote\s+from'
                ],
                'quality_score': 9.0,
                'reliability': 'very_high'
            },
            'expert_sources': {
                'indicators': [
                    'expert', 'specialist', 'professor', 'researcher',
                    'scientist', 'doctor', 'analyst', 'authority',
                    'academic', 'scholar', 'professional'
                ],
                'patterns': [
                    r'dr\.\s+\w+',
                    r'professor\s+\w+',
                    r'expert\s+in\s+\w+',
                    r'leading\s+researcher'
                ],
                'quality_score': 8.0,
                'reliability': 'high'
            },
            'institutional_sources': {
                'indicators': [
                    'university', 'institute', 'organization', 'agency',
                    'department', 'ministry', 'bureau', 'commission',
                    'foundation', 'association', 'society'
                ],
                'patterns': [
                    r'university\s+of\s+\w+',
                    r'institute\s+for\s+\w+',
                    r'department\s+of\s+\w+',
                    r'national\s+\w+\s+agency'
                ],
                'quality_score': 7.5,
                'reliability': 'high'
            },
            'journalistic_sources': {
                'indicators': [
                    'reporter', 'correspondent', 'journalist', 'investigation',
                    'news team', 'editorial board', 'fact-check', 'verification'
                ],
                'patterns': [
                    r'reported\s+by\s+\w+',
                    r'investigation\s+by',
                    r'fact-checked\s+by',
                    r'verified\s+by'
                ],
                'quality_score': 7.0,
                'reliability': 'medium_high'
            },
            'anonymous_sources': {
                'indicators': [
                    'anonymous', 'unnamed', 'confidential', 'source close to',
                    'insider', 'official who spoke', 'person familiar'
                ],
                'patterns': [
                    r'anonymous\s+source',
                    r'unnamed\s+official',
                    r'source\s+close\s+to',
                    r'person\s+familiar\s+with'
                ],
                'quality_score': 4.0,
                'reliability': 'medium_low'
            },
            'social_media_sources': {
                'indicators': [
                    'tweet', 'facebook post', 'social media', 'instagram',
                    'tiktok', 'youtube', 'blog post', 'forum'
                ],
                'patterns': [
                    r'tweeted\s+that',
                    r'posted\s+on\s+facebook',
                    r'social\s+media\s+post',
                    r'viral\s+video'
                ],
                'quality_score': 3.0,
                'reliability': 'low'
            }
        }
    
    def _initialize_evidence_strength_indicators(self) -> Dict[str, List[str]]:
        """
        📊 EVIDENCE STRENGTH INDICATORS - What makes evidence strong
        """
        return {
            'strong_evidence': [
                'peer-reviewed', 'published study', 'clinical trial',
                'controlled experiment', 'statistical analysis', 'meta-analysis',
                'longitudinal study', 'randomized trial', 'systematic review',
                'large sample size', 'multiple sources', 'corroborating evidence',
                'independent verification', 'replication', 'consensus'
            ],
            'moderate_evidence': [
                'survey', 'poll', 'observational study', 'case study',
                'anecdotal evidence', 'expert opinion', 'analysis',
                'comparison', 'correlation', 'pattern', 'trend'
            ],
            'weak_evidence': [
                'rumor', 'speculation', 'allegation', 'claim',
                'unverified report', 'single source', 'isolated incident',
                'personal opinion', 'hearsay', 'assumption'
            ],
            'quantitative_indicators': [
                'sample size', 'confidence interval', 'margin of error',
                'statistical significance', 'p-value', 'standard deviation',
                'regression analysis', 'correlation coefficient'
            ]
        }
    
    def _initialize_verification_indicators(self) -> Dict[str, List[str]]:
        """
        📊 VERIFICATION INDICATORS - Evidence of fact-checking and verification
        """
        return {
            'high_verification': [
                'fact-checked', 'verified', 'confirmed', 'substantiated',
                'corroborated', 'cross-referenced', 'independently verified',
                'multiple sources confirm', 'official confirmation'
            ],
            'moderate_verification': [
                'reported by multiple outlets', 'consistent with',
                'similar reports', 'appears to confirm', 'tends to support'
            ],
            'low_verification': [
                'unconfirmed', 'unverified', 'alleged', 'claimed',
                'reportedly', 'supposedly', 'purportedly'
            ],
            'contradictory_evidence': [
                'disputed', 'contradicted', 'refuted', 'debunked',
                'conflicting reports', 'inconsistent with'
            ]
        }
    
    def _initialize_transparency_indicators(self) -> Dict[str, List[str]]:
        """
        📊 TRANSPARENCY INDICATORS - Openness about methods and limitations
        """
        return {
            'high_transparency': [
                'methodology', 'limitations', 'conflicts of interest',
                'funding source', 'data available', 'full disclosure',
                'transparent process', 'open access'
            ],
            'moderate_transparency': [
                'some limitations', 'partial disclosure', 'general methodology',
                'brief description'
            ],
            'low_transparency': [
                'no methodology provided', 'undisclosed funding',
                'conflicts not mentioned', 'limited information'
            ]
        }
    
    def assess_evidence_quality(self, article_text: str, extracted_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        📊 COMPREHENSIVE EVIDENCE QUALITY ASSESSMENT WITH CONFIG
        
        Systematic assessment of evidence quality using all criteria with performance tracking.
        """
        start_time = time.time()
        text_lower = article_text.lower()
        
        # Source quality assessment
        source_assessment = self._assess_source_quality(text_lower)
        
        # Evidence strength assessment
        strength_assessment = self._assess_evidence_strength(text_lower)
        
        # Verification assessment
        verification_assessment = self._assess_verification_level(text_lower)
        
        # Transparency assessment
        transparency_assessment = self._assess_transparency(text_lower)
        
        # Claims-based assessment
        claims_assessment = self._assess_claims_quality(extracted_claims)
        
        # Calculate overall scores with config weights
        overall_scores = self._calculate_overall_scores(
            source_assessment, strength_assessment, verification_assessment,
            transparency_assessment, claims_assessment
        )
        
        # Performance tracking
        processing_time = time.time() - start_time
        total_indicators = (source_assessment['total_indicators'] + 
                          strength_assessment['total_indicators'] +
                          verification_assessment['total_indicators'] +
                          transparency_assessment['total_indicators'])
        
        self.criteria_stats['total_assessments'] += 1
        self.criteria_stats['total_quality_indicators_found'] += total_indicators
        self.criteria_stats['analysis_time_total'] += processing_time
        
        return {
            'source_quality_assessment': source_assessment,
            'evidence_strength_assessment': strength_assessment,
            'verification_assessment': verification_assessment,
            'transparency_assessment': transparency_assessment,
            'claims_quality_assessment': claims_assessment,
            'overall_quality_score': overall_scores['overall_score'],
            'source_quality_score': overall_scores['source_score'],
            'completeness_score': overall_scores['completeness_score'],
            'quality_summary': overall_scores['quality_summary'],
            'quality_indicators_found': total_indicators,
            'analysis_time_ms': round(processing_time * 1000, 2),
            'config_applied': bool(self.config)
        }
    
    def _assess_source_quality(self, text_lower: str) -> Dict[str, Any]:
        """Assess source quality based on source type indicators"""
        source_scores = {}
        source_counts = {}
        total_indicators = 0
        
        for source_type, info in self.source_quality_indicators.items():
            count = 0
            
            # Check indicators
            for indicator in info['indicators']:
                if indicator in text_lower:
                    count += 1
                    total_indicators += 1
            
            # Check patterns
            for pattern in info['patterns']:
                matches = re.findall(pattern, text_lower)
                if matches:
                    count += len(matches)
                    total_indicators += len(matches)
            
            if count > 0:
                source_scores[source_type] = info['quality_score'] * min(1.0, count / 3)
                source_counts[source_type] = count
        
        # Calculate weighted average score
        if source_scores:
            total_weight = sum(source_counts.values())
            weighted_score = sum(score * source_counts[source_type] 
                               for source_type, score in source_scores.items()) / total_weight
        else:
            weighted_score = 5.0  # Default neutral score
        
        return {
            'source_scores': source_scores,
            'source_counts': source_counts,
            'weighted_average_score': round(weighted_score, 2),
            'total_indicators': total_indicators,
            'dominant_source_type': max(source_counts, key=source_counts.get) if source_counts else 'none'
        }
    
    def _assess_evidence_strength(self, text_lower: str) -> Dict[str, Any]:
        """Assess evidence strength based on methodology and rigor indicators"""
        strength_counts = {}
        
        for strength_level, indicators in self.evidence_strength_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            strength_counts[strength_level] = count
        
        # Calculate strength score with config weights
        strength_weights = self.config.get('strength_weights', {
            'strong_evidence': 3.0,
            'moderate_evidence': 1.5,
            'weak_evidence': -1.0,
            'quantitative_indicators': 2.0
        }) if self.config else {
            'strong_evidence': 3.0,
            'moderate_evidence': 1.5,
            'weak_evidence': -1.0,
            'quantitative_indicators': 2.0
        }
        
        weighted_strength = sum(
            strength_counts.get(level, 0) * weight 
            for level, weight in strength_weights.items()
        )
        
        # Normalize to 0-10 scale
        strength_score = max(0, min(10, 5 + (weighted_strength * 0.5)))
        
        return {
            'strength_counts': strength_counts,
            'weighted_strength_score': weighted_strength,
            'normalized_score': round(strength_score, 2),
            'total_indicators': sum(strength_counts.values())
        }
    
    def _assess_verification_level(self, text_lower: str) -> Dict[str, Any]:
        """Assess verification and fact-checking indicators"""
        verification_counts = {}
        
        for verification_level, indicators in self.verification_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            verification_counts[verification_level] = count
        
        # Calculate verification score
        verification_weights = {
            'high_verification': 3.0,
            'moderate_verification': 1.0,
            'low_verification': -0.5,
            'contradictory_evidence': -2.0
        }
        
        weighted_verification = sum(
            verification_counts.get(level, 0) * weight 
            for level, weight in verification_weights.items()
        )
        
        verification_score = max(0, min(10, 5 + (weighted_verification * 0.8)))
        
        return {
            'verification_counts': verification_counts,
            'weighted_verification_score': weighted_verification,
            'normalized_score': round(verification_score, 2),
            'total_indicators': sum(verification_counts.values())
        }
    
    def _assess_transparency(self, text_lower: str) -> Dict[str, Any]:
        """Assess transparency and disclosure indicators"""
        transparency_counts = {}
        
        for transparency_level, indicators in self.transparency_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            transparency_counts[transparency_level] = count
        
        # Calculate transparency score
        transparency_weights = {
            'high_transparency': 2.5,
            'moderate_transparency': 1.0,
            'low_transparency': -1.0
        }
        
        weighted_transparency = sum(
            transparency_counts.get(level, 0) * weight 
            for level, weight in transparency_weights.items()
        )
        
        transparency_score = max(0, min(10, 5 + (weighted_transparency * 0.7)))
        
        return {
            'transparency_counts': transparency_counts,
            'weighted_transparency_score': weighted_transparency,
            'normalized_score': round(transparency_score, 2),
            'total_indicators': sum(transparency_counts.values())
        }
    
    def _assess_claims_quality(self, extracted_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess quality based on extracted claims characteristics"""
        if not extracted_claims:
            return {
                'total_claims': 0,
                'average_verifiability': 0,
                'high_verifiability_count': 0,
                'claims_quality_score': 0
            }
        
        verifiability_scores = [claim.get('verifiability_score', 5) for claim in extracted_claims]
        average_verifiability = sum(verifiability_scores) / len(verifiability_scores)
        high_verifiability_count = sum(1 for score in verifiability_scores if score >= 7)
        
        # Claims quality score based on quantity and verifiability
        quantity_factor = min(1.0, len(extracted_claims) / 5)  # Optimal around 5 claims
        verifiability_factor = average_verifiability / 10
        
        claims_quality_score = (quantity_factor * 0.3 + verifiability_factor * 0.7) * 10
        
        return {
            'total_claims': len(extracted_claims),
            'average_verifiability': round(average_verifiability, 2),
            'high_verifiability_count': high_verifiability_count,
            'claims_quality_score': round(claims_quality_score, 2),
            'verifiability_distribution': {
                'high': sum(1 for s in verifiability_scores if s >= 7),
                'medium': sum(1 for s in verifiability_scores if 4 <= s < 7),
                'low': sum(1 for s in verifiability_scores if s < 4)
            }
        }
    
    def _calculate_overall_scores(self, source_assessment: Dict, strength_assessment: Dict,
                                 verification_assessment: Dict, transparency_assessment: Dict,
                                 claims_assessment: Dict) -> Dict[str, Any]:
        """Calculate overall quality scores with config weights"""
        # Component scores
        source_score = source_assessment['weighted_average_score']
        strength_score = strength_assessment['normalized_score']
        verification_score = verification_assessment['normalized_score']
        transparency_score = transparency_assessment['normalized_score']
        claims_score = claims_assessment['claims_quality_score']
        
        # Overall score weights from config
        overall_weights = self.config.get('overall_weights', {
            'source_quality': 0.3,
            'evidence_strength': 0.25,
            'verification': 0.2,
            'transparency': 0.15,
            'claims_quality': 0.1
        }) if self.config else {
            'source_quality': 0.3,
            'evidence_strength': 0.25,
            'verification': 0.2,
            'transparency': 0.15,
            'claims_quality': 0.1
        }
        
        overall_score = (
            source_score * overall_weights['source_quality'] +
            strength_score * overall_weights['evidence_strength'] +
            verification_score * overall_weights['verification'] +
            transparency_score * overall_weights['transparency'] +
            claims_score * overall_weights['claims_quality']
        )
        
        # Quality summary
        if overall_score >= 8.0:
            quality_summary = "Excellent evidence quality with strong sources and methodology"
        elif overall_score >= 6.5:
            quality_summary = "Good evidence quality with reliable sources"
        elif overall_score >= 5.0:
            quality_summary = "Moderate evidence quality with some limitations"
        elif overall_score >= 3.5:
            quality_summary = "Poor evidence quality with significant concerns"
        else:
            quality_summary = "Very poor evidence quality requiring substantial verification"
        
        return {
            'overall_score': round(overall_score, 2),
            'source_score': round(source_score, 2),
            'completeness_score': round((strength_score + verification_score) / 2, 2),
            'quality_summary': quality_summary,
            'component_scores': {
                'source_quality': round(source_score, 2),
                'evidence_strength': round(strength_score, 2),
                'verification': round(verification_score, 2),
                'transparency': round(transparency_score, 2),
                'claims_quality': round(claims_score, 2)
            },
            'weights_used': overall_weights
        }
    
    def get_criteria_statistics(self) -> Dict[str, Any]:
        """Get comprehensive criteria statistics"""
        base_stats = {
            'source_quality_categories': len(self.source_quality_indicators),
            'evidence_strength_categories': len(self.evidence_strength_indicators),
            'verification_levels': len(self.verification_indicators),
            'transparency_levels': len(self.transparency_indicators),
            'total_quality_indicators': sum(
                len(indicators['indicators']) + len(indicators['patterns'])
                for indicators in self.source_quality_indicators.values()
            ) + sum(
                len(indicators) for category in self.evidence_strength_indicators.values()
                for indicators in [category] if isinstance(category, list)
            )
        }
        
        # Add performance stats
        performance_stats = self.criteria_stats.copy()
        if performance_stats['total_assessments'] > 0:
            performance_stats['average_assessment_time_ms'] = round(
                (performance_stats['analysis_time_total'] / performance_stats['total_assessments']) * 1000, 2
            )
            performance_stats['average_indicators_per_assessment'] = round(
                performance_stats['total_quality_indicators_found'] / performance_stats['total_assessments'], 2
            )
        
        return {**base_stats, 'performance_stats': performance_stats}

# Testing
if __name__ == "__main__":
    """Test evidence quality criteria with config"""
    test_config = {
        'strength_weights': {
            'strong_evidence': 3.5,
            'moderate_evidence': 1.0,
            'weak_evidence': -1.5,
            'quantitative_indicators': 2.5
        },
        'overall_weights': {
            'source_quality': 0.4,
            'evidence_strength': 0.3,
            'verification': 0.2,
            'transparency': 0.1,
            'claims_quality': 0.0
        }
    }
    
    criteria = EvidenceQualityCriteria(test_config)
    
    test_text = """
    According to a peer-reviewed study published in Nature by Dr. Sarah Johnson
    from Harvard University, the clinical trial with 2,400 participants showed
    statistically significant results with p-value < 0.001. The research was
    independently verified by multiple institutions and the methodology is
    fully disclosed with data available for review.
    """
    
    test_claims = [
        {'verifiability_score': 9, 'text': 'Peer-reviewed study in Nature'},
        {'verifiability_score': 8, 'text': 'Clinical trial with 2,400 participants'},
        {'verifiability_score': 7, 'text': 'Statistically significant with p < 0.001'}
    ]
    
    assessment = criteria.assess_evidence_quality(test_text, test_claims)
    
    print(f"Evidence quality assessment results:")
    print(f"Overall quality score: {assessment['overall_quality_score']:.1f}/10")
    print(f"Source quality score: {assessment['source_quality_score']:.1f}/10")
    print(f"Quality summary: {assessment['quality_summary']}")
    print(f"Quality indicators found: {assessment['quality_indicators_found']}")
    
    stats = criteria.get_criteria_statistics()
    print(f"Criteria database contains {stats['total_quality_indicators']} quality indicators")
