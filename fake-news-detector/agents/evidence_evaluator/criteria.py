# agents/evidence_evaluator/criteria.py

"""
Evidence Quality Criteria Assessment - Production Ready

Enhanced rule-based evidence quality assessment system with structured logging,
error handling, and configurable parameters for production reliability.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional

from .exceptions import EvidenceAssessmentError


class EvidenceQualityCriteria:
    """
    Production-ready evidence quality assessment system for news articles.
    
    Provides systematic criteria for evaluating evidence quality based on
    source types, evidence strength, verification potential, and transparency
    with enhanced error handling and structured logging.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize evidence quality criteria system with production configuration.

        Args:
            config: Optional configuration for evidence assessment weights and thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.EvidenceQualityCriteria")
        
        # Initialize assessment criteria with error handling
        try:
            self.source_quality_indicators = self._initialize_source_quality_indicators()
            self.evidence_strength_indicators = self._initialize_evidence_strength_indicators()
            self.verification_indicators = self._initialize_verification_indicators()
            self.transparency_indicators = self._initialize_transparency_indicators()
            
            self.logger.info(f"Evidence quality criteria initialized with {len(self.source_quality_indicators)} source types")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize evidence quality criteria: {str(e)}")
            raise EvidenceAssessmentError(
                f"Initialization failed: {str(e)}",
                assessment_type="initialization"
            )

        # Configurable weights and thresholds
        self.strength_weights = self.config.get('strength_weights', {
            'strong_evidence': 3.0,
            'moderate_evidence': 1.5,
            'weak_evidence': -1.0,
            'quantitative_indicators': 2.0
        })
        
        self.overall_weights = self.config.get('overall_weights', {
            'source_quality': 0.3,
            'evidence_strength': 0.25,
            'verification': 0.2,
            'transparency': 0.15,
            'claims_quality': 0.1
        })
        
        # Validation thresholds
        self.min_article_length = self.config.get('min_article_length', 50)
        self.max_claims_for_processing = self.config.get('max_claims_for_processing', 20)

        # Performance metrics
        self.assessment_count = 0
        self.total_processing_time = 0.0
        self.total_indicators_found = 0
        self.error_count = 0

    def _initialize_source_quality_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize source quality assessment criteria."""
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
        """Initialize evidence strength assessment criteria."""
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
        """Initialize verification potential assessment criteria."""
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
        """Initialize transparency assessment criteria."""
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

    def assess_evidence_quality(self, 
                              article_text: str, 
                              extracted_claims: List[Dict[str, Any]], 
                              session_id: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive evidence quality assessment with error handling.

        Args:
            article_text: Article content to assess
            extracted_claims: List of extracted claims with metadata
            session_id: Optional session ID for tracking

        Returns:
            Dictionary containing quality assessment results
        """
        start_time = time.time()
        
        # Input validation with structured logging
        if not article_text or not isinstance(article_text, str):
            self.error_count += 1
            self.logger.error(f"Invalid article text for quality assessment: {type(article_text).__name__}", 
                            extra={'session_id': session_id})
            raise EvidenceAssessmentError(
                f"Article text must be non-empty string, got {type(article_text).__name__}",
                assessment_type="input_validation",
                session_id=session_id
            )

        if len(article_text.strip()) < self.min_article_length:
            self.logger.warning(f"Article text very short: {len(article_text)} characters", 
                              extra={'session_id': session_id})

        if not isinstance(extracted_claims, list):
            self.logger.warning(f"Invalid claims format: {type(extracted_claims).__name__}", 
                              extra={'session_id': session_id})
            extracted_claims = []

        self.logger.info(f"Starting evidence quality assessment", 
                        extra={
                            'session_id': session_id,
                            'article_length': len(article_text),
                            'claims_count': len(extracted_claims)
                        })

        try:
            text_lower = article_text.lower()
            
            # Limit claims for processing efficiency
            claims_to_process = extracted_claims[:self.max_claims_for_processing]
            if len(extracted_claims) > self.max_claims_for_processing:
                self.logger.info(f"Limited claims processing to {self.max_claims_for_processing} out of {len(extracted_claims)}", 
                               extra={'session_id': session_id})

            # Perform individual assessments with error handling
            source_assessment = self._assess_source_quality(text_lower, session_id)
            strength_assessment = self._assess_evidence_strength(text_lower, session_id)
            verification_assessment = self._assess_verification_level(text_lower, session_id)
            transparency_assessment = self._assess_transparency(text_lower, session_id)
            claims_assessment = self._assess_claims_quality(claims_to_process, session_id)

            # Calculate overall scores with error handling
            overall_scores = self._calculate_overall_scores(
                source_assessment, strength_assessment, verification_assessment,
                transparency_assessment, claims_assessment, session_id
            )

            # Update performance metrics
            processing_time = time.time() - start_time
            self.assessment_count += 1
            self.total_processing_time += processing_time
            
            total_indicators = (
                source_assessment.get('total_indicators', 0) +
                strength_assessment.get('total_indicators', 0) +
                verification_assessment.get('total_indicators', 0) +
                transparency_assessment.get('total_indicators', 0)
            )
            self.total_indicators_found += total_indicators

            self.logger.info(f"Evidence quality assessment completed successfully", 
                           extra={
                               'session_id': session_id,
                               'processing_time': round(processing_time * 1000, 2),
                               'overall_score': overall_scores.get('overall_score', 0),
                               'indicators_found': total_indicators
                           })

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
                'processing_time_ms': round(processing_time * 1000, 2)
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.error_count += 1
            
            self.logger.error(f"Evidence quality assessment failed: {str(e)}", 
                            extra={
                                'session_id': session_id,
                                'processing_time': round(processing_time * 1000, 2),
                                'error_type': type(e).__name__
                            })
            
            # Return fallback results instead of crashing
            return self._create_fallback_assessment(session_id)

    def _assess_source_quality(self, text_lower: str, session_id: str = None) -> Dict[str, Any]:
        """Assess source quality based on source type indicators with error handling."""
        try:
            source_scores = {}
            source_counts = {}
            total_indicators = 0

            for source_type, info in self.source_quality_indicators.items():
                try:
                    count = 0
                    
                    # Check text indicators
                    for indicator in info['indicators']:
                        if indicator in text_lower:
                            count += 1
                            total_indicators += 1
                    
                    # Check regex patterns with error handling
                    for pattern in info['patterns']:
                        try:
                            matches = re.findall(pattern, text_lower)
                            if matches:
                                count += len(matches)
                                total_indicators += len(matches)
                        except re.error as regex_error:
                            self.logger.warning(f"Regex error in source quality pattern: {str(regex_error)}", 
                                              extra={'session_id': session_id})
                            continue
                    
                    if count > 0:
                        # Calculate score with diminishing returns for excessive counts
                        source_scores[source_type] = info['quality_score'] * min(1.0, count / 3)
                        source_counts[source_type] = count
                        
                except Exception as source_error:
                    self.logger.warning(f"Error assessing source type {source_type}: {str(source_error)}", 
                                      extra={'session_id': session_id})
                    continue

            # Calculate weighted average score
            if source_scores:
                total_weight = sum(source_counts.values())
                weighted_score = sum(
                    score * source_counts[source_type]
                    for source_type, score in source_scores.items()
                ) / total_weight
            else:
                weighted_score = 5.0  # Default neutral score

            return {
                'source_scores': source_scores,
                'source_counts': source_counts,
                'weighted_average_score': round(weighted_score, 2),
                'total_indicators': total_indicators,
                'dominant_source_type': max(source_counts, key=source_counts.get) if source_counts else 'none'
            }
            
        except Exception as e:
            self.logger.error(f"Error in source quality assessment: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'source_scores': {},
                'source_counts': {},
                'weighted_average_score': 5.0,
                'total_indicators': 0,
                'dominant_source_type': 'error'
            }

    def _assess_evidence_strength(self, text_lower: str, session_id: str = None) -> Dict[str, Any]:
        """Assess evidence strength based on methodology and rigor indicators with error handling."""
        try:
            strength_counts = {}
            for strength_level, indicators in self.evidence_strength_indicators.items():
                count = sum(1 for indicator in indicators if indicator in text_lower)
                strength_counts[strength_level] = count

            # Calculate weighted strength score using configurable weights
            weighted_strength = sum(
                strength_counts.get(level, 0) * weight
                for level, weight in self.strength_weights.items()
            )

            # Normalize to 0-10 scale
            strength_score = max(0, min(10, 5 + (weighted_strength * 0.5)))

            return {
                'strength_counts': strength_counts,
                'weighted_strength_score': weighted_strength,
                'normalized_score': round(strength_score, 2),
                'total_indicators': sum(strength_counts.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error in evidence strength assessment: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'strength_counts': {},
                'weighted_strength_score': 0.0,
                'normalized_score': 5.0,
                'total_indicators': 0
            }

    def _assess_verification_level(self, text_lower: str, session_id: str = None) -> Dict[str, Any]:
        """Assess verification and fact-checking indicators with error handling."""
        try:
            verification_counts = {}
            for verification_level, indicators in self.verification_indicators.items():
                count = sum(1 for indicator in indicators if indicator in text_lower)
                verification_counts[verification_level] = count

            # Calculate verification score with fixed weights
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
            
        except Exception as e:
            self.logger.error(f"Error in verification assessment: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'verification_counts': {},
                'weighted_verification_score': 0.0,
                'normalized_score': 5.0,
                'total_indicators': 0
            }

    def _assess_transparency(self, text_lower: str, session_id: str = None) -> Dict[str, Any]:
        """Assess transparency and disclosure indicators with error handling."""
        try:
            transparency_counts = {}
            for transparency_level, indicators in self.transparency_indicators.items():
                count = sum(1 for indicator in indicators if indicator in text_lower)
                transparency_counts[transparency_level] = count

            # Calculate transparency score with fixed weights
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
            
        except Exception as e:
            self.logger.error(f"Error in transparency assessment: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'transparency_counts': {},
                'weighted_transparency_score': 0.0,
                'normalized_score': 5.0,
                'total_indicators': 0
            }

    def _assess_claims_quality(self, extracted_claims: List[Dict[str, Any]], session_id: str = None) -> Dict[str, Any]:
        """Assess quality based on extracted claims characteristics with error handling."""
        try:
            if not extracted_claims:
                return {
                    'total_claims': 0,
                    'average_verifiability': 0,
                    'high_verifiability_count': 0,
                    'claims_quality_score': 0,
                    'verifiability_distribution': {'high': 0, 'medium': 0, 'low': 0}
                }

            verifiability_scores = []
            for claim in extracted_claims:
                if isinstance(claim, dict):
                    score = claim.get('verifiability_score', 5)
                    # Validate score range
                    if isinstance(score, (int, float)) and 0 <= score <= 10:
                        verifiability_scores.append(score)
                    else:
                        verifiability_scores.append(5)  # Default score
                else:
                    verifiability_scores.append(5)  # Default for invalid claims

            average_verifiability = sum(verifiability_scores) / len(verifiability_scores)
            high_verifiability_count = sum(1 for score in verifiability_scores if score >= 7)

            # Calculate claims quality score
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
            
        except Exception as e:
            self.logger.error(f"Error in claims quality assessment: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'total_claims': len(extracted_claims) if extracted_claims else 0,
                'average_verifiability': 5.0,
                'high_verifiability_count': 0,
                'claims_quality_score': 5.0,
                'verifiability_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }

    def _calculate_overall_scores(self,
                                source_assessment: Dict[str, Any],
                                strength_assessment: Dict[str, Any],
                                verification_assessment: Dict[str, Any],
                                transparency_assessment: Dict[str, Any],
                                claims_assessment: Dict[str, Any],
                                session_id: str = None) -> Dict[str, Any]:
        """Calculate overall quality scores with configurable weights and error handling."""
        try:
            # Extract component scores with safe defaults
            source_score = source_assessment.get('weighted_average_score', 5.0)
            strength_score = strength_assessment.get('normalized_score', 5.0)
            verification_score = verification_assessment.get('normalized_score', 5.0)
            transparency_score = transparency_assessment.get('normalized_score', 5.0)
            claims_score = claims_assessment.get('claims_quality_score', 5.0)

            # Calculate overall score using configurable weights
            overall_score = (
                source_score * self.overall_weights['source_quality'] +
                strength_score * self.overall_weights['evidence_strength'] +
                verification_score * self.overall_weights['verification'] +
                transparency_score * self.overall_weights['transparency'] +
                claims_score * self.overall_weights['claims_quality']
            )

            # Ensure score is in valid range
            overall_score = max(0, min(10, overall_score))

            # Generate quality summary based on score
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
                'weights_used': self.overall_weights
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating overall scores: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'overall_score': 5.0,
                'source_score': 5.0,
                'completeness_score': 5.0,
                'quality_summary': "Quality assessment unavailable due to processing error",
                'component_scores': {
                    'source_quality': 5.0,
                    'evidence_strength': 5.0,
                    'verification': 5.0,
                    'transparency': 5.0,
                    'claims_quality': 5.0
                },
                'weights_used': self.overall_weights
            }

    def _create_fallback_assessment(self, session_id: str = None) -> Dict[str, Any]:
        """Create fallback assessment results when processing fails."""
        self.logger.warning(f"Using fallback results for evidence quality assessment", 
                          extra={'session_id': session_id})
        
        return {
            'source_quality_assessment': {
                'source_scores': {},
                'source_counts': {},
                'weighted_average_score': 5.0,
                'total_indicators': 0,
                'dominant_source_type': 'error'
            },
            'evidence_strength_assessment': {
                'strength_counts': {},
                'weighted_strength_score': 0.0,
                'normalized_score': 5.0,
                'total_indicators': 0
            },
            'verification_assessment': {
                'verification_counts': {},
                'weighted_verification_score': 0.0,
                'normalized_score': 5.0,
                'total_indicators': 0
            },
            'transparency_assessment': {
                'transparency_counts': {},
                'weighted_transparency_score': 0.0,
                'normalized_score': 5.0,
                'total_indicators': 0
            },
            'claims_quality_assessment': {
                'total_claims': 0,
                'average_verifiability': 5.0,
                'high_verifiability_count': 0,
                'claims_quality_score': 5.0,
                'verifiability_distribution': {'high': 0, 'medium': 0, 'low': 0}
            },
            'overall_quality_score': 5.0,
            'source_quality_score': 5.0,
            'completeness_score': 5.0,
            'quality_summary': 'Evidence quality assessment unavailable due to processing error',
            'quality_indicators_found': 0,
            'processing_time_ms': 0
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        avg_processing_time = (
            self.total_processing_time / self.assessment_count
            if self.assessment_count > 0 else 0
        )
        
        avg_indicators_per_assessment = (
            self.total_indicators_found / self.assessment_count
            if self.assessment_count > 0 else 0
        )
        
        return {
            'assessments_completed': self.assessment_count,
            'total_processing_time_seconds': round(self.total_processing_time, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'total_indicators_found': self.total_indicators_found,
            'average_indicators_per_assessment': round(avg_indicators_per_assessment, 2),
            'error_count': self.error_count,
            'error_rate': round(self.error_count / max(self.assessment_count, 1) * 100, 2),
            'criteria_categories': {
                'source_quality_types': len(self.source_quality_indicators),
                'evidence_strength_types': len(self.evidence_strength_indicators),
                'verification_levels': len(self.verification_indicators),
                'transparency_levels': len(self.transparency_indicators)
            },
            'configuration': {
                'strength_weights': self.strength_weights,
                'overall_weights': self.overall_weights,
                'min_article_length': self.min_article_length,
                'max_claims_for_processing': self.max_claims_for_processing
            }
        }

    def get_assessment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive assessment statistics."""
        total_indicators = sum(
            len(indicators['indicators']) + len(indicators['patterns'])
            for indicators in self.source_quality_indicators.values()
        ) + sum(
            len(indicators) for category in self.evidence_strength_indicators.values()
            for indicators in [category] if isinstance(category, list)
        )

        return {
            'total_quality_indicators': total_indicators,
            'source_quality_categories': len(self.source_quality_indicators),
            'evidence_strength_categories': len(self.evidence_strength_indicators),
            'verification_categories': len(self.verification_indicators),
            'transparency_categories': len(self.transparency_indicators),
            'config_customization_enabled': bool(self.config),
            'performance_tracking_enabled': True,
            'error_handling_enabled': True
        }


# Testing functionality
if __name__ == "__main__":
    """Test evidence quality criteria system with production configuration."""
    import logging
    
    # Setup production logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Test configuration
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
    
    print("=== EVIDENCE QUALITY CRITERIA TEST ===")
    
    try:
        criteria = EvidenceQualityCriteria(test_config)
        test_session_id = "criteria_test_101112"
        
        test_article = """
        According to a peer-reviewed study published in Nature by Dr. Sarah Johnson
        from Harvard University, the clinical trial with 2,400 participants showed
        statistically significant results with p-value < 0.001. The research was
        independently verified by multiple institutions and the methodology is
        fully disclosed with data available for review. The funding sources and
        potential conflicts of interest are clearly stated in the publication.
        """
        
        test_claims = [
            {'verifiability_score': 9, 'text': 'Peer-reviewed study in Nature'},
            {'verifiability_score': 8, 'text': 'Clinical trial with 2,400 participants'},
            {'verifiability_score': 7, 'text': 'Statistically significant with p < 0.001'}
        ]
        
        print(f"Analyzing article: {test_article[:100]}...")
        
        assessment = criteria.assess_evidence_quality(test_article, test_claims, test_session_id)
        
        print("✅ Evidence Quality Assessment Results:")
        print(f"  Overall Score: {assessment['overall_quality_score']:.1f}/10")
        print(f"  Source Quality: {assessment['source_quality_score']:.1f}/10")
        print(f"  Completeness Score: {assessment['completeness_score']:.1f}/10")
        print(f"  Quality Summary: {assessment['quality_summary']}")
        print(f"  Processing Time: {assessment['processing_time_ms']:.1f}ms")
        print(f"  Quality Indicators Found: {assessment['quality_indicators_found']}")
        
        # Test performance metrics
        print("\n📊 Performance Metrics:")
        metrics = criteria.get_performance_metrics()
        print(f"  Assessments completed: {metrics['assessments_completed']}")
        print(f"  Average processing time: {metrics['average_processing_time_ms']:.1f}ms")
        print(f"  Error rate: {metrics['error_rate']:.1f}%")
        print(f"  Average indicators per assessment: {metrics['average_indicators_per_assessment']:.1f}")
        
        # Test error handling
        print("\n🔧 Error Handling Test:")
        try:
            error_assessment = criteria.assess_evidence_quality(None, test_claims, test_session_id)
            print("❌ Error handling failed - should have raised exception")
        except EvidenceAssessmentError as e:
            print(f"✅ Error handling working: {e.message}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    
    print("\n✅ Evidence quality criteria tests completed")
