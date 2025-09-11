# agents/evidence_evaluator/fallacy_detection.py

"""
Logical Fallacy Detection - Production Ready

Enhanced rule-based fallacy detection system with structured logging,
error handling, and configurable parameters for production reliability.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional

from .exceptions import FallacyDetectionError


class LogicalFallacyDetector:
    """
    Production-ready logical fallacy detection system for news articles.
    
    Identifies common logical fallacies and assesses reasoning quality
    using rule-based pattern matching with enhanced error handling
    and structured logging for production environments.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize logical fallacy detector with production configuration.

        Args:
            config: Optional configuration for fallacy detection weights and thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.LogicalFallacyDetector")
        
        # Initialize fallacy detection systems
        try:
            self.logical_fallacies = self._initialize_logical_fallacies()
            self.reasoning_patterns = self._initialize_reasoning_patterns()
            self.argument_quality_indicators = self._initialize_argument_quality_indicators()
            
            self.logger.info(f"Fallacy detector initialized with {len(self.logical_fallacies)} fallacy types")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fallacy detection patterns: {str(e)}")
            raise FallacyDetectionError(
                f"Initialization failed: {str(e)}",
                detection_stage="initialization"
            )

        # Configurable weights and thresholds
        self.confidence_multiplier = self.config.get('confidence_multiplier', 0.15)
        self.fallacy_penalty_multiplier = self.config.get('fallacy_penalty_multiplier', 1.5)
        
        self.reasoning_weights = self.config.get('reasoning_weights', {
            'strong_reasoning': 2.0,
            'weak_reasoning': -1.5,
            'speculative_reasoning': -0.5,
            'emotional_reasoning': -1.0
        })
        
        self.health_thresholds = self.config.get('health_thresholds', {
            'excellent': 8.0,
            'good': 6.0, 
            'fair': 4.0,
            'poor': 2.0
        })

        # Performance metrics
        self.detection_count = 0
        self.total_processing_time = 0.0
        self.total_fallacies_detected = 0
        self.error_count = 0

    def _initialize_logical_fallacies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize logical fallacy detection patterns."""
        return {
            'ad_hominem': {
                'patterns': [
                    r'you\s+can\'t\s+trust\s+\w+\s+because\s+they',
                    r'\w+\s+is\s+known\s+for\s+being',
                    r'consider\s+the\s+source',
                    r'coming\s+from\s+someone\s+who'
                ],
                'indicators': [
                    'liar', 'dishonest', 'corrupt', 'biased', 'hypocrite',
                    'can\'t trust', 'consider the source', 'look who\'s talking'
                ],
                'severity': 'high',
                'description': 'Attacking the person making the argument instead of the argument itself'
            },
            'strawman': {
                'patterns': [
                    r'they\s+want\s+to\s+\w+\s+all',
                    r'what\s+they\s+really\s+mean\s+is',
                    r'their\s+real\s+agenda\s+is',
                    r'they\'re\s+trying\s+to\s+\w+\s+our'
                ],
                'indicators': [
                    'they want to ban', 'they want to eliminate', 'they want to destroy',
                    'what they really mean', 'their real agenda', 'they\'re trying to'
                ],
                'severity': 'high',
                'description': 'Misrepresenting someone\'s argument to make it easier to attack'
            },
            'false_dichotomy': {
                'patterns': [
                    r'either\s+\w+\s+or\s+\w+',
                    r'you\'re\s+either\s+\w+\s+or\s+\w+',
                    r'only\s+two\s+choices',
                    r'no\s+middle\s+ground'
                ],
                'indicators': [
                    'either or', 'only choice', 'two options', 'must choose',
                    'no alternative', 'black and white', 'all or nothing'
                ],
                'severity': 'medium',
                'description': 'Presenting only two choices when more options exist'
            },
            'hasty_generalization': {
                'patterns': [
                    r'all\s+\w+\s+are\s+\w+',
                    r'every\s+\w+\s+does',
                    r'this\s+always\s+happens',
                    r'they\s+never\s+\w+'
                ],
                'indicators': [
                    'all', 'every', 'always', 'never', 'none',
                    'everyone', 'no one', 'everything', 'nothing'
                ],
                'severity': 'medium',
                'description': 'Drawing broad conclusions from limited examples'
            },
            'appeal_to_authority': {
                'patterns': [
                    r'experts\s+agree\s+that',
                    r'studies\s+show\s+that',
                    r'scientists\s+say',
                    r'authorities\s+confirm'
                ],
                'indicators': [
                    'experts say', 'studies show', 'research proves',
                    'authorities confirm', 'specialists agree'
                ],
                'severity': 'low',
                'description': 'Using authority claims without proper evidence'
            },
            'slippery_slope': {
                'patterns': [
                    r'if\s+we\s+allow\s+\w+,\s+then',
                    r'this\s+will\s+lead\s+to',
                    r'next\s+thing\s+you\s+know',
                    r'where\s+does\s+it\s+end'
                ],
                'indicators': [
                    'will lead to', 'slippery slope', 'next thing',
                    'where does it end', 'domino effect'
                ],
                'severity': 'medium',
                'description': 'Claiming that one event will lead to extreme consequences'
            },
            'bandwagon': {
                'patterns': [
                    r'everyone\s+is\s+\w+ing',
                    r'most\s+people\s+agree',
                    r'popular\s+opinion\s+is',
                    r'the\s+majority\s+believes'
                ],
                'indicators': [
                    'everyone is doing', 'most people agree', 'popular opinion',
                    'majority believes', 'everyone knows', 'common sense'
                ],
                'severity': 'low',
                'description': 'Arguing that something is right because many people believe it'
            },
            'post_hoc': {
                'patterns': [
                    r'after\s+\w+,\s+therefore',
                    r'since\s+\w+\s+happened,\s+\w+',
                    r'because\s+\w+\s+came\s+after',
                    r'following\s+\w+,\s+we\s+saw'
                ],
                'indicators': [
                    'after therefore', 'since happened', 'because came after',
                    'correlation proves', 'must have caused'
                ],
                'severity': 'medium',
                'description': 'Assuming that because B followed A, A caused B'
            },
            'red_herring': {
                'patterns': [
                    r'but\s+what\s+about',
                    r'the\s+real\s+issue\s+is',
                    r'instead\s+of\s+talking\s+about',
                    r'let\'s\s+focus\s+on'
                ],
                'indicators': [
                    'but what about', 'real issue is', 'instead of',
                    'let\'s focus on', 'more important'
                ],
                'severity': 'medium',
                'description': 'Diverting attention from the main issue'
            },
            'circular_reasoning': {
                'patterns': [
                    r'because\s+it\'s\s+true',
                    r'obviously\s+correct',
                    r'it\'s\s+true\s+because',
                    r'proven\s+by\s+the\s+fact'
                ],
                'indicators': [
                    'because it\'s true', 'obviously correct', 'self-evident',
                    'proven by the fact', 'speaks for itself'
                ],
                'severity': 'high',
                'description': 'Using the conclusion as evidence for the premise'
            }
        }

    def _initialize_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Initialize reasoning quality assessment patterns."""
        return {
            'strong_reasoning': [
                'evidence shows', 'data indicates', 'research demonstrates',
                'studies confirm', 'analysis reveals', 'findings suggest',
                'empirical evidence', 'peer-reviewed', 'controlled study',
                'statistical significance', 'correlation', 'causation established'
            ],
            'weak_reasoning': [
                'obviously', 'clearly', 'everyone knows', 'common sense',
                'without a doubt', 'it\'s obvious', 'stands to reason',
                'goes without saying', 'any fool can see'
            ],
            'speculative_reasoning': [
                'might', 'could', 'possibly', 'perhaps', 'maybe',
                'seems like', 'appears to', 'suggests that',
                'one could argue', 'it\'s possible'
            ],
            'emotional_reasoning': [
                'outrageous', 'shocking', 'disgusting', 'terrible',
                'unbelievable', 'ridiculous', 'absurd', 'insane'
            ]
        }

    def _initialize_argument_quality_indicators(self) -> Dict[str, List[str]]:
        """Initialize argument quality assessment indicators."""
        return {
            'good_argument_indicators': [
                'evidence suggests', 'data supports', 'research indicates',
                'multiple sources', 'peer-reviewed', 'systematic analysis',
                'controlled for', 'confidence interval', 'limitations include',
                'however', 'on the other hand', 'alternatively'
            ],
            'poor_argument_indicators': [
                'proves beyond doubt', 'absolutely certain', 'no question',
                'anyone can see', 'it\'s obvious', 'common sense',
                'trust me', 'believe me', 'take my word'
            ],
            'nuanced_thinking': [
                'however', 'although', 'while', 'despite', 'nevertheless',
                'on one hand', 'complexity', 'nuanced', 'multifaceted'
            ],
            'absolute_thinking': [
                'always', 'never', 'all', 'none', 'completely',
                'totally', 'absolutely', 'definitely', 'certainly'
            ]
        }

    def detect_fallacies(self, text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Detect logical fallacies and assess reasoning quality with comprehensive error handling.

        Args:
            text: Text content to analyze for logical fallacies
            session_id: Optional session ID for tracking

        Returns:
            Dictionary containing fallacy detection results and reasoning assessment
        """
        start_time = time.time()
        
        # Input validation with structured logging
        if not text or not isinstance(text, str):
            self.error_count += 1
            self.logger.error(f"Invalid input for fallacy detection: {type(text).__name__}", 
                            extra={'session_id': session_id})
            raise FallacyDetectionError(
                f"Text input must be non-empty string, got {type(text).__name__}",
                detection_stage="input_validation",
                session_id=session_id
            )

        self.logger.info(f"Starting fallacy detection", 
                        extra={
                            'session_id': session_id,
                            'text_length': len(text),
                            'fallacy_types': len(self.logical_fallacies)
                        })

        try:
            text_lower = text.lower()
            
            # Step 1: Detect specific logical fallacies
            detected_fallacies = self._detect_logical_fallacies(text_lower, session_id)
            
            # Step 2: Assess reasoning quality
            reasoning_assessment = self._assess_reasoning_quality(text_lower, session_id)
            
            # Step 3: Assess argument quality
            argument_assessment = self._assess_argument_quality(text_lower, session_id)
            
            # Step 4: Calculate overall logical health score
            logical_health_score = self._calculate_logical_health_score(
                detected_fallacies, reasoning_assessment, argument_assessment, session_id
            )
            
            # Step 5: Create fallacy summary
            fallacy_summary = self._create_fallacy_summary(detected_fallacies, logical_health_score)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.detection_count += 1
            self.total_processing_time += processing_time
            self.total_fallacies_detected += len(detected_fallacies)
            
            self.logger.info(f"Fallacy detection completed successfully", 
                           extra={
                               'session_id': session_id,
                               'processing_time': round(processing_time * 1000, 2),
                               'fallacies_detected': len(detected_fallacies),
                               'logical_health_score': logical_health_score.get('logical_health_score', 0),
                               'high_severity_fallacies': len([f for f in detected_fallacies if f.get('severity') == 'high'])
                           })
            
            return {
                'detected_fallacies': detected_fallacies,
                'reasoning_assessment': reasoning_assessment,
                'argument_assessment': argument_assessment,
                'logical_health_score': logical_health_score,
                'fallacy_summary': fallacy_summary,
                'total_fallacies_detected': len(detected_fallacies),
                'processing_time_ms': round(processing_time * 1000, 2)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.error_count += 1
            
            self.logger.error(f"Fallacy detection failed: {str(e)}", 
                            extra={
                                'session_id': session_id,
                                'processing_time': round(processing_time * 1000, 2),
                                'error_type': type(e).__name__
                            })
            
            # Return fallback results instead of crashing
            return self._create_fallback_results(session_id)

    def _detect_logical_fallacies(self, text_lower: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Detect specific logical fallacies in text with error handling."""
        detected = []
        
        try:
            for fallacy_name, fallacy_info in self.logical_fallacies.items():
                try:
                    matches = []
                    confidence_score = 0
                    
                    # Check pattern matches with error handling
                    for pattern in fallacy_info['patterns']:
                        try:
                            pattern_matches = re.findall(pattern, text_lower, re.IGNORECASE)
                            if pattern_matches:
                                matches.extend([f"Pattern: {match}" for match in pattern_matches[:2]])
                                confidence_score += len(pattern_matches) * 2
                        except re.error as regex_error:
                            self.logger.warning(f"Regex error in fallacy {fallacy_name} pattern: {str(regex_error)}", 
                                              extra={'session_id': session_id})
                            continue
                    
                    # Check indicator matches
                    for indicator in fallacy_info['indicators']:
                        if indicator in text_lower:
                            matches.append(f"Indicator: {indicator}")
                            confidence_score += 1
                    
                    if matches:
                        # Calculate confidence with configurable multiplier
                        confidence = min(1.0, confidence_score * self.confidence_multiplier)
                        
                        detected.append({
                            'fallacy_type': fallacy_name,
                            'confidence': round(confidence, 2),
                            'severity': fallacy_info['severity'],
                            'description': fallacy_info['description'],
                            'evidence': matches[:3],  # Limit evidence items
                            'match_count': len(matches)
                        })
                        
                except Exception as fallacy_error:
                    self.logger.warning(f"Error detecting fallacy {fallacy_name}: {str(fallacy_error)}", 
                                      extra={'session_id': session_id})
                    continue
            
            # Sort by confidence
            detected.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.debug(f"Detected {len(detected)} fallacies", 
                            extra={'session_id': session_id})
            
            return detected
            
        except Exception as e:
            self.logger.error(f"Critical error in fallacy detection: {str(e)}", 
                            extra={'session_id': session_id})
            return []

    def _assess_reasoning_quality(self, text_lower: str, session_id: str = None) -> Dict[str, Any]:
        """Assess overall quality of reasoning in the text with error handling."""
        try:
            reasoning_counts = {}
            for reasoning_type, indicators in self.reasoning_patterns.items():
                count = sum(1 for indicator in indicators if indicator in text_lower)
                reasoning_counts[reasoning_type] = count
            
            # Calculate weighted reasoning score using configurable weights
            weighted_score = sum(
                reasoning_counts.get(reasoning_type, 0) * weight
                for reasoning_type, weight in self.reasoning_weights.items()
            )
            
            # Normalize to 0-10 scale
            reasoning_score = max(0, min(10, 5 + (weighted_score * 0.5)))
            
            return {
                'reasoning_counts': reasoning_counts,
                'weighted_reasoning_score': weighted_score,
                'normalized_score': round(reasoning_score, 2),
                'total_reasoning_indicators': sum(reasoning_counts.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error in reasoning quality assessment: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'reasoning_counts': {},
                'weighted_reasoning_score': 0.0,
                'normalized_score': 5.0,
                'total_reasoning_indicators': 0
            }

    def _assess_argument_quality(self, text_lower: str, session_id: str = None) -> Dict[str, Any]:
        """Assess argument quality based on structure and nuance with error handling."""
        try:
            quality_counts = {}
            for quality_type, indicators in self.argument_quality_indicators.items():
                count = sum(1 for indicator in indicators if indicator in text_lower)
                quality_counts[quality_type] = count
            
            # Calculate argument quality score with fixed weights
            quality_weights = {
                'good_argument_indicators': 1.5,
                'poor_argument_indicators': -1.0,
                'nuanced_thinking': 1.0,
                'absolute_thinking': -0.5
            }
            
            weighted_quality = sum(
                quality_counts.get(quality_type, 0) * weight
                for quality_type, weight in quality_weights.items()
            )
            
            quality_score = max(0, min(10, 5 + (weighted_quality * 0.6)))
            
            return {
                'quality_counts': quality_counts,
                'weighted_quality_score': weighted_quality,
                'normalized_score': round(quality_score, 2),
                'total_quality_indicators': sum(quality_counts.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error in argument quality assessment: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'quality_counts': {},
                'weighted_quality_score': 0.0,
                'normalized_score': 5.0,
                'total_quality_indicators': 0
            }

    def _calculate_logical_health_score(self,
                                      detected_fallacies: List[Dict[str, Any]],
                                      reasoning_assessment: Dict[str, Any],
                                      argument_assessment: Dict[str, Any],
                                      session_id: str = None) -> Dict[str, Any]:
        """Calculate overall logical health score with error handling."""
        try:
            # Calculate fallacy penalty with configurable weights
            fallacy_penalty = 0
            severity_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
            
            for fallacy in detected_fallacies:
                penalty = fallacy.get('confidence', 0) * severity_multiplier.get(fallacy.get('severity', 'medium'), 1.0)
                fallacy_penalty += penalty
            
            # Get component scores
            reasoning_score = reasoning_assessment.get('normalized_score', 5.0)
            argument_score = argument_assessment.get('normalized_score', 5.0)
            
            # Calculate base score
            base_score = (reasoning_score + argument_score) / 2
            
            # Apply fallacy penalty with configurable multiplier
            final_score = max(0, base_score - (fallacy_penalty * self.fallacy_penalty_multiplier))
            
            # Determine health level using configurable thresholds
            if final_score >= self.health_thresholds['excellent']:
                health_level = "EXCELLENT"
            elif final_score >= self.health_thresholds['good']:
                health_level = "GOOD"
            elif final_score >= self.health_thresholds['fair']:
                health_level = "FAIR"
            elif final_score >= self.health_thresholds['poor']:
                health_level = "POOR"
            else:
                health_level = "VERY POOR"
            
            return {
                'logical_health_score': round(final_score, 2),
                'health_level': health_level,
                'reasoning_component': round(reasoning_score, 2),
                'argument_component': round(argument_score, 2),
                'fallacy_penalty': round(fallacy_penalty, 2),
                'fallacy_count': len(detected_fallacies),
                'high_severity_fallacies': len([f for f in detected_fallacies if f.get('severity') == 'high'])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating logical health score: {str(e)}", 
                            extra={'session_id': session_id})
            return {
                'logical_health_score': 5.0,
                'health_level': "UNKNOWN",
                'reasoning_component': 5.0,
                'argument_component': 5.0,
                'fallacy_penalty': 0.0,
                'fallacy_count': len(detected_fallacies),
                'high_severity_fallacies': 0
            }

    def _create_fallacy_summary(self,
                              detected_fallacies: List[Dict[str, Any]],
                              logical_health_score: Dict[str, Any]) -> str:
        """Create formatted fallacy summary with error handling."""
        try:
            if not detected_fallacies:
                return (f"No logical fallacies detected. "
                       f"Logical health: {logical_health_score.get('health_level', 'UNKNOWN')} "
                       f"({logical_health_score.get('logical_health_score', 0):.1f}/10)")
            
            summary_lines = [
                "LOGICAL FALLACY ANALYSIS",
                f"Total Fallacies: {len(detected_fallacies)}",
                f"Logical Health: {logical_health_score.get('health_level', 'UNKNOWN')} "
                f"({logical_health_score.get('logical_health_score', 0):.1f}/10)",
                ""
            ]
            
            # Add fallacy details (limit to top 5)
            severity_indicators = {
                'high': "High Risk",
                'medium': "Medium Risk", 
                'low': "Low Risk"
            }
            
            for i, fallacy in enumerate(detected_fallacies[:5], 1):
                severity_indicator = severity_indicators.get(fallacy.get('severity', 'medium'), "Unknown Risk")
                
                summary_lines.extend([
                    f"Fallacy {i}: {fallacy.get('fallacy_type', 'Unknown').replace('_', ' ').title()}",
                    f"  Severity: {severity_indicator}",
                    f"  Confidence: {fallacy.get('confidence', 0):.2f}",
                    f"  Description: {fallacy.get('description', 'No description available')}",
                    ""
                ])
            
            if len(detected_fallacies) > 5:
                summary_lines.append(f"... and {len(detected_fallacies) - 5} more fallacies detected")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Error creating fallacy summary: {str(e)}")
            return f"Fallacy summary unavailable due to processing error. Total fallacies: {len(detected_fallacies)}"

    def _create_fallback_results(self, session_id: str = None) -> Dict[str, Any]:
        """Create fallback results when detection fails."""
        self.logger.warning(f"Using fallback results for fallacy detection", 
                          extra={'session_id': session_id})
        
        return {
            'detected_fallacies': [],
            'reasoning_assessment': {
                'reasoning_counts': {},
                'weighted_reasoning_score': 0.0,
                'normalized_score': 5.0,
                'total_reasoning_indicators': 0
            },
            'argument_assessment': {
                'quality_counts': {},
                'weighted_quality_score': 0.0,
                'normalized_score': 5.0,
                'total_quality_indicators': 0
            },
            'logical_health_score': {
                'logical_health_score': 5.0,
                'health_level': "UNKNOWN",
                'reasoning_component': 5.0,
                'argument_component': 5.0,
                'fallacy_penalty': 0.0,
                'fallacy_count': 0,
                'high_severity_fallacies': 0
            },
            'fallacy_summary': "Fallacy detection unavailable due to processing error",
            'total_fallacies_detected': 0,
            'processing_time_ms': 0
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        avg_processing_time = (
            self.total_processing_time / self.detection_count
            if self.detection_count > 0 else 0
        )
        
        avg_fallacies_per_detection = (
            self.total_fallacies_detected / self.detection_count
            if self.detection_count > 0 else 0
        )
        
        return {
            'detections_completed': self.detection_count,
            'total_processing_time_seconds': round(self.total_processing_time, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'total_fallacies_detected': self.total_fallacies_detected,
            'average_fallacies_per_detection': round(avg_fallacies_per_detection, 2),
            'error_count': self.error_count,
            'error_rate': round(self.error_count / max(self.detection_count, 1) * 100, 2),
            'fallacy_categories': {
                'total_fallacy_types': len(self.logical_fallacies),
                'reasoning_pattern_types': len(self.reasoning_patterns),
                'argument_quality_categories': len(self.argument_quality_indicators)
            },
            'configuration': {
                'confidence_multiplier': self.confidence_multiplier,
                'fallacy_penalty_multiplier': self.fallacy_penalty_multiplier,
                'health_thresholds': self.health_thresholds
            }
        }

    def get_detector_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detector statistics."""
        total_detection_patterns = sum(
            len(fallacy_info['patterns']) + len(fallacy_info['indicators'])
            for fallacy_info in self.logical_fallacies.values()
        )
        
        return {
            'fallacy_types_count': len(self.logical_fallacies),
            'total_detection_patterns': total_detection_patterns,
            'reasoning_pattern_count': sum(len(patterns) for patterns in self.reasoning_patterns.values()),
            'argument_quality_indicators_count': sum(len(indicators) for indicators in self.argument_quality_indicators.values()),
            'config_customization_enabled': bool(self.config),
            'performance_tracking_enabled': True,
            'error_handling_enabled': True
        }


# Testing functionality
if __name__ == "__main__":
    """Test logical fallacy detector with production configuration."""
    import logging
    
    # Setup production logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Test configuration
    test_config = {
        'confidence_multiplier': 0.2,
        'reasoning_weights': {
            'strong_reasoning': 2.5,
            'weak_reasoning': -2.0,
            'speculative_reasoning': -0.3,
            'emotional_reasoning': -1.5
        },
        'fallacy_penalty_multiplier': 2.0,
        'health_thresholds': {
            'excellent': 8.5,
            'good': 6.5, 
            'fair': 4.5,
            'poor': 2.5
        }
    }
    
    print("=== LOGICAL FALLACY DETECTOR TEST ===")
    
    try:
        detector = LogicalFallacyDetector(test_config)
        test_session_id = "fallacy_test_789"
        
        test_text = """
        You can't trust Dr. Smith because he's obviously biased and corrupt.
        Everyone knows that all experts are just trying to control us these days.
        Either you believe the science or you're against progress - there's no middle ground here.
        After the new policy was implemented, crime rates went up, so clearly the policy caused more crime.
        This is just common sense that any fool can see.
        """
        
        print(f"Analyzing text: {test_text[:100]}...")
        
        results = detector.detect_fallacies(test_text, test_session_id)
        
        print("✅ Fallacy Detection Results:")
        print(f"  Total fallacies detected: {results['total_fallacies_detected']}")
        print(f"  Logical health score: {results['logical_health_score']['logical_health_score']:.1f}/10")
        print(f"  Health level: {results['logical_health_score']['health_level']}")
        print(f"  High severity fallacies: {results['logical_health_score']['high_severity_fallacies']}")
        print(f"  Processing time: {results['processing_time_ms']:.1f}ms")
        
        print("\nDetected fallacies:")
        for fallacy in results['detected_fallacies'][:3]:  # Show top 3
            print(f"  • {fallacy['fallacy_type']}: {fallacy['confidence']:.2f} confidence "
                  f"({fallacy['severity']} severity)")
        
        # Test performance metrics
        print("\n📊 Performance Metrics:")
        metrics = detector.get_performance_metrics()
        print(f"  Detections completed: {metrics['detections_completed']}")
        print(f"  Average processing time: {metrics['average_processing_time_ms']:.1f}ms")
        print(f"  Error rate: {metrics['error_rate']:.1f}%")
        
        # Test error handling
        print("\n🔧 Error Handling Test:")
        try:
            error_results = detector.detect_fallacies(None, test_session_id)
            print("❌ Error handling failed - should have raised exception")
        except FallacyDetectionError as e:
            print(f"✅ Error handling working: {e.message}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    
    print("\n✅ Fallacy detector tests completed")
