# agents/evidence_evaluator/fallacy_detection.py

"""
Logical Fallacy Detection

Detects logical fallacies and reasoning errors in news articles to assess
argument quality and logical consistency. Provides systematic fallacy
identification and reasoning quality evaluation.
"""

import re
import time
import logging
from typing import Dict, List, Any


class LogicalFallacyDetector:
    """
    Logical fallacy detection system for news articles.
    
    Identifies common logical fallacies and assesses reasoning quality
    to evaluate the logical consistency of arguments in news content.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize logical fallacy detector.
        
        Args:
            config: Optional configuration for fallacy detection weights and thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize fallacy detection systems
        self.logical_fallacies = self._initialize_logical_fallacies()
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        self.argument_quality_indicators = self._initialize_argument_quality_indicators()
        
        # Performance metrics
        self.detection_count = 0
        self.total_processing_time = 0.0

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

    def detect_fallacies(self, text: str) -> Dict[str, Any]:
        """
        Detect logical fallacies and assess reasoning quality.
        
        Args:
            text: Text content to analyze for logical fallacies
            
        Returns:
            Dictionary containing fallacy detection results and reasoning assessment
        """
        start_time = time.time()
        text_lower = text.lower()
        
        # Detect specific logical fallacies
        detected_fallacies = self._detect_logical_fallacies(text_lower)
        
        # Assess reasoning quality
        reasoning_assessment = self._assess_reasoning_quality(text_lower)
        
        # Assess argument quality
        argument_assessment = self._assess_argument_quality(text_lower)
        
        # Calculate overall logical health score
        logical_health_score = self._calculate_logical_health_score(
            detected_fallacies, reasoning_assessment, argument_assessment
        )
        
        # Create fallacy summary
        fallacy_summary = self._create_fallacy_summary(detected_fallacies, logical_health_score)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.detection_count += 1
        self.total_processing_time += processing_time
        
        return {
            'detected_fallacies': detected_fallacies,
            'reasoning_assessment': reasoning_assessment,
            'argument_assessment': argument_assessment,
            'logical_health_score': logical_health_score,
            'fallacy_summary': fallacy_summary,
            'total_fallacies_detected': len(detected_fallacies),
            'processing_time_ms': round(processing_time * 1000, 2)
        }

    def _detect_logical_fallacies(self, text_lower: str) -> List[Dict[str, Any]]:
        """Detect specific logical fallacies in text."""
        detected = []
        
        for fallacy_name, fallacy_info in self.logical_fallacies.items():
            matches = []
            confidence_score = 0
            
            # Check pattern matches
            for pattern in fallacy_info['patterns']:
                pattern_matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if pattern_matches:
                    matches.extend([f"Pattern: {match}" for match in pattern_matches[:2]])
                    confidence_score += len(pattern_matches) * 2
            
            # Check indicator matches
            for indicator in fallacy_info['indicators']:
                if indicator in text_lower:
                    matches.append(f"Indicator: {indicator}")
                    confidence_score += 1
            
            if matches:
                # Calculate confidence with config multiplier
                confidence_multiplier = self.config.get('confidence_multiplier', 0.15)
                confidence = min(1.0, confidence_score * confidence_multiplier)
                
                detected.append({
                    'fallacy_type': fallacy_name,
                    'confidence': round(confidence, 2),
                    'severity': fallacy_info['severity'],
                    'description': fallacy_info['description'],
                    'evidence': matches[:3],  # Limit evidence items
                    'match_count': len(matches)
                })
        
        # Sort by confidence
        detected.sort(key=lambda x: x['confidence'], reverse=True)
        return detected

    def _assess_reasoning_quality(self, text_lower: str) -> Dict[str, Any]:
        """Assess overall quality of reasoning in the text."""
        reasoning_counts = {}
        
        for reasoning_type, indicators in self.reasoning_patterns.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            reasoning_counts[reasoning_type] = count
        
        # Get reasoning weights from config or use defaults
        reasoning_weights = self.config.get('reasoning_weights', {
            'strong_reasoning': 2.0,
            'weak_reasoning': -1.5,
            'speculative_reasoning': -0.5,
            'emotional_reasoning': -1.0
        })
        
        # Calculate weighted reasoning score
        weighted_score = sum(
            reasoning_counts.get(reasoning_type, 0) * weight
            for reasoning_type, weight in reasoning_weights.items()
        )
        
        # Normalize to 0-10 scale
        reasoning_score = max(0, min(10, 5 + (weighted_score * 0.5)))
        
        return {
            'reasoning_counts': reasoning_counts,
            'weighted_reasoning_score': weighted_score,
            'normalized_score': round(reasoning_score, 2),
            'total_reasoning_indicators': sum(reasoning_counts.values())
        }

    def _assess_argument_quality(self, text_lower: str) -> Dict[str, Any]:
        """Assess argument quality based on structure and nuance."""
        quality_counts = {}
        
        for quality_type, indicators in self.argument_quality_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            quality_counts[quality_type] = count
        
        # Calculate argument quality score
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

    def _calculate_logical_health_score(self, 
                                      detected_fallacies: List[Dict[str, Any]],
                                      reasoning_assessment: Dict[str, Any],
                                      argument_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall logical health score."""
        # Calculate fallacy penalty
        fallacy_penalty = 0
        severity_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
        
        for fallacy in detected_fallacies:
            penalty = fallacy['confidence'] * severity_multiplier.get(fallacy['severity'], 1.0)
            fallacy_penalty += penalty
        
        # Get component scores
        reasoning_score = reasoning_assessment['normalized_score']
        argument_score = argument_assessment['normalized_score']
        
        # Calculate base score
        base_score = (reasoning_score + argument_score) / 2
        
        # Apply fallacy penalty
        penalty_multiplier = self.config.get('fallacy_penalty_multiplier', 1.5)
        final_score = max(0, base_score - (fallacy_penalty * penalty_multiplier))
        
        # Determine health level thresholds
        health_thresholds = self.config.get('health_thresholds', {
            'excellent': 8.0, 'good': 6.0, 'fair': 4.0, 'poor': 2.0
        })
        
        if final_score >= health_thresholds['excellent']:
            health_level = "EXCELLENT"
        elif final_score >= health_thresholds['good']:
            health_level = "GOOD"
        elif final_score >= health_thresholds['fair']:
            health_level = "FAIR"
        elif final_score >= health_thresholds['poor']:
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
            'high_severity_fallacies': len([f for f in detected_fallacies if f['severity'] == 'high'])
        }

    def _create_fallacy_summary(self, 
                              detected_fallacies: List[Dict[str, Any]], 
                              logical_health_score: Dict[str, Any]) -> str:
        """Create formatted fallacy summary."""
        if not detected_fallacies:
            return f"No logical fallacies detected. Logical health: {logical_health_score['health_level']} ({logical_health_score['logical_health_score']:.1f}/10)"
        
        summary_lines = [
            "LOGICAL FALLACY ANALYSIS",
            f"Total Fallacies: {len(detected_fallacies)}",
            f"Logical Health: {logical_health_score['health_level']} ({logical_health_score['logical_health_score']:.1f}/10)",
            ""
        ]
        
        # Add fallacy details (limit to top 5)
        for i, fallacy in enumerate(detected_fallacies[:5], 1):
            severity_indicator = {
                'high': "High Risk",
                'medium': "Medium Risk", 
                'low': "Low Risk"
            }.get(fallacy['severity'], "Unknown")
            
            summary_lines.extend([
                f"Fallacy {i}: {fallacy['fallacy_type'].replace('_', ' ').title()}",
                f"  Severity: {severity_indicator}",
                f"  Confidence: {fallacy['confidence']:.2f}",
                f"  Description: {fallacy['description']}",
                ""
            ])
        
        if len(detected_fallacies) > 5:
            summary_lines.append(f"... and {len(detected_fallacies) - 5} more fallacies detected")
        
        return "\n".join(summary_lines)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the fallacy detector."""
        avg_processing_time = (
            self.total_processing_time / self.detection_count 
            if self.detection_count > 0 else 0
        )
        
        return {
            'detections_completed': self.detection_count,
            'total_processing_time_seconds': round(self.total_processing_time, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'fallacy_categories': {
                'total_fallacy_types': len(self.logical_fallacies),
                'reasoning_pattern_types': len(self.reasoning_patterns),
                'argument_quality_categories': len(self.argument_quality_indicators)
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
            'config_customization_enabled': bool(self.config)
        }


# Testing functionality
if __name__ == "__main__":
    """Test logical fallacy detector."""
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
            'excellent': 8.5, 'good': 6.5, 'fair': 4.5, 'poor': 2.5
        }
    }
    
    detector = LogicalFallacyDetector(test_config)
    
    test_text = """
    You can't trust Dr. Smith because he's obviously biased. Everyone knows
    that all experts are corrupt these days. Either you believe the science
    or you're against progress - there's no middle ground. After the new
    policy was implemented, crime rates went up, therefore the policy caused
    more crime. This is clearly common sense that any fool can see.
    """
    
    results = detector.detect_fallacies(test_text)
    
    print("Fallacy Detection Results:")
    print(f"Total fallacies detected: {results['total_fallacies_detected']}")
    print(f"Logical health score: {results['logical_health_score']['logical_health_score']:.1f}/10")
    print(f"Health level: {results['logical_health_score']['health_level']}")
    print(f"High severity fallacies: {results['logical_health_score']['high_severity_fallacies']}")
    print(f"Processing time: {results['processing_time_ms']:.1f}ms")
    
    print("\nDetected fallacies:")
    for fallacy in results['detected_fallacies']:
        print(f"  - {fallacy['fallacy_type']}: {fallacy['confidence']:.2f} confidence ({fallacy['severity']} severity)")
