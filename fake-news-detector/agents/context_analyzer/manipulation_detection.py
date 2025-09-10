# agents/context_analyzer/manipulation_detection.py

"""
Manipulation Detection Module

Detects manipulation techniques, propaganda methods, and logical fallacies
in news articles to assess argument quality and identify persuasion tactics.
Provides systematic detection of common manipulation strategies.
"""

import re
import time
import logging
from typing import Dict, List, Any, Tuple


class ManipulationDetector:
    """
    Manipulation and propaganda detection system for news articles.
    
    Identifies manipulation techniques, propaganda methods, and logical fallacies
    to assess the quality of arguments and detect potential misinformation tactics.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize manipulation detector.
        
        Args:
            config: Optional configuration for detection parameters and thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize detection systems
        self.propaganda_techniques = self._initialize_propaganda_techniques()
        self.manipulation_patterns = self._initialize_manipulation_patterns()
        self.logical_fallacies = self._initialize_logical_fallacies()
        
        # Performance metrics
        self.detection_count = 0
        self.total_processing_time = 0.0
        self.total_techniques_found = 0

    def _initialize_propaganda_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Initialize propaganda technique detection patterns."""
        return {
            'name_calling': {
                'indicators': ['extremist', 'radical', 'terrorist', 'criminal', 'corrupt'],
                'patterns': [r'so-called\s+\w+', r'\w+\s+is\s+nothing\s+but'],
                'severity': 'medium',
                'description': 'Attaching negative labels to discredit opponents'
            },
            'glittering_generalities': {
                'indicators': ['freedom', 'democracy', 'justice', 'truth', 'patriot'],
                'patterns': [r'fight\s+for\s+\w+', r'defend\s+our\s+\w+'],
                'severity': 'low',
                'description': 'Using emotionally appealing but vague words'
            },
            'transfer': {
                'indicators': ['flag', 'cross', 'symbol', 'heritage', 'tradition'],
                'patterns': [r'our\s+\w+\s+values', r'american\s+\w+'],
                'severity': 'medium',
                'description': 'Connecting ideas with positive or negative symbols'
            },
            'testimonial': {
                'indicators': ['celebrity', 'expert', 'endorses', 'recommends', 'supports'],
                'patterns': [r'\w+\s+endorses', r'according\s+to\s+\w+\s+expert'],
                'severity': 'low',
                'description': 'Using celebrity or authority endorsements inappropriately'
            },
            'plain_folks': {
                'indicators': ['ordinary', 'common', 'working class', 'regular people', 'folks'],
                'patterns': [r'just\s+like\s+you', r'ordinary\s+\w+'],
                'severity': 'low',
                'description': 'Appealing to common people values to gain trust'
            },
            'card_stacking': {
                'indicators': ['only', 'never mentions', 'ignores', 'hides', 'conceals'],
                'patterns': [r'they\s+don\'t\s+want\s+you\s+to\s+know', r'hidden\s+truth'],
                'severity': 'high',
                'description': 'Presenting only one side of an argument'
            },
            'bandwagon': {
                'indicators': ['everyone', 'everybody', 'all', 'join', 'movement'],
                'patterns': [r'everyone\s+is\s+\w+ing', r'join\s+the\s+\w+'],
                'severity': 'medium',
                'description': 'Encouraging following the crowd or majority'
            },
            'fear_mongering': {
                'indicators': ['danger', 'threat', 'crisis', 'emergency', 'catastrophe'],
                'patterns': [r'if\s+we\s+don\'t\s+act', r'time\s+is\s+running\s+out'],
                'severity': 'high',
                'description': 'Creating fear to motivate specific actions'
            },
            'false_dilemma': {
                'indicators': ['either', 'only choice', 'must choose', 'no alternative'],
                'patterns': [r'either\s+\w+\s+or\s+\w+', r'you\'re\s+either\s+\w+\s+or\s+\w+'],
                'severity': 'high',
                'description': 'Presenting only two options when more exist'
            },
            'ad_hominem': {
                'indicators': ['liar', 'dishonest', 'corrupt', 'incompetent', 'biased'],
                'patterns': [r'you\s+can\'t\s+trust\s+\w+\s+because', r'\w+\s+is\s+known\s+for'],
                'severity': 'high',
                'description': 'Attacking the person instead of addressing the argument'
            }
        }

    def _initialize_manipulation_patterns(self) -> Dict[str, List[str]]:
        """Initialize modern manipulation technique patterns."""
        return {
            'astroturfing': [
                'grassroots movement', 'citizen group', 'concerned citizens',
                'independent organization', 'community initiative'
            ],
            'gaslighting': [
                'you\'re imagining things', 'that never happened', 'you\'re overreacting',
                'you\'re being paranoid', 'that\'s not what I said'
            ],
            'whataboutism': [
                'what about', 'but what about', 'what about when', 'remember when',
                'you didn\'t complain when'
            ],
            'strawman': [
                'they want to ban', 'they want to eliminate', 'they want to destroy',
                'their real agenda is', 'what they really mean'
            ],
            'false_flag': [
                'false flag', 'staged', 'crisis actor', 'inside job',
                'government conspiracy'
            ],
            'cherry_picking': [
                'one study shows', 'some experts say', 'according to one report',
                'a recent poll found', 'data suggests'
            ]
        }

    def _initialize_logical_fallacies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize logical fallacy detection patterns."""
        return {
            'hasty_generalization': {
                'patterns': [r'all\s+\w+\s+are', r'this\s+always\s+happens', r'they\s+never'],
                'indicators': ['all', 'always', 'never', 'every', 'none'],
                'severity': 'medium'
            },
            'post_hoc': {
                'patterns': [r'after\s+\w+,\s+therefore', r'since\s+\w+\s+happened'],
                'indicators': ['therefore', 'thus', 'consequently', 'as a result'],
                'severity': 'medium'
            },
            'circular_reasoning': {
                'patterns': [r'because\s+it\'s\s+true', r'obviously\s+correct'],
                'indicators': ['obviously', 'clearly', 'self-evident', 'common sense'],
                'severity': 'high'
            },
            'appeal_to_authority': {
                'patterns': [r'expert\s+says', r'study\s+proves', r'scientist\s+confirms'],
                'indicators': ['expert', 'authority', 'study', 'research'],
                'severity': 'low'
            },
            'slippery_slope': {
                'patterns': [r'if\s+we\s+allow\s+\w+,\s+then', r'this\s+will\s+lead\s+to'],
                'indicators': ['will lead to', 'next thing', 'slippery slope'],
                'severity': 'medium'
            }
        }

    def get_manipulation_report(self, text: str) -> Dict[str, Any]:
        """
        Generate comprehensive manipulation detection report.
        
        Args:
            text: Article text to analyze for manipulation techniques
            
        Returns:
            Dictionary containing manipulation analysis results
        """
        start_time = time.time()
        
        # Detect different types of manipulation
        propaganda_results = self._detect_propaganda_techniques(text)
        manipulation_results = self._detect_manipulation_patterns(text)
        fallacy_results = self._detect_logical_fallacies(text)
        
        # Calculate overall manipulation metrics
        total_techniques = (
            len(propaganda_results['detected']) +
            len(manipulation_results['detected']) +
            len(fallacy_results['detected'])
        )
        
        # Calculate manipulation score
        manipulation_score = self._calculate_manipulation_score(
            propaganda_results, manipulation_results, fallacy_results
        )
        
        # Assess risk level
        risk_level = self._assess_risk_level(manipulation_score)
        
        # Count high severity techniques
        high_severity_count = self._count_high_severity_techniques(
            propaganda_results, fallacy_results
        )
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.detection_count += 1
        self.total_processing_time += processing_time
        self.total_techniques_found += total_techniques
        
        return {
            'propaganda_techniques': propaganda_results,
            'manipulation_patterns': manipulation_results,
            'logical_fallacies': fallacy_results,
            'overall_manipulation_score': manipulation_score,
            'risk_level': risk_level,
            'techniques_summary': {
                'total_techniques_detected': total_techniques,
                'high_severity_count': high_severity_count,
                'processing_time_ms': round(processing_time * 1000, 2)
            }
        }

    def _detect_propaganda_techniques(self, text: str) -> Dict[str, Any]:
        """Detect propaganda techniques with enhanced pattern matching."""
        text_lower = text.lower()
        detected = {}
        
        for technique, info in self.propaganda_techniques.items():
            score = 0
            matches = []
            
            # Check indicators
            for indicator in info['indicators']:
                if indicator in text_lower:
                    score += 1
                    matches.append(f"Indicator: {indicator}")
            
            # Check patterns
            for pattern in info['patterns']:
                pattern_matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if pattern_matches:
                    score += len(pattern_matches) * 2  # Patterns weighted higher
                    matches.extend([f"Pattern: {match}" for match in pattern_matches[:2]])
            
            if score > 0:
                # Calculate confidence with configurable multiplier
                confidence_multiplier = self.config.get('confidence_multiplier', 0.15)
                confidence = min(1.0, score * confidence_multiplier)
                
                detected[technique] = {
                    'score': score,
                    'confidence': round(confidence, 2),
                    'severity': info['severity'],
                    'matches': matches[:3],  # Limit matches
                    'description': info['description']
                }
        
        # Sort by confidence
        detected = dict(sorted(detected.items(), key=lambda x: x[1]['confidence'], reverse=True))
        
        return {
            'detected': detected,
            'total_techniques': len(detected),
            'high_confidence': [t for t, d in detected.items() if d['confidence'] > 0.6]
        }

    def _detect_manipulation_patterns(self, text: str) -> Dict[str, Any]:
        """Detect modern manipulation patterns."""
        text_lower = text.lower()
        detected = {}
        
        for pattern_type, indicators in self.manipulation_patterns.items():
            matches = []
            for indicator in indicators:
                if indicator in text_lower:
                    matches.append(indicator)
            
            if matches:
                confidence = min(1.0, len(matches) / 3)
                detected[pattern_type] = {
                    'matches': matches[:3],
                    'confidence': round(confidence, 2),
                    'count': len(matches)
                }
        
        return {
            'detected': detected,
            'total_patterns': len(detected)
        }

    def _detect_logical_fallacies(self, text: str) -> Dict[str, Any]:
        """Detect logical fallacies with enhanced analysis."""
        text_lower = text.lower()
        detected = {}
        
        for fallacy, info in self.logical_fallacies.items():
            score = 0
            evidence = []
            
            # Check patterns
            for pattern in info['patterns']:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    score += len(matches)
                    evidence.extend(matches[:2])
            
            # Check indicators
            for indicator in info['indicators']:
                if indicator in text_lower:
                    score += 0.5
                    if len(evidence) < 3:
                        evidence.append(f"Indicator: {indicator}")
            
            if score > 0:
                confidence = min(1.0, score / 3)
                detected[fallacy] = {
                    'confidence': round(confidence, 2),
                    'severity': info['severity'],
                    'evidence': evidence[:3]  # Limit evidence items
                }
        
        return {
            'detected': detected,
            'total_fallacies': len(detected)
        }

    def _calculate_manipulation_score(self, 
                                    propaganda_results: Dict[str, Any],
                                    manipulation_results: Dict[str, Any], 
                                    fallacy_results: Dict[str, Any]) -> float:
        """Calculate overall manipulation score with configurable weights."""
        # Get weights from config or use defaults
        weights = self.config.get('scoring_weights', {
            'propaganda': 0.4,
            'manipulation': 0.4,
            'fallacies': 0.2
        })
        
        # Calculate component scores
        propaganda_score = self._score_propaganda_techniques(propaganda_results)
        manipulation_score = self._score_manipulation_patterns(manipulation_results)
        fallacy_score = self._score_logical_fallacies(fallacy_results)
        
        # Weighted overall score
        overall_score = (
            propaganda_score * weights['propaganda'] +
            manipulation_score * weights['manipulation'] +
            fallacy_score * weights['fallacies']
        )
        
        return round(min(10.0, overall_score), 2)

    def _score_propaganda_techniques(self, results: Dict[str, Any]) -> float:
        """Score propaganda techniques based on severity and confidence."""
        if not results['detected']:
            return 0.0
        
        total_score = 0
        severity_multiplier = {'low': 1, 'medium': 2, 'high': 3}
        
        for technique, data in results['detected'].items():
            severity_weight = severity_multiplier.get(data['severity'], 1)
            total_score += data['confidence'] * severity_weight
        
        return min(10.0, total_score * 2)

    def _score_manipulation_patterns(self, results: Dict[str, Any]) -> float:
        """Score manipulation patterns based on frequency and confidence."""
        if not results['detected']:
            return 0.0
        
        total_score = sum(data['confidence'] for data in results['detected'].values())
        return min(10.0, total_score * 2.5)

    def _score_logical_fallacies(self, results: Dict[str, Any]) -> float:
        """Score logical fallacies based on severity and confidence."""
        if not results['detected']:
            return 0.0
        
        total_score = 0
        severity_multiplier = {'low': 1, 'medium': 1.5, 'high': 2}
        
        for fallacy, data in results['detected'].items():
            severity_weight = severity_multiplier.get(data['severity'], 1)
            total_score += data['confidence'] * severity_weight
        
        return min(10.0, total_score * 2)

    def _assess_risk_level(self, score: float) -> str:
        """Assess manipulation risk level from score."""
        # Get thresholds from config or use defaults
        thresholds = self.config.get('risk_thresholds', {
            'minimal': 2, 'low': 4, 'medium': 6, 'high': 8
        })
        
        if score <= thresholds['minimal']:
            return "MINIMAL"
        elif score <= thresholds['low']:
            return "LOW"
        elif score <= thresholds['medium']:
            return "MEDIUM"
        elif score <= thresholds['high']:
            return "HIGH"
        else:
            return "CRITICAL"

    def _count_high_severity_techniques(self, 
                                      propaganda_results: Dict[str, Any], 
                                      fallacy_results: Dict[str, Any]) -> int:
        """Count high severity manipulation techniques."""
        count = 0
        
        # Count high severity propaganda
        for technique, data in propaganda_results['detected'].items():
            if data['severity'] == 'high':
                count += 1
        
        # Count high severity fallacies
        for fallacy, data in fallacy_results['detected'].items():
            if data['severity'] == 'high':
                count += 1
        
        return count

    def analyze_propaganda_by_category(self, text: str) -> Dict[str, Any]:
        """
        Analyze propaganda techniques by category for detailed reporting.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with propaganda analysis by category
        """
        results = self._detect_propaganda_techniques(text)
        
        # Group by severity
        by_severity = {'low': [], 'medium': [], 'high': []}
        for technique, data in results['detected'].items():
            by_severity[data['severity']].append({
                'technique': technique,
                'confidence': data['confidence'],
                'description': data['description']
            })
        
        # Calculate category risk
        category_risk = {
            'emotional_appeals': 0,
            'logical_manipulation': 0,
            'social_pressure': 0
        }
        
        emotional_techniques = ['fear_mongering', 'glittering_generalities', 'transfer']
        logical_techniques = ['false_dilemma', 'card_stacking', 'ad_hominem']
        social_techniques = ['bandwagon', 'plain_folks', 'testimonial']
        
        for technique, data in results['detected'].items():
            confidence = data['confidence']
            if technique in emotional_techniques:
                category_risk['emotional_appeals'] += confidence
            elif technique in logical_techniques:
                category_risk['logical_manipulation'] += confidence
            elif technique in social_techniques:
                category_risk['social_pressure'] += confidence
        
        return {
            'by_severity': by_severity,
            'category_risk_scores': {k: round(v, 2) for k, v in category_risk.items()},
            'highest_risk_category': max(category_risk, key=category_risk.get),
            'total_techniques_detected': len(results['detected'])
        }

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        avg_processing_time = (
            self.total_processing_time / self.detection_count 
            if self.detection_count > 0 else 0
        )
        
        avg_techniques_per_detection = (
            self.total_techniques_found / self.detection_count
            if self.detection_count > 0 else 0
        )
        
        return {
            'total_detections_completed': self.detection_count,
            'total_processing_time_seconds': round(self.total_processing_time, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'total_techniques_found': self.total_techniques_found,
            'average_techniques_per_detection': round(avg_techniques_per_detection, 2),
            'detection_database_sizes': {
                'propaganda_techniques_count': len(self.propaganda_techniques),
                'manipulation_patterns_count': len(self.manipulation_patterns),
                'logical_fallacies_count': len(self.logical_fallacies),
                'total_detection_methods': (
                    len(self.propaganda_techniques) +
                    len(self.manipulation_patterns) +
                    len(self.logical_fallacies)
                )
            }
        }

    def validate_detection_patterns(self) -> Dict[str, Any]:
        """Validate detection pattern database integrity."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check propaganda techniques
        for technique, data in self.propaganda_techniques.items():
            if not data.get('indicators') and not data.get('patterns'):
                validation_results['issues'].append(f"Propaganda technique '{technique}' has no indicators or patterns")
                validation_results['valid'] = False
            
            if not data.get('severity') or data['severity'] not in ['low', 'medium', 'high']:
                validation_results['issues'].append(f"Propaganda technique '{technique}' has invalid severity")
                validation_results['valid'] = False
        
        # Check manipulation patterns
        for pattern_type, indicators in self.manipulation_patterns.items():
            if not indicators:
                validation_results['issues'].append(f"Manipulation pattern '{pattern_type}' has no indicators")
                validation_results['valid'] = False
        
        # Check logical fallacies
        for fallacy, data in self.logical_fallacies.items():
            if not data.get('indicators') and not data.get('patterns'):
                validation_results['issues'].append(f"Logical fallacy '{fallacy}' has no indicators or patterns")
                validation_results['valid'] = False
        
        return validation_results


# Testing functionality
if __name__ == "__main__":
    """Test manipulation detector functionality."""
    
    # Initialize detector with test configuration
    test_config = {
        'confidence_multiplier': 0.2,
        'scoring_weights': {
            'propaganda': 0.5,
            'manipulation': 0.3,
            'fallacies': 0.2
        },
        'risk_thresholds': {
            'minimal': 1, 'low': 3, 'medium': 5, 'high': 7
        }
    }
    
    detector = ManipulationDetector(test_config)
    
    # Test analysis
    test_text = """
    Don't let the corrupt establishment fool you! This is clearly a false flag
    operation designed to distract patriots from the real truth. Either you're
    with us or you're against America. What about all the lies they told before?
    Every real American knows this is obviously a conspiracy by extremists.
    """
    
    print("=== MANIPULATION DETECTION TEST ===")
    report = detector.get_manipulation_report(test_text)
    
    print(f"Overall manipulation score: {report['overall_manipulation_score']}/10")
    print(f"Risk level: {report['risk_level']}")
    print(f"Total techniques detected: {report['techniques_summary']['total_techniques_detected']}")
    print(f"High severity techniques: {report['techniques_summary']['high_severity_count']}")
    
    # Test propaganda category analysis
    propaganda_analysis = detector.analyze_propaganda_by_category(test_text)
    print(f"\nHighest risk category: {propaganda_analysis['highest_risk_category']}")
    print(f"Category risk scores: {propaganda_analysis['category_risk_scores']}")
    
    # Show statistics
    stats = detector.get_detection_statistics()
    print(f"\n=== STATISTICS ===")
    print(f"Total detections: {stats['total_detections_completed']}")
    print(f"Average processing time: {stats['average_processing_time_ms']:.1f}ms")
    print(f"Detection methods available: {stats['detection_database_sizes']['total_detection_methods']}")
    
    # Validate detection patterns
    validation = detector.validate_detection_patterns()
    print(f"\nPattern validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['issues']:
        print(f"Issues found: {validation['issues']}")
