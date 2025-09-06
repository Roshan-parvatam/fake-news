# agents/context_analyzer/manipulation_detection.py
"""
Manipulation Detection for Context Analyzer Agent - Config Enhanced

Enhanced manipulation and propaganda detection with better performance tracking
and configuration awareness.
"""

from typing import Dict, List, Any
import re
import logging
import time

class ManipulationDetector:
    """
    🚨 ENHANCED MANIPULATION DETECTION WITH CONFIG AWARENESS
    
    This class detects manipulation techniques and propaganda methods
    in news articles with enhanced performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the manipulation detector with optional config
        
        Args:
            config: Optional configuration for manipulation detection
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize detection systems
        self.propaganda_techniques = self._initialize_propaganda_techniques()
        self.manipulation_patterns = self._initialize_manipulation_patterns()
        self.logical_fallacies = self._initialize_logical_fallacies()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'total_techniques_found': 0,
            'analysis_time_total': 0.0,
            'config_applied': bool(config)
        }
        
        self.logger.info(f"✅ ManipulationDetector initialized with {len(self.propaganda_techniques)} propaganda techniques")
    
    def _initialize_propaganda_techniques(self) -> Dict[str, Dict[str, Any]]:
        """
        🚨 PROPAGANDA TECHNIQUES DATABASE - Enhanced with more techniques
        """
        return {
            'name_calling': {
                'indicators': ['extremist', 'radical', 'terrorist', 'criminal', 'corrupt'],
                'patterns': [r'so-called\s+\w+', r'\w+\s+is\s+nothing\s+but'],
                'severity': 'medium',
                'description': 'Attaching negative labels to discredit'
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
                'description': 'Connecting ideas with positive/negative symbols'
            },
            'testimonial': {
                'indicators': ['celebrity', 'expert', 'endorses', 'recommends', 'supports'],
                'patterns': [r'\w+\s+endorses', r'according\s+to\s+\w+\s+expert'],
                'severity': 'low',
                'description': 'Using celebrity or authority endorsements'
            },
            'plain_folks': {
                'indicators': ['ordinary', 'common', 'working class', 'regular people', 'folks'],
                'patterns': [r'just\s+like\s+you', r'ordinary\s+\w+'],
                'severity': 'low',
                'description': 'Appealing to common people values'
            },
            'card_stacking': {
                'indicators': ['only', 'never mentions', 'ignores', 'hides', 'conceals'],
                'patterns': [r'they\s+don\'t\s+want\s+you\s+to\s+know', r'hidden\s+truth'],
                'severity': 'high',
                'description': 'Presenting only one side of argument'
            },
            'bandwagon': {
                'indicators': ['everyone', 'everybody', 'all', 'join', 'movement'],
                'patterns': [r'everyone\s+is\s+\w+ing', r'join\s+the\s+\w+'],
                'severity': 'medium',
                'description': 'Encouraging following the crowd'
            },
            'fear_mongering': {
                'indicators': ['danger', 'threat', 'crisis', 'emergency', 'catastrophe'],
                'patterns': [r'if\s+we\s+don\'t\s+act', r'time\s+is\s+running\s+out'],
                'severity': 'high',
                'description': 'Creating fear to motivate action'
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
                'description': 'Attacking person instead of argument'
            }
        }
    
    def _initialize_manipulation_patterns(self) -> Dict[str, List[str]]:
        """
        🚨 MANIPULATION PATTERN DATABASE - Modern manipulation techniques
        """
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
        """
        🚨 LOGICAL FALLACY DATABASE - Common reasoning errors
        """
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
        🚨 COMPREHENSIVE MANIPULATION ANALYSIS WITH CONFIG
        
        Generates complete manipulation detection report with performance tracking.
        """
        start_time = time.time()
        
        # Detect propaganda techniques
        propaganda_results = self.detect_propaganda_techniques(text)
        
        # Detect manipulation patterns
        manipulation_results = self.detect_manipulation_patterns(text)
        
        # Detect logical fallacies
        fallacy_results = self.detect_logical_fallacies(text)
        
        # Calculate overall manipulation score
        overall_score = self.calculate_manipulation_score(
            propaganda_results, manipulation_results, fallacy_results
        )
        
        # Performance tracking
        processing_time = time.time() - start_time
        total_techniques = (len(propaganda_results['detected']) + 
                          len(manipulation_results['detected']) + 
                          len(fallacy_results['detected']))
        
        self.detection_stats['total_detections'] += 1
        self.detection_stats['total_techniques_found'] += total_techniques
        self.detection_stats['analysis_time_total'] += processing_time
        
        return {
            'propaganda_techniques': propaganda_results,
            'manipulation_patterns': manipulation_results,
            'logical_fallacies': fallacy_results,
            'overall_manipulation_score': overall_score,
            'risk_level': self._assess_risk_level(overall_score),
            'techniques_summary': {
                'total_techniques_detected': total_techniques,
                'high_severity_count': self._count_high_severity(propaganda_results, manipulation_results, fallacy_results),
                'analysis_time_ms': round(processing_time * 1000, 2)
            },
            'config_applied': bool(self.config)
        }
    
    def detect_propaganda_techniques(self, text: str) -> Dict[str, Any]:
        """Detect propaganda techniques with enhanced pattern matching"""
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
                confidence = min(1.0, score / 5)  # Normalize confidence
                detected[technique] = {
                    'score': score,
                    'confidence': round(confidence, 2),
                    'severity': info['severity'],
                    'matches': matches[:3],  # Limit matches
                    'description': info['description']
                }
        
        return {
            'detected': detected,
            'total_techniques': len(detected),
            'high_confidence': [t for t, d in detected.items() if d['confidence'] > 0.6]
        }
    
    def detect_manipulation_patterns(self, text: str) -> Dict[str, Any]:
        """Detect modern manipulation patterns"""
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
    
    def detect_logical_fallacies(self, text: str) -> Dict[str, Any]:
        """Detect logical fallacies with enhanced analysis"""
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
                    'evidence': evidence
                }
        
        return {
            'detected': detected,
            'total_fallacies': len(detected)
        }
    
    def calculate_manipulation_score(self, propaganda_results: Dict, manipulation_results: Dict, fallacy_results: Dict) -> float:
        """Calculate overall manipulation score with config weights"""
        # Default weights
        propaganda_weight = self.config.get('propaganda_weight', 0.4) if self.config else 0.4
        manipulation_weight = self.config.get('manipulation_weight', 0.4) if self.config else 0.4
        fallacy_weight = self.config.get('fallacy_weight', 0.2) if self.config else 0.2
        
        # Calculate component scores
        propaganda_score = self._score_propaganda(propaganda_results)
        manipulation_score = self._score_manipulation(manipulation_results)
        fallacy_score = self._score_fallacies(fallacy_results)
        
        # Weighted overall score
        overall_score = (
            propaganda_score * propaganda_weight +
            manipulation_score * manipulation_weight +
            fallacy_score * fallacy_weight
        )
        
        return round(min(10.0, overall_score), 2)
    
    def _score_propaganda(self, results: Dict) -> float:
        """Score propaganda techniques"""
        if not results['detected']:
            return 0.0
        
        total_score = 0
        for technique, data in results['detected'].items():
            severity_multiplier = {'low': 1, 'medium': 2, 'high': 3}
            total_score += data['confidence'] * severity_multiplier.get(data['severity'], 1)
        
        return min(10.0, total_score * 2)
    
    def _score_manipulation(self, results: Dict) -> float:
        """Score manipulation patterns"""
        if not results['detected']:
            return 0.0
        
        total_score = sum(data['confidence'] for data in results['detected'].values())
        return min(10.0, total_score * 2.5)
    
    def _score_fallacies(self, results: Dict) -> float:
        """Score logical fallacies"""
        if not results['detected']:
            return 0.0
        
        total_score = 0
        for fallacy, data in results['detected'].items():
            severity_multiplier = {'low': 1, 'medium': 1.5, 'high': 2}
            total_score += data['confidence'] * severity_multiplier.get(data['severity'], 1)
        
        return min(10.0, total_score * 2)
    
    def _assess_risk_level(self, score: float) -> str:
        """Assess manipulation risk level with config thresholds"""
        thresholds = self.config.get('risk_thresholds', {
            'minimal': 2, 'low': 4, 'medium': 6, 'high': 8
        }) if self.config else {'minimal': 2, 'low': 4, 'medium': 6, 'high': 8}
        
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
    
    def _count_high_severity(self, propaganda_results: Dict, manipulation_results: Dict, fallacy_results: Dict) -> int:
        """Count high severity techniques found"""
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
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics"""
        base_stats = {
            'propaganda_techniques_count': len(self.propaganda_techniques),
            'manipulation_patterns_count': len(self.manipulation_patterns),
            'logical_fallacies_count': len(self.logical_fallacies),
            'total_detection_methods': (len(self.propaganda_techniques) + 
                                      len(self.manipulation_patterns) + 
                                      len(self.logical_fallacies))
        }
        
        # Add performance stats
        performance_stats = self.detection_stats.copy()
        if performance_stats['total_detections'] > 0:
            performance_stats['average_detection_time_ms'] = round(
                (performance_stats['analysis_time_total'] / performance_stats['total_detections']) * 1000, 2
            )
            performance_stats['average_techniques_per_detection'] = round(
                performance_stats['total_techniques_found'] / performance_stats['total_detections'], 2
            )
        
        return {**base_stats, 'performance_stats': performance_stats}

# Testing
if __name__ == "__main__":
    """Test manipulation detector with config"""
    test_config = {
        'propaganda_weight': 0.5,
        'manipulation_weight': 0.3,
        'fallacy_weight': 0.2,
        'risk_thresholds': {'minimal': 1, 'low': 3, 'medium': 5, 'high': 7}
    }
    
    detector = ManipulationDetector(test_config)
    
    test_text = """
    Don't let the corrupt establishment fool you! This is clearly a false flag
    operation designed to distract patriots from the real truth. Either you're 
    with us or you're against America. What about all the lies they told before?
    Every real American knows this is obviously a conspiracy.
    """
    
    report = detector.get_manipulation_report(test_text)
    
    print(f"Manipulation analysis results:")
    print(f"Overall manipulation score: {report['overall_manipulation_score']}/10")
    print(f"Risk level: {report['risk_level']}")
    print(f"Total techniques detected: {report['techniques_summary']['total_techniques_detected']}")
    print(f"High severity count: {report['techniques_summary']['high_severity_count']}")
    
    stats = detector.get_detection_statistics()
    print(f"Detector has {stats['total_detection_methods']} detection methods")
