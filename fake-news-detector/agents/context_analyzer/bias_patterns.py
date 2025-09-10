# agents/context_analyzer/bias_patterns.py

"""
Bias Pattern Database

Systematic bias detection patterns for news articles providing comprehensive
pattern recognition for political, emotional, linguistic, and framing biases.
Enhanced with performance tracking and configurable detection algorithms.
"""

import re
import time
import logging
from typing import Dict, List, Any, Tuple


class BiasPatternDatabase:
    """
    Comprehensive bias pattern database for news article analysis.
    
    Provides systematic detection of various bias types including political,
    emotional, selection, linguistic, and framing biases using pattern matching
    and keyword analysis.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize bias pattern database.
        
        Args:
            config: Optional configuration for bias detection parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize pattern databases
        self.bias_indicators = self._initialize_bias_indicators()
        self.emotional_keywords = self._initialize_emotional_keywords()
        self.framing_patterns = self._initialize_framing_patterns()
        self.linguistic_patterns = self._initialize_linguistic_patterns()
        
        # Performance tracking
        self.analysis_count = 0
        self.total_processing_time = 0.0
        self.total_indicators_found = 0

    def _initialize_bias_indicators(self) -> Dict[str, List[str]]:
        """Initialize comprehensive bias indicator patterns."""
        return {
            'political_left': [
                'progressive', 'liberal', 'social justice', 'equality', 'diversity',
                'climate action', 'workers rights', 'universal healthcare', 'gun control',
                'minimum wage', 'reproductive rights', 'lgbtq', 'immigration reform',
                'green energy', 'wealth inequality', 'systemic racism', 'social safety net'
            ],
            'political_right': [
                'conservative', 'traditional values', 'law and order', 'fiscal responsibility',
                'free market', 'small government', 'pro-life', 'second amendment',
                'border security', 'national defense', 'family values', 'religious freedom',
                'tax cuts', 'deregulation', 'patriotic', 'america first'
            ],
            'fear_words': [
                'dangerous', 'threat', 'crisis', 'catastrophe', 'emergency', 'disaster',
                'chaos', 'collapse', 'destruction', 'devastating', 'alarming', 'terrifying',
                'nightmare', 'horror', 'panic', 'terror', 'menace', 'peril'
            ],
            'anger_words': [
                'outrageous', 'scandal', 'betrayal', 'corrupt', 'fraud', 'disgrace',
                'shameful', 'appalling', 'disgusting', 'infuriating', 'enraging',
                'shocking', 'insulting', 'offensive', 'unacceptable', 'despicable'
            ],
            'loaded_positive': [
                'hero', 'patriot', 'champion', 'defender', 'warrior', 'crusader',
                'savior', 'brilliant', 'genius', 'outstanding', 'exceptional',
                'remarkable', 'inspiring', 'courageous', 'noble', 'righteous'
            ],
            'loaded_negative': [
                'extremist', 'radical', 'terrorist', 'criminal', 'thug', 'villain',
                'enemy', 'traitor', 'corrupt', 'dishonest', 'incompetent',
                'dangerous', 'reckless', 'irresponsible', 'disgraceful', 'shameful'
            ],
            'certainty_absolute': [
                'always', 'never', 'everyone', 'nobody', 'all', 'none',
                'completely', 'totally', 'absolutely', 'definitely', 'certainly',
                'undoubtedly', 'obviously', 'clearly', 'without question'
            ],
            'certainty_uncertain': [
                'might', 'could', 'possibly', 'perhaps', 'maybe', 'allegedly',
                'reportedly', 'supposedly', 'apparently', 'seemingly', 'likely',
                'probably', 'potentially', 'presumably', 'ostensibly'
            ]
        }

    def _initialize_emotional_keywords(self) -> Dict[str, List[str]]:
        """Initialize emotional keyword patterns for manipulation detection."""
        return {
            'fear': [
                'terror', 'panic', 'dread', 'nightmare', 'horror', 'frightening',
                'scary', 'threatening', 'menacing', 'ominous', 'sinister',
                'chilling', 'spine-tingling', 'bone-chilling', 'blood-curdling'
            ],
            'anger': [
                'rage', 'fury', 'outrage', 'wrath', 'indignation', 'ire',
                'livid', 'fuming', 'incensed', 'infuriated', 'enraged',
                'mad', 'angry', 'irate', 'upset', 'annoyed'
            ],
            'sadness': [
                'tragic', 'heartbreaking', 'devastating', 'sorrowful', 'mournful',
                'melancholy', 'depressing', 'gloomy', 'bleak', 'somber',
                'grieving', 'lamenting', 'suffering', 'anguish', 'despair'
            ],
            'joy': [
                'amazing', 'incredible', 'fantastic', 'wonderful', 'marvelous',
                'spectacular', 'outstanding', 'excellent', 'brilliant',
                'success', 'victory', 'triumph', 'celebration', 'jubilant'
            ],
            'disgust': [
                'disgusting', 'revolting', 'repulsive', 'sickening', 'nauseating',
                'vile', 'foul', 'repugnant', 'loathsome', 'abhorrent',
                'shameful', 'despicable', 'contemptible', 'offensive'
            ],
            'pride': [
                'proud', 'patriotic', 'honor', 'dignity', 'respect', 'noble',
                'heroic', 'admirable', 'exemplary', 'distinguished',
                'prestigious', 'renowned', 'celebrated', 'acclaimed'
            ]
        }

    def _initialize_framing_patterns(self) -> Dict[str, List[str]]:
        """Initialize framing pattern indicators."""
        return {
            'victim_framing': [
                'victim', 'victimized', 'oppressed', 'persecuted', 'targeted',
                'attacked', 'bullied', 'harassed', 'discriminated', 'marginalized'
            ],
            'hero_framing': [
                'hero', 'champion', 'defender', 'protector', 'guardian',
                'savior', 'rescuer', 'liberator', 'crusader', 'warrior'
            ],
            'villain_framing': [
                'villain', 'enemy', 'threat', 'danger', 'menace',
                'perpetrator', 'aggressor', 'oppressor', 'tyrant', 'dictator'
            ],
            'crisis_framing': [
                'crisis', 'emergency', 'urgent', 'critical', 'dire',
                'catastrophic', 'devastating', 'alarming', 'pressing', 'immediate'
            ],
            'solution_framing': [
                'solution', 'answer', 'fix', 'remedy', 'cure',
                'resolve', 'address', 'tackle', 'handle', 'deal with'
            ]
        }

    def _initialize_linguistic_patterns(self) -> Dict[str, List[str]]:
        """Initialize linguistic style indicators."""
        return {
            'hyperbolic': [
                'most', 'best', 'worst', 'greatest', 'terrible', 'awful',
                'incredible', 'unbelievable', 'shocking', 'stunning',
                'massive', 'huge', 'enormous', 'gigantic', 'tiny'
            ],
            'urgency': [
                'now', 'immediately', 'urgent', 'critical', 'emergency',
                'breaking', 'alert', 'rush', 'hurry', 'deadline',
                'last chance', 'final warning', 'act now'
            ],
            'exclusive': [
                'exclusive', 'secret', 'hidden', 'revealed', 'exposed',
                'leaked', 'confidential', 'classified', 'insider',
                'behind closed doors', 'off the record'
            ]
        }

    def analyze_bias_patterns(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive bias pattern analysis.
        
        Args:
            text: Article text to analyze
            
        Returns:
            Dictionary containing bias analysis results
        """
        start_time = time.time()
        text_lower = text.lower()
        
        # Analyze bias indicators
        bias_counts = {}
        for bias_type, keywords in self.bias_indicators.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            bias_counts[bias_type] = count
        
        # Analyze emotional content
        emotional_counts = {}
        for emotion, keywords in self.emotional_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotional_counts[emotion] = count
        
        # Analyze framing patterns
        framing_counts = {}
        for frame_type, keywords in self.framing_patterns.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            framing_counts[frame_type] = count
        
        # Analyze linguistic patterns
        linguistic_counts = {}
        for pattern_type, keywords in self.linguistic_patterns.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            linguistic_counts[pattern_type] = count
        
        # Find specific indicators
        indicators_found = []
        for bias_type, keywords in self.bias_indicators.items():
            found_keywords = [keyword for keyword in keywords if keyword in text_lower]
            if found_keywords:
                indicators_found.extend([(bias_type, word) for word in found_keywords[:3]])
        
        # Calculate emotional profile
        emotional_profile = self._analyze_emotional_profile(emotional_counts)
        
        # Calculate bias intensity
        total_bias_indicators = sum(bias_counts.values())
        bias_intensity_multiplier = self.config.get('bias_intensity_multiplier', 0.3)
        bias_intensity = min(10, total_bias_indicators * bias_intensity_multiplier)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.analysis_count += 1
        self.total_processing_time += processing_time
        self.total_indicators_found += total_bias_indicators
        
        return {
            'bias_counts': bias_counts,
            'emotional_counts': emotional_counts,
            'framing_counts': framing_counts,
            'linguistic_counts': linguistic_counts,
            'indicators_found': indicators_found[:15],  # Limit output
            'emotional_profile': emotional_profile,
            'bias_intensity_score': round(bias_intensity, 2),
            'total_bias_indicators': total_bias_indicators,
            'processing_time_ms': round(processing_time * 1000, 2)
        }

    def _analyze_emotional_profile(self, emotional_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Analyze emotional content profile and manipulation risk.
        
        Args:
            emotional_counts: Dictionary of emotion counts
            
        Returns:
            Dictionary with emotional profile analysis
        """
        total_emotional = sum(emotional_counts.values())
        
        if total_emotional == 0:
            return {
                'primary_emotion': 'neutral',
                'intensity': 'none',
                'manipulation_risk': 'low',
                'total_emotional_indicators': 0
            }
        
        # Find primary emotion
        primary_emotion = max(emotional_counts, key=emotional_counts.get)
        primary_count = emotional_counts[primary_emotion]
        
        # Determine intensity level
        intensity_thresholds = self.config.get('intensity_thresholds', {
            'low': 2, 'moderate': 5, 'high': 8, 'extreme': 12
        })
        
        if total_emotional <= intensity_thresholds['low']:
            intensity = 'low'
        elif total_emotional <= intensity_thresholds['moderate']:
            intensity = 'moderate'
        elif total_emotional <= intensity_thresholds['high']:
            intensity = 'high'
        else:
            intensity = 'extreme'
        
        # Assess manipulation risk
        high_risk_emotions = ['fear', 'anger', 'disgust']
        high_risk_count = sum(emotional_counts.get(emotion, 0) for emotion in high_risk_emotions)
        
        if high_risk_count > 5:
            manipulation_risk = 'high'
        elif high_risk_count > 2:
            manipulation_risk = 'medium'
        else:
            manipulation_risk = 'low'
        
        return {
            'primary_emotion': primary_emotion,
            'primary_emotion_count': primary_count,
            'intensity': intensity,
            'total_emotional_indicators': total_emotional,
            'manipulation_risk': manipulation_risk,
            'high_risk_emotion_count': high_risk_count,
            'emotional_distribution': emotional_counts
        }

    def detect_political_bias(self, text: str) -> Dict[str, Any]:
        """
        Focused political bias detection.
        
        Args:
            text: Text to analyze for political bias
            
        Returns:
            Dictionary with political bias analysis
        """
        text_lower = text.lower()
        
        left_indicators = sum(1 for keyword in self.bias_indicators['political_left'] if keyword in text_lower)
        right_indicators = sum(1 for keyword in self.bias_indicators['political_right'] if keyword in text_lower)
        
        total_political = left_indicators + right_indicators
        
        if total_political == 0:
            return {
                'political_leaning': 'neutral',
                'bias_strength': 0,
                'confidence': 0.0
            }
        
        # Determine political leaning
        if left_indicators > right_indicators * 2:
            leaning = 'left'
            strength = left_indicators
        elif right_indicators > left_indicators * 2:
            leaning = 'right' 
            strength = right_indicators
        else:
            leaning = 'mixed'
            strength = total_political
        
        # Calculate confidence based on indicator count and ratio
        confidence = min(1.0, total_political / 10)
        
        return {
            'political_leaning': leaning,
            'bias_strength': strength,
            'left_indicators': left_indicators,
            'right_indicators': right_indicators,
            'confidence': round(confidence, 2)
        }

    def analyze_loaded_language(self, text: str) -> Dict[str, Any]:
        """
        Analyze loaded and emotional language usage.
        
        Args:
            text: Text to analyze for loaded language
            
        Returns:
            Dictionary with loaded language analysis
        """
        text_lower = text.lower()
        
        positive_loaded = sum(1 for word in self.bias_indicators['loaded_positive'] if word in text_lower)
        negative_loaded = sum(1 for word in self.bias_indicators['loaded_negative'] if word in text_lower)
        
        total_loaded = positive_loaded + negative_loaded
        
        if total_loaded == 0:
            return {
                'loaded_language_detected': False,
                'emotional_tone': 'neutral',
                'manipulation_potential': 'low'
            }
        
        # Determine emotional tone
        if positive_loaded > negative_loaded:
            tone = 'positive'
        elif negative_loaded > positive_loaded:
            tone = 'negative'
        else:
            tone = 'mixed'
        
        # Assess manipulation potential
        if total_loaded >= 5:
            manipulation_potential = 'high'
        elif total_loaded >= 3:
            manipulation_potential = 'medium'
        else:
            manipulation_potential = 'low'
        
        return {
            'loaded_language_detected': True,
            'emotional_tone': tone,
            'positive_loaded_count': positive_loaded,
            'negative_loaded_count': negative_loaded,
            'total_loaded_words': total_loaded,
            'manipulation_potential': manipulation_potential
        }

    def get_bias_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bias detection statistics."""
        avg_processing_time = (
            self.total_processing_time / self.analysis_count 
            if self.analysis_count > 0 else 0
        )
        
        avg_indicators_per_analysis = (
            self.total_indicators_found / self.analysis_count
            if self.analysis_count > 0 else 0
        )
        
        return {
            'total_analyses_completed': self.analysis_count,
            'total_processing_time_seconds': round(self.total_processing_time, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'total_indicators_found': self.total_indicators_found,
            'average_indicators_per_analysis': round(avg_indicators_per_analysis, 2),
            'pattern_database_sizes': {
                'bias_indicator_types': len(self.bias_indicators),
                'emotional_keyword_types': len(self.emotional_keywords),
                'framing_pattern_types': len(self.framing_patterns),
                'linguistic_pattern_types': len(self.linguistic_patterns),
                'total_bias_indicators': sum(len(indicators) for indicators in self.bias_indicators.values()),
                'total_emotional_keywords': sum(len(keywords) for keywords in self.emotional_keywords.values())
            }
        }

    def validate_patterns(self) -> Dict[str, Any]:
        """Validate pattern database integrity and completeness."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for empty pattern lists
        for category, patterns in self.bias_indicators.items():
            if not patterns:
                validation_results['issues'].append(f"Empty pattern list: {category}")
                validation_results['valid'] = False
        
        # Check for duplicate patterns across categories
        all_patterns = []
        for patterns in self.bias_indicators.values():
            all_patterns.extend(patterns)
        
        duplicates = set([x for x in all_patterns if all_patterns.count(x) > 1])
        if duplicates:
            validation_results['warnings'].append(f"Duplicate patterns found: {list(duplicates)}")
        
        # Check pattern quality (length, complexity)
        short_patterns = [p for patterns in self.bias_indicators.values() for p in patterns if len(p) < 3]
        if short_patterns:
            validation_results['warnings'].append(f"Very short patterns: {short_patterns}")
        
        return validation_results


# Testing functionality
if __name__ == "__main__":
    """Test bias pattern database functionality."""
    
    # Initialize database
    test_config = {
        'bias_intensity_multiplier': 0.4,
        'intensity_thresholds': {'low': 3, 'moderate': 6, 'high': 10, 'extreme': 15}
    }
    
    bias_db = BiasPatternDatabase(test_config)
    
    # Test analysis
    test_text = """
    This outrageous scandal exposes the corrupt establishment's lies!
    The radical left-wing media is spreading fear and panic among patriots.
    Every conservative American must wake up to this devastating crisis
    and defend our traditional values against these extremist attacks.
    """
    
    print("=== BIAS PATTERN ANALYSIS TEST ===")
    result = bias_db.analyze_bias_patterns(test_text)
    
    print(f"Total bias indicators: {result['total_bias_indicators']}")
    print(f"Bias intensity score: {result['bias_intensity_score']}/10")
    print(f"Primary emotion: {result['emotional_profile']['primary_emotion']}")
    print(f"Emotional intensity: {result['emotional_profile']['intensity']}")
    print(f"Manipulation risk: {result['emotional_profile']['manipulation_risk']}")
    
    # Test political bias detection
    political_analysis = bias_db.detect_political_bias(test_text)
    print(f"\nPolitical leaning: {political_analysis['political_leaning']}")
    print(f"Bias strength: {political_analysis['bias_strength']}")
    print(f"Confidence: {political_analysis['confidence']}")
    
    # Test loaded language analysis
    loaded_analysis = bias_db.analyze_loaded_language(test_text)
    print(f"\nLoaded language detected: {loaded_analysis['loaded_language_detected']}")
    print(f"Emotional tone: {loaded_analysis['emotional_tone']}")
    print(f"Manipulation potential: {loaded_analysis['manipulation_potential']}")
    
    # Show statistics
    stats = bias_db.get_bias_statistics()
    print(f"\n=== STATISTICS ===")
    print(f"Total analyses: {stats['total_analyses_completed']}")
    print(f"Average processing time: {stats['average_processing_time_ms']:.1f}ms")
    print(f"Pattern database size: {stats['pattern_database_sizes']['total_bias_indicators']} indicators")
    
    # Validate patterns
    validation = bias_db.validate_patterns()
    print(f"\nPattern validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
