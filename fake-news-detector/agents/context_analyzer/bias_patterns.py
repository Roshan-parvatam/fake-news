# agents/context_analyzer/bias_patterns.py
"""
Bias Pattern Database for Context Analyzer Agent - Config Enhanced

Enhanced bias detection patterns with better performance tracking
and configuration awareness.
"""

from typing import Dict, List, Any
import re
import logging
import time

class BiasPatternDatabase:
    """
    📊 ENHANCED BIAS PATTERN DATABASE WITH CONFIG AWARENESS
    
    This class manages patterns for detecting different types of bias
    in news articles with enhanced performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the bias pattern database with optional config
        
        Args:
            config: Optional configuration for bias detection
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize pattern systems
        self.bias_indicators = self._initialize_bias_indicators()
        self.emotional_keywords = self._initialize_emotional_keywords()
        self.framing_patterns = self._initialize_framing_patterns()
        self.linguistic_patterns = self._initialize_linguistic_patterns()
        
        # Performance tracking
        self.bias_stats = {
            'total_analyses': 0,
            'total_bias_indicators_found': 0,
            'analysis_time_total': 0.0,
            'config_applied': bool(config)
        }
        
        self.logger.info(f"✅ BiasPatternDatabase initialized with {len(self.bias_indicators)} bias types")
    
    def _initialize_bias_indicators(self) -> Dict[str, List[str]]:
        """
        📊 BIAS INDICATOR DATABASE - Enhanced with more patterns
        """
        return {
            'political_left': [
                'progressive', 'liberal', 'social justice', 'equality', 'diversity',
                'climate action', 'workers rights', 'universal healthcare', 'gun control',
                'minimum wage', 'reproductive rights', 'lgbtq', 'immigration reform',
                'green energy', 'wealth inequality', 'systemic racism', 'defund police'
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
        """
        📊 EMOTIONAL KEYWORD DATABASE - Comprehensive emotion detection
        """
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
        """
        📊 FRAMING PATTERN DATABASE - How stories are structured
        """
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
        """
        📊 LINGUISTIC PATTERN DATABASE - Language style indicators
        """
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
        🔍 ENHANCED BIAS PATTERN ANALYSIS WITH CONFIG
        
        Comprehensive bias detection using all pattern systems with performance tracking.
        """
        start_time = time.time()
        text_lower = text.lower()
        
        # Bias indicator counting
        bias_counts = {}
        for bias_type, keywords in self.bias_indicators.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            bias_counts[bias_type] = count
        
        # Emotional keyword counting
        emotional_counts = {}
        for emotion, keywords in self.emotional_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotional_counts[emotion] = count
        
        # Framing pattern counting
        framing_counts = {}
        for frame_type, keywords in self.framing_patterns.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            framing_counts[frame_type] = count
        
        # Linguistic pattern counting
        linguistic_counts = {}
        for pattern_type, keywords in self.linguistic_patterns.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            linguistic_counts[pattern_type] = count
        
        # Find specific indicators
        indicators_found = []
        for bias_type, keywords in self.bias_indicators.items():
            found = [keyword for keyword in keywords if keyword in text_lower]
            if found:
                indicators_found.extend([(bias_type, word) for word in found[:3]])  # Limit per type
        
        # Get emotional profile
        emotional_profile = self.get_emotional_profile(emotional_counts)
        
        # Calculate overall bias intensity with config multiplier
        multiplier = self.config.get('bias_intensity_multiplier', 0.3) if self.config else 0.3
        total_bias_indicators = sum(bias_counts.values())
        bias_intensity = min(10, total_bias_indicators * multiplier)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.bias_stats['total_analyses'] += 1
        self.bias_stats['total_bias_indicators_found'] += total_bias_indicators
        self.bias_stats['analysis_time_total'] += processing_time
        
        return {
            'bias_counts': bias_counts,
            'emotional_counts': emotional_counts,
            'framing_counts': framing_counts,
            'linguistic_counts': linguistic_counts,
            'indicators_found': indicators_found[:15],  # Limit output
            'emotional_profile': emotional_profile,
            'bias_intensity_score': round(bias_intensity, 2),
            'total_bias_indicators': total_bias_indicators,
            'analysis_time_ms': round(processing_time * 1000, 2),
            'config_applied': bool(self.config)
        }
    
    def get_emotional_profile(self, emotional_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        📊 ENHANCED EMOTIONAL PROFILE ANALYSIS
        
        Analyze emotional content profile with intensity assessment.
        """
        total_emotional = sum(emotional_counts.values())
        
        if total_emotional == 0:
            return {
                'primary_emotion': 'neutral',
                'intensity': 'none',
                'emotional_distribution': emotional_counts,
                'manipulation_risk': 'low'
            }
        
        # Find primary emotion
        primary_emotion = max(emotional_counts, key=emotional_counts.get)
        primary_count = emotional_counts[primary_emotion]
        
        # Determine intensity level with config thresholds
        intensity_thresholds = self.config.get('intensity_thresholds', {
            'low': 2, 'moderate': 5, 'high': 8, 'extreme': 12
        }) if self.config else {'low': 2, 'moderate': 5, 'high': 8, 'extreme': 12}
        
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
            'emotional_distribution': emotional_counts,
            'manipulation_risk': manipulation_risk,
            'high_risk_emotion_count': high_risk_count
        }
    
    def get_bias_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about bias detection"""
        base_stats = {
            'bias_indicator_types': len(self.bias_indicators),
            'total_bias_indicators': sum(len(indicators) for indicators in self.bias_indicators.values()),
            'emotional_keyword_types': len(self.emotional_keywords),
            'total_emotional_keywords': sum(len(keywords) for keywords in self.emotional_keywords.values()),
            'framing_pattern_types': len(self.framing_patterns),
            'linguistic_pattern_types': len(self.linguistic_patterns)
        }
        
        # Add performance stats
        performance_stats = self.bias_stats.copy()
        if performance_stats['total_analyses'] > 0:
            performance_stats['average_analysis_time_ms'] = round(
                (performance_stats['analysis_time_total'] / performance_stats['total_analyses']) * 1000, 2
            )
            performance_stats['average_indicators_per_analysis'] = round(
                performance_stats['total_bias_indicators_found'] / performance_stats['total_analyses'], 2
            )
        
        return {**base_stats, 'performance_stats': performance_stats}

# Testing
if __name__ == "__main__":
    """Test bias pattern database with config"""
    test_config = {
        'bias_intensity_multiplier': 0.4,
        'intensity_thresholds': {'low': 3, 'moderate': 6, 'high': 10, 'extreme': 15}
    }
    
    bias_db = BiasPatternDatabase(test_config)
    
    test_text = """
    This outrageous scandal exposes the corrupt establishment's lies! 
    The radical left-wing media is spreading fear and panic among patriots.
    Every conservative American must wake up to this devastating crisis.
    """
    
    result = bias_db.analyze_bias_patterns(test_text)
    
    print(f"Bias analysis results:")
    print(f"Total bias indicators: {result['total_bias_indicators']}")
    print(f"Bias intensity score: {result['bias_intensity_score']}/10")
    print(f"Primary emotion: {result['emotional_profile']['primary_emotion']}")
    print(f"Emotional intensity: {result['emotional_profile']['intensity']}")
    print(f"Manipulation risk: {result['emotional_profile']['manipulation_risk']}")
    
    stats = bias_db.get_bias_statistics()
    print(f"Database contains {stats['total_bias_indicators']} bias indicators")
