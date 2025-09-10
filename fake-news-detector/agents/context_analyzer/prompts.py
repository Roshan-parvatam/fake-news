# agents/context_analyzer/prompts.py

"""
Context Analyzer Prompts Module

Industry-standard prompt templates for context analysis with consistent
numerical scoring, bias detection, manipulation analysis, and framing assessment.
Uses Chain-of-Thought and structured output patterns for reliable results.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PromptResponse:
    """Structured response container for prompt outputs."""
    content: str
    metadata: Dict[str, Any]

class BiasDetectionPrompts:
    """Bias detection prompts with consistent scoring enforcement."""
    
    @staticmethod
    def comprehensive_bias_analysis(article_text: str, source: str, topic_domain: str, 
                                  prediction: str, confidence: float) -> str:
        """
        Generate comprehensive bias detection prompt with consistent scoring.
        Fixes the main issue of text analysis not matching numerical scores.
        """
        return f"""You are a professional media bias analyst conducting systematic bias assessment.

ARTICLE TO ANALYZE:
{article_text[:2000]}

CONTEXT:
- Source: {source}
- Topic: {topic_domain}
- Classification: {prediction} ({confidence:.1%} confidence)

ANALYSIS FRAMEWORK:

## Step 1: Political Bias Assessment
Analyze for:
- Left-leaning language (progressive, social justice, climate action, etc.)
- Right-leaning language (conservative, traditional values, law and order, etc.)
- Neutral language vs. partisan framing

## Step 2: Selection Bias Assessment  
Analyze for:
- Cherry-picked facts or quotes
- Missing opposing viewpoints
- Selective use of sources
- Context omissions

## Step 3: Linguistic Bias Assessment
Analyze for:
- Loaded language (positive/negative descriptors)
- Emotional appeals vs. factual presentation
- Absolute vs. nuanced language
- Framing techniques

## Step 4: Source Bias Assessment
Analyze for:
- Source diversity and balance
- Expert selection patterns
- Attribution quality
- Institutional bias indicators

REQUIRED OUTPUT FORMAT:

### NUMERICAL SCORES (0-100 scale):
POLITICAL_BIAS: [0-100 where 0=completely neutral, 100=extremely partisan]
SELECTION_BIAS: [0-100 where 0=comprehensive coverage, 100=highly selective]
LINGUISTIC_BIAS: [0-100 where 0=neutral language, 100=heavily loaded language]
OVERALL_BIAS: [0-100 overall bias intensity]

### DETAILED ANALYSIS:

## Political Bias Evidence
[Provide specific examples from the article. Your analysis MUST justify your POLITICAL_BIAS score above.]

## Selection Bias Evidence
[Identify what perspectives/facts are included or omitted. Your analysis MUST justify your SELECTION_BIAS score above.]

## Linguistic Bias Evidence
[Analyze word choice and framing techniques. Your analysis MUST justify your LINGUISTIC_BIAS score above.]

## Overall Assessment
[Provide summary that perfectly matches your OVERALL_BIAS score above.]

CRITICAL CONSISTENCY REQUIREMENTS:
1. Your numerical scores MUST exactly match your written analysis
2. If you write "minimal bias", OVERALL_BIAS must be 25 or lower
3. If you write "moderate bias", OVERALL_BIAS must be 30-60
4. If you write "significant bias", OVERALL_BIAS must be 60-80
5. If you write "extreme bias", OVERALL_BIAS must be 80+
6. Provide specific text examples to justify each score

DOUBLE-CHECK: Verify your scores match your written explanations before finishing."""

    @staticmethod
    def political_bias_assessment(article_text: str, political_context: str) -> str:
        """Focused political bias detection prompt."""
        return f"""Assess political bias in this article using systematic analysis.

ARTICLE: {article_text[:1500]}
CONTEXT: {political_context}

ANALYSIS FRAMEWORK:

## Left-Leaning Indicators
- Progressive policy language
- Social justice framing
- Environmental activism terms
- Labor/worker rights emphasis
- Inclusive language patterns

## Right-Leaning Indicators  
- Conservative value language
- Traditional institution emphasis
- Security/law enforcement focus
- Business/market friendly terms
- National sovereignty themes

## Neutral Indicators
- Balanced source representation
- Multiple perspective inclusion
- Factual presentation style
- Objective language use

OUTPUT FORMAT:
POLITICAL_LEANING: [Left/Right/Center/Mixed]
INTENSITY: [Subtle/Moderate/Strong/Extreme]
CONFIDENCE: [0-100]

EVIDENCE:
[Specific examples from text supporting your assessment]"""

class ManipulationDetectionPrompts:
    """Manipulation and propaganda detection prompts."""
    
    @staticmethod
    def emotional_manipulation_analysis(article_text: str, emotional_indicators: Dict[str, Any]) -> str:
        """Detect emotional manipulation techniques with scoring."""
        return f"""Analyze emotional manipulation techniques in this article.

ARTICLE: {article_text[:2000]}
DETECTED_INDICATORS: {emotional_indicators}

MANIPULATION ASSESSMENT FRAMEWORK:

## Fear-Based Appeals
- Threat language and crisis framing
- Urgency creation and time pressure
- Catastrophic outcome predictions
- Safety/security concerns amplification

## Anger-Based Appeals  
- Outrage generation and injustice framing
- Enemy identification and us-vs-them language
- Moral violation emphasis
- Blame assignment patterns

## Hope-Based Manipulation
- Unrealistic solution promises
- Savior figure promotion
- Easy fix oversimplification
- False optimism injection

## Emotional Intensity Scoring
- Subtle emotional appeals (20-40)
- Moderate emotional content (40-60)  
- Strong emotional manipulation (60-80)
- Extreme emotional exploitation (80-100)

REQUIRED OUTPUT:

MANIPULATION_SCORE: [0-100]
PRIMARY_EMOTION: [Fear/Anger/Hope/Sadness/Other]
TECHNIQUES_DETECTED: [List specific manipulation methods]

DETAILED_ANALYSIS:
[Explain how emotions are used to influence readers. Analysis must justify your MANIPULATION_SCORE.]

MANIPULATION_RISK: [Low/Medium/High/Critical]"""

    @staticmethod
    def propaganda_technique_detection(article_text: str) -> str:
        """Detect specific propaganda techniques with examples."""
        return f"""Identify propaganda techniques in this article using systematic analysis.

ARTICLE: {article_text[:2000]}

PROPAGANDA TECHNIQUE CHECKLIST:

## Classical Techniques
- Name Calling: Negative labels to discredit
- Glittering Generalities: Vague positive concepts
- Transfer: Association with symbols/authority
- Testimonial: Celebrity/expert endorsement misuse
- Plain Folks: False common people appeals
- Card Stacking: One-sided argument presentation
- Bandwagon: Follow the crowd pressure

## Modern Techniques  
- Astroturfing: Fake grassroots movements
- Gaslighting: Reality distortion attempts
- Whataboutism: Deflection through comparison
- Strawman: Misrepresenting opposing views
- False Dilemma: Artificial choice limitation

ANALYSIS OUTPUT:

TECHNIQUES_DETECTED: [List each technique found]
TECHNIQUE_INTENSITY: [0-100 for overall propaganda level]

For each detected technique:
- TECHNIQUE: [Name]
- EXAMPLES: [Specific text from article]  
- IMPACT: [How it influences readers]
- SEVERITY: [Low/Medium/High]

OVERALL_ASSESSMENT:
[Summary of propaganda usage and reader impact risk]"""

class FramingAnalysisPrompts:
    """Framing and narrative structure analysis prompts."""
    
    @staticmethod
    def narrative_framing_analysis(article_text: str, context: Dict[str, Any]) -> str:
        """Analyze how the story is framed and structured."""
        return f"""Analyze the narrative framing and story structure of this article.

ARTICLE: {article_text[:2000]}
CONTEXT: {context}

FRAMING ANALYSIS FRAMEWORK:

## Problem Definition Framing
- How is the central issue characterized?
- What aspects are emphasized or minimized?
- Whose perspective defines the problem?

## Causal Attribution Framing
- Who/what is presented as responsible?
- Are causes oversimplified or comprehensive?
- Is blame assignment fair and evidence-based?

## Solution Framing
- What solutions are presented or implied?
- Whose interests do proposed solutions serve?
- Are alternative approaches acknowledged?

## Stakeholder Framing
- How are different parties characterized?
- Who gets sympathetic vs. critical treatment?
- Whose voices are privileged or marginalized?

## Temporal Framing
- Is this presented as urgent or routine?
- Historical context inclusion or omission?
- Future implications emphasized?

OUTPUT FORMAT:

FRAMING_TYPE: [Crisis/Conflict/Progress/Human Interest/Other]
PERSPECTIVE_BIAS: [0-100 where 0=balanced, 100=one-sided]

FRAMING_ELEMENTS:
- PROBLEM: [How issue is defined]
- CAUSE: [Attribution of responsibility]  
- SOLUTION: [Proposed remedies]
- STAKEHOLDERS: [How parties are characterized]

FRAMING_IMPACT: [How framing influences reader interpretation]
ALTERNATIVE_FRAMINGS: [Other ways this could be presented]"""

class StructuredOutputPrompts:
    """Prompts that enforce structured JSON output for programmatic processing."""
    
    @staticmethod
    def comprehensive_context_analysis(article_text: str, source: str, prediction: str, confidence: float) -> str:
        """Generate complete context analysis with structured scoring."""
        return f"""Conduct comprehensive context analysis with consistent numerical scoring.

ARTICLE: {article_text[:2500]}
SOURCE: {source}
PREDICTION: {prediction} ({confidence:.1%})

COMPLETE ANALYSIS REQUIRED:

## Bias Assessment
Analyze political, selection, and linguistic bias patterns.

## Manipulation Assessment  
Identify emotional manipulation and propaganda techniques.

## Credibility Assessment
Evaluate source reliability and factual presentation quality.

## Risk Assessment
Assess potential for misinformation spread and harm.

MANDATORY OUTPUT FORMAT - EXACT JSON STRUCTURE:

{{
  "scores": {{
    "bias": [0-100],
    "manipulation": [0-100], 
    "credibility": [0-100],
    "risk": [0-100]
  }},
  "analysis": {{
    "bias_explanation": "Your bias analysis here - must justify bias score",
    "manipulation_explanation": "Your manipulation analysis here - must justify manipulation score", 
    "credibility_explanation": "Your credibility analysis here - must justify credibility score",
    "risk_explanation": "Your risk analysis here - must justify risk score"
  }},
  "detected_techniques": [
    "List specific bias/manipulation techniques found"
  ],
  "risk_level": "LOW/MEDIUM/HIGH/CRITICAL",
  "recommendation": "Clear guidance for readers"
}}

CRITICAL SCORING RULES:
- Scores 0-25: Use words like "minimal", "low", "slight"
- Scores 26-50: Use words like "moderate", "some", "noticeable"  
- Scores 51-75: Use words like "significant", "high", "concerning"
- Scores 76-100: Use words like "extreme", "severe", "dangerous"

Your explanations MUST use language that matches your numerical scores exactly."""

class DomainSpecificPrompts:
    """Domain-specific prompts for different content types."""
    
    POLITICAL_CONTEXT_PROMPT = """
    For political content, focus on:
    - Partisan language and framing
    - Policy position bias
    - Political figure characterization
    - Electoral implications emphasis
    - Ideological positioning indicators
    
    Political bias indicators:
    - LEFT: progressive, social justice, climate action, workers rights
    - RIGHT: conservative, traditional values, law and order, fiscal responsibility
    - CENTER: balanced representation, multiple perspectives, neutral framing
    """
    
    HEALTH_CONTEXT_PROMPT = """
    For health/medical content, focus on:
    - Scientific accuracy vs. sensationalism
    - Expert source quality and balance
    - Risk communication appropriateness
    - Commercial bias indicators
    - Public health vs. individual interest framing
    """
    
    ECONOMIC_CONTEXT_PROMPT = """  
    For economic content, focus on:
    - Market ideology bias (pro-business vs. pro-regulation)
    - Class perspective bias (investor vs. worker viewpoint)
    - Policy impact framing completeness
    - Statistical presentation accuracy
    - Corporate interest alignment
    """

class PromptValidator:
    """Validation methods for prompt inputs and outputs."""
    
    @staticmethod
    def validate_score_consistency(analysis_text: str, scores: Dict[str, int]) -> bool:
        """Validate that numerical scores match textual analysis."""
        text_lower = analysis_text.lower()
        
        for score_type, score_value in scores.items():
            # Check for consistency violations
            if score_value <= 25 and any(word in text_lower for word in ['high', 'significant', 'extreme', 'severe']):
                return False
            if score_value >= 75 and any(word in text_lower for word in ['minimal', 'low', 'slight', 'neutral']):
                return False
                
        return True
    
    @staticmethod
    def extract_numerical_scores(text: str) -> Dict[str, int]:
        """Extract numerical scores from LLM response."""
        import re
        scores = {}
        
        # Score extraction patterns
        patterns = {
            'bias': r'(?:BIAS_SCORE|OVERALL_BIAS|bias.*?):\s*(\d+)',
            'manipulation': r'(?:MANIPULATION_SCORE|manipulation.*?):\s*(\d+)', 
            'credibility': r'(?:CREDIBILITY_SCORE|credibility.*?):\s*(\d+)',
            'risk': r'(?:RISK_SCORE|risk.*?):\s*(\d+)'
        }
        
        for score_type, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[score_type] = min(100, max(0, int(match.group(1))))
        
        return scores

def get_context_prompt_template(prompt_type: str, **kwargs) -> str:
    """
    Get specific context analysis prompt template.
    
    Args:
        prompt_type: Type of analysis needed
        **kwargs: Parameters for prompt formatting
        
    Returns:
        Formatted prompt string
    """
    prompt_mapping = {
        'bias_detection': BiasDetectionPrompts.comprehensive_bias_analysis,
        'political_bias': BiasDetectionPrompts.political_bias_assessment,
        'emotional_manipulation': ManipulationDetectionPrompts.emotional_manipulation_analysis,
        'propaganda_detection': ManipulationDetectionPrompts.propaganda_technique_detection,
        'framing_analysis': FramingAnalysisPrompts.narrative_framing_analysis,
        'comprehensive_analysis': StructuredOutputPrompts.comprehensive_context_analysis,
    }
    
    if prompt_type not in prompt_mapping:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompt_mapping[prompt_type](**kwargs)

def validate_context_analysis_output(analysis_text: str, scores: Dict[str, int]) -> bool:
    """
    Validate context analysis output for consistency.
    
    Args:
        analysis_text: Generated analysis text
        scores: Numerical scores dictionary
        
    Returns:
        True if consistent, False if inconsistent
    """
    validator = PromptValidator()
    return validator.validate_score_consistency(analysis_text, scores)

# Testing functionality
if __name__ == "__main__":
    """Test context analyzer prompts."""
    
    test_article = """
    The corrupt establishment politicians are once again betraying hardworking Americans!
    This outrageous scandal exposes their lies while patriots demand justice.
    Every real American must wake up to this crisis before it's too late.
    """
    
    # Test comprehensive bias analysis
    bias_prompt = BiasDetectionPrompts.comprehensive_bias_analysis(
        article_text=test_article,
        source="Test News",
        topic_domain="political", 
        prediction="FAKE",
        confidence=0.85
    )
    
    print("=== BIAS DETECTION PROMPT ===")
    print(bias_prompt[:500] + "...")
    
    # Test manipulation detection
    manipulation_prompt = ManipulationDetectionPrompts.emotional_manipulation_analysis(
        article_text=test_article,
        emotional_indicators={'anger_detected': True, 'fear_detected': True}
    )
    
    print("\n=== MANIPULATION DETECTION PROMPT ===")
    print(manipulation_prompt[:500] + "...")
    
    # Test structured output
    structured_prompt = StructuredOutputPrompts.comprehensive_context_analysis(
        article_text=test_article,
        source="Test Source",
        prediction="FAKE",
        confidence=0.85
    )
    
    print("\n=== STRUCTURED OUTPUT PROMPT ===") 
    print(structured_prompt[:500] + "...")
    
    print("\n=== PROMPT VALIDATION ===")
    test_scores = {'bias': 85, 'manipulation': 75, 'credibility': 25, 'risk': 80}
    test_analysis = "This article shows extreme bias and severe manipulation with minimal credibility."
    
    is_consistent = validate_context_analysis_output(test_analysis, test_scores)
    print(f"Score consistency check: {'✓ PASSED' if is_consistent else '✗ FAILED'}")
