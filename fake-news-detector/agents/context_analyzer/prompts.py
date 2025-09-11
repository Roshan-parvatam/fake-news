# agents/context_analyzer/prompts.py

"""
Context Analyzer Prompts Module - Production Ready

Industry-standard prompt templates for context analysis with consistent
numerical scoring, bias detection, manipulation analysis, and framing assessment.
Enhanced with safety filter handling, structured outputs, and comprehensive
error prevention for reliable production use.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .exceptions import (
    PromptGenerationError,
    InputValidationError,
    SafetyFilterError,
    raise_input_validation_error,
    raise_prompt_generation_error,
    raise_safety_filter_error
)


@dataclass
class PromptResponse:
    """Structured response container for prompt outputs with metadata."""
    content: str
    metadata: Dict[str, Any]
    prompt_type: str
    generation_time: float
    safety_level: str = "standard"
    session_id: str = None
    
    def __post_init__(self):
        """Validate prompt response after initialization."""
        if not self.content or len(self.content.strip()) < 50:
            raise PromptGenerationError("Generated prompt content too short or empty")


class BiasDetectionPrompts:
    """
    Production-ready bias detection prompts with safety handling.
    
    Addresses Gemini safety filter blocking issues with careful language
    and structured output formatting for reliable parsing.
    """

    @staticmethod
    def comprehensive_bias_analysis(article_text: str, 
                                  source: str, 
                                  topic_domain: str,
                                  prediction: str, 
                                  confidence: float,
                                  session_id: str = None) -> str:
        """
        Generate comprehensive bias analysis with consistent scoring.
        
        Enhanced to prevent Gemini safety filter blocking while maintaining
        thorough bias detection capabilities.
        """
        logger = logging.getLogger(f"{__name__}.BiasDetectionPrompts")
        
        try:
            # Input validation with detailed error messages
            if not isinstance(article_text, str) or len(article_text.strip()) < 20:
                raise_input_validation_error(
                    "article_text",
                    "Must be non-empty string with at least 20 characters",
                    article_text,
                    session_id=session_id
                )

            # Truncate article text for prompt efficiency
            article_context = article_text[:1200] if len(article_text) > 1200 else article_text

            logger.debug(f"Generating comprehensive bias analysis prompt", 
                        extra={'session_id': session_id, 'domain': topic_domain})

            # Enhanced prompt with safety-conscious language and clear structure
            prompt = f"""You are a professional media analysis specialist conducting systematic bias assessment for journalistic integrity.

ARTICLE CLASSIFICATION: {prediction} (Confidence: {confidence:.1%})
CONTENT DOMAIN: {topic_domain.title()}
SOURCE ATTRIBUTION: {source}

ARTICLE CONTENT FOR ANALYSIS:
{article_context}

COMPREHENSIVE BIAS ASSESSMENT FRAMEWORK:

## Political Perspective Analysis

Analyze the article's political positioning:
- Progressive vs. conservative language patterns
- Policy position framing and emphasis
- Political figure characterization approach
- Ideological terminology and context usage

## Information Selection Analysis

Evaluate content selection patterns:
- Fact inclusion and omission patterns
- Source diversity and representation
- Perspective balance and completeness
- Context provision and background information

## Language Choice Analysis

Assess linguistic presentation:
- Descriptive vs. evaluative language usage
- Emotional vs. neutral terminology selection
- Absolute vs. nuanced statement patterns
- Professional vs. opinionated tone consistency

## Source Attribution Analysis

Review source handling:
- Authority and expertise representation
- Viewpoint diversity in sourcing
- Quote selection and contextualization
- Attribution balance and fairness

REQUIRED STRUCTURED OUTPUT:

### NUMERICAL ASSESSMENTS (0-100 scale):

POLITICAL_BIAS: [0-100 where 0=completely neutral, 100=strongly partisan]
SELECTION_BIAS: [0-100 where 0=comprehensive coverage, 100=highly selective]
LINGUISTIC_BIAS: [0-100 where 0=neutral language, 100=heavily loaded language]
SOURCE_BIAS: [0-100 where 0=balanced sourcing, 100=one-sided sources]
OVERALL_BIAS: [0-100 overall bias intensity]

### DETAILED PROFESSIONAL ANALYSIS:

## Political Assessment Evidence
[Provide specific examples from the article that justify your POLITICAL_BIAS score. Your analysis must align with the numerical score provided.]

## Selection Assessment Evidence  
[Identify content inclusion/omission patterns that justify your SELECTION_BIAS score. Your analysis must match the numerical assessment.]

## Language Assessment Evidence
[Analyze word choice and phrasing that justifies your LINGUISTIC_BIAS score. Your written assessment must correspond to the numerical score.]

## Source Assessment Evidence
[Evaluate source handling that justifies your SOURCE_BIAS score. Your analysis must be consistent with the numerical rating.]

## Overall Professional Conclusion
[Provide summary assessment that precisely matches your OVERALL_BIAS score.]

CONSISTENCY REQUIREMENTS:
1. Numerical scores must exactly correspond to written analysis
2. Use "minimal" for scores 0-25, "moderate" for 26-60, "significant" for 61-85, "extreme" for 86-100
3. Include specific textual evidence for each assessment category
4. Maintain professional analytical tone throughout
5. Ensure all scores reflect the same analytical framework

VERIFICATION: Review all scores against written explanations before completion."""

            logger.info(f"Comprehensive bias analysis prompt generated successfully", 
                       extra={'session_id': session_id, 'prompt_length': len(prompt)})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate comprehensive bias analysis prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Comprehensive bias analysis prompt generation failed: {str(e)}",
                prompt_type="comprehensive_bias_analysis",
                session_id=session_id
            )

    @staticmethod
    def political_bias_assessment(article_text: str, political_context: str, session_id: str = None) -> str:
        """Enhanced political bias detection with safety-conscious language."""
        logger = logging.getLogger(f"{__name__}.BiasDetectionPrompts")
        
        try:
            article_context = article_text[:1000] if len(article_text) > 1000 else article_text

            prompt = f"""Conduct professional political perspective analysis for journalistic assessment.

ARTICLE CONTENT:
{article_context}

POLITICAL CONTEXT: {political_context}

ANALYTICAL FRAMEWORK:

## Progressive Perspective Indicators
- Policy approaches emphasizing social programs and regulation
- Environmental sustainability and climate action emphasis
- Social equity and inclusion language patterns
- Labor and worker protection terminology
- International cooperation and diplomacy focus

## Conservative Perspective Indicators  
- Traditional institution and value emphasis
- Market-based policy approach language
- Individual responsibility and self-reliance themes
- Security and law enforcement priority language
- National sovereignty and independence themes

## Centrist Perspective Indicators
- Balanced policy approach presentation
- Multiple stakeholder consideration
- Pragmatic problem-solving language
- Bipartisan cooperation emphasis
- Moderate reform and compromise themes

REQUIRED OUTPUT FORMAT:

POLITICAL_PERSPECTIVE: [Progressive/Conservative/Centrist/Mixed/Neutral]
INTENSITY_LEVEL: [Subtle/Moderate/Strong/Pronounced]  
CONFIDENCE_ASSESSMENT: [0-100]

SUPPORTING_EVIDENCE:
[List specific examples from the text that support your assessment]

CONTEXTUAL_CONSIDERATIONS:
[Note any factors that influenced your analysis]"""

            logger.info(f"Political bias assessment prompt generated", 
                       extra={'session_id': session_id, 'context': political_context})
            
            return prompt

        except Exception as e:
            logger.error(f"Political bias assessment prompt generation failed: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Political bias assessment prompt failed: {str(e)}",
                prompt_type="political_bias_assessment",
                session_id=session_id
            )


class ManipulationDetectionPrompts:
    """Production-ready manipulation and propaganda detection prompts."""

    @staticmethod
    def emotional_manipulation_analysis(article_text: str, 
                                      emotional_indicators: Dict[str, Any], 
                                      session_id: str = None) -> str:
        """Detect emotional manipulation with safety-conscious language."""
        logger = logging.getLogger(f"{__name__}.ManipulationDetectionPrompts")
        
        try:
            article_context = article_text[:1500] if len(article_text) > 1500 else article_text
            indicators_summary = ', '.join(f"{k}: {v}" for k, v in emotional_indicators.items()) if emotional_indicators else "No specific indicators provided"

            prompt = f"""Analyze communication techniques and emotional appeals in professional content assessment.

ARTICLE CONTENT:
{article_context}

DETECTED_INDICATORS: {indicators_summary}

PROFESSIONAL ASSESSMENT FRAMEWORK:

## Concern-Based Communication Analysis
- Urgency language and time pressure creation  
- Risk communication and safety concern emphasis
- Problem identification and threat awareness
- Crisis framing and emergency situation language

## Motivational Communication Analysis
- Action-oriented language and call-to-action patterns
- Solution presentation and resolution emphasis
- Community engagement and collective response appeals
- Support mobilization and participation encouragement

## Emotional Resonance Analysis  
- Personal relevance and individual impact emphasis
- Shared values and common interest appeals
- Identity-based connection and group belonging themes
- Empathy and compassion activation language

## Persuasive Technique Assessment
- Evidence presentation and support provision
- Authority and expertise citation patterns
- Social proof and consensus indication methods
- Logic and reasoning structure analysis

REQUIRED OUTPUT FORMAT:

MANIPULATION_ASSESSMENT: [0-100 where 0=straightforward communication, 100=heavy manipulation]
PRIMARY_TECHNIQUE: [Concern-Based/Motivational/Emotional/Persuasive]
COMMUNICATION_METHODS: [List specific techniques identified]
PROFESSIONAL_ANALYSIS: [Detailed explanation matching your numerical assessment]
IMPACT_ASSESSMENT: [Low/Medium/High/Critical]"""

            logger.info(f"Emotional manipulation analysis prompt generated", 
                       extra={'session_id': session_id, 'indicators_count': len(emotional_indicators)})
            
            return prompt

        except Exception as e:
            logger.error(f"Emotional manipulation analysis prompt failed: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Emotional manipulation analysis failed: {str(e)}",
                prompt_type="emotional_manipulation_analysis",
                session_id=session_id
            )

    @staticmethod
    def propaganda_technique_detection(article_text: str, session_id: str = None) -> str:
        """Detect propaganda techniques with professional framing."""
        logger = logging.getLogger(f"{__name__}.ManipulationDetectionPrompts")
        
        try:
            article_context = article_text[:1500] if len(article_text) > 1500 else article_text

            prompt = f"""Identify communication influence techniques using professional media analysis standards.

ARTICLE CONTENT:
{article_context}

INFLUENCE TECHNIQUE ASSESSMENT FRAMEWORK:

## Authority Appeal Techniques
- Expert testimony and credential emphasis
- Institutional endorsement and official support
- Celebrity and public figure association
- Professional recommendation and guidance

## Social Consensus Techniques  
- Popular opinion and majority viewpoint emphasis
- Community support and widespread acceptance
- Trend identification and movement participation  
- Peer validation and group conformity appeals

## Emotional Connection Techniques
- Personal story and individual experience sharing
- Value alignment and principle correspondence
- Identity recognition and group belonging
- Shared interest and common goal emphasis

## Information Presentation Techniques
- Selective fact emphasis and highlight patterns
- Comparative analysis and option presentation  
- Timeline and sequence organization methods
- Context provision and background information

PROFESSIONAL OUTPUT FORMAT:

TECHNIQUES_IDENTIFIED: [List each technique found with confidence level]
TECHNIQUE_INTENSITY: [0-100 for overall influence technique usage]
COMMUNICATION_STRATEGY: [Authority-Based/Social-Proof/Emotional/Information-Focused]
PROFESSIONAL_ASSESSMENT: [Detailed analysis of communication approach]
INFLUENCE_EFFECTIVENESS: [Low/Medium/High]"""

            logger.info(f"Propaganda technique detection prompt generated", 
                       extra={'session_id': session_id})
            
            return prompt

        except Exception as e:
            logger.error(f"Propaganda technique detection prompt failed: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Propaganda technique detection failed: {str(e)}",
                prompt_type="propaganda_technique_detection",
                session_id=session_id
            )


class FramingAnalysisPrompts:
    """Professional framing and narrative structure analysis prompts."""

    @staticmethod
    def narrative_framing_analysis(article_text: str, context: Dict[str, Any], session_id: str = None) -> str:
        """Analyze narrative framing with comprehensive structure."""
        logger = logging.getLogger(f"{__name__}.FramingAnalysisPrompts")
        
        try:
            article_context = article_text[:1500] if len(article_text) > 1500 else article_text
            context_summary = json.dumps(context, indent=2) if context else "No specific context provided"

            prompt = f"""Conduct professional narrative structure and framing analysis.

ARTICLE CONTENT:
{article_context}

CONTEXT INFORMATION:
{context_summary}

NARRATIVE ANALYSIS FRAMEWORK:

## Issue Definition Analysis
- Central problem identification and characterization
- Scope definition and boundary establishment  
- Stakeholder identification and role assignment
- Impact assessment and consequence emphasis

## Causal Framework Analysis  
- Responsibility attribution and agency assignment
- Contributing factor identification and weighting
- Historical context inclusion and timeline emphasis
- Systemic vs. individual factor balance

## Solution Approach Analysis
- Resolution method presentation and feasibility
- Implementation pathway description and requirements
- Success metric definition and measurement approach
- Alternative option acknowledgment and comparison

## Stakeholder Characterization Analysis
- Party representation and role description
- Interest alignment and conflict identification  
- Power dynamic presentation and authority recognition
- Voice amplification and perspective prioritization

PROFESSIONAL OUTPUT FORMAT:

FRAMING_APPROACH: [Problem-Focused/Solution-Oriented/Stakeholder-Centered/Process-Focused]
PERSPECTIVE_BALANCE: [0-100 where 0=highly one-sided, 100=comprehensively balanced]
FRAMING_ELEMENTS:
- ISSUE: [How the central matter is characterized]
- CAUSATION: [How responsibility and contributing factors are presented]  
- RESOLUTION: [How solutions and pathways forward are framed]
- STAKEHOLDERS: [How different parties and interests are characterized]

IMPACT_ASSESSMENT: [Analysis of how framing influences reader understanding]
ALTERNATIVE_APPROACHES: [Other ways this topic could be presented]"""

            logger.info(f"Narrative framing analysis prompt generated", 
                       extra={'session_id': session_id, 'context_provided': bool(context)})
            
            return prompt

        except Exception as e:
            logger.error(f"Narrative framing analysis prompt failed: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Narrative framing analysis failed: {str(e)}",
                prompt_type="narrative_framing_analysis",
                session_id=session_id
            )


class StructuredOutputPrompts:
    """Prompts that enforce structured JSON output for programmatic processing."""

    @staticmethod
    def comprehensive_context_analysis(article_text: str, 
                                     source: str, 
                                     prediction: str, 
                                     confidence: float,
                                     session_id: str = None) -> str:
        """Generate complete context analysis with consistent scoring."""
        logger = logging.getLogger(f"{__name__}.StructuredOutputPrompts")
        
        try:
            article_context = article_text[:2000] if len(article_text) > 2000 else article_text

            prompt = f"""Conduct comprehensive professional content analysis with structured scoring.

ARTICLE CONTENT:
{article_context}

SOURCE: {source}
CLASSIFICATION: {prediction} ({confidence:.1%} confidence)

COMPLETE PROFESSIONAL ANALYSIS REQUIRED:

## Content Quality Assessment
Analyze information presentation quality, source citation, factual accuracy indicators, and professional standards compliance.

## Perspective Balance Assessment  
Evaluate viewpoint representation, stakeholder inclusion, alternative perspective acknowledgment, and comprehensive coverage.

## Communication Approach Assessment
Review language choice, tone consistency, professional presentation, and audience consideration.

## Information Reliability Assessment
Assess source credibility, evidence quality, verification indicators, and trustworthiness markers.

MANDATORY STRUCTURED OUTPUT FORMAT:

{{
  "assessment_scores": {{
    "bias": [0-100],
    "manipulation": [0-100], 
    "credibility": [0-100],
    "risk": [0-100]
  }},
  "detailed_analysis": {{
    "bias_explanation": "Professional assessment justifying bias score",
    "manipulation_explanation": "Communication technique analysis justifying manipulation score", 
    "credibility_explanation": "Information reliability assessment justifying credibility score",
    "risk_explanation": "Overall risk evaluation justifying risk score"
  }},
  "identified_techniques": [
    "List specific communication techniques and presentation methods identified"
  ],
  "risk_classification": "LOW/MEDIUM/HIGH/CRITICAL",
  "professional_recommendation": "Clear guidance for information consumers and media professionals"
}}

CRITICAL SCORING CONSISTENCY RULES:
- Scores 0-25: Use terms like "minimal", "low", "limited", "slight"  
- Scores 26-50: Use terms like "moderate", "some", "noticeable", "present"
- Scores 51-75: Use terms like "significant", "considerable", "substantial", "pronounced"
- Scores 76-100: Use terms like "extreme", "severe", "extensive", "pervasive"

Your explanations must use language that precisely corresponds to your numerical scores."""

            logger.info(f"Comprehensive context analysis prompt generated", 
                       extra={'session_id': session_id, 'article_length': len(article_text)})
            
            return prompt

        except Exception as e:
            logger.error(f"Comprehensive context analysis prompt failed: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Comprehensive context analysis failed: {str(e)}",
                prompt_type="comprehensive_context_analysis",
                session_id=session_id
            )


class DomainSpecificPrompts:
    """Domain-specific prompts for different content types with professional guidance."""

    # Enhanced domain-specific guidance with professional standards
    DOMAIN_GUIDANCE = {
        'health': """
HEALTH DOMAIN PROFESSIONAL STANDARDS:

Content Assessment Focus:
- Medical accuracy and evidence-based information presentation
- Professional medical source citation and authority verification  
- Risk communication appropriateness and safety consideration
- Patient-centered language and accessibility assessment
- Clinical research citation and methodology acknowledgment

Professional Verification Approach:
- Peer-reviewed medical literature cross-reference
- Medical professional authority and credential verification
- Clinical trial registration and methodology validation
- Health organization endorsement and guideline alignment
- Medical ethics and patient safety consideration

Quality Indicators:
- Evidence-based claims with appropriate medical source citation
- Professional medical terminology usage and explanation quality
- Risk-benefit communication balance and clarity
- Patient decision-making support and information adequacy
- Medical professional oversight and clinical validation
""",

        'politics': """
POLITICAL DOMAIN PROFESSIONAL STANDARDS:

Content Assessment Focus:
- Political position representation and viewpoint balance
- Policy analysis depth and stakeholder consideration
- Election information accuracy and process explanation
- Political figure characterization and biographical accuracy
- Governmental process explanation and civic education quality

Professional Verification Approach:
- Official government source verification and document citation
- Policy position accuracy through official statement review
- Electoral process information through authoritative source validation
- Political claim verification through multiple independent sources
- Civic information accuracy through educational institution validation

Quality Indicators:
- Multiple political perspective inclusion and fair representation
- Policy impact analysis with stakeholder consideration
- Electoral process explanation with accuracy and clarity
- Political figure representation with factual accuracy
- Governmental function explanation with educational value
""",

        'science': """
SCIENCE DOMAIN PROFESSIONAL STANDARDS:

Content Assessment Focus:
- Scientific methodology explanation and research process description
- Peer review status and publication venue assessment
- Research finding interpretation and limitation acknowledgment  
- Scientific consensus representation and uncertainty communication
- Technical accuracy and accessibility balance

Professional Verification Approach:
- Peer-reviewed publication verification and impact assessment
- Research methodology evaluation and replication consideration
- Scientific institution affiliation and credibility verification
- Expert consensus assessment and disagreement acknowledgment
- Data interpretation accuracy and statistical validity review

Quality Indicators:
- Research methodology transparency and appropriate limitation discussion
- Scientific uncertainty communication and confidence level indication
- Peer review status acknowledgment and publication quality assessment
- Expert consensus representation with minority viewpoint acknowledgment
- Technical accuracy with appropriate accessibility and explanation quality
""",

        'technology': """
TECHNOLOGY DOMAIN PROFESSIONAL STANDARDS:

Content Assessment Focus:  
- Technical accuracy and specification verification
- Implementation feasibility and practical consideration
- Privacy and security implication assessment
- User impact analysis and accessibility consideration
- Innovation context and comparative analysis quality

Professional Verification Approach:
- Technical documentation review and specification validation
- Industry expert consultation and professional assessment
- Security analysis through established framework application
- User experience research and practical testing consideration
- Market analysis through reliable data source verification

Quality Indicators:
- Technical specification accuracy with appropriate detail level
- Security and privacy consideration with risk assessment
- User impact analysis with accessibility and usability focus
- Implementation guidance with practical consideration
- Innovation assessment with appropriate context and comparison
"""
    }

    @classmethod
    def get_domain_guidance(cls, domain: str) -> str:
        """Get professional guidance for specific domain analysis."""
        return cls.DOMAIN_GUIDANCE.get(domain, """
GENERAL DOMAIN PROFESSIONAL STANDARDS:

Apply standard professional analysis principles:
- Multiple independent source verification and cross-reference
- Expert authority validation and credential assessment
- Information accuracy verification through reliable source consultation
- Comprehensive perspective inclusion and bias identification
- Professional ethics and transparency standard maintenance
""")


class SafetyEnhancedPrompts:
    """Safety-enhanced prompts designed to prevent content filter issues."""

    @staticmethod
    def institutional_fallback_analysis(article_summary: str, 
                                      domain: str,
                                      session_id: str = None) -> str:
        """Generate institutional analysis when AI analysis faces restrictions."""
        logger = logging.getLogger(f"{__name__}.SafetyEnhancedPrompts")
        
        try:
            logger.info(f"Generating institutional fallback analysis for domain: {domain}", 
                       extra={'session_id': session_id})

            prompt = f"""Conduct professional institutional content assessment for media analysis standards.

CONTENT SUMMARY: {article_summary[:800]}
SUBJECT DOMAIN: {domain.title()}

INSTITUTIONAL ASSESSMENT FRAMEWORK:

## Professional Media Standards Review
Evaluate content against established journalistic integrity standards, information accuracy requirements, source attribution practices, and professional presentation guidelines.

## Educational Content Value Assessment  
Analyze information educational merit, knowledge contribution, learning objective support, and public information service quality.

## Communication Effectiveness Review
Assess message clarity, audience consideration, accessibility standards, and professional communication practice adherence.

## Information Quality Framework
Review factual accuracy indicators, source reliability assessment, verification standard compliance, and professional validation methods.

STRUCTURED PROFESSIONAL OUTPUT:

CONTENT_QUALITY_ASSESSMENT: [0-100 professional quality rating]
EDUCATIONAL_VALUE: [0-100 information service rating]  
COMMUNICATION_EFFECTIVENESS: [0-100 presentation quality rating]
OVERALL_PROFESSIONAL_RATING: [0-100 comprehensive assessment]

PROFESSIONAL_ANALYSIS:
- Content Quality: [Detailed assessment justifying quality rating]
- Educational Merit: [Information value analysis justifying educational rating]
- Communication Approach: [Presentation analysis justifying effectiveness rating]
- Overall Assessment: [Comprehensive evaluation justifying overall rating]

INSTITUTIONAL_RECOMMENDATION:
[Professional guidance for media professionals and information consumers]"""

            logger.info(f"Institutional fallback analysis prompt generated", 
                       extra={'session_id': session_id, 'domain': domain})
            
            return prompt

        except Exception as e:
            logger.error(f"Institutional fallback analysis prompt failed: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Institutional fallback analysis failed: {str(e)}",
                prompt_type="institutional_fallback_analysis",
                session_id=session_id
            )


# Main prompt template access function with enhanced error handling
def get_context_prompt_template(prompt_type: str, 
                               session_id: str = None,
                               **kwargs) -> str:
    """
    Get specific context analysis prompt template with production error handling.

    Args:
        prompt_type: Type of prompt needed
        session_id: Optional session ID for tracking
        **kwargs: Parameters for prompt formatting

    Returns:
        Formatted prompt string

    Raises:
        PromptGenerationError: If prompt generation fails
        InputValidationError: If invalid prompt type provided
    """
    logger = logging.getLogger(f"{__name__}.get_context_prompt_template")
    start_time = time.time()
    
    try:
        # Enhanced prompt mapping with error handling
        prompt_mapping = {
            'comprehensive_analysis': StructuredOutputPrompts.comprehensive_context_analysis,
            'bias_detection': BiasDetectionPrompts.comprehensive_bias_analysis,
            'political_bias': BiasDetectionPrompts.political_bias_assessment,
            'emotional_manipulation': ManipulationDetectionPrompts.emotional_manipulation_analysis,
            'propaganda_detection': ManipulationDetectionPrompts.propaganda_technique_detection,
            'framing_analysis': FramingAnalysisPrompts.narrative_framing_analysis,
            'institutional_fallback': SafetyEnhancedPrompts.institutional_fallback_analysis,
        }

        if prompt_type not in prompt_mapping:
            available_types = list(prompt_mapping.keys())
            raise_input_validation_error(
                "prompt_type",
                f"Unknown prompt type '{prompt_type}'. Available: {available_types}",
                prompt_type,
                session_id=session_id
            )

        logger.debug(f"Generating prompt of type: {prompt_type}", 
                    extra={'session_id': session_id, 'kwargs_count': len(kwargs)})

        # Generate prompt with error handling
        prompt_function = prompt_mapping[prompt_type]
        prompt = prompt_function(session_id=session_id, **kwargs)

        generation_time = time.time() - start_time
        
        # Validate generated prompt
        if not prompt or len(prompt.strip()) < 100:
            raise PromptGenerationError(
                f"Generated prompt too short or empty for type: {prompt_type}",
                prompt_type=prompt_type,
                session_id=session_id
            )

        logger.info(f"Prompt generated successfully", 
                   extra={
                       'session_id': session_id,
                       'prompt_type': prompt_type,
                       'generation_time': round(generation_time * 1000, 2),
                       'prompt_length': len(prompt)
                   })

        return prompt

    except (InputValidationError, PromptGenerationError):
        # Re-raise validation and generation errors
        raise
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"Unexpected error generating prompt: {str(e)}", 
                    extra={
                        'session_id': session_id,
                        'prompt_type': prompt_type,
                        'generation_time': round(generation_time * 1000, 2),
                        'error_type': type(e).__name__
                    })
        raise PromptGenerationError(
            f"Prompt generation failed for {prompt_type}: {str(e)}",
            prompt_type=prompt_type,
            session_id=session_id
        )


def validate_context_analysis_output(analysis_text: str, scores: Dict[str, int]) -> bool:
    """
    Validate context analysis output for consistency between text and scores.

    Args:
        analysis_text: Generated analysis text
        scores: Dictionary of numerical scores

    Returns:
        True if consistent, False if inconsistent
    """
    if not isinstance(analysis_text, str) or not analysis_text.strip():
        return False
    
    if not isinstance(scores, dict) or not scores:
        return False

    text_lower = analysis_text.lower()
    
    # Check each score for basic consistency
    for score_type, score_value in scores.items():
        if not isinstance(score_value, (int, float)) or not (0 <= score_value <= 100):
            return False
            
        # Check for obvious inconsistencies
        if score_value <= 25:  # Low scores
            if any(phrase in text_lower for phrase in ['high', 'significant', 'extreme', 'severe']):
                if score_type in text_lower or f'{score_type.replace("_", " ")}' in text_lower:
                    return False
        elif score_value >= 75:  # High scores
            if any(phrase in text_lower for phrase in ['minimal', 'low', 'slight', 'neutral']):
                if score_type in text_lower or f'{score_type.replace("_", " ")}' in text_lower:
                    return False
    
    return True


def get_domain_guidance(domain: str) -> str:
    """
    Get domain-specific professional guidance for analysis.

    Args:
        domain: Domain type (health, politics, science, technology, etc.)

    Returns:
        Professional guidance text with analysis standards
    """
    return DomainSpecificPrompts.get_domain_guidance(domain)


# Performance and monitoring helpers
def get_prompt_statistics() -> Dict[str, Any]:
    """Get comprehensive prompt system statistics for monitoring."""
    return {
        'available_prompt_types': [
            'comprehensive_analysis',
            'bias_detection', 
            'political_bias',
            'emotional_manipulation',
            'propaganda_detection',
            'framing_analysis',
            'institutional_fallback'
        ],
        'domain_guidance_available': list(DomainSpecificPrompts.DOMAIN_GUIDANCE.keys()),
        'safety_features': {
            'safety_conscious_language': True,
            'institutional_fallback': True,
            'structured_output_enforcement': True,
            'consistency_validation': True
        },
        'prompt_capabilities': [
            'Professional media analysis standards',
            'Safety filter avoidance with institutional language',
            'Consistent numerical scoring enforcement',
            'Domain-specific analysis guidance',
            'Structured JSON output formatting',
            'Session tracking and error handling'
        ]
    }


# Testing functionality
if __name__ == "__main__":
    """Test context analyzer prompts with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    test_session_id = "prompt_test_context_001"
    
    print("=== CONTEXT ANALYZER PROMPTS TEST ===")
    
    try:
        # Test comprehensive analysis prompt
        print("--- Comprehensive Analysis Prompt Test ---")
        comprehensive_prompt = get_context_prompt_template(
            'comprehensive_analysis',
            session_id=test_session_id,
            article_text="Recent studies show promising developments in renewable energy technology with significant policy implications for environmental sustainability.",
            source="Science Daily",
            prediction="REAL",
            confidence=0.87
        )
        
        print(f"✅ Comprehensive analysis prompt generated: {len(comprehensive_prompt)} characters")
        print(f"Preview: {comprehensive_prompt[:200]}...")

        # Test bias detection prompt
        print("\n--- Bias Detection Prompt Test ---")
        bias_prompt = get_context_prompt_template(
            'bias_detection',
            session_id=test_session_id,
            article_text="Political party leadership announces new policy initiative with bipartisan support mechanisms.",
            source="Political News",
            topic_domain="politics",
            prediction="REAL", 
            confidence=0.75
        )
        
        print(f"✅ Bias detection prompt generated: {len(bias_prompt)} characters")

        # Test manipulation detection prompt
        print("\n--- Manipulation Detection Prompt Test ---")
        manipulation_prompt = get_context_prompt_template(
            'emotional_manipulation',
            session_id=test_session_id,
            article_text="Community leaders work together to address local concerns through collaborative problem-solving approaches.",
            emotional_indicators={'concern': 0.6, 'motivation': 0.4}
        )
        
        print(f"✅ Manipulation detection prompt generated: {len(manipulation_prompt)} characters")

        # Test framing analysis prompt
        print("\n--- Framing Analysis Prompt Test ---")
        framing_prompt = get_context_prompt_template(
            'framing_analysis',
            session_id=test_session_id,
            article_text="Economic policy changes aim to balance growth objectives with social considerations.",
            context={'domain': 'economics', 'scope': 'policy_analysis'}
        )
        
        print(f"✅ Framing analysis prompt generated: {len(framing_prompt)} characters")

        # Test institutional fallback prompt
        print("\n--- Institutional Fallback Prompt Test ---")
        fallback_prompt = get_context_prompt_template(
            'institutional_fallback',
            session_id=test_session_id,
            article_summary="Technology developments in healthcare show promise for improved patient outcomes",
            domain="health"
        )
        
        print(f"✅ Institutional fallback prompt generated: {len(fallback_prompt)} characters")

        # Test domain guidance
        print("\n--- Domain Guidance Test ---")
        health_guidance = get_domain_guidance('health')
        print(f"✅ Health domain guidance: {len(health_guidance)} characters")
        
        politics_guidance = get_domain_guidance('politics')
        print(f"✅ Politics domain guidance: {len(politics_guidance)} characters")

        # Test consistency validation
        print("\n--- Consistency Validation Test ---")
        
        # Test consistent analysis
        consistent_analysis = "This article demonstrates minimal bias with professional presentation and high credibility."
        consistent_scores = {'bias': 20, 'manipulation': 15, 'credibility': 85, 'risk': 25}
        
        is_consistent = validate_context_analysis_output(consistent_analysis, consistent_scores)
        print(f"Consistent analysis validation: {'✅ PASSED' if is_consistent else '❌ FAILED'}")
        
        # Test inconsistent analysis (should fail)
        inconsistent_analysis = "This article shows extreme bias and severe manipulation with minimal credibility."
        inconsistent_scores = {'bias': 15, 'manipulation': 10, 'credibility': 90, 'risk': 5}
        
        is_inconsistent = validate_context_analysis_output(inconsistent_analysis, inconsistent_scores)
        print(f"Inconsistent analysis detection: {'✅ PASSED' if not is_inconsistent else '❌ FAILED'}")

        # Test error handling
        print("\n--- Error Handling Test ---")
        try:
            invalid_prompt = get_context_prompt_template('invalid_type', session_id=test_session_id)
            print("❌ Should have failed with invalid prompt type")
        except InputValidationError:
            print("✅ Invalid prompt type properly rejected")

        # Test prompt statistics
        print("\n--- Prompt Statistics Test ---")
        stats = get_prompt_statistics()
        print(f"Available prompt types: {len(stats['available_prompt_types'])}")
        print(f"Domain guidance available: {len(stats['domain_guidance_available'])}")
        print(f"Safety features: {stats['safety_features']}")

        print("\n✅ Context analyzer prompt tests completed successfully!")

    except Exception as e:
        print(f"❌ Prompt test failed: {str(e)}")
        import traceback
        traceback.print_exc()
