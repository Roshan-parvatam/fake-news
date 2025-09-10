# agents/llm_explanation/prompts.py

"""
LLM Explanation Agent Prompts

Production-ready prompt templates for generating human-readable explanations
of fake news detection results. Provides structured prompts for different
analysis depths and explanation types with professional formatting.
"""

from typing import Dict, Any, Optional


class ExplanationPrompts:
    """
    Core prompt templates for LLM explanation generation.
    
    Provides professional, structured prompts for explaining fake news
    detection results with varying levels of detail and analysis depth.
    """

    @staticmethod
    def main_explanation_prompt(article_text: str, prediction: str, confidence: float,
                              source: str, date: str, subject: str) -> str:
        """
        Generate main explanation prompt for article analysis.
        
        Args:
            article_text: Article content to analyze
            prediction: REAL or FAKE classification
            confidence: Confidence score (0.0-1.0)
            source: Article source
            date: Publication date
            subject: Article subject/domain
            
        Returns:
            Formatted prompt for main explanation generation
        """
        return f"""You are an expert fact-checking analyst providing clear, accessible explanations of news article credibility assessments.

ARTICLE ANALYSIS REQUEST:

Article Text:
{article_text}

Classification Result: {prediction}
Confidence Level: {confidence:.1%}
Source: {source}
Publication Date: {date}
Subject Category: {subject}

EXPLANATION REQUIREMENTS:

Provide a comprehensive but accessible explanation covering:

1. **Classification Summary**
   - Clear statement of the credibility assessment
   - Key factors that led to this classification
   - Overall confidence level and what it means

2. **Supporting Evidence**
   - Specific elements that support the classification
   - Language patterns, source credibility, or factual inconsistencies
   - Corroborating or contradicting information

3. **Credibility Assessment**
   - Source reliability evaluation
   - Cross-reference with known facts or reliable sources
   - Identification of potential red flags or verification markers

4. **Limitations and Uncertainties**
   - Areas where confidence might be lower
   - Information that would strengthen the assessment
   - Alternative interpretations to consider

5. **Verification Recommendations**
   - Specific steps readers can take to verify claims
   - Reliable sources to consult for fact-checking
   - Warning signs to watch for in similar content

Format your response in clear, conversational language that non-experts can understand while maintaining analytical rigor."""

    @staticmethod
    def detailed_analysis_prompt(article_text: str, prediction: str, 
                               confidence: float, metadata: Dict[str, Any]) -> str:
        """
        Generate detailed forensic analysis prompt.
        
        Args:
            article_text: Article content
            prediction: Classification result
            confidence: Confidence score
            metadata: Additional context and metadata
            
        Returns:
            Formatted prompt for detailed analysis generation
        """
        metadata_str = str(metadata) if metadata else "No additional metadata available"
        
        return f"""You are a forensic news analyst conducting an in-depth investigation of article credibility and misinformation patterns.

FORENSIC ANALYSIS REQUEST:

Article Content:
{article_text}

Initial Assessment: {prediction}
Confidence Score: {confidence:.1%}
Additional Context: {metadata_str}

DETAILED ANALYSIS FRAMEWORK:

Conduct a comprehensive forensic examination covering:

## 1. Content Analysis
- **Claim Verification**: Fact-check specific assertions and statistics
- **Source Attribution**: Evaluate the credibility of cited sources and experts
- **Evidence Quality**: Assess the strength and reliability of supporting evidence
- **Logical Consistency**: Identify logical fallacies or inconsistent reasoning

## 2. Linguistic and Stylistic Analysis
- **Language Patterns**: Examine word choice, emotional appeals, and rhetorical devices
- **Writing Quality**: Assess grammar, style, and professional presentation
- **Sensationalism Indicators**: Identify clickbait, hyperbole, or inflammatory language
- **Bias Detection**: Analyze potential political, commercial, or ideological bias

## 3. Contextual Investigation
- **Timing Analysis**: Consider publication timing relative to related events
- **Distribution Patterns**: Examine how and where the content has spread
- **Similar Content**: Identify patterns with known misinformation campaigns
- **Motivation Assessment**: Analyze potential reasons for content creation

## 4. Technical Verification
- **Media Authenticity**: Assess any images, videos, or documents for manipulation
- **Link Analysis**: Verify external links and reference materials
- **Metadata Examination**: Review publication details and technical indicators
- **Cross-Platform Verification**: Check consistency across different platforms

## 5. Risk Assessment
- **Harm Potential**: Evaluate potential impact if false information spreads
- **Vulnerability Analysis**: Identify demographics most likely to be misled
- **Correction Difficulty**: Assess how hard misinformation would be to counter
- **Amplification Risk**: Predict likelihood of viral spread

## 6. Confidence Calibration
- **Certainty Factors**: Elements that increase confidence in the assessment
- **Uncertainty Areas**: Aspects requiring further investigation
- **Alternative Scenarios**: Plausible alternative interpretations
- **Verification Gaps**: Missing information that would strengthen analysis

Provide detailed findings with specific examples and evidence. Include recommendations for additional verification steps and monitoring."""

    @staticmethod
    def confidence_analysis_prompt(article_text: str, prediction: str, confidence: float) -> str:
        """
        Generate confidence level analysis prompt.
        
        Args:
            article_text: Article content
            prediction: Classification result
            confidence: Confidence score to analyze
            
        Returns:
            Formatted prompt for confidence analysis generation
        """
        return f"""You are an AI assessment specialist evaluating the appropriateness and reliability of confidence levels in automated news credibility analysis.

CONFIDENCE EVALUATION REQUEST:

Article Text:
{article_text}

AI Classification: {prediction}
Reported Confidence: {confidence:.1%}

CONFIDENCE ANALYSIS FRAMEWORK:

Provide a thorough evaluation of the confidence level appropriateness:

## 1. Confidence Appropriateness Assessment
- **Justification Analysis**: Does the confidence level match the strength of available evidence?
- **Comparative Evaluation**: How does this confidence compare to similar cases?
- **Calibration Check**: Are there signs of overconfidence or underconfidence?
- **Consistency Review**: Is the confidence level internally consistent with the classification?

## 2. Supporting Evidence Strength
- **Clear Indicators**: Strong evidence that supports high confidence
- **Ambiguous Elements**: Factors that introduce uncertainty
- **Missing Information**: Gaps that prevent higher confidence
- **Contradictory Signals**: Evidence that might support alternative classifications

## 3. Uncertainty Factors
- **Source Ambiguity**: Unclear or unverifiable source information
- **Content Complexity**: Nuanced claims requiring expert knowledge
- **Context Dependencies**: Factors that might change interpretation
- **Time Sensitivity**: Information that might become outdated

## 4. Risk Analysis
- **False Positive Risk**: Probability of incorrectly labeling true content as false
- **False Negative Risk**: Probability of missing actual misinformation
- **Impact Consideration**: Consequences of potential misclassification
- **Threshold Evaluation**: Whether confidence meets decision thresholds

## 5. Improvement Recommendations
- **Additional Verification**: Steps that could increase confidence
- **Expert Consultation**: Areas requiring human expert review
- **Data Requirements**: Additional information needed for better assessment
- **Model Limitations**: Known constraints affecting confidence

## 6. Practical Implications
- **User Guidance**: How users should interpret this confidence level
- **Decision Support**: Whether confidence supports automated vs. manual review
- **Communication Strategy**: How to effectively convey uncertainty to end users
- **Monitoring Needs**: Indicators for ongoing assessment quality

Conclude with specific recommendations for confidence interpretation and any suggested threshold adjustments."""

    @staticmethod
    def source_assessment_prompt(source: str, article_context: str) -> str:
        """
        Generate source reliability assessment prompt.
        
        Args:
            source: Source name or URL
            article_context: Context about the article
            
        Returns:
            Formatted prompt for source assessment
        """
        return f"""You are a media credibility specialist evaluating news source reliability and trustworthiness.

SOURCE EVALUATION REQUEST:

Source: {source}
Article Context: {article_context}

COMPREHENSIVE SOURCE ASSESSMENT:

## 1. Source Identification and Classification
- **Media Type**: News outlet, blog, social media, academic, government, etc.
- **Ownership Structure**: Corporate ownership, funding sources, independence level
- **Established Reputation**: Historical track record and industry standing
- **Editorial Standards**: Evidence of fact-checking, correction policies, editorial oversight

## 2. Credibility Indicators
- **Transparency**: Clear authorship, contact information, publication standards
- **Verification Practices**: Evidence of source verification and fact-checking
- **Correction History**: How the source handles errors and corrections
- **Expert Recognition**: Citations by other credible sources and institutions

## 3. Bias and Reliability Assessment
- **Political Bias**: Evidence of partisan lean or balanced reporting
- **Commercial Interests**: Potential conflicts of interest or sponsored content
- **Accuracy Record**: History of factual reporting vs. misinformation
- **Sensationalism Level**: Tendency toward clickbait or inflammatory content

## 4. Technical and Operational Factors
- **Website Quality**: Professional presentation, security, technical standards
- **Social Media Presence**: Engagement patterns, follower authenticity
- **Content Consistency**: Quality and reliability across different topics
- **Update Frequency**: Timeliness and regular content publication

## 5. Contextual Evaluation
- **Subject Matter Expertise**: Specialization relevant to the article topic
- **Geographic Relevance**: Local knowledge and access for location-specific stories
- **Historical Context**: Performance during major news events
- **Peer Recognition**: Citations and references by other reputable sources

## 6. Red Flags and Warning Signs
- **Anonymity Issues**: Lack of clear authorship or contact information
- **Extreme Claims**: Tendency to publish sensational or unverified claims
- **Isolation Patterns**: Claims not reported by other credible sources
- **Technical Issues**: Poor website quality, suspicious domain patterns

Provide a clear reliability rating (HIGH/MEDIUM/LOW) with detailed justification and recommendations for readers."""


class AdaptivePrompts:
    """
    Adaptive prompts that adjust based on content type and analysis needs.
    """

    @staticmethod
    def get_domain_specific_prompt(domain: str, base_prompt: str) -> str:
        """
        Adapt base prompt for specific domains.
        
        Args:
            domain: Content domain (health, politics, science, etc.)
            base_prompt: Base prompt to adapt
            
        Returns:
            Domain-adapted prompt
        """
        domain_additions = {
            'health': "\n\nIMPORTANT: Pay special attention to medical claims, treatment recommendations, and health statistics. Verify against established medical sources and highlight any potentially dangerous health misinformation.",
            'politics': "\n\nIMPORTANT: Consider political bias, election-related claims, and policy statements. Cross-reference with official government sources and fact-checking organizations.",
            'science': "\n\nIMPORTANT: Evaluate scientific claims, research citations, and statistical interpretations. Verify against peer-reviewed sources and established scientific consensus.",
            'economics': "\n\nIMPORTANT: Assess financial claims, market predictions, and economic statistics. Cross-reference with official economic data and financial institutions.",
            'technology': "\n\nIMPORTANT: Evaluate technical claims, cybersecurity information, and technology trends. Verify against established technology sources and expert analysis."
        }
        
        addition = domain_additions.get(domain.lower(), "")
        return base_prompt + addition

    @staticmethod
    def get_confidence_adjusted_prompt(confidence: float, base_prompt: str) -> str:
        """
        Adjust prompt based on confidence level.
        
        Args:
            confidence: Confidence score (0.0-1.0)
            base_prompt: Base prompt to adjust
            
        Returns:
            Confidence-adjusted prompt
        """
        if confidence < 0.6:
            addition = "\n\nNOTE: Given the lower confidence level, emphasize uncertainties and alternative interpretations. Recommend additional verification steps."
        elif confidence > 0.9:
            addition = "\n\nNOTE: Given the high confidence level, explain why the assessment is particularly strong but still acknowledge any limitations."
        else:
            addition = "\n\nNOTE: Provide balanced analysis acknowledging both supporting evidence and areas of uncertainty."
        
        return base_prompt + addition


def get_explanation_prompt(prompt_type: str, **kwargs) -> str:
    """
    Get specific explanation prompt by type.
    
    Args:
        prompt_type: Type of prompt ('main', 'detailed', 'confidence', 'source')
        **kwargs: Parameters for prompt formatting
        
    Returns:
        Formatted prompt string
    """
    if prompt_type == 'main':
        return ExplanationPrompts.main_explanation_prompt(**kwargs)
    elif prompt_type == 'detailed':
        return ExplanationPrompts.detailed_analysis_prompt(**kwargs)
    elif prompt_type == 'confidence':
        return ExplanationPrompts.confidence_analysis_prompt(**kwargs)
    elif prompt_type == 'source':
        return ExplanationPrompts.source_assessment_prompt(**kwargs)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def validate_prompt_parameters(prompt_type: str, **kwargs) -> tuple[bool, str]:
    """
    Validate that required parameters are provided for prompt type.
    
    Args:
        prompt_type: Type of prompt to validate
        **kwargs: Parameters to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_params = {
        'main': ['article_text', 'prediction', 'confidence', 'source', 'date', 'subject'],
        'detailed': ['article_text', 'prediction', 'confidence', 'metadata'],
        'confidence': ['article_text', 'prediction', 'confidence'],
        'source': ['source', 'article_context']
    }
    
    if prompt_type not in required_params:
        return False, f"Unknown prompt type: {prompt_type}"
    
    missing_params = [param for param in required_params[prompt_type] if param not in kwargs]
    
    if missing_params:
        return False, f"Missing required parameters: {missing_params}"
    
    return True, ""


# Testing functionality
if __name__ == "__main__":
    """Test explanation prompt functionality."""
    
    # Test main explanation prompt
    test_prompt = ExplanationPrompts.main_explanation_prompt(
        article_text="A new study claims that drinking 10 cups of coffee daily can extend lifespan by 50 years.",
        prediction="FAKE",
        confidence=0.89,
        source="HealthBlog.net",
        date="2025-01-15",
        subject="Health"
    )
    
    print("=== MAIN EXPLANATION PROMPT TEST ===")
    print(test_prompt[:300] + "...")
    
    # Test validation
    is_valid, error = validate_prompt_parameters(
        'main',
        article_text="test",
        prediction="FAKE",
        confidence=0.8,
        source="test",
        date="2025-01-01",
        subject="test"
    )
    
    print(f"\n=== VALIDATION TEST ===")
    print(f"Validation result: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    if not is_valid:
        print(f"Error: {error}")
    
    # Test adaptive prompts
    adapted_prompt = AdaptivePrompts.get_domain_specific_prompt("health", "Base prompt text")
    print(f"\n=== ADAPTIVE PROMPT TEST ===")
    print("Domain adaptation working:", "medical claims" in adapted_prompt)
    
    print("\n=== LLM EXPLANATION PROMPTS TESTING COMPLETED ===")
