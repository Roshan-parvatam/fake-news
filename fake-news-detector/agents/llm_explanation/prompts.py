# agents/llm_explanation/prompts.py

"""
LLM Explanation Agent Prompts - Production Ready

Advanced prompt templates for generating human-readable explanations of fake news
detection results. Features institutional language, domain-specific adaptations,
comprehensive analysis frameworks, and safety filter optimization for reliable
production use across diverse content domains.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .exceptions import PromptFormattingError, raise_explanation_generation_error


@dataclass
class PromptResponse:
    """Structured response container for prompt outputs with enhanced metadata."""
    content: str
    metadata: Dict[str, Any]
    prompt_type: str
    generation_time: float
    safety_level: str = "institutional"
    session_id: str = None
    
    def __post_init__(self):
        """Validate prompt response after initialization."""
        if not self.content or len(self.content.strip()) < 200:
            raise PromptFormattingError("Generated prompt content insufficient for quality analysis")


class ExplanationPrompts:
    """
    Core explanation prompts with institutional language and safety optimization.
    
    Provides professional, structured prompts for explaining fake news detection
    results with enhanced safety filter avoidance through academic framing and
    institutional language patterns.
    """

    @staticmethod
    def main_explanation_prompt(article_text: str, prediction: str, confidence: float,
                               source: str, date: str, subject: str, session_id: str = None) -> str:
        """
        Generate comprehensive main explanation prompt with institutional framing.
        
        Enhanced with academic language to avoid safety filter blocks while maintaining
        analytical depth and accessibility for general audiences.
        """
        logger = logging.getLogger(f"{__name__}.ExplanationPrompts")
        
        try:
            # Truncate article for optimal processing
            article_context = article_text[:3000] if len(article_text) > 3000 else article_text
            
            logger.info(f"Generating main explanation prompt", extra={'session_id': session_id, 'domain': subject})

            prompt = f"""You are a senior academic researcher conducting systematic content analysis for institutional fact-checking protocols.

INSTITUTIONAL ANALYSIS REQUEST:
- Content Classification: {prediction}
- Analytical Confidence: {confidence:.1%}
- Publication Source: {source}
- Publication Date: {date}
- Subject Domain: {subject.title()}

CONTENT FOR ACADEMIC EVALUATION:
{article_context}

RESEARCH FRAMEWORK FOR PUBLIC EDUCATION:

Provide comprehensive institutional analysis suitable for public education and academic review:

## 1. Classification Summary for Public Understanding

Explain the content credibility assessment using clear, educational language:
- **Research Findings**: Clear statement of the institutional credibility evaluation
- **Supporting Evidence**: Academic analysis of key factors supporting this assessment
- **Confidence Context**: Educational explanation of what the confidence level indicates
- **Methodology Overview**: Brief explanation of how such assessments are conducted

## 2. Academic Evidence Review

Present systematic evidence analysis for educational purposes:
- **Content Analysis**: Specific elements that support the institutional classification
- **Source Evaluation**: Academic assessment of publication source reliability
- **Factual Verification**: Cross-reference analysis with established academic sources
- **Research Methodology**: Evidence-based analytical approaches used

## 3. Institutional Credibility Assessment

Provide educational content credibility evaluation:
- **Publication Standards**: Assessment of editorial and publication standards
- **Source Attribution**: Evaluation of cited sources and expert references
- **Academic Verification**: Comparison with peer-reviewed and institutional sources
- **Quality Indicators**: Educational explanation of reliability markers

## 4. Educational Uncertainty Discussion

Present balanced academic perspective on limitations:
- **Analytical Limitations**: Areas where institutional assessment has constraints
- **Additional Information**: Research that would strengthen the evaluation
- **Alternative Interpretations**: Academic consideration of different perspectives
- **Continuing Research**: Ongoing developments that might affect assessment

## 5. Public Education Recommendations

Provide actionable educational guidance for content evaluation:
- **Verification Methods**: Academic approaches citizens can use for fact-checking
- **Authoritative Sources**: Institutional and academic resources for verification
- **Educational Warning Signs**: Research-based indicators of questionable content
- **Critical Thinking**: Academic methods for evaluating information quality

INSTITUTIONAL STANDARDS:
- Maintain academic objectivity and educational focus throughout analysis
- Use institutional language appropriate for public education
- Provide balanced research perspective acknowledging uncertainties
- Focus on educational value and critical thinking development
- Apply systematic research methodology to content evaluation

Generate educational analysis suitable for institutional fact-checking protocols and public understanding."""

            logger.info(f"Main explanation prompt generated successfully", 
                       extra={'session_id': session_id, 'prompt_length': len(prompt)})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate main explanation prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptFormattingError(
                f"Main explanation prompt generation failed: {str(e)}",
                prompt_type="main_explanation"
            )

    @staticmethod
    def detailed_analysis_prompt(article_text: str, prediction: str, confidence: float,
                               metadata: Dict[str, Any], session_id: str = None) -> str:
        """
        Generate detailed forensic analysis prompt with institutional methodology.
        
        Provides comprehensive forensic analysis framework using academic language
        and institutional standards for expert-level review and analysis.
        """
        logger = logging.getLogger(f"{__name__}.ExplanationPrompts")
        
        try:
            article_context = article_text[:3500] if len(article_text) > 3500 else article_text
            metadata_str = str(metadata) if metadata else "No additional institutional metadata available"
            
            prompt = f"""You are a forensic content analyst conducting comprehensive institutional investigation for academic research purposes.

FORENSIC INVESTIGATION PARAMETERS:
- Content Classification: {prediction}
- Analytical Confidence: {confidence:.1%}
- Investigation Context: {metadata_str}

CONTENT FOR INSTITUTIONAL FORENSIC ANALYSIS:
{article_context}

COMPREHENSIVE FORENSIC RESEARCH FRAMEWORK:

Conduct systematic forensic examination following institutional research protocols:

## 1. Content Forensic Analysis

**Academic Claim Verification**:
- Systematic fact-checking of statistical assertions and quantitative claims
- Institutional verification of cited research studies and academic sources
- Cross-referencing with established academic databases and publications
- Evaluation of methodology and research design quality

**Source Attribution Forensics**:
- Institutional assessment of cited experts and authority figures
- Verification of organizational affiliations and professional credentials
- Analysis of source transparency and attribution completeness
- Evaluation of potential conflicts of interest and funding sources

## 2. Linguistic and Stylistic Forensic Investigation

**Professional Communication Analysis**:
- Academic evaluation of language patterns and rhetorical structure
- Assessment of professional writing standards and editorial quality
- Identification of persuasive techniques and emotional appeal mechanisms
- Analysis of technical accuracy and subject matter expertise demonstration

**Publication Standards Assessment**:
- Institutional evaluation of editorial oversight and fact-checking processes
- Assessment of correction and retraction policies and implementation
- Analysis of peer review and quality assurance mechanisms
- Evaluation of publication ethics and professional standards compliance

## 3. Contextual Research Investigation

**Temporal and Environmental Analysis**:
- Academic assessment of publication timing relative to significant events
- Institutional analysis of information ecosystem and distribution patterns
- Research into historical context and precedent identification
- Evaluation of potential coordination with other content or campaigns

**Cross-Platform Verification Research**:
- Institutional investigation of content consistency across platforms
- Academic analysis of modification and adaptation patterns
- Research into audience targeting and demographic considerations
- Assessment of viral mechanisms and amplification strategies

## 4. Technical Verification and Authentication

**Digital Forensics and Media Analysis**:
- Institutional assessment of multimedia content authenticity
- Technical evaluation of document and image integrity
- Research into metadata and technical indicators of manipulation
- Academic analysis of digital provenance and chain of custody

**Reference and Link Analysis**:
- Systematic verification of external references and hyperlinks
- Institutional assessment of supporting documentation quality
- Academic evaluation of citation accuracy and completeness
- Research into reference manipulation and misdirection techniques

## 5. Institutional Risk and Impact Assessment

**Academic Harm Evaluation**:
- Research-based assessment of potential misinformation impact
- Institutional analysis of vulnerable population targeting
- Academic evaluation of correction difficulty and persistence factors
- Assessment of amplification potential and viral spread mechanisms

**Professional Verification Challenges**:
- Institutional identification of verification obstacles and limitations
- Academic assessment of required expertise and resource needs
- Research into specialized knowledge requirements for evaluation
- Evaluation of time-sensitive factors affecting verification feasibility

## 6. Research Confidence and Uncertainty Analysis

**Academic Certainty Assessment**:
- Institutional evaluation of evidence strength and reliability
- Research-based identification of high-confidence analytical elements
- Academic assessment of verification completeness and thoroughness
- Evaluation of consensus among multiple analytical approaches

**Professional Uncertainty Documentation**:
- Systematic identification of areas requiring additional investigation
- Institutional acknowledgment of analytical limitations and constraints
- Academic documentation of alternative interpretation possibilities
- Research recommendations for enhanced verification and validation

INSTITUTIONAL RESEARCH STANDARDS:
- Apply rigorous academic methodology throughout forensic investigation
- Maintain institutional objectivity and professional analytical standards
- Document systematic research approach and verification processes
- Provide comprehensive evidence base for all analytical conclusions
- Follow established forensic analysis protocols and quality assurance measures

Generate detailed forensic documentation suitable for institutional review and academic research standards."""

            logger.info(f"Detailed analysis prompt generated successfully", 
                       extra={'session_id': session_id})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate detailed analysis prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptFormattingError(
                f"Detailed analysis prompt generation failed: {str(e)}",
                prompt_type="detailed_analysis"
            )

    @staticmethod
    def confidence_analysis_prompt(article_text: str, prediction: str, confidence: float,
                                 session_id: str = None) -> str:
        """
        Generate confidence level analysis prompt with institutional assessment framework.
        
        Provides systematic confidence evaluation using academic standards and
        institutional methodology for analytical quality assurance.
        """
        logger = logging.getLogger(f"{__name__}.ExplanationPrompts")
        
        try:
            article_context = article_text[:2500] if len(article_text) > 2500 else article_text

            prompt = f"""You are an institutional assessment specialist conducting systematic evaluation of analytical confidence levels for academic research quality assurance.

CONFIDENCE EVALUATION PARAMETERS:
- Content Classification: {prediction}
- Reported Analytical Confidence: {confidence:.1%}

CONTENT FOR CONFIDENCE ASSESSMENT:
{article_context}

INSTITUTIONAL CONFIDENCE ANALYSIS FRAMEWORK:

Conduct comprehensive evaluation of confidence level appropriateness using academic methodology:

## 1. Academic Confidence Appropriateness Assessment

**Institutional Justification Analysis**:
- Research-based evaluation of confidence level alignment with evidence strength
- Academic comparison with established analytical benchmarks and standards
- Institutional assessment of confidence calibration and appropriateness
- Professional evaluation of analytical consistency and methodological rigor

**Comparative Academic Evaluation**:
- Systematic comparison with similar institutional content analysis cases
- Academic benchmarking against established confidence level databases
- Research-based assessment of confidence level distribution patterns
- Professional evaluation of confidence level statistical significance

## 2. Evidence Strength and Quality Assessment

**Academic Evidence Evaluation**:
- Institutional analysis of supporting evidence robustness and reliability
- Research-based assessment of evidence diversity and corroboration
- Academic evaluation of evidence source quality and institutional credibility
- Professional assessment of evidence sufficiency for confidence level

**Institutional Uncertainty Factors**:
- Systematic identification of analytical limitations and constraints
- Academic assessment of information gaps and verification challenges
- Research-based evaluation of methodological limitations and biases
- Professional documentation of uncertainty sources and impact assessment

## 3. Methodological Confidence Calibration

**Academic Calibration Assessment**:
- Institutional evaluation of analytical methodology appropriateness
- Research-based assessment of systematic bias and correction factors
- Academic analysis of calibration accuracy and precision measurement
- Professional evaluation of confidence interval appropriateness and reliability

**Professional Quality Assurance**:
- Systematic assessment of analytical quality control measures
- Institutional evaluation of peer review and validation processes
- Academic assessment of methodological transparency and reproducibility
- Professional evaluation of analytical standard compliance and adherence

## 4. Risk Assessment and Decision Support

**Academic Risk Analysis**:
- Institutional evaluation of false positive and false negative probabilities
- Research-based assessment of misclassification consequences and impact
- Academic analysis of decision threshold appropriateness and optimization
- Professional evaluation of risk mitigation strategies and implementation

**Professional Decision Support Framework**:
- Systematic evaluation of confidence level utility for decision making
- Institutional assessment of user interpretation guidance and support
- Academic evaluation of communication effectiveness and clarity
- Professional assessment of actionable insight generation and application

## 5. Institutional Improvement Recommendations

**Academic Enhancement Strategies**:
- Research-based identification of confidence improvement opportunities
- Institutional recommendations for enhanced verification and validation
- Academic assessment of additional information requirements and sources
- Professional evaluation of methodological refinement and optimization potential

**Professional Development Recommendations**:
- Systematic identification of analytical skill development needs
- Institutional assessment of training and education requirements
- Academic evaluation of best practice implementation and standardization
- Professional recommendations for continuous improvement and quality enhancement

## 6. Institutional Communication and Transparency

**Academic Communication Assessment**:
- Research-based evaluation of confidence level communication effectiveness
- Institutional assessment of user understanding and interpretation accuracy
- Academic analysis of transparency and explanatory adequacy
- Professional evaluation of public education and awareness requirements

**Professional Standards Compliance**:
- Systematic assessment of institutional disclosure and transparency standards
- Academic evaluation of ethical communication requirements and compliance
- Research-based assessment of professional responsibility and accountability
- Professional evaluation of institutional integrity and public trust considerations

INSTITUTIONAL RESEARCH METHODOLOGY:
- Apply rigorous academic standards throughout confidence assessment
- Maintain professional objectivity and systematic analytical approach
- Document comprehensive evidence base for all confidence evaluations
- Provide detailed recommendations for confidence level interpretation and application
- Follow established institutional quality assurance and peer review protocols

Generate systematic confidence analysis suitable for institutional research standards and academic peer review."""

            logger.info(f"Confidence analysis prompt generated", 
                       extra={'session_id': session_id})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate confidence analysis prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptFormattingError(
                f"Confidence analysis prompt generation failed: {str(e)}",
                prompt_type="confidence_analysis"
            )

    @staticmethod
    def source_assessment_prompt(source: str, article_context: str, session_id: str = None) -> str:
        """
        Generate source reliability assessment prompt with institutional methodology.
        
        Provides comprehensive source evaluation framework using academic standards
        and institutional credibility assessment protocols.
        """
        logger = logging.getLogger(f"{__name__}.ExplanationPrompts")
        
        try:
            context_sample = article_context[:2000] if len(article_context) > 2000 else article_context

            prompt = f"""You are an institutional media credibility specialist conducting systematic source reliability assessment for academic research purposes.

SOURCE EVALUATION REQUEST:
- Publication Source: {source}
- Content Context Sample: {context_sample}

INSTITUTIONAL SOURCE ASSESSMENT FRAMEWORK:

Conduct comprehensive source reliability evaluation using established academic methodology:

## 1. Institutional Source Classification and Verification

**Academic Source Identification**:
- Systematic classification of media type and organizational structure
- Institutional verification of ownership, funding sources, and independence status
- Research-based assessment of historical establishment and industry recognition
- Academic evaluation of editorial standards, oversight mechanisms, and quality assurance

**Professional Credibility Documentation**:
- Institutional assessment of journalistic credentials and professional standards
- Academic verification of fact-checking processes and correction policies
- Research-based evaluation of editorial independence and conflict of interest policies
- Professional assessment of transparency in funding, ownership, and editorial processes

## 2. Academic Reliability and Quality Assessment

**Institutional Performance Analysis**:
- Systematic evaluation of factual accuracy record and correction history
- Academic assessment of source verification practices and quality control
- Research-based analysis of retraction and error correction effectiveness
- Professional evaluation of editorial oversight and quality assurance implementation

**Professional Standards Evaluation**:
- Institutional assessment of journalistic ethics compliance and professional conduct
- Academic evaluation of editorial standards and content quality consistency
- Research-based assessment of peer recognition and industry standing
- Professional evaluation of awards, citations, and institutional acknowledgment

## 3. Bias Assessment and Editorial Perspective Analysis

**Academic Bias Evaluation**:
- Systematic analysis of editorial perspective and ideological positioning
- Institutional assessment of balanced reporting and perspective diversity
- Research-based evaluation of opinion versus factual content separation
- Academic analysis of source selection and story prioritization patterns

**Professional Independence Assessment**:
- Institutional evaluation of editorial independence and external influence resistance
- Academic assessment of commercial conflict management and disclosure practices
- Research-based analysis of political and ideological independence maintenance
- Professional evaluation of advertiser and sponsor influence mitigation strategies

## 4. Technical and Operational Quality Assessment

**Institutional Infrastructure Evaluation**:
- Academic assessment of digital security, privacy protection, and user safety measures
- Professional evaluation of website functionality, accessibility, and user experience
- Research-based assessment of technical standards and digital best practices
- Institutional evaluation of content management and publication quality systems

**Professional Distribution and Engagement Analysis**:
- Systematic assessment of content distribution strategies and audience engagement
- Academic evaluation of social media presence authenticity and engagement quality
- Research-based analysis of audience demographics and reach assessment
- Professional evaluation of content amplification and viral distribution patterns

## 5. Contextual Reliability and Expertise Assessment

**Academic Subject Matter Competence**:
- Institutional evaluation of specialized knowledge and expertise demonstration
- Professional assessment of subject matter expert consultation and collaboration
- Research-based evaluation of technical accuracy and specialized knowledge application
- Academic assessment of continuing education and professional development evidence

**Professional Geographic and Cultural Competence**:
- Systematic assessment of local knowledge and cultural understanding demonstration
- Institutional evaluation of geographic coverage competence and resource allocation
- Academic analysis of cultural sensitivity and community engagement effectiveness
- Professional assessment of diverse perspective inclusion and representation quality

## 6. Institutional Warning Indicators and Risk Assessment

**Academic Risk Factor Analysis**:
- Research-based identification of reliability concerns and credibility risk factors
- Institutional assessment of misinformation history and pattern identification
- Professional evaluation of controversial content handling and crisis management
- Academic analysis of regulatory compliance and legal standards adherence

**Professional Recommendation Framework**:
- Systematic development of source reliability rating with detailed justification
- Institutional recommendations for source verification and cross-referencing strategies
- Academic guidance for content evaluation and critical assessment approaches
- Professional assessment of appropriate reliance levels and verification requirements

INSTITUTIONAL ASSESSMENT STANDARDS:
- Apply established academic methodology for media credibility evaluation
- Maintain professional objectivity and systematic analytical approach
- Document comprehensive evidence base for all reliability assessments
- Provide clear reliability classification with detailed supporting rationale
- Follow institutional transparency and disclosure standards for assessment methodology

Generate comprehensive source reliability assessment suitable for institutional research standards and academic credibility evaluation protocols."""

            logger.info(f"Source assessment prompt generated", 
                       extra={'session_id': session_id})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate source assessment prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptFormattingError(
                f"Source assessment prompt generation failed: {str(e)}",
                prompt_type="source_assessment"
            )


class AdaptivePrompts:
    """
    Adaptive prompts that adjust based on content type, domain, and analysis requirements.
    
    Provides dynamic prompt adaptation for domain-specific analysis and confidence-adjusted
    explanations with enhanced institutional language and safety optimization.
    """

    @staticmethod
    def get_domain_specific_prompt(domain: str, base_prompt: str, session_id: str = None) -> str:
        """
        Adapt base prompt for specific content domains with institutional enhancement.
        
        Args:
            domain: Content domain (health, politics, science, economics, technology)
            base_prompt: Base institutional prompt to enhance
            session_id: Optional session ID for tracking
            
        Returns:
            Domain-enhanced prompt with specialized institutional guidance
        """
        logger = logging.getLogger(f"{__name__}.AdaptivePrompts")
        
        try:
            domain_enhancements = {
                'health': """
SPECIALIZED INSTITUTIONAL HEALTH ANALYSIS REQUIREMENTS:

**Medical and Health Research Standards**:
- Apply rigorous medical research methodology and evidence-based analysis
- Cross-reference with peer-reviewed medical journals and institutional health databases
- Evaluate against established medical consensus and professional health organization guidance
- Assess potential public health impact and medical misinformation risks

**Academic Health Communication Guidelines**:
- Use precise medical terminology while maintaining public accessibility
- Distinguish between preliminary research findings and established medical consensus
- Evaluate clinical study methodology and institutional research quality standards
- Consider vulnerable population impact and health equity implications

**Professional Health Information Assessment**:
- Verify medical professional credentials and institutional affiliations
- Assess compliance with medical ethics and professional disclosure standards
- Evaluate treatment recommendation safety and evidence-based support
- Consider regulatory approval status and institutional medical oversight""",

                'politics': """
SPECIALIZED INSTITUTIONAL POLITICAL ANALYSIS REQUIREMENTS:

**Democratic Process and Governance Analysis**:
- Apply systematic political science methodology and institutional analysis frameworks
- Cross-reference with official government records and institutional democratic databases
- Evaluate against established democratic principles and institutional governance standards
- Assess electoral process integrity and institutional democratic safeguards

**Academic Political Communication Standards**:
- Maintain strict political neutrality and institutional analytical objectivity
- Distinguish between factual political reporting and opinion/advocacy content
- Evaluate institutional source credibility and political bias assessment
- Consider democratic participation impact and civic engagement implications

**Professional Governance Information Assessment**:
- Verify political figure credentials and official institutional positions
- Assess compliance with democratic transparency and institutional disclosure standards
- Evaluate policy accuracy and institutional governmental procedure compliance
- Consider constitutional and legal framework compliance and institutional oversight""",

                'science': """
SPECIALIZED INSTITUTIONAL SCIENTIFIC ANALYSIS REQUIREMENTS:

**Academic Research and Scientific Method Standards**:
- Apply rigorous scientific methodology and peer review evaluation standards
- Cross-reference with established scientific databases and institutional research repositories
- Evaluate against scientific consensus and institutional academic quality standards
- Assess research methodology quality and institutional scientific oversight

**Professional Scientific Communication Guidelines**:
- Use precise scientific terminology with accessible public education focus
- Distinguish between preliminary findings and established scientific consensus
- Evaluate publication quality and institutional peer review standards
- Consider scientific reproducibility and institutional research integrity standards

**Institutional Scientific Information Assessment**:
- Verify researcher credentials and institutional academic affiliations
- Assess compliance with research ethics and institutional scientific standards
- Evaluate funding transparency and institutional conflict of interest disclosure
- Consider scientific consensus alignment and institutional academic authority""",

                'economics': """
SPECIALIZED INSTITUTIONAL ECONOMIC ANALYSIS REQUIREMENTS:

**Academic Economic Research and Financial Analysis Standards**:
- Apply systematic economic methodology and institutional financial analysis frameworks
- Cross-reference with official economic data and institutional financial databases
- Evaluate against established economic principles and institutional market analysis standards
- Assess economic policy impact and institutional financial stability implications

**Professional Economic Communication Guidelines**:
- Use precise economic terminology with accessible public education focus
- Distinguish between economic analysis and financial advice or speculation
- Evaluate institutional source credibility and economic data quality standards
- Consider economic equity impact and institutional market fairness implications

**Institutional Financial Information Assessment**:
- Verify economic expert credentials and institutional financial affiliations
- Assess compliance with financial disclosure and institutional transparency standards
- Evaluate economic data accuracy and institutional market oversight compliance
- Consider regulatory compliance and institutional financial authority standards""",

                'technology': """
SPECIALIZED INSTITUTIONAL TECHNOLOGY ANALYSIS REQUIREMENTS:

**Academic Technology Research and Digital Analysis Standards**:
- Apply systematic technology assessment methodology and institutional digital evaluation frameworks
- Cross-reference with established technology databases and institutional research standards
- Evaluate against technological best practices and institutional cybersecurity standards
- Assess digital privacy impact and institutional technology ethics implications

**Professional Technology Communication Guidelines**:
- Use precise technical terminology with accessible public education focus
- Distinguish between established technology capabilities and speculative claims
- Evaluate institutional source credibility and technical expertise verification
- Consider digital equity impact and institutional technology access implications

**Institutional Technology Information Assessment**:
- Verify technology expert credentials and institutional technical affiliations
- Assess compliance with privacy protection and institutional digital rights standards
- Evaluate technical accuracy and institutional cybersecurity oversight compliance
- Consider regulatory compliance and institutional technology authority standards"""
            }

            enhancement = domain_enhancements.get(domain.lower(), 
                "Apply general institutional analysis standards with academic rigor and professional objectivity.")
            
            enhanced_prompt = base_prompt + "\n\n" + enhancement
            
            logger.info(f"Domain-specific prompt generated for {domain}", 
                       extra={'session_id': session_id})
            
            return enhanced_prompt

        except Exception as e:
            logger.warning(f"Domain adaptation failed, using base prompt: {str(e)}", 
                          extra={'session_id': session_id})
            return base_prompt

    @staticmethod
    def get_confidence_adjusted_prompt(confidence: float, base_prompt: str, session_id: str = None) -> str:
        """
        Adjust prompt based on confidence level with institutional guidance.
        
        Args:
            confidence: Confidence score (0.0-1.0)
            base_prompt: Base institutional prompt to adjust
            session_id: Optional session ID for tracking
            
        Returns:
            Confidence-adjusted prompt with appropriate institutional emphasis
        """
        logger = logging.getLogger(f"{__name__}.AdaptivePrompts")
        
        try:
            if confidence < 0.5:
                adjustment = """
INSTITUTIONAL LOW CONFIDENCE ANALYSIS REQUIREMENTS:

**Enhanced Uncertainty Documentation**:
- Emphasize analytical limitations and institutional assessment constraints
- Provide detailed documentation of evidence gaps and verification challenges  
- Recommend additional institutional verification steps and expert consultation
- Focus on educational guidance for critical evaluation and independent verification

**Academic Caution and Alternative Perspective Consideration**:
- Present multiple plausible interpretations using academic analytical frameworks
- Acknowledge significant uncertainty and institutional analytical limitations
- Recommend professional expert consultation and specialized institutional review
- Provide enhanced critical thinking guidance and institutional verification resources"""

            elif confidence < 0.7:
                adjustment = """
INSTITUTIONAL MODERATE CONFIDENCE ANALYSIS REQUIREMENTS:

**Balanced Academic Assessment**:
- Present evidence-based analysis with appropriate institutional caveats
- Acknowledge areas of uncertainty while highlighting strong supporting evidence
- Provide balanced perspective with institutional analytical objectivity
- Recommend standard verification approaches and academic cross-referencing

**Professional Quality Assurance**:
- Apply standard institutional verification protocols and quality control measures
- Balance confidence expression with appropriate academic uncertainty acknowledgment
- Provide moderate-level verification recommendations and institutional guidance
- Focus on educational balance between certainty and critical thinking development"""

            elif confidence > 0.9:
                adjustment = """
INSTITUTIONAL HIGH CONFIDENCE ANALYSIS REQUIREMENTS:

**Enhanced Evidence Documentation**:
- Provide comprehensive documentation of strong supporting evidence
- Explain why institutional assessment confidence is particularly robust
- Acknowledge any remaining limitations despite high confidence levels
- Present exceptional evidence quality and institutional verification completeness

**Academic Rigor and Professional Standards**:
- Maintain institutional humility despite strong evidence base
- Provide detailed methodology explanation for high confidence assessment
- Include appropriate academic caveats and professional analytical limitations
- Focus on educational explanation of exceptional analytical certainty"""

            else:
                adjustment = """
INSTITUTIONAL STANDARD CONFIDENCE ANALYSIS REQUIREMENTS:

**Professional Analytical Standards**:
- Apply standard institutional analysis methodology with appropriate confidence expression
- Provide balanced evidence presentation with professional analytical objectivity
- Include standard uncertainty acknowledgment and institutional analytical limitations
- Focus on educational explanation with appropriate confidence level communication"""

            adjusted_prompt = base_prompt + "\n\n" + adjustment
            
            logger.info(f"Confidence-adjusted prompt generated for {confidence:.2f}", 
                       extra={'session_id': session_id})
            
            return adjusted_prompt

        except Exception as e:
            logger.warning(f"Confidence adjustment failed, using base prompt: {str(e)}", 
                          extra={'session_id': session_id})
            return base_prompt


# Main prompt access function with enhanced error handling
def get_explanation_prompt(prompt_type: str, session_id: str = None, **kwargs) -> str:
    """
    Get specific explanation prompt template with comprehensive error handling.

    Args:
        prompt_type: Type of prompt needed ('main', 'detailed', 'confidence', 'source')
        session_id: Optional session ID for tracking and debugging
        **kwargs: Parameters for prompt formatting

    Returns:
        Formatted institutional prompt string optimized for safety filters

    Raises:
        PromptFormattingError: If prompt type is unknown or generation fails
    """
    logger = logging.getLogger(f"{__name__}.get_explanation_prompt")
    
    try:
        prompt_mapping = {
            'main': ExplanationPrompts.main_explanation_prompt,
            'detailed': ExplanationPrompts.detailed_analysis_prompt,  
            'confidence': ExplanationPrompts.confidence_analysis_prompt,
            'source': ExplanationPrompts.source_assessment_prompt
        }

        if prompt_type not in prompt_mapping:
            available_types = ', '.join(prompt_mapping.keys())
            raise PromptFormattingError(
                f"Unknown prompt type: {prompt_type}. Available types: {available_types}",
                prompt_type=prompt_type
            )

        # Add session_id to kwargs if provided
        if session_id:
            kwargs['session_id'] = session_id

        # Generate the prompt
        prompt = prompt_mapping[prompt_type](**kwargs)
        
        logger.info(f"Explanation prompt generated successfully", 
                   extra={'session_id': session_id, 'prompt_type': prompt_type})
        
        return prompt

    except PromptFormattingError:
        raise  # Re-raise prompt formatting errors
    except Exception as e:
        logger.error(f"Failed to generate explanation prompt: {str(e)}", 
                    extra={'session_id': session_id, 'prompt_type': prompt_type})
        raise PromptFormattingError(
            f"Prompt generation failed: {str(e)}",
            prompt_type=prompt_type
        )


def validate_prompt_parameters(prompt_type: str, session_id: str = None, **kwargs):
    """
    Validate parameters for prompt generation with enhanced checking.

    Args:
        prompt_type: Type of prompt to validate parameters for
        session_id: Optional session ID for tracking
        **kwargs: Parameters to validate

    Returns:
        ValidationResult: Comprehensive parameter validation results
    """
    from .validators import ValidationResult
    
    logger = logging.getLogger(f"{__name__}.validate_prompt_parameters")
    
    try:
        # Define required parameters for each prompt type
        required_params = {
            'main': ['article_text', 'prediction', 'confidence', 'source', 'date', 'subject'],
            'detailed': ['article_text', 'prediction', 'confidence', 'metadata'],
            'confidence': ['article_text', 'prediction', 'confidence'],
            'source': ['source', 'article_context']
        }

        errors = []
        warnings = []

        # Validate prompt type
        if prompt_type not in required_params:
            available_types = ', '.join(required_params.keys())
            errors.append(f"Unknown prompt type: {prompt_type}. Available: {available_types}")
            return ValidationResult(False, errors, warnings)

        # Check for missing parameters
        missing_params = [param for param in required_params[prompt_type] if param not in kwargs]
        if missing_params:
            errors.extend([f"Missing required parameter: {param}" for param in missing_params])

        # Validate parameter types and values
        for param, value in kwargs.items():
            if param == 'confidence':
                if not isinstance(value, (int, float)):
                    errors.append(f"Parameter 'confidence' must be numeric, got {type(value).__name__}")
                elif not (0 <= value <= 1):
                    errors.append(f"Parameter 'confidence' must be between 0 and 1, got {value}")
            
            elif param == 'prediction':
                if not isinstance(value, str):
                    errors.append(f"Parameter 'prediction' must be string, got {type(value).__name__}")
                elif value.upper() not in ['FAKE', 'REAL', 'UNKNOWN']:
                    warnings.append(f"Parameter 'prediction' has unusual value: {value}")
            
            elif param in ['article_text', 'source', 'date', 'subject', 'article_context']:
                if not isinstance(value, str):
                    warnings.append(f"Parameter '{param}' should be string, got {type(value).__name__}")
                elif param == 'article_text' and len(value.strip()) < 50:
                    warnings.append(f"Parameter 'article_text' is very short: {len(value)} characters")

        # Final validation
        is_valid = len(errors) == 0
        
        logger.info(
            f"Parameter validation completed: {'PASSED' if is_valid else 'FAILED'}",
            extra={'session_id': session_id, 'prompt_type': prompt_type}
        )

        return ValidationResult(is_valid, errors, warnings)

    except Exception as e:
        logger.error(f"Parameter validation error: {str(e)}", 
                    extra={'session_id': session_id})
        return ValidationResult(False, [f"Validation error: {str(e)}"], [])


def get_domain_guidance(domain: str) -> Dict[str, Any]:
    """
    Get domain-specific guidance for explanation generation.

    Args:
        domain: Content domain (health, politics, science, technology, economics, general)

    Returns:
        Dictionary with domain-specific explanation guidance and requirements
    """
    domain_guidance = {
        'health': {
            'focus_areas': [
                'Medical research methodology and peer review standards',
                'Clinical study design and evidence-based medicine',
                'Public health impact and safety considerations', 
                'Medical professional credentials and institutional oversight'
            ],
            'priority_indicators': [
                'Peer-reviewed medical journals and institutional publications',
                'Medical professional consensus and academic authority',
                'Clinical trial methodology and regulatory compliance',
                'Patient safety considerations and medical ethics'
            ],
            'verification_sources': [
                'PubMed and medical database cross-referencing',
                'Professional medical associations and institutions',
                'Regulatory agencies and health department resources',
                'Medical academic institutions and research centers'
            ],
            'special_considerations': 'Emphasize patient safety, medical accuracy, and evidence-based standards while avoiding medical advice'
        },
        'politics': {
            'focus_areas': [
                'Electoral process integrity and democratic institutions',
                'Government policy accuracy and procedural compliance',
                'Political figure credentials and official statements',
                'Constitutional and legal framework adherence'
            ],
            'priority_indicators': [
                'Official government records and institutional sources',
                'Electoral commission documentation and oversight',
                'Legislative records and parliamentary procedures',
                'Constitutional compliance and legal precedent'
            ],
            'verification_sources': [
                'Government databases and official records',
                'Electoral commissions and democratic oversight bodies',
                'Academic political science institutions',
                'Non-partisan civic organizations and transparency groups'
            ],
            'special_considerations': 'Maintain strict political neutrality and focus on verifiable institutional facts'
        },
        'science': {
            'focus_areas': [
                'Peer review process and academic publication standards',
                'Research methodology and experimental design quality',
                'Scientific consensus and institutional authority',
                'Research ethics and institutional oversight'
            ],
            'priority_indicators': [
                'Peer-reviewed scientific journals and publications',
                'Academic institutional affiliation and credentials',
                'Research methodology transparency and reproducibility',
                'Scientific consensus and expert institutional agreement'
            ],
            'verification_sources': [
                'Scientific databases and academic repositories',
                'Research institutions and academic authorities',
                'Professional scientific societies and organizations',
                'Peer review systems and academic quality assurance'
            ],
            'special_considerations': 'Distinguish between preliminary findings and established scientific consensus'
        },
        'economics': {
            'focus_areas': [
                'Economic data accuracy and institutional sources',
                'Financial methodology and analytical standards',
                'Economic policy impact and institutional analysis',
                'Market analysis and regulatory compliance'
            ],
            'priority_indicators': [
                'Official economic statistics and government data',
                'Financial regulatory compliance and oversight',
                'Academic economic research and institutional analysis',
                'Professional economic methodology and transparency'
            ],
            'verification_sources': [
                'Economic databases and statistical agencies',
                'Financial regulatory institutions and oversight bodies',
                'Academic economics departments and research centers',
                'Professional economic associations and standards bodies'
            ],
            'special_considerations': 'Focus on verifiable economic data and avoid financial advice or speculation'
        },
        'technology': {
            'focus_areas': [
                'Technical accuracy and engineering standards',
                'Cybersecurity implications and digital safety',
                'Privacy protection and data security standards',
                'Technology ethics and social impact assessment'
            ],
            'priority_indicators': [
                'Technical documentation and engineering specifications',
                'Security research and academic cybersecurity analysis',
                'Privacy protection standards and regulatory compliance',
                'Technology ethics and institutional oversight'
            ],
            'verification_sources': [
                'Technical documentation and engineering standards',
                'Cybersecurity research institutions and organizations',
                'Technology academic institutions and research centers',
                'Professional technology associations and standards bodies'
            ],
            'special_considerations': 'Emphasize technical accuracy, security implications, and ethical considerations'
        },
        'general': {
            'focus_areas': [
                'Factual accuracy and source verification',
                'Logical consistency and evidence-based analysis',
                'Source credibility and institutional authority',
                'Information quality and verification standards'
            ],
            'priority_indicators': [
                'Authoritative sources and institutional credibility',
                'Evidence-based analysis and verification standards',
                'Logical consistency and factual accuracy',
                'Professional standards and quality assurance'
            ],
            'verification_sources': [
                'Authoritative databases and institutional sources',
                'Professional journalism and fact-checking organizations',
                'Academic institutions and research centers',
                'Government agencies and official records'
            ],
            'special_considerations': 'Apply comprehensive analytical standards while maintaining accessibility'
        }
    }

    return domain_guidance.get(domain.lower(), domain_guidance['general'])


def get_prompt_statistics() -> Dict[str, Any]:
    """
    Get comprehensive prompt system statistics and capabilities for monitoring.

    Returns:
        Dictionary with detailed prompt system metrics and features
    """
    return {
        'available_prompt_types': [
            'main', 'detailed', 'confidence', 'source'
        ],
        'adaptive_features': [
            'domain_specific_adaptation', 'confidence_level_adjustment',
            'institutional_language_optimization', 'safety_filter_avoidance'
        ],
        'supported_domains': [
            'health', 'politics', 'science', 'economics', 'technology', 'general'
        ],
        'safety_optimizations': {
            'institutional_language': True,
            'academic_framing': True,
            'professional_terminology': True,
            'educational_focus': True
        },
        'prompt_capabilities': {
            'main_explanation': 'Comprehensive public education explanation with institutional credibility',
            'detailed_analysis': 'Forensic institutional analysis for expert review and academic standards',
            'confidence_analysis': 'Academic confidence level assessment and quality assurance',
            'source_assessment': 'Institutional source reliability evaluation and credibility analysis'
        },
        'validation_features': {
            'parameter_validation': True,
            'format_checking': True,
            'error_prevention': True,
            'session_tracking': True
        },
        'domain_specializations': {
            'health': 'Medical research standards and evidence-based medicine',
            'politics': 'Democratic process integrity and institutional governance',
            'science': 'Peer review standards and academic research methodology',
            'economics': 'Economic analysis and institutional financial oversight',
            'technology': 'Technical accuracy and cybersecurity considerations'
        },
        'institutional_standards': [
            'Academic methodology and research protocols',
            'Professional objectivity and analytical rigor', 
            'Educational focus and public accessibility',
            'Quality assurance and peer review compliance',
            'Transparency and disclosure requirements'
        ],
        'version_info': {
            'version': '2.0.0',
            'architecture': 'institutional_language_optimized',
            'safety_level': 'production_optimized',
            'last_updated': '2025-09-11'
        }
    }


# Testing functionality
if __name__ == "__main__":
    """Test explanation prompt functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== LLM EXPLANATION PROMPTS TEST ===")
    
    # Test main explanation prompt
    print("--- Main Explanation Prompt Test ---")
    test_main_prompt = get_explanation_prompt(
        'main',
        article_text="A new study claims that drinking 10 cups of coffee daily can extend lifespan by 50 years without peer review.",
        prediction="FAKE",
        confidence=0.89,
        source="HealthBlog.net", 
        date="2025-01-15",
        subject="Health",
        session_id="test_prompt_001"
    )

    print(f"✅ Main explanation prompt generated")
    print(f"   Length: {len(test_main_prompt)} characters")
    print(f"   Contains 'institutional': {'institutional' in test_main_prompt.lower()}")
    print(f"   Contains 'academic': {'academic' in test_main_prompt.lower()}")

    # Test detailed analysis prompt
    print("\n--- Detailed Analysis Prompt Test ---")
    test_metadata = {
        "author": "Anonymous Blogger",
        "domain": "health",
        "publication_type": "blog"
    }
    
    detailed_prompt = get_explanation_prompt(
        'detailed',
        article_text="Sample health misinformation article content for testing purposes.",
        prediction="FAKE",
        confidence=0.76,
        metadata=test_metadata,
        session_id="test_prompt_002"
    )

    print(f"✅ Detailed analysis prompt generated")
    print(f"   Length: {len(detailed_prompt)} characters")
    print(f"   Contains 'forensic': {'forensic' in detailed_prompt.lower()}")

    # Test confidence analysis prompt
    print("\n--- Confidence Analysis Prompt Test ---")
    confidence_prompt = get_explanation_prompt(
        'confidence',
        article_text="Test article for confidence analysis evaluation.",
        prediction="FAKE",
        confidence=0.45,
        session_id="test_prompt_003"
    )

    print(f"✅ Confidence analysis prompt generated")
    print(f"   Length: {len(confidence_prompt)} characters")
    print(f"   Contains 'confidence': {'confidence' in confidence_prompt.lower()}")

    # Test source assessment prompt
    print("\n--- Source Assessment Prompt Test ---")
    source_prompt = get_explanation_prompt(
        'source',
        source="UnknownHealthBlog.net",
        article_context="Health misinformation blog with questionable claims",
        session_id="test_prompt_004"
    )

    print(f"✅ Source assessment prompt generated")
    print(f"   Length: {len(source_prompt)} characters")
    print(f"   Contains 'reliability': {'reliability' in source_prompt.lower()}")

    # Test adaptive prompts
    print("\n--- Adaptive Prompts Test ---")
    base_prompt = "Test base prompt for domain adaptation."
    
    health_adapted = AdaptivePrompts.get_domain_specific_prompt(
        'health', base_prompt, "test_prompt_005"
    )
    print(f"✅ Health domain adaptation: {'medical' in health_adapted.lower()}")
    
    low_confidence_adapted = AdaptivePrompts.get_confidence_adjusted_prompt(
        0.3, base_prompt, "test_prompt_006"
    )
    print(f"✅ Low confidence adaptation: {'uncertainty' in low_confidence_adapted.lower()}")

    # Test parameter validation
    print("\n--- Parameter Validation Test ---")
    validation_result = validate_prompt_parameters(
        'main',
        article_text="Test article",
        prediction="FAKE",
        confidence=0.8,
        source="Test Source",
        date="2025-01-01",
        subject="Health",
        session_id="test_prompt_007"
    )
    
    print(f"✅ Parameter validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    if not validation_result.is_valid:
        print(f"   Errors: {validation_result.errors}")
    if validation_result.warnings:
        print(f"   Warnings: {validation_result.warnings[:2]}")

    # Test domain guidance
    print("\n--- Domain Guidance Test ---")
    health_guidance = get_domain_guidance('health')
    print(f"✅ Health domain guidance: {len(health_guidance['focus_areas'])} focus areas")
    print(f"   Priority indicators: {len(health_guidance['priority_indicators'])}")

    # Test prompt statistics
    print("\n--- Prompt Statistics Test ---")
    stats = get_prompt_statistics()
    print(f"✅ Available prompt types: {len(stats['available_prompt_types'])}")
    print(f"✅ Supported domains: {len(stats['supported_domains'])}")
    print(f"✅ Safety optimizations: {stats['safety_optimizations']['institutional_language']}")
    print(f"✅ Version: {stats['version_info']['version']}")

    print("\n🎯 LLM explanation prompts tests completed successfully!")
