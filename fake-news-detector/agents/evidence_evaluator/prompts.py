# agents/evidence_evaluator/prompts.py

"""
Evidence Evaluator Prompts Module - Production Ready

Enhanced prompt templates with state-of-the-art prompt engineering techniques,
fallback prompts, validation, and production-level error handling.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .exceptions import PromptGenerationError, raise_prompt_generation_error


@dataclass
class PromptResponse:
    """Structured response container for prompt outputs with metadata."""
    content: str
    metadata: Dict[str, Any]
    prompt_type: str
    tokens_used: Optional[int] = None
    generation_time: Optional[float] = None


class EvidenceVerificationPrompts:
    """
    Enhanced evidence verification prompts with Chain-of-Thought reasoning,
    Few-Shot examples, and robust fallback handling.
    """

    @staticmethod
    def generate_verification_sources(article_text: str, 
                                    claims: List[Dict[str, Any]], 
                                    session_id: str = None) -> str:
        """
        Generate specific verification sources using advanced prompt engineering.
        
        Uses Chain-of-Thought reasoning, Few-Shot examples, and structured output.
        
        Args:
            article_text: Article content for verification
            claims: Claims to verify with metadata
            session_id: Optional session ID for tracking
        
        Returns:
            Formatted prompt string for LLM processing
        
        Raises:
            PromptGenerationError: If prompt generation fails
        """
        logger = logging.getLogger(f"{__name__}.EvidenceVerificationPrompts")
        
        try:
            # Validate inputs
            if not article_text or not isinstance(article_text, str):
                raise_prompt_generation_error(
                    'verification_sources', 
                    "Article text must be non-empty string",
                    {'article_text_type': type(article_text).__name__},
                    session_id
                )
            
            if not claims or not isinstance(claims, list):
                raise_prompt_generation_error(
                    'verification_sources',
                    "Claims must be non-empty list", 
                    {'claims_type': type(claims).__name__},
                    session_id
                )
            
            # Prepare claims with safety checks
            formatted_claims = []
            for i, claim in enumerate(claims[:5]):  # Limit to 5 claims for focus
                if isinstance(claim, dict) and claim.get('text'):
                    claim_text = str(claim['text']).strip()
                    if claim_text:
                        formatted_claims.append(f"{i+1}. {claim_text}")
            
            if not formatted_claims:
                raise_prompt_generation_error(
                    'verification_sources',
                    "No valid claims found for verification",
                    {'claims_count': len(claims)},
                    session_id
                )
            
            claims_text = "\n".join(formatted_claims)
            article_excerpt = article_text[:1200].strip()
            
            # Enhanced prompt with CoT and Few-Shot learning
            prompt = f"""You are Dr. Sarah Chen, a senior fact-checking specialist with 15+ years of experience in investigative journalism and academic research verification. You have worked with Reuters, Associated Press, and the International Fact-Checking Network.

<thinking>
My task is to generate exactly 5 highly specific, actionable verification sources for the given claims. For each claim, I need to:

1. IDENTIFY: What is the core factual assertion that needs verification?
2. CATEGORIZE: What type of evidence would best verify this claim? (official data, research study, expert statement, etc.)
3. SOURCE: Which authoritative institution would be the most credible source for this type of claim?
4. SPECIFY: What would be the exact URL path structure for finding this information?
5. ASSESS: How confident am I that this source exists and would contain the needed information?

I must avoid generic homepage URLs and instead provide specific, deep-linked URLs that would directly address each claim.
</thinking>

ARTICLE EXCERPT:
{article_excerpt}

CLAIMS TO VERIFY:
{claims_text}

EXAMPLES OF EXCELLENT VERIFICATION SOURCES:

Example 1:
**CLAIM**: "FDA approved Pfizer COVID-19 vaccine for adults in August 2021"
**INSTITUTION**: U.S. Food and Drug Administration
**SPECIFIC_URL**: https://www.fda.gov/news-events/press-announcements/fda-approves-first-covid-19-vaccine
**VERIFICATION_TYPE**: official_data
**SEARCH_STRATEGY**: "Pfizer COVID-19 vaccine approval" site:fda.gov
**CONFIDENCE**: 0.95

Example 2:
**CLAIM**: "Harvard study shows 40% reduction in cardiovascular events"
**INSTITUTION**: Harvard T.H. Chan School of Public Health
**SPECIFIC_URL**: https://www.hsph.harvard.edu/news/press-releases/2023/cardiovascular-study-results/
**VERIFICATION_TYPE**: research_study
**SEARCH_STRATEGY**: "cardiovascular reduction study" site:hsph.harvard.edu
**CONFIDENCE**: 0.85

Now provide exactly 5 verification sources using this EXACT format:

## VERIFICATION SOURCE 1
**CLAIM**: "Exact quote from article being verified"
**INSTITUTION**: Name of most authoritative organization for this claim type
**SPECIFIC_URL**: https://domain.org/exact-path/to/relevant-content
**VERIFICATION_TYPE**: primary_source|expert_analysis|official_data|research_study
**SEARCH_STRATEGY**: "specific keywords" site:domain.org
**CONFIDENCE**: 0.X (decimal from 0.1 to 1.0)

## VERIFICATION SOURCE 2
[Continue same format for all 5 sources...]

CRITICAL REQUIREMENTS:
1. URLs must be SPECIFIC pages, never homepages (❌ https://cdc.gov ✅ https://cdc.gov/vaccines/covid-19/clinical-considerations/managing-anaphylaxis.html)
2. Each institution must be the MOST AUTHORITATIVE source for that specific claim type
3. Confidence scores must reflect realistic likelihood of finding the information at that exact URL
4. Search strategies must be specific enough to find the exact information needed
5. Verification types must match the nature of the evidence being sought

Generate exactly 5 sources now:"""

            logger.info(f"Generated verification sources prompt", 
                       extra={
                           'session_id': session_id,
                           'article_length': len(article_text),
                           'claims_count': len(formatted_claims),
                           'prompt_length': len(prompt)
                       })
            
            return prompt
            
        except PromptGenerationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in verification sources prompt generation: {str(e)}", 
                        extra={'session_id': session_id})
            raise_prompt_generation_error(
                'verification_sources',
                f"Prompt generation failed: {str(e)}",
                {'error_type': type(e).__name__},
                session_id
            )

    @staticmethod
    def generate_verification_sources_fallback(article_text: str, 
                                             claims: List[Dict[str, Any]], 
                                             session_id: str = None) -> str:
        """
        Fallback prompt for verification sources when main prompt fails.
        
        Simplified but reliable prompt for basic verification source generation.
        """
        logger = logging.getLogger(f"{__name__}.EvidenceVerificationPrompts")
        logger.warning(f"Using fallback verification sources prompt", 
                      extra={'session_id': session_id})
        
        try:
            # Basic validation and preparation
            claims_text = "No specific claims provided"
            if claims and isinstance(claims, list):
                valid_claims = []
                for i, claim in enumerate(claims[:3]):  # Limit to 3 for fallback
                    if isinstance(claim, dict) and claim.get('text'):
                        valid_claims.append(f"- {claim['text']}")
                if valid_claims:
                    claims_text = "\n".join(valid_claims)
            
            article_excerpt = str(article_text)[:800] if article_text else "No article provided"
            
            return f"""Generate 3 verification sources for fact-checking the following content.

CONTENT: {article_excerpt}

CLAIMS: 
{claims_text}

For each claim, provide:
1. Institution: Authoritative source name
2. URL: Specific verification URL (not homepage)
3. Confidence: 0.1 to 1.0

Format each as:
Source 1: [Institution] - [URL] - Confidence: [0.X]
Source 2: [Institution] - [URL] - Confidence: [0.X]
Source 3: [Institution] - [URL] - Confidence: [0.X]"""

        except Exception as e:
            logger.error(f"Fallback prompt generation failed: {str(e)}", 
                        extra={'session_id': session_id})
            return "Generate verification sources for the provided content."

    @staticmethod
    def assess_source_quality(article_text: str, 
                            sources: List[str], 
                            session_id: str = None) -> str:
        """
        Enhanced source quality assessment prompt with systematic evaluation framework.
        """
        logger = logging.getLogger(f"{__name__}.EvidenceVerificationPrompts")
        
        try:
            if not sources:
                sources = ["No sources provided"]
            
            sources_list = "\n".join([f"• {source}" for source in sources[:10]])
            article_excerpt = str(article_text)[:800] if article_text else "No content provided"
            
            prompt = f"""You are Prof. Michael Rodriguez, Director of the Digital Media Literacy Institute and expert in source credibility assessment with 20+ years of experience evaluating information sources across academic, journalistic, and policy contexts.

<evaluation_framework>
Your systematic evaluation will assess sources across four key dimensions:

1. INSTITUTIONAL AUTHORITY: Government agencies > Academic institutions > Professional organizations > Established media
2. DOMAIN EXPERTISE: Direct subject matter relevance and recognized specialization
3. TRANSPARENCY: Methodology disclosure, funding sources, bias acknowledgment, correction policies
4. VERIFICATION POTENTIAL: Primary vs secondary sources, independent corroboration availability
</evaluation_framework>

CONTENT BEING EVALUATED:
{article_excerpt}

IDENTIFIED SOURCES:
{sources_list}

TASK: Conduct systematic source quality analysis using this structure:

## AUTHORITY ASSESSMENT (Weight: 35%)
For each source, evaluate:
- **Institutional Credibility**: Rate government/academic/professional/independent (1-10)
- **Domain Expertise**: Relevance to specific claims (1-10)
- **Recognition Factor**: Peer acknowledgment and reputation (1-10)
- **Track Record**: Historical accuracy and reliability (1-10)

## TRANSPARENCY EVALUATION (Weight: 25%)
Assess each source for:
- **Methodology Disclosure**: How information was gathered and validated
- **Funding Transparency**: Financial backing and potential conflicts
- **Bias Indicators**: Acknowledged limitations and perspectives
- **Correction Policy**: How errors are handled and communicated

## VERIFICATION POTENTIAL (Weight: 25%)
Analyze:
- **Primary vs Secondary**: Original research vs reported information
- **Independent Access**: Can readers verify claims independently?
- **Corroboration**: Are multiple independent sources available?
- **Currency**: Information recency and ongoing relevance

## COMPLETENESS CHECK (Weight: 15%)
Evaluate:
- **Coverage Gaps**: What perspectives or evidence types are missing?
- **Source Diversity**: Geographic, institutional, and methodological variety
- **Stakeholder Representation**: Whose voices are included/excluded?

OUTPUT STRUCTURE:
**OVERALL SOURCE QUALITY SCORE**: X/10

**STRONGEST SOURCES** (Top 3 with individual scores):
1. [Source name]: X/10 - [Brief justification]
2. [Source name]: X/10 - [Brief justification]
3. [Source name]: X/10 - [Brief justification]

**SIGNIFICANT CONCERNS** (Issues to address):
• [Specific concern]: [Impact on credibility]
• [Specific concern]: [Impact on credibility]

**VERIFICATION STRATEGY** (For readers):
• [Step 1]: [Specific action to verify information]
• [Step 2]: [Additional verification method]
• [Step 3]: [Cross-reference approach]

**MISSING SOURCE TYPES** (To strengthen analysis):
• [Type]: [Why needed and where to find]
• [Type]: [Why needed and where to find]"""

            logger.info(f"Generated source quality assessment prompt", 
                       extra={
                           'session_id': session_id,
                           'sources_count': len(sources),
                           'article_length': len(article_text) if article_text else 0
                       })
            
            return prompt
            
        except Exception as e:
            logger.error(f"Source quality prompt generation failed: {str(e)}", 
                        extra={'session_id': session_id})
            return EvidenceVerificationPrompts.assess_source_quality_fallback(sources, session_id)

    @staticmethod
    def assess_source_quality_fallback(sources: List[str], session_id: str = None) -> str:
        """Fallback source quality assessment prompt."""
        logger = logging.getLogger(f"{__name__}.EvidenceVerificationPrompts")
        logger.warning(f"Using fallback source quality prompt", extra={'session_id': session_id})
        
        sources_text = "No sources provided"
        if sources:
            sources_text = "\n".join([f"- {source}" for source in sources[:5]])
        
        return f"""Evaluate the quality and credibility of these sources:

{sources_text}

Rate each source 1-10 and explain:
1. Authority level (government, academic, media, etc.)
2. Relevance to claims being verified
3. Potential bias or limitations
4. Overall reliability assessment

Provide summary with overall score out of 10."""


class LogicalConsistencyPrompts:
    """Enhanced prompts for logical consistency and reasoning quality analysis."""

    @staticmethod
    def analyze_logical_consistency(article_text: str, 
                                  claims: List[str], 
                                  session_id: str = None) -> str:
        """
        Advanced logical consistency analysis with systematic reasoning framework.
        """
        logger = logging.getLogger(f"{__name__}.LogicalConsistencyPrompts")
        
        try:
            if not claims:
                claims = ["No claims provided"]
            
            key_claims = "\n".join([f"• {claim}" for claim in claims[:6]])
            article_excerpt = str(article_text)[:1000] if article_text else "No content provided"
            
            prompt = f"""You are Dr. Elena Vasquez, Professor of Logic and Critical Thinking at MIT with expertise in argument analysis, formal logic, and reasoning assessment. You've authored 3 books on logical fallacies and have 25+ years of experience evaluating argument quality.

<reasoning_framework>
Your systematic analysis will evaluate logical structure across:
1. PREMISE QUALITY: Foundation strength and reasonableness
2. LOGICAL FLOW: Valid inference patterns and connections
3. EVIDENCE-CONCLUSION LINKS: Support strength and relevance
4. INTERNAL CONSISTENCY: Contradiction detection and resolution
5. ALTERNATIVE EXPLANATIONS: Consideration of other possibilities
</reasoning_framework>

CONTENT FOR ANALYSIS:
{article_excerpt}

KEY CLAIMS:
{key_claims}

ANALYSIS TASKS:

## 1. ARGUMENT STRUCTURE EVALUATION
Examine the logical architecture:
- **Premise Quality**: Are foundational assumptions reasonable and well-supported?
- **Inference Validity**: Do conclusions follow logically from premises?
- **Evidence Strength**: How well does presented evidence support each conclusion?
- **Logical Gaps**: Where are connections missing or insufficiently supported?

## 2. INTERNAL CONSISTENCY ANALYSIS
Check for contradictions and alignment:
- **Claim Contradictions**: Do any claims conflict with each other?
- **Timeline Consistency**: Are dates, sequences, and chronology accurate?
- **Numerical Coherence**: Do statistics and quantitative claims align?
- **Definitional Consistency**: Are key terms used consistently throughout?

## 3. REASONING QUALITY ASSESSMENT
Evaluate thinking patterns:
- **Causal Reasoning**: Are cause-effect relationships properly established?
- **Generalization Validity**: Are broad conclusions justified by evidence scope?
- **Comparative Logic**: Are comparisons appropriate and fair?
- **Alternative Consideration**: Are other explanations acknowledged and addressed?

## 4. CRITICAL REASONING INDICATORS
Look for quality markers:
- **Qualification Language**: Appropriate use of "may," "suggests," "indicates"
- **Uncertainty Acknowledgment**: Recognition of limitations and unknowns
- **Source Integration**: How well are multiple perspectives synthesized?
- **Counterargument Handling**: How are opposing views addressed?

OUTPUT FORMAT:
**LOGICAL CONSISTENCY SCORE**: X/10

**REASONING STRENGTHS**:
• [Strength 1]: [Specific example and impact]
• [Strength 2]: [Specific example and impact]
• [Strength 3]: [Specific example and impact]

**LOGICAL WEAKNESSES**:
• [Weakness 1]: [Specific issue and improvement needed]
• [Weakness 2]: [Specific issue and improvement needed]
• [Weakness 3]: [Specific issue and improvement needed]

**CRITICAL GAPS**:
• [Gap 1]: [Missing logical connection and consequence]
• [Gap 2]: [Missing logical connection and consequence]

**REASONING QUALITY INDICATORS**:
• Qualification Language: [Assessment]
• Evidence Integration: [Assessment]  
• Alternative Consideration: [Assessment]

**IMPROVEMENT RECOMMENDATIONS**:
1. [Specific suggestion for strengthening logic]
2. [Specific suggestion for addressing gaps]
3. [Specific suggestion for enhancing reasoning]"""

            logger.info(f"Generated logical consistency analysis prompt", 
                       extra={
                           'session_id': session_id,
                           'claims_count': len(claims),
                           'article_length': len(article_text) if article_text else 0
                       })
            
            return prompt
            
        except Exception as e:
            logger.error(f"Logical consistency prompt generation failed: {str(e)}", 
                        extra={'session_id': session_id})
            return LogicalConsistencyPrompts.analyze_logical_consistency_fallback(claims, session_id)

    @staticmethod
    def analyze_logical_consistency_fallback(claims: List[str], session_id: str = None) -> str:
        """Fallback logical consistency analysis prompt."""
        logger = logging.getLogger(f"{__name__}.LogicalConsistencyPrompts")
        logger.warning(f"Using fallback logical consistency prompt", extra={'session_id': session_id})
        
        claims_text = "No claims provided"
        if claims:
            claims_text = "\n".join([f"- {claim}" for claim in claims[:5]])
        
        return f"""Analyze the logical consistency of these claims:

{claims_text}

Check for:
1. Internal contradictions
2. Logical flow between claims
3. Evidence-conclusion relationships
4. Missing logical connections

Provide:
- Overall logical consistency score (1-10)
- Main strengths in reasoning
- Key logical weaknesses or gaps
- Suggestions for improvement"""


class EvidenceGapPrompts:
    """Enhanced prompts for identifying evidence gaps and completeness issues."""

    @staticmethod
    def identify_evidence_gaps(article_text: str, 
                             claims: List[Dict[str, Any]], 
                             session_id: str = None) -> str:
        """
        Comprehensive evidence gap analysis with specific recommendations.
        """
        logger = logging.getLogger(f"{__name__}.EvidenceGapPrompts")
        
        try:
            if not claims:
                claims_summary = "No claims provided for analysis"
            else:
                claims_summary = "\n".join([
                    f"• {claim.get('text', 'Unknown claim')[:100]} (Priority: {claim.get('priority', 'Unknown')})"
                    for claim in claims[:5]
                ])
            
            article_excerpt = str(article_text)[:1000] if article_text else "No content provided"
            
            prompt = f"""You are Dr. James Patterson, Senior Research Methodology Consultant with 18+ years experience in evidence assessment for academic institutions, policy organizations, and investigative journalism teams. You specialize in identifying research gaps and recommending evidence strengthening strategies.

<gap_analysis_framework>
You will systematically identify missing evidence across:
1. QUANTITATIVE EVIDENCE: Statistics, data, measurements
2. QUALITATIVE EVIDENCE: Expert opinions, stakeholder perspectives  
3. VERIFICATION EVIDENCE: Sources, corroboration, methodology
4. CONTEXTUAL EVIDENCE: Background, comparison, temporal factors
</gap_analysis_framework>

ARTICLE CONTENT:
{article_excerpt}

CLAIMS BEING ANALYZED:
{claims_summary}

COMPREHENSIVE GAP ANALYSIS:

## 1. QUANTITATIVE EVIDENCE GAPS
Identify missing numerical support:
- **Statistical Evidence**: What key statistics would strengthen claims?
- **Data Sources**: Which authoritative datasets should be referenced?
- **Comparative Data**: What benchmarks or control groups are missing?
- **Sample Size Analysis**: Are study populations adequate for conclusions?
- **Measurement Clarity**: What specific metrics should be defined?

## 2. QUALITATIVE EVIDENCE GAPS  
Assess missing perspectives and context:
- **Expert Voices**: Which subject matter authorities should be consulted?
- **Stakeholder Input**: Whose perspectives are absent from the analysis?
- **Historical Context**: What background information is missing?
- **Alternative Viewpoints**: Which opposing or nuanced positions are ignored?
- **Case Studies**: What specific examples would illustrate key points?

## 3. VERIFICATION EVIDENCE GAPS
Examine source and methodology issues:
- **Primary Sources**: What original documents or data are missing?
- **Independent Confirmation**: What lacks corroboration from separate sources?
- **Methodology Disclosure**: Which research methods are unexplained?
- **Replication Evidence**: What findings need independent verification?
- **Source Diversity**: Which types of authoritative sources are underrepresented?

## 4. CONTEXTUAL EVIDENCE GAPS
Consider broader context needs:
- **Temporal Context**: What timeline or historical information is absent?
- **Geographic Scope**: Which regional or demographic contexts are missing?
- **Regulatory Framework**: What legal or policy background is needed?
- **Economic Factors**: Which financial implications aren't addressed?
- **Comparative Analysis**: What relevant comparisons to similar situations are absent?

OUTPUT STRUCTURE:
**EVIDENCE COMPLETENESS SCORE**: X/10

**CRITICAL GAPS** (Must Address - Impact on Credibility: High):
1. [Gap Type]: [Specific missing evidence] → [Impact: Why this matters]
2. [Gap Type]: [Specific missing evidence] → [Impact: Why this matters]
3. [Gap Type]: [Specific missing evidence] → [Impact: Why this matters]

**IMPORTANT GAPS** (Should Address - Impact on Credibility: Medium):
1. [Gap Type]: [Missing evidence] → [Impact: How this would improve analysis]
2. [Gap Type]: [Missing evidence] → [Impact: How this would improve analysis]

**EVIDENCE STRENGTHENING ROADMAP**:

**For Content Creators**:
• [Action 1]: [Specific research or reporting needed]
• [Action 2]: [Additional sources to contact]
• [Action 3]: [Data collection recommendations]

**For Readers/Fact-Checkers**:
• [Verification Step 1]: [Where to find missing information]
• [Verification Step 2]: [How to independently confirm claims]
• [Verification Step 3]: [Cross-reference strategies]

**PRIORITY RECOMMENDATIONS** (Ranked by Impact):
1. **Highest Priority**: [Gap] - [Why critical] - [Where to find evidence]
2. **High Priority**: [Gap] - [Why important] - [Where to find evidence]  
3. **Medium Priority**: [Gap] - [Why helpful] - [Where to find evidence]"""

            logger.info(f"Generated evidence gaps analysis prompt", 
                       extra={
                           'session_id': session_id,
                           'claims_count': len(claims),
                           'article_length': len(article_text) if article_text else 0
                       })
            
            return prompt
            
        except Exception as e:
            logger.error(f"Evidence gaps prompt generation failed: {str(e)}", 
                        extra={'session_id': session_id})
            return EvidenceGapPrompts.identify_evidence_gaps_fallback(claims, session_id)

    @staticmethod
    def identify_evidence_gaps_fallback(claims: List[Dict[str, Any]], session_id: str = None) -> str:
        """Fallback evidence gaps analysis prompt."""
        logger = logging.getLogger(f"{__name__}.EvidenceGapPrompts")
        logger.warning(f"Using fallback evidence gaps prompt", extra={'session_id': session_id})
        
        claims_count = len(claims) if claims else 0
        
        return f"""Identify evidence gaps in the provided content.

Claims analyzed: {claims_count}

Look for missing:
1. Statistical data or quantitative evidence
2. Expert opinions or authoritative sources  
3. Independent verification or corroboration
4. Historical context or background information
5. Alternative perspectives or counterarguments

Provide:
- Evidence completeness score (1-10)
- Top 3 critical gaps that should be addressed
- Recommendations for finding missing information
- Impact assessment of each gap on overall credibility"""


class StructuredOutputPrompts:
    """Enhanced prompts for structured JSON output with validation."""

    @staticmethod
    def extract_verification_data(analysis_text: str, session_id: str = None) -> str:
        """
        Enhanced structured data extraction with validation and error handling.
        """
        logger = logging.getLogger(f"{__name__}.StructuredOutputPrompts")
        
        try:
            if not analysis_text or not isinstance(analysis_text, str):
                raise_prompt_generation_error(
                    'structured_output',
                    "Analysis text must be non-empty string",
                    {'analysis_text_type': type(analysis_text).__name__},
                    session_id
                )
            
            analysis_excerpt = analysis_text[:2000] if len(analysis_text) > 2000 else analysis_text
            
            prompt = f"""You are a data extraction specialist. Extract key information from the analysis and format it as valid JSON.

ANALYSIS TEXT TO PROCESS:
{analysis_excerpt}

Extract information into this EXACT JSON structure (all fields required):

{{
  "verification_sources": [
    {{
      "claim": "exact claim text from analysis",
      "url": "specific verification URL (never homepage)",
      "institution": "authoritative organization name",
      "confidence": 0.8,
      "verification_type": "primary_source",
      "quality_score": 0.9
    }}
  ],
  "source_quality": {{
    "overall_score": 7.5,
    "strongest_sources": ["source1", "source2", "source3"],
    "quality_concerns": ["concern1", "concern2"]
  }},
  "logical_consistency": {{
    "consistency_score": 8.2,
    "reasoning_strengths": ["strength1", "strength2"],
    "logical_weaknesses": ["weakness1", "weakness2"]
  }},
  "evidence_gaps": {{
    "completeness_score": 6.8,
    "critical_gaps": ["gap1", "gap2"],
    "recommendations": ["rec1", "rec2"]
  }}
}}

STRICT REQUIREMENTS:
- All scores must be decimals between 0.1 and 10.0
- Arrays limited to maximum 3 items each
- URLs must be specific pages, never homepages
- All text fields must be meaningful, not placeholder text
- Output must be valid JSON only - no additional text

Extract and format the data now:"""

            logger.info(f"Generated structured output extraction prompt", 
                       extra={
                           'session_id': session_id,
                           'analysis_length': len(analysis_text)
                       })
            
            return prompt
            
        except PromptGenerationError:
            raise
        except Exception as e:
            logger.error(f"Structured output prompt generation failed: {str(e)}", 
                        extra={'session_id': session_id})
            return StructuredOutputPrompts.extract_verification_data_fallback(session_id)

    @staticmethod
    def extract_verification_data_fallback(session_id: str = None) -> str:
        """Fallback structured output extraction prompt."""
        logger = logging.getLogger(f"{__name__}.StructuredOutputPrompts")
        logger.warning(f"Using fallback structured output prompt", extra={'session_id': session_id})
        
        return """{
  "verification_sources": [],
  "source_quality": {
    "overall_score": 5.0,
    "strongest_sources": [],
    "quality_concerns": ["Analysis unavailable"]
  },
  "logical_consistency": {
    "consistency_score": 5.0,
    "reasoning_strengths": [],
    "logical_weaknesses": ["Analysis unavailable"]
  },
  "evidence_gaps": {
    "completeness_score": 5.0,
    "critical_gaps": ["Full analysis unavailable"],
    "recommendations": ["Retry analysis"]
  }
}"""


class DomainSpecificPrompts:
    """Enhanced domain-specific prompt templates with specialized instructions."""

    MEDICAL_VERIFICATION_ENHANCED = """
For MEDICAL/HEALTH claims, prioritize these authoritative sources in order:

**TIER 1 (Highest Authority)**:
• PubMed/MEDLINE: https://pubmed.ncbi.nlm.nih.gov/[PMID] (peer-reviewed research)
• FDA Official: https://www.fda.gov/news-events/press-announcements/[specific-announcement]
• CDC Guidelines: https://www.cdc.gov/[specific-health-topic]/[guidance-page]
• WHO Position: https://www.who.int/news-room/[specific-statement]

**TIER 2 (High Authority)**:
• NIH Institutes: https://www.[institute].nih.gov/[specific-research-findings]
• Medical Journals: https://www.[journal].com/article/[doi-or-id]
• Professional Associations: https://www.[association].org/[position-statements]

**VERIFICATION STRATEGIES**:
• Search: "[specific drug/treatment/condition]" site:pubmed.ncbi.nlm.nih.gov
• Cross-reference: Multiple independent clinical studies
• Check: FDA approval status and safety communications
• Validate: WHO/CDC official position statements

**RED FLAGS TO AVOID**:
• Personal health blogs or testimonials
• Non-peer-reviewed preprint servers
• Commercial websites selling products
• Social media health claims
"""

    POLITICAL_VERIFICATION_ENHANCED = """
For POLITICAL/POLICY claims, prioritize these authoritative sources:

**TIER 1 (Highest Authority)**:
• Congress Records: https://www.congress.gov/bill/[congress]/[bill-type]/[number]
• Government Data: https://data.gov/dataset/[specific-dataset]
• Official Agencies: https://www.[agency].gov/[specific-policy-or-data]
• Federal Register: https://www.federalregister.gov/documents/[date]/[document-id]

**TIER 2 (High Authority)**:
• Congressional Budget Office: https://www.cbo.gov/publication/[report-number]
• Government Accountability Office: https://www.gao.gov/products/[report-id]
• Bureau of Statistics: https://www.[bureau].gov/[specific-statistics]

**VERIFICATION STRATEGIES**:
• Search: "[policy/bill name]" site:congress.gov
• Cross-check: Official vote records and legislative history  
• Validate: Government agency implementation data
• Confirm: Multiple official government sources

**RED FLAGS TO AVOID**:
• Partisan advocacy websites
• Unsourced political blogs
• Social media political claims
• Opinion pieces without factual basis
"""

    SCIENTIFIC_VERIFICATION_ENHANCED = """
For SCIENTIFIC/RESEARCH claims, prioritize these authoritative sources:

**TIER 1 (Highest Authority)**:
• Peer-reviewed Journals: https://www.[journal].com/articles/[doi]
• ArXiv (Physics/Math): https://arxiv.org/abs/[paper-id]
• University Research: https://www.[university].edu/research/[specific-study]
• Government Research: https://www.[agency].gov/research/[study-id]

**TIER 2 (High Authority)**:
• Scientific Organizations: https://www.[organization].org/[position-statement]
• Research Institutions: https://www.[institution].org/publications/[report]
• Academic Repositories: https://[repository].[university].edu/[paper-id]

**VERIFICATION STRATEGIES**:
• Search: "[research topic] peer reviewed" site:edu
• Cross-reference: Citation index and replication studies
• Check: Journal impact factor and peer review process
• Validate: Independent confirmation by other researchers

**QUALITY INDICATORS**:
• Peer review status clearly stated
• Methodology section detailed
• Data availability mentioned
• Conflict of interest disclosed
• Sample size adequate for conclusions
"""


class PromptValidator:
    """Enhanced prompt validation with comprehensive error checking."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize prompt validator with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.PromptValidator")

    def validate_url_specificity(self, url: str) -> bool:
        """
        Enhanced URL specificity validation with detailed pattern checking.
        
        Args:
            url: URL to validate for specificity
            
        Returns:
            True if URL is specific, False if generic
        """
        if not url or not isinstance(url, str):
            return False
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url.strip())
        except:
            return False
        
        # Enhanced generic patterns
        generic_patterns = [
            r'^/?$',                          # Root path only
            r'^/index\.html?$',               # Index pages
            r'^/home/?$',                     # Home pages  
            r'^/main/?$',                     # Main pages
            r'^/default\.html?$',             # Default pages
            r'^/about/?$',                    # About pages
            r'^/news/?$',                     # General news sections
            r'^/research/?$',                 # General research sections
            r'^/publications/?$',             # General publications
            r'^/press/?$',                    # General press sections
            r'^/media/?$',                    # General media sections
        ]
        
        path = parsed.path.lower() if parsed.path else ""
        
        for pattern in generic_patterns:
            if re.match(pattern, path):
                self.logger.debug(f"URL failed specificity check: {url} matched pattern {pattern}")
                return False
        
        # URL should have meaningful path beyond domain
        path_segments = [seg for seg in path.split('/') if seg]
        if len(path_segments) < 2:
            self.logger.debug(f"URL failed specificity check: {url} has insufficient path segments")
            return False
        
        # Positive indicators of specificity
        specific_indicators = [
            'article', 'study', 'research', 'report', 'publication', 'paper',
            'analysis', 'findings', 'results', 'data', 'statistics', 'document',
            'announcement', 'statement', 'press-release', 'guidance', 'policy'
        ]
        
        has_specific_indicator = any(indicator in path.lower() for indicator in specific_indicators)
        
        # Query parameters or fragments indicate specificity
        has_specificity_markers = bool(parsed.query or parsed.fragment)
        
        is_specific = len(path_segments) >= 2 and (has_specific_indicator or has_specificity_markers)
        
        self.logger.debug(f"URL specificity validation: {url} -> {is_specific}")
        return is_specific

    def extract_confidence_score(self, text: str) -> float:
        """
        Enhanced confidence score extraction with multiple pattern support.
        
        Args:
            text: Text to extract confidence score from
            
        Returns:
            Extracted confidence score or default value
        """
        if not text or not isinstance(text, str):
            return 0.5
        
        # Multiple patterns for confidence extraction
        confidence_patterns = [
            r'confidence[:\s]+([0-9]\.[0-9]+)',
            r'confidence[:\s]+([0-9]+\.?[0-9]*)',
            r'confidence[:\s]+([0-9]+)%',
            r'score[:\s]+([0-9]\.[0-9]+)',
            r'rating[:\s]+([0-9]\.[0-9]+)',
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize percentage scores
                    if score > 1.0:
                        score = score / 100.0
                    # Ensure valid range
                    return max(0.1, min(1.0, score))
                except ValueError:
                    continue
        
        self.logger.debug(f"No confidence score found in text, using default 0.5")
        return 0.5  # Default moderate confidence

    def validate_prompt_parameters(self, prompt_type: str, **kwargs) -> bool:
        """
        Validate parameters for prompt generation.
        
        Args:
            prompt_type: Type of prompt being generated
            **kwargs: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        required_params = {
            'verification_sources': ['article_text', 'claims'],
            'source_quality': ['article_text', 'sources'],
            'logical_consistency': ['article_text', 'claims'],
            'evidence_gaps': ['article_text', 'claims'],
            'structured_output': ['analysis_text']
        }
        
        if prompt_type not in required_params:
            self.logger.warning(f"Unknown prompt type for validation: {prompt_type}")
            return False
        
        missing_params = []
        for param in required_params[prompt_type]:
            if param not in kwargs or not kwargs[param]:
                missing_params.append(param)
        
        if missing_params:
            self.logger.error(f"Missing required parameters for {prompt_type}: {missing_params}")
            return False
        
        self.logger.debug(f"Prompt parameters validated for {prompt_type}")
        return True


def get_prompt_template(prompt_type: str, session_id: str = None, **kwargs) -> str:
    """
    Enhanced prompt template retrieval with validation and fallback handling.
    
    Args:
        prompt_type: Type of prompt needed
        session_id: Optional session ID for tracking
        **kwargs: Parameters for prompt formatting
        
    Returns:
        Formatted prompt string
        
    Raises:
        PromptGenerationError: If prompt generation fails
    """
    logger = logging.getLogger(f"{__name__}.get_prompt_template")
    
    try:
        # Validate prompt type and parameters
        validator = PromptValidator()
        if not validator.validate_prompt_parameters(prompt_type, **kwargs):
            raise_prompt_generation_error(
                prompt_type,
                f"Invalid parameters for prompt type: {prompt_type}",
                {'provided_params': list(kwargs.keys())},
                session_id
            )
        
        # Prompt mapping with fallback support
        prompt_mapping = {
            'verification_sources': (
                EvidenceVerificationPrompts.generate_verification_sources,
                EvidenceVerificationPrompts.generate_verification_sources_fallback
            ),
            'source_quality': (
                EvidenceVerificationPrompts.assess_source_quality,
                EvidenceVerificationPrompts.assess_source_quality_fallback
            ),
            'logical_consistency': (
                LogicalConsistencyPrompts.analyze_logical_consistency,
                LogicalConsistencyPrompts.analyze_logical_consistency_fallback
            ),
            'evidence_gaps': (
                EvidenceGapPrompts.identify_evidence_gaps,
                EvidenceGapPrompts.identify_evidence_gaps_fallback
            ),
            'structured_output': (
                StructuredOutputPrompts.extract_verification_data,
                StructuredOutputPrompts.extract_verification_data_fallback
            )
        }
        
        if prompt_type not in prompt_mapping:
            raise_prompt_generation_error(
                prompt_type,
                f"Unknown prompt type: {prompt_type}",
                {'available_types': list(prompt_mapping.keys())},
                session_id
            )
        
        # Try main prompt generation
        main_func, fallback_func = prompt_mapping[prompt_type]
        
        try:
            # Add session_id to kwargs if the function supports it
            import inspect
            sig = inspect.signature(main_func)
            if 'session_id' in sig.parameters:
                kwargs['session_id'] = session_id
            
            prompt = main_func(**kwargs)
            
            logger.info(f"Generated {prompt_type} prompt successfully", 
                       extra={
                           'session_id': session_id,
                           'prompt_length': len(prompt)
                       })
            
            return prompt
            
        except Exception as main_error:
            logger.warning(f"Main prompt generation failed for {prompt_type}, trying fallback: {str(main_error)}", 
                          extra={'session_id': session_id})
            
            # Try fallback prompt
            try:
                sig = inspect.signature(fallback_func)
                if 'session_id' in sig.parameters:
                    kwargs['session_id'] = session_id
                
                fallback_prompt = fallback_func(**kwargs)
                
                logger.info(f"Generated fallback {prompt_type} prompt", 
                           extra={'session_id': session_id})
                
                return fallback_prompt
                
            except Exception as fallback_error:
                logger.error(f"Both main and fallback prompt generation failed for {prompt_type}", 
                            extra={'session_id': session_id})
                raise_prompt_generation_error(
                    prompt_type,
                    f"All prompt generation methods failed. Main: {str(main_error)}, Fallback: {str(fallback_error)}",
                    {'main_error': str(main_error), 'fallback_error': str(fallback_error)},
                    session_id
                )
        
    except PromptGenerationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prompt template retrieval: {str(e)}", 
                    extra={'session_id': session_id})
        raise_prompt_generation_error(
            prompt_type,
            f"Unexpected error: {str(e)}",
            {'error_type': type(e).__name__},
            session_id
        )


# Testing functionality
if __name__ == "__main__":
    """Test prompt generation functionality."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    test_session_id = "prompt_test_456"
    
    # Test verification sources prompt
    print("=== VERIFICATION SOURCES PROMPT TEST ===")
    try:
        test_article = "According to a Harvard study published in Nature, the treatment showed 85% efficacy in clinical trials with 2,400 participants."
        test_claims = [
            {'text': 'Harvard study shows treatment efficacy', 'verifiability_score': 8},
            {'text': 'Nature publication with clinical trial data', 'verifiability_score': 9}
        ]
        
        prompt = get_prompt_template(
            'verification_sources',
            article_text=test_article,
            claims=test_claims,
            session_id=test_session_id
        )
        
        print(f"✅ Generated verification sources prompt ({len(prompt)} characters)")
        print(f"Preview: {prompt[:200]}...")
        
    except Exception as e:
        print(f"❌ Verification sources prompt test failed: {str(e)}")
    
    # Test URL validation
    print("\n=== URL VALIDATION TEST ===")
    validator = PromptValidator()
    test_urls = [
        'https://www.nature.com/articles/nature12345',  # Should pass
        'https://www.cdc.gov/',  # Should fail (homepage)
        'https://pubmed.ncbi.nlm.nih.gov/12345678/',  # Should pass
    ]
    
    for url in test_urls:
        is_specific = validator.validate_url_specificity(url)
        print(f"URL: {url} -> {'✅ Specific' if is_specific else '❌ Generic'}")
    
    # Test confidence extraction
    print("\n=== CONFIDENCE EXTRACTION TEST ===")
    test_texts = [
        "The confidence is 0.85 for this source",
        "Confidence: 7.5/10 based on analysis",
        "High confidence score of 90%",
        "No confidence information available"
    ]
    
    for text in test_texts:
        confidence = validator.extract_confidence_score(text)
        print(f"Text: '{text}' -> Confidence: {confidence}")
    
    print("\n✅ Prompt system tests completed")
