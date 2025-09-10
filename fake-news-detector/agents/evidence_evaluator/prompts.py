# agents/evidence_evaluator/prompts.py

"""
Evidence Evaluator Prompts Module

Industry-standard prompt templates for evidence evaluation with structured output
and URL specificity enforcement. Uses Chain-of-Thought and Few-Shot patterns.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PromptResponse:
    """Structured response container for prompt outputs."""
    content: str
    metadata: Dict[str, Any]

class EvidenceVerificationPrompts:
    """Evidence verification prompts with URL specificity enforcement."""
    
    @staticmethod
    def generate_verification_sources(article_text: str, claims: List[Dict[str, Any]]) -> str:
        """
        Generate specific verification sources with actionable URLs.
        Uses Chain-of-Thought and structured output patterns.
        """
        claims_text = "\n".join([f"{i+1}. {claim.get('text', '')}" for i, claim in enumerate(claims[:5])])
        
        return f"""You are a fact-checking specialist providing SPECIFIC verification sources for exact claims.

ARTICLE CONTENT: {article_text[:1200]}

CLAIMS TO VERIFY:
{claims_text}

TASK: Provide exactly 5 specific verification sources using this EXACT format:

## VERIFICATION SOURCE 1
**CLAIM**: "Exact quote from article being verified"
**INSTITUTION**: Name of authoritative organization
**SPECIFIC_URL**: https://domain.org/exact-page-that-verifies-this-claim
**VERIFICATION_TYPE**: primary_source|expert_analysis|official_data|research_study
**SEARCH_STRATEGY**: "exact keywords" site:domain.org
**CONFIDENCE**: 0.X (decimal from 0.1 to 1.0)

## VERIFICATION SOURCE 2
[Continue same format...]

CRITICAL REQUIREMENTS:
1. URLs must be SPECIFIC to the claim, not homepage
2. Each URL should directly address the exact claim
3. Use authoritative domains: .gov, .edu, major institutions
4. Provide specific search terms to find the verification
5. Match institution credibility to claim type

EXAMPLES OF GOOD URLS:
- https://www.cdc.gov/vaccines/covid-19/clinical-considerations/managing-anaphylaxis.html
- https://pubmed.ncbi.nlm.nih.gov/34289274/
- https://www.fda.gov/news-events/press-announcements/fda-approves-first-covid-19-vaccine

EXAMPLES OF BAD URLS (DO NOT USE):
- https://www.cdc.gov/ (too general)
- https://pubmed.ncbi.nlm.nih.gov/ (homepage only)
- https://www.fda.gov/ (not specific)

OUTPUT: Provide exactly 5 sources in the specified format."""

    @staticmethod
    def assess_source_quality(article_text: str, sources: List[str]) -> str:
        """Assess source quality with structured evaluation."""
        sources_list = "\n".join([f"- {source}" for source in sources[:10]])
        
        return f"""Analyze source quality using systematic evaluation framework.

CONTENT: {article_text[:800]}

IDENTIFIED SOURCES:
{sources_list}

EVALUATION FRAMEWORK:

## AUTHORITY ASSESSMENT
For each source, evaluate:
- **Institutional Credibility**: Government/Academic/Professional/Independent
- **Domain Expertise**: Directly relevant to claims
- **Recognition**: Peer-acknowledged authority
- **Track Record**: History of accurate information

## TRANSPARENCY EVALUATION  
- **Methodology Disclosure**: How information was gathered
- **Funding Sources**: Financial backing transparency  
- **Bias Indicators**: Potential conflicts of interest
- **Correction Policy**: How errors are handled

## VERIFICATION POTENTIAL
- **Primary vs Secondary**: Original vs. reported information
- **Accessibility**: Can readers verify claims independently
- **Corroboration**: Multiple independent sources available
- **Recency**: Information currency and relevance

OUTPUT FORMAT:
**OVERALL SOURCE QUALITY**: X/10
**STRONGEST SOURCES**: [List top 3 with scores]
**WEAKEST SOURCES**: [List concerning sources with reasons]
**VERIFICATION RECOMMENDATIONS**: [Specific steps for readers]
**MISSING SOURCE TYPES**: [What additional sources needed]"""

class LogicalConsistencyPrompts:
    """Prompts for logical consistency and reasoning quality analysis."""
    
    @staticmethod
    def analyze_logical_consistency(article_text: str, claims: List[str]) -> str:
        """Analyze logical consistency using structured reasoning framework."""
        key_claims = "\n".join([f"• {claim}" for claim in claims[:6]])
        
        return f"""Conduct systematic logical consistency analysis.

CONTENT: {article_text[:1000]}

KEY CLAIMS:
{key_claims}

ANALYSIS FRAMEWORK:

## ARGUMENT STRUCTURE EVALUATION
1. **Premise Quality**: Are foundational assumptions reasonable?
2. **Evidence-Conclusion Link**: Does evidence actually support conclusions?
3. **Logical Flow**: Do conclusions follow logically from premises?
4. **Missing Links**: Where are logical gaps or leaps?

## INTERNAL CONSISTENCY CHECK
- **Claim Contradictions**: Do claims contradict each other?
- **Timeline Consistency**: Do dates and sequences align?
- **Numerical Consistency**: Do statistics add up correctly?
- **Definitional Consistency**: Are terms used consistently?

## REASONING QUALITY ASSESSMENT
- **Causal Claims**: Are cause-effect relationships supported?
- **Generalization Validity**: Are broad conclusions justified?
- **Comparison Fairness**: Are comparisons appropriate and fair?
- **Alternative Explanations**: Are other possibilities considered?

OUTPUT FORMAT:
**LOGICAL CONSISTENCY SCORE**: X/10
**STRONG REASONING ELEMENTS**: [Well-supported arguments]
**LOGICAL WEAKNESSES**: [Specific reasoning problems]
**CRITICAL GAPS**: [Missing logical connections]
**FALLACY ALERTS**: [Potential logical fallacies identified]
**READER GUIDANCE**: [What to question or verify]"""

class EvidenceGapPrompts:
    """Prompts for identifying evidence gaps and completeness issues."""
    
    @staticmethod
    def identify_evidence_gaps(article_text: str, claims: List[Dict[str, Any]]) -> str:
        """Identify critical evidence gaps with specific recommendations."""
        claims_summary = "\n".join([
            f"• {claim.get('text', '')[:100]} (Priority: {claim.get('priority', 'Unknown')})" 
            for claim in claims[:5]
        ])
        
        return f"""Identify critical evidence gaps using systematic analysis.

ARTICLE: {article_text[:1000]}

CLAIMS ANALYZED:
{claims_summary}

GAP ANALYSIS FRAMEWORK:

## QUANTITATIVE EVIDENCE GAPS
- **Missing Statistics**: What numbers would strengthen claims?
- **Absent Data Sources**: What datasets should be referenced?
- **Comparative Context**: What comparisons are missing?
- **Sample Size Issues**: Are study populations adequate?

## QUALITATIVE EVIDENCE GAPS  
- **Expert Perspectives**: Which authorities should be consulted?
- **Stakeholder Views**: Whose voices are absent?
- **Historical Context**: What background information is missing?
- **Alternative Viewpoints**: What opposing perspectives are ignored?

## VERIFICATION EVIDENCE GAPS
- **Primary Source Access**: What original documents are missing?
- **Independent Confirmation**: What hasn't been corroborated?
- **Methodology Transparency**: What processes aren't explained?
- **Replication Evidence**: What findings lack independent verification?

## CONTEXTUAL EVIDENCE GAPS
- **Temporal Context**: What timeline information is missing?
- **Geographic Context**: What location-specific data is absent?
- **Regulatory Context**: What legal/policy background is missing?
- **Economic Context**: What financial implications aren't addressed?

OUTPUT FORMAT:
**EVIDENCE COMPLETENESS SCORE**: X/10
**CRITICAL GAPS** (Must Address):
- [Gap 1]: Specific missing evidence with impact
- [Gap 2]: Specific missing evidence with impact

**IMPORTANT GAPS** (Should Address):
- [Gap 1]: Missing evidence that would improve confidence
- [Gap 2]: Missing evidence that would improve confidence

**GAP-FILLING RECOMMENDATIONS**:
- **For Publishers**: What additional reporting needed
- **For Readers**: Where to find missing information
- **Verification Strategy**: How to independently confirm claims"""

class StructuredOutputPrompts:
    """Prompts that enforce structured JSON output for programmatic processing."""
    
    @staticmethod
    def extract_verification_data(analysis_text: str) -> str:
        """Extract structured verification data from analysis text."""
        return f"""Extract structured verification data from the analysis.

ANALYSIS TEXT: {analysis_text}

Extract information into this EXACT JSON structure:

{{
  "verification_sources": [
    {{
      "claim": "exact claim text",
      "url": "specific verification URL", 
      "institution": "authoritative organization",
      "confidence": 0.X,
      "verification_type": "primary_source|expert_analysis|official_data|research_study",
      "quality_score": 0.X
    }}
  ],
  "source_quality": {{
    "overall_score": X.X,
    "strongest_sources": ["source1", "source2"],
    "quality_concerns": ["concern1", "concern2"]
  }},
  "logical_consistency": {{
    "consistency_score": X.X,
    "reasoning_strengths": ["strength1", "strength2"], 
    "logical_weaknesses": ["weakness1", "weakness2"]
  }},
  "evidence_gaps": {{
    "completeness_score": X.X,
    "critical_gaps": ["gap1", "gap2"],
    "recommendations": ["rec1", "rec2"]
  }}
}}

REQUIREMENTS:
- All scores as decimals (0.1 to 1.0)
- URLs must be specific, not homepages
- Arrays limited to top 3 items each
- Valid JSON format only"""

class DomainSpecificPrompts:
    """Domain-specific prompt templates for different content types."""
    
    MEDICAL_VERIFICATION = """
    For medical claims, prioritize:
    - PubMed/MEDLINE database searches
    - FDA official statements and approvals
    - CDC guidelines and recommendations  
    - WHO official positions
    - Peer-reviewed medical journals
    - Professional medical association statements
    
    SPECIFIC DOMAINS:
    - https://pubmed.ncbi.nlm.nih.gov/ (research studies)
    - https://www.fda.gov/news-events/ (regulatory actions)
    - https://www.cdc.gov/ (health guidelines)
    - https://www.who.int/ (global health positions)
    """
    
    POLITICAL_VERIFICATION = """
    For political claims, prioritize:
    - Official government websites (.gov domains)
    - Congressional records and voting databases
    - Official policy documents and legislation
    - Government statistics and data portals
    - Official statements from agencies
    - Verified electoral data sources
    
    SPECIFIC DOMAINS:
    - https://www.congress.gov/ (legislative information)
    - https://www.gpo.gov/ (official government documents)
    - https://data.gov/ (government datasets)
    - https://www.fec.gov/ (electoral information)
    """
    
    SCIENTIFIC_VERIFICATION = """
    For scientific claims, prioritize:
    - Peer-reviewed journal articles
    - University research publications
    - Government research agencies (NSF, NIH, NASA)
    - Professional scientific organizations
    - Institutional research repositories
    - Independent research institutions
    
    SPECIFIC DOMAINS:
    - https://pubmed.ncbi.nlm.nih.gov/ (life sciences)
    - https://arxiv.org/ (preprint research)
    - https://www.nsf.gov/ (National Science Foundation)
    - https://www.nature.com/ (Nature publications)
    """

class PromptValidator:
    """Validation methods for prompt inputs and outputs."""
    
    @staticmethod
    def validate_url_specificity(url: str) -> bool:
        """Validate that URL is specific and not just a homepage."""
        generic_patterns = [
            r'https?://[^/]+/?$',  # Just domain with optional trailing slash
            r'https?://[^/]+/index\.html?$',  # Index pages
            r'https?://[^/]+/home/?$',  # Home pages
        ]
        
        import re
        for pattern in generic_patterns:
            if re.match(pattern, url):
                return False
        
        # URL should have meaningful path beyond domain
        return len(url.split('/')) > 3
    
    @staticmethod
    def extract_confidence_score(text: str) -> float:
        """Extract confidence score from analysis text."""
        import re
        confidence_pattern = r'confidence[:\s]+([0-9]\.[0-9]+)'
        match = re.search(confidence_pattern, text.lower())
        if match:
            return float(match.group(1))
        return 0.5  # Default moderate confidence

def get_prompt_template(prompt_type: str, **kwargs) -> str:
    """
    Get specific prompt template with parameters.
    
    Args:
        prompt_type: Type of prompt needed
        **kwargs: Parameters for prompt formatting
        
    Returns:
        Formatted prompt string
    """
    prompt_mapping = {
        'verification_sources': EvidenceVerificationPrompts.generate_verification_sources,
        'source_quality': EvidenceVerificationPrompts.assess_source_quality,
        'logical_consistency': LogicalConsistencyPrompts.analyze_logical_consistency,
        'evidence_gaps': EvidenceGapPrompts.identify_evidence_gaps,
        'structured_output': StructuredOutputPrompts.extract_verification_data,
    }
    
    if prompt_type not in prompt_mapping:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompt_mapping[prompt_type](**kwargs)
# agents/evidence_evaluator/prompts.py

"""
Evidence Evaluator Prompts Module

Industry-standard prompt templates for evidence evaluation with structured output
and URL specificity enforcement. Uses Chain-of-Thought and Few-Shot patterns.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PromptResponse:
    """Structured response container for prompt outputs."""
    content: str
    metadata: Dict[str, Any]

class EvidenceVerificationPrompts:
    """Evidence verification prompts with URL specificity enforcement."""
    
    @staticmethod
    def generate_verification_sources(article_text: str, claims: List[Dict[str, Any]]) -> str:
        """
        Generate specific verification sources with actionable URLs.
        Uses Chain-of-Thought and structured output patterns.
        """
        claims_text = "\n".join([f"{i+1}. {claim.get('text', '')}" for i, claim in enumerate(claims[:5])])
        
        return f"""You are a fact-checking specialist providing SPECIFIC verification sources for exact claims.

ARTICLE CONTENT: {article_text[:1200]}

CLAIMS TO VERIFY:
{claims_text}

TASK: Provide exactly 5 specific verification sources using this EXACT format:

## VERIFICATION SOURCE 1
**CLAIM**: "Exact quote from article being verified"
**INSTITUTION**: Name of authoritative organization
**SPECIFIC_URL**: https://domain.org/exact-page-that-verifies-this-claim
**VERIFICATION_TYPE**: primary_source|expert_analysis|official_data|research_study
**SEARCH_STRATEGY**: "exact keywords" site:domain.org
**CONFIDENCE**: 0.X (decimal from 0.1 to 1.0)

## VERIFICATION SOURCE 2
[Continue same format...]

CRITICAL REQUIREMENTS:
1. URLs must be SPECIFIC to the claim, not homepage
2. Each URL should directly address the exact claim
3. Use authoritative domains: .gov, .edu, major institutions
4. Provide specific search terms to find the verification
5. Match institution credibility to claim type

EXAMPLES OF GOOD URLS:
- https://www.cdc.gov/vaccines/covid-19/clinical-considerations/managing-anaphylaxis.html
- https://pubmed.ncbi.nlm.nih.gov/34289274/
- https://www.fda.gov/news-events/press-announcements/fda-approves-first-covid-19-vaccine

EXAMPLES OF BAD URLS (DO NOT USE):
- https://www.cdc.gov/ (too general)
- https://pubmed.ncbi.nlm.nih.gov/ (homepage only)
- https://www.fda.gov/ (not specific)

OUTPUT: Provide exactly 5 sources in the specified format."""

    @staticmethod
    def assess_source_quality(article_text: str, sources: List[str]) -> str:
        """Assess source quality with structured evaluation."""
        sources_list = "\n".join([f"- {source}" for source in sources[:10]])
        
        return f"""Analyze source quality using systematic evaluation framework.

CONTENT: {article_text[:800]}

IDENTIFIED SOURCES:
{sources_list}

EVALUATION FRAMEWORK:

## AUTHORITY ASSESSMENT
For each source, evaluate:
- **Institutional Credibility**: Government/Academic/Professional/Independent
- **Domain Expertise**: Directly relevant to claims
- **Recognition**: Peer-acknowledged authority
- **Track Record**: History of accurate information

## TRANSPARENCY EVALUATION  
- **Methodology Disclosure**: How information was gathered
- **Funding Sources**: Financial backing transparency  
- **Bias Indicators**: Potential conflicts of interest
- **Correction Policy**: How errors are handled

## VERIFICATION POTENTIAL
- **Primary vs Secondary**: Original vs. reported information
- **Accessibility**: Can readers verify claims independently
- **Corroboration**: Multiple independent sources available
- **Recency**: Information currency and relevance

OUTPUT FORMAT:
**OVERALL SOURCE QUALITY**: X/10
**STRONGEST SOURCES**: [List top 3 with scores]
**WEAKEST SOURCES**: [List concerning sources with reasons]
**VERIFICATION RECOMMENDATIONS**: [Specific steps for readers]
**MISSING SOURCE TYPES**: [What additional sources needed]"""

class LogicalConsistencyPrompts:
    """Prompts for logical consistency and reasoning quality analysis."""
    
    @staticmethod
    def analyze_logical_consistency(article_text: str, claims: List[str]) -> str:
        """Analyze logical consistency using structured reasoning framework."""
        key_claims = "\n".join([f"• {claim}" for claim in claims[:6]])
        
        return f"""Conduct systematic logical consistency analysis.

CONTENT: {article_text[:1000]}

KEY CLAIMS:
{key_claims}

ANALYSIS FRAMEWORK:

## ARGUMENT STRUCTURE EVALUATION
1. **Premise Quality**: Are foundational assumptions reasonable?
2. **Evidence-Conclusion Link**: Does evidence actually support conclusions?
3. **Logical Flow**: Do conclusions follow logically from premises?
4. **Missing Links**: Where are logical gaps or leaps?

## INTERNAL CONSISTENCY CHECK
- **Claim Contradictions**: Do claims contradict each other?
- **Timeline Consistency**: Do dates and sequences align?
- **Numerical Consistency**: Do statistics add up correctly?
- **Definitional Consistency**: Are terms used consistently?

## REASONING QUALITY ASSESSMENT
- **Causal Claims**: Are cause-effect relationships supported?
- **Generalization Validity**: Are broad conclusions justified?
- **Comparison Fairness**: Are comparisons appropriate and fair?
- **Alternative Explanations**: Are other possibilities considered?

OUTPUT FORMAT:
**LOGICAL CONSISTENCY SCORE**: X/10
**STRONG REASONING ELEMENTS**: [Well-supported arguments]
**LOGICAL WEAKNESSES**: [Specific reasoning problems]
**CRITICAL GAPS**: [Missing logical connections]
**FALLACY ALERTS**: [Potential logical fallacies identified]
**READER GUIDANCE**: [What to question or verify]"""

class EvidenceGapPrompts:
    """Prompts for identifying evidence gaps and completeness issues."""
    
    @staticmethod
    def identify_evidence_gaps(article_text: str, claims: List[Dict[str, Any]]) -> str:
        """Identify critical evidence gaps with specific recommendations."""
        claims_summary = "\n".join([
            f"• {claim.get('text', '')[:100]} (Priority: {claim.get('priority', 'Unknown')})" 
            for claim in claims[:5]
        ])
        
        return f"""Identify critical evidence gaps using systematic analysis.

ARTICLE: {article_text[:1000]}

CLAIMS ANALYZED:
{claims_summary}

GAP ANALYSIS FRAMEWORK:

## QUANTITATIVE EVIDENCE GAPS
- **Missing Statistics**: What numbers would strengthen claims?
- **Absent Data Sources**: What datasets should be referenced?
- **Comparative Context**: What comparisons are missing?
- **Sample Size Issues**: Are study populations adequate?

## QUALITATIVE EVIDENCE GAPS  
- **Expert Perspectives**: Which authorities should be consulted?
- **Stakeholder Views**: Whose voices are absent?
- **Historical Context**: What background information is missing?
- **Alternative Viewpoints**: What opposing perspectives are ignored?

## VERIFICATION EVIDENCE GAPS
- **Primary Source Access**: What original documents are missing?
- **Independent Confirmation**: What hasn't been corroborated?
- **Methodology Transparency**: What processes aren't explained?
- **Replication Evidence**: What findings lack independent verification?

## CONTEXTUAL EVIDENCE GAPS
- **Temporal Context**: What timeline information is missing?
- **Geographic Context**: What location-specific data is absent?
- **Regulatory Context**: What legal/policy background is missing?
- **Economic Context**: What financial implications aren't addressed?

OUTPUT FORMAT:
**EVIDENCE COMPLETENESS SCORE**: X/10
**CRITICAL GAPS** (Must Address):
- [Gap 1]: Specific missing evidence with impact
- [Gap 2]: Specific missing evidence with impact

**IMPORTANT GAPS** (Should Address):
- [Gap 1]: Missing evidence that would improve confidence
- [Gap 2]: Missing evidence that would improve confidence

**GAP-FILLING RECOMMENDATIONS**:
- **For Publishers**: What additional reporting needed
- **For Readers**: Where to find missing information
- **Verification Strategy**: How to independently confirm claims"""

class StructuredOutputPrompts:
    """Prompts that enforce structured JSON output for programmatic processing."""
    
    @staticmethod
    def extract_verification_data(analysis_text: str) -> str:
        """Extract structured verification data from analysis text."""
        return f"""Extract structured verification data from the analysis.

ANALYSIS TEXT: {analysis_text}

Extract information into this EXACT JSON structure:

{{
  "verification_sources": [
    {{
      "claim": "exact claim text",
      "url": "specific verification URL", 
      "institution": "authoritative organization",
      "confidence": 0.X,
      "verification_type": "primary_source|expert_analysis|official_data|research_study",
      "quality_score": 0.X
    }}
  ],
  "source_quality": {{
    "overall_score": X.X,
    "strongest_sources": ["source1", "source2"],
    "quality_concerns": ["concern1", "concern2"]
  }},
  "logical_consistency": {{
    "consistency_score": X.X,
    "reasoning_strengths": ["strength1", "strength2"], 
    "logical_weaknesses": ["weakness1", "weakness2"]
  }},
  "evidence_gaps": {{
    "completeness_score": X.X,
    "critical_gaps": ["gap1", "gap2"],
    "recommendations": ["rec1", "rec2"]
  }}
}}

REQUIREMENTS:
- All scores as decimals (0.1 to 1.0)
- URLs must be specific, not homepages
- Arrays limited to top 3 items each
- Valid JSON format only"""

class DomainSpecificPrompts:
    """Domain-specific prompt templates for different content types."""
    
    MEDICAL_VERIFICATION = """
    For medical claims, prioritize:
    - PubMed/MEDLINE database searches
    - FDA official statements and approvals
    - CDC guidelines and recommendations  
    - WHO official positions
    - Peer-reviewed medical journals
    - Professional medical association statements
    
    SPECIFIC DOMAINS:
    - https://pubmed.ncbi.nlm.nih.gov/ (research studies)
    - https://www.fda.gov/news-events/ (regulatory actions)
    - https://www.cdc.gov/ (health guidelines)
    - https://www.who.int/ (global health positions)
    """
    
    POLITICAL_VERIFICATION = """
    For political claims, prioritize:
    - Official government websites (.gov domains)
    - Congressional records and voting databases
    - Official policy documents and legislation
    - Government statistics and data portals
    - Official statements from agencies
    - Verified electoral data sources
    
    SPECIFIC DOMAINS:
    - https://www.congress.gov/ (legislative information)
    - https://www.gpo.gov/ (official government documents)
    - https://data.gov/ (government datasets)
    - https://www.fec.gov/ (electoral information)
    """
    
    SCIENTIFIC_VERIFICATION = """
    For scientific claims, prioritize:
    - Peer-reviewed journal articles
    - University research publications
    - Government research agencies (NSF, NIH, NASA)
    - Professional scientific organizations
    - Institutional research repositories
    - Independent research institutions
    
    SPECIFIC DOMAINS:
    - https://pubmed.ncbi.nlm.nih.gov/ (life sciences)
    - https://arxiv.org/ (preprint research)
    - https://www.nsf.gov/ (National Science Foundation)
    - https://www.nature.com/ (Nature publications)
    """

class PromptValidator:
    """Validation methods for prompt inputs and outputs."""
    
    @staticmethod
    def validate_url_specificity(url: str) -> bool:
        """Validate that URL is specific and not just a homepage."""
        generic_patterns = [
            r'https?://[^/]+/?$',  # Just domain with optional trailing slash
            r'https?://[^/]+/index\.html?$',  # Index pages
            r'https?://[^/]+/home/?$',  # Home pages
        ]
        
        import re
        for pattern in generic_patterns:
            if re.match(pattern, url):
                return False
        
        # URL should have meaningful path beyond domain
        return len(url.split('/')) > 3
    
    @staticmethod
    def extract_confidence_score(text: str) -> float:
        """Extract confidence score from analysis text."""
        import re
        confidence_pattern = r'confidence[:\s]+([0-9]\.[0-9]+)'
        match = re.search(confidence_pattern, text.lower())
        if match:
            return float(match.group(1))
        return 0.5  # Default moderate confidence

def get_prompt_template(prompt_type: str, **kwargs) -> str:
    """
    Get specific prompt template with parameters.
    
    Args:
        prompt_type: Type of prompt needed
        **kwargs: Parameters for prompt formatting
        
    Returns:
        Formatted prompt string
    """
    prompt_mapping = {
        'verification_sources': EvidenceVerificationPrompts.generate_verification_sources,
        'source_quality': EvidenceVerificationPrompts.assess_source_quality,
        'logical_consistency': LogicalConsistencyPrompts.analyze_logical_consistency,
        'evidence_gaps': EvidenceGapPrompts.identify_evidence_gaps,
        'structured_output': StructuredOutputPrompts.extract_verification_data,
    }
    
    if prompt_type not in prompt_mapping:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompt_mapping[prompt_type](**kwargs)
