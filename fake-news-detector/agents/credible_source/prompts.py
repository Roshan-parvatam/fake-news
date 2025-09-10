# agents/credible_source/prompts.py

"""
Credible Source Agent Prompts Module

Industry-standard prompt templates for credible source recommendations,
reliability assessments, verification strategies, and fact-checking guidance.
Enhanced with safety handling and specific output formatting.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PromptResponse:
    """Structured response container for prompt outputs."""
    content: str
    metadata: Dict[str, Any]

class SourceRecommendationPrompts:
    """Source recommendation prompts with safety handling and structured output."""
    
    @staticmethod
    def contextual_source_analysis(article_text: str, extracted_claims: List[Dict[str, Any]], 
                                 domain: str, evidence_score: float) -> str:
        """
        Generate contextual source recommendations with safety-aware language.
        Addresses Gemini safety filter blocking issues.
        """
        # Prepare claims summary
        claim_topics = []
        for i, claim in enumerate(extracted_claims[:6], 1):
            claim_text = claim.get('text', 'Unknown claim')[:100]
            claim_type = claim.get('claim_type', 'General')
            priority = claim.get('priority', 3)
            if priority <= 2:  # Focus on high priority
                claim_topics.append(f"{i}. {claim_type}: {claim_text}")
        
        if not claim_topics:
            claim_topics = ["1. General verification needed for article claims"]
        
        claims_summary = "\n".join(claim_topics)
        
        return f"""You are an information verification specialist helping journalists identify authoritative sources.

ARTICLE DOMAIN: {domain}
EVIDENCE QUALITY: {evidence_score}/10

SPECIFIC CLAIMS REQUIRING VERIFICATION:
{claims_summary}

ARTICLE CONTEXT (FIRST 600 CHARS):
{article_text[:600]}

TASK: Identify specific institutional sources and expert contacts for verifying these claims.

REQUIRED OUTPUT FORMAT:

PRIMARY VERIFICATION SOURCES:
1. [Institution Name] - [Department/Division]
   • Relevance: [Why this source can verify the specific claims]
   • Access Method: [How to contact or access information]
   • URL: [Official website if available]

2. [Different Institution] - [Specific Unit]
   • Relevance: [Connection to the claims being verified] 
   • Access Method: [Contact method or information access]
   • URL: [Official website if available]

EXPERT CONTACTS:
• [Expert Name/Title]: [Institution] - [Area of expertise relevant to claims]
• [Different Expert]: [Organization] - [Specific domain knowledge]

INSTITUTIONAL DATABASES:
• [Database/Repository]: [Institution] - [Type of information available]
• [Research Database]: [Organization] - [Relevance to verification]

GUIDELINES:
- Focus on sources that can verify the SPECIFIC claims listed
- Provide actionable contact methods when possible
- Prioritize government, academic, and professional organizations
- Include both primary sources and expert contacts
- Ensure sources match the article domain ({domain})

Generate recommendations that directly address the verification needs."""

    @staticmethod
    def reliability_assessment_structured(article_text: str, recommended_sources: List[Dict[str, Any]]) -> str:
        """Generate structured reliability assessment for recommended sources."""
        # Prepare source list
        source_entries = []
        for i, source in enumerate(recommended_sources[:8], 1):
            name = source.get('name', 'Unknown Source')
            source_type = source.get('type', 'Unknown')
            reliability = source.get('reliability_score', 'N/A')
            source_entries.append(f"{i}. {name} (Type: {source_type}, Reliability: {reliability}/10)")
        
        sources_text = "\n".join(source_entries) if source_entries else "No specific sources provided"
        
        return f"""Assess the reliability and suitability of these sources for fact-checking.

ARTICLE CONTEXT:
{article_text[:800]}

RECOMMENDED SOURCES:
{sources_text}

ASSESSMENT REQUIREMENTS:

## Source Reliability Analysis
Evaluate each source's:
- Institutional authority and credibility
- Domain expertise relevance
- Independence and bias considerations
- Accessibility for verification

## Overall Assessment
Provide:
- Reliability score (1-10) with justification
- Source diversity evaluation
- Coverage gaps identification
- Additional source recommendations if needed

## Verification Feasibility
Assess:
- How easily can these sources be contacted?
- What type of information can each provide?
- Expected response time and accessibility

FORMAT: Provide structured analysis with clear sections and numerical scoring."""

class VerificationStrategyPrompts:
    """Verification strategy and fact-checking guidance prompts."""
    
    @staticmethod
    def verification_strategy_systematic(extracted_claims: List[Dict[str, Any]], 
                                       domain_analysis: Dict[str, Any], 
                                       evidence_evaluation: Dict[str, Any]) -> str:
        """Generate systematic verification strategies for extracted claims."""
        # Prepare priority claims
        priority_claims = []
        for claim in extracted_claims[:6]:
            claim_text = claim.get('text', 'Unknown claim')[:120]
            priority = claim.get('priority', 3)
            verifiability = claim.get('verifiability_score', 5)
            claim_type = claim.get('claim_type', 'General')
            priority_claims.append(f"• {claim_type} (Priority {priority}): {claim_text} [Verifiability: {verifiability}/10]")
        
        claims_text = "\n".join(priority_claims) if priority_claims else "No priority claims identified"
        domain = domain_analysis.get('primary_domain', 'general')
        evidence_score = evidence_evaluation.get('overall_evidence_score', 5)
        
        return f"""Create systematic verification strategies for these claims.

PRIORITY CLAIMS FOR VERIFICATION:
{claims_text}

CONTEXT INFORMATION:
• Domain: {domain}
• Overall Evidence Quality: {evidence_score}/10
• Domain Confidence: {domain_analysis.get('confidence', 0.5):.2f}

VERIFICATION STRATEGY REQUIREMENTS:

## Claim-Specific Strategies
For each high-priority claim, provide:
1. Primary verification approach
2. Recommended authoritative sources
3. Specific search terms and databases
4. Expected verification timeline
5. Alternative verification methods

## Source Contact Strategy
Outline:
- Order of source contact (most authoritative first)
- Specific questions to ask each source type
- Documentation requirements
- Follow-up procedures

## Evidence Collection Framework
Detail:
- What types of evidence to seek
- How to validate source credentials  
- Cross-verification methods
- Documentation standards

## Timeline and Workflow
Provide:
- Estimated verification time per claim
- Parallel vs sequential verification approach
- Milestone checkpoints
- Quality assurance steps

Generate actionable, step-by-step verification protocols."""

    @staticmethod
    def fact_check_guidance_comprehensive(priority_claims: List[str], 
                                        available_sources: List[Dict[str, Any]],
                                        contextual_sources: List[Dict[str, Any]]) -> str:
        """Generate comprehensive fact-checking guidance."""
        # Prepare claims list
        claims_list = "\n".join([f"{i+1}. {claim[:100]}" for i, claim in enumerate(priority_claims[:5])])
        
        # Prepare source summaries
        database_sources = [f"• {s.get('name', 'Unknown')}: {s.get('type', 'Unknown type')}" 
                          for s in available_sources[:5]]
        contextual_source_list = [f"• {s.get('name', 'Unknown')}: {s.get('relevance', 'Contextually relevant')[:60]}" 
                                for s in contextual_sources[:5]]
        
        db_sources_text = "\n".join(database_sources) if database_sources else "No database sources available"
        ctx_sources_text = "\n".join(contextual_source_list) if contextual_source_list else "No contextual sources available"
        
        return f"""Provide comprehensive fact-checking guidance for systematic claim verification.

HIGH-PRIORITY CLAIMS TO VERIFY:
{claims_list}

AVAILABLE DATABASE SOURCES:
{db_sources_text}

CONTEXTUAL SOURCES IDENTIFIED:
{ctx_sources_text}

FACT-CHECKING PROTOCOL:

## Verification Priority Order
Rank claims by:
1. Potential harm if misinformation spreads
2. Verifiability using available sources
3. Public interest and impact
4. Evidence quality and accessibility

## Step-by-Step Verification Process
For each claim:
1. **Initial Assessment**
   - Identify claim type and verification requirements
   - Select most appropriate sources from available options
   - Determine expected evidence types

2. **Source Contact Protocol**
   - Begin with highest-authority institutional sources
   - Prepare specific questions for each source type
   - Document all communications and responses

3. **Evidence Evaluation**
   - Cross-reference information across multiple sources
   - Verify source credentials and expertise
   - Assess evidence quality and reliability

4. **Red Flags and Warning Signs**
   - Identify common misinformation patterns
   - Watch for source bias or conflicts of interest
   - Note inconsistencies across sources

## Quality Assurance Checklist
- [ ] Multiple independent sources consulted
- [ ] Primary sources accessed when possible
- [ ] Expert opinions sought for technical claims
- [ ] All sources properly documented
- [ ] Potential conflicts of interest identified

## Documentation Standards
Record:
- Complete source contact information
- Exact quotes and attributions
- Date and time of all communications
- Supporting documents and evidence
- Verification confidence level for each claim

Generate detailed, actionable fact-checking procedures."""

class DomainSpecificPrompts:
    """Domain-specific prompts for targeted source recommendations."""
    
    HEALTH_DOMAIN_GUIDANCE = """
    For health-related claims, prioritize:
    - Government health agencies (CDC, FDA, WHO)
    - Peer-reviewed medical literature (PubMed, medical journals)
    - Academic medical institutions
    - Professional medical associations
    - Clinical research databases
    
    Avoid: Social media health claims, non-peer-reviewed studies, commercial health sites
    """
    
    POLITICAL_DOMAIN_GUIDANCE = """
    For political claims, prioritize:
    - Official government sources and documents
    - Nonpartisan research organizations
    - Legislative databases and voting records
    - Campaign finance records (FEC)
    - Fact-checking organizations with established credibility
    
    Seek multiple perspectives across the political spectrum when possible.
    """
    
    SCIENTIFIC_DOMAIN_GUIDANCE = """
    For scientific claims, prioritize:
    - Peer-reviewed journal publications
    - Academic research institutions
    - Professional scientific organizations
    - Government research agencies (NSF, NIH, NOAA)
    - Established scientific databases
    
    Verify: Study methodology, sample sizes, peer-review status, replication studies.
    """
    
    ECONOMIC_DOMAIN_GUIDANCE = """
    For economic claims, prioritize:
    - Government economic agencies (BLS, Federal Reserve, Treasury)
    - Established economic research institutions
    - Academic economics departments
    - International economic organizations (IMF, World Bank)
    - Peer-reviewed economic journals
    
    Consider: Data sources, methodology, potential bias in economic analysis.
    """

class SafetyEnhancedPrompts:
    """Safety-enhanced prompts designed to avoid triggering content filters."""
    
    @staticmethod
    def institutional_fallback_sources(claims_summary: str, domain: str) -> str:
        """Generate institutional source recommendations when AI analysis is blocked."""
        return f"""Identify institutional verification resources for fact-checking purposes.

CLAIMS REQUIRING VERIFICATION:
{claims_summary}

SUBJECT DOMAIN: {domain}

INSTITUTIONAL RESOURCE CATEGORIES:

## Government Agencies
List relevant government agencies and departments with:
- Agency name and specific division
- Type of information they provide
- Official contact methods
- Website URLs

## Academic Institutions  
Identify universities and research centers with:
- Institution name and relevant departments
- Areas of expertise
- Faculty expert directories
- Research database access

## Professional Organizations
List professional associations with:
- Organization name and focus area
- Expert member directories
- Position statements and guidelines
- Contact information for media relations

## Research Databases
Identify databases and repositories with:
- Database name and scope
- Search capabilities
- Access requirements
- Data reliability indicators

Provide practical, actionable institutional resources for verification."""

def get_source_prompt_template(prompt_type: str, **kwargs) -> str:
    """
    Get specific source recommendation prompt template.
    
    Args:
        prompt_type: Type of prompt needed
        **kwargs: Parameters for prompt formatting
        
    Returns:
        Formatted prompt string
    """
    prompt_mapping = {
        'contextual_analysis': SourceRecommendationPrompts.contextual_source_analysis,
        'reliability_assessment': SourceRecommendationPrompts.reliability_assessment_structured,
        'verification_strategy': VerificationStrategyPrompts.verification_strategy_systematic,
        'fact_check_guidance': VerificationStrategyPrompts.fact_check_guidance_comprehensive,
        'institutional_fallback': SafetyEnhancedPrompts.institutional_fallback_sources,
    }
    
    if prompt_type not in prompt_mapping:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompt_mapping[prompt_type](**kwargs)

def get_domain_guidance(domain: str) -> str:
    """
    Get domain-specific guidance for source selection.
    
    Args:
        domain: Domain type (health, politics, science, economics, etc.)
        
    Returns:
        Domain-specific guidance text
    """
    domain_guidance = {
        'health': DomainSpecificPrompts.HEALTH_DOMAIN_GUIDANCE,
        'politics': DomainSpecificPrompts.POLITICAL_DOMAIN_GUIDANCE,
        'science': DomainSpecificPrompts.SCIENTIFIC_DOMAIN_GUIDANCE,
        'economics': DomainSpecificPrompts.ECONOMIC_DOMAIN_GUIDANCE,
    }
    
    return domain_guidance.get(domain, "General verification principles apply.")

# Testing functionality
if __name__ == "__main__":
    """Test credible source prompts."""
    
    # Test contextual source analysis
    test_claims = [
        {
            'text': 'COVID-19 vaccines contain tracking microchips',
            'claim_type': 'Medical',
            'priority': 1,
            'verifiability_score': 2
        },
        {
            'text': 'Study shows 95% efficacy rate',
            'claim_type': 'Research',
            'priority': 2,
            'verifiability_score': 8
        }
    ]
    
    contextual_prompt = SourceRecommendationPrompts.contextual_source_analysis(
        article_text="Recent claims about COVID-19 vaccines have emerged from various sources...",
        extracted_claims=test_claims,
        domain="health",
        evidence_score=6.5
    )
    
    print("=== CONTEXTUAL SOURCE ANALYSIS PROMPT ===")
    print(contextual_prompt[:500] + "...")
    
    # Test verification strategy
    domain_analysis = {'primary_domain': 'health', 'confidence': 0.85}
    evidence_evaluation = {'overall_evidence_score': 6.5}
    
    strategy_prompt = VerificationStrategyPrompts.verification_strategy_systematic(
        test_claims, domain_analysis, evidence_evaluation
    )
    
    print("\n=== VERIFICATION STRATEGY PROMPT ===")
    print(strategy_prompt[:500] + "...")
    
    # Test domain guidance
    health_guidance = get_domain_guidance('health')
    print(f"\n=== HEALTH DOMAIN GUIDANCE ===")
    print(health_guidance)
    
    print("\n=== PROMPT TESTING COMPLETED ===")
