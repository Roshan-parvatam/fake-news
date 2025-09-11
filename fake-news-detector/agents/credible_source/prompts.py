# agents/credible_source/prompts.py

"""
Credible Source Agent Prompts Module - Production Ready

Enhanced prompt templates for credible source recommendations, reliability assessments,
verification strategies, and fact-checking guidance with production-level safety handling,
structured outputs, and comprehensive error prevention.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .exceptions import (
    PromptGenerationError,
    InputValidationError,
    raise_input_validation_error
)


@dataclass
class PromptResponse:
    """Structured response container for prompt outputs with metadata."""
    content: str
    metadata: Dict[str, Any]
    prompt_type: str
    generation_time: float
    safety_level: str = "standard"
    
    def __post_init__(self):
        """Validate prompt response after initialization."""
        if not self.content or len(self.content.strip()) < 10:
            raise PromptGenerationError("Generated prompt content too short or empty")


class SourceRecommendationPrompts:
    """
    Production-ready source recommendation prompts with safety handling.
    
    Addresses Gemini safety filter blocking issues with careful language
    and structured output formatting for reliable parsing.
    """

    @staticmethod
    def contextual_source_analysis(article_text: str, 
                                 extracted_claims: List[Dict[str, Any]],
                                 domain: str, 
                                 evidence_score: float,
                                 session_id: str = None) -> str:
        """
        Generate contextual source recommendations with safety-aware language.
        
        Enhanced to prevent Gemini safety filter blocking while maintaining
        comprehensive source recommendation capabilities.
        """
        logger = logging.getLogger(f"{__name__}.SourceRecommendationPrompts")
        
        try:
            # Input validation with detailed error messages
            if not isinstance(article_text, str) or len(article_text.strip()) < 20:
                raise_input_validation_error(
                    "article_text",
                    "Must be non-empty string with at least 20 characters",
                    len(article_text.strip()) if article_text else 0,
                    session_id
                )

            if not isinstance(extracted_claims, list):
                logger.warning(f"Invalid claims format, using empty list", extra={'session_id': session_id})
                extracted_claims = []

            # Prepare claims summary with safety-conscious language
            claim_topics = []
            for i, claim in enumerate(extracted_claims[:6], 1):  # Limit to 6 for performance
                try:
                    claim_text = claim.get('text', 'Unknown claim')[:120]  # Truncate long claims
                    claim_type = claim.get('claim_type', 'General')
                    priority = claim.get('priority', 3)
                    
                    # Only include high-priority claims for focused analysis
                    if priority <= 2:
                        # Use neutral, professional language
                        claim_topics.append(f"{i}. {claim_type} claim: {claim_text}")
                        
                except Exception as claim_error:
                    logger.debug(f"Error processing claim {i}: {str(claim_error)}", 
                               extra={'session_id': session_id})
                    continue

            if not claim_topics:
                claim_topics = ["1. General verification needed for article content"]

            claims_summary = "\n".join(claim_topics)
            
            # Truncate article text for prompt efficiency
            article_context = article_text[:800] if len(article_text) > 800 else article_text

            logger.debug(f"Generating contextual analysis prompt for domain: {domain}", 
                        extra={'session_id': session_id, 'claims_count': len(claim_topics)})

            # Enhanced prompt with safety-conscious language and clear structure
            prompt = f"""You are a professional information verification specialist assisting journalists and researchers with source identification.

ARTICLE DOMAIN: {domain.title()}
EVIDENCE QUALITY ASSESSMENT: {evidence_score}/10

CONTENT REQUIRING VERIFICATION:
{claims_summary}

ARTICLE CONTEXT (First 800 characters):
{article_context}

TASK: Identify authoritative institutional sources and expert contacts for professional fact-checking and verification of the above content.

REQUIRED STRUCTURED OUTPUT:

## PRIMARY INSTITUTIONAL SOURCES

1. **[Institution Name] - [Department/Division]**
   - Relevance: [Specific connection to the content requiring verification]
   - Contact Method: [How to reach for official information]
   - Official Website: [URL if available]
   - Type of Information: [What kind of verification they can provide]

2. **[Different Institution] - [Relevant Department]**
   - Relevance: [Why this source is appropriate for verification]
   - Contact Method: [Professional contact information or process]
   - Official Website: [Official URL]
   - Type of Information: [Specific expertise area]

## EXPERT CONTACTS AND ORGANIZATIONS

• **[Expert Title/Role]**: [Institution] - [Specific area of expertise]
• **[Different Expert]**: [Organization] - [Relevant domain knowledge]
• **[Professional Organization]**: [Contact method] - [Area of authority]

## OFFICIAL DATABASES AND REPOSITORIES

• **[Database Name]**: [Managing Institution] - [Type of official records available]
• **[Research Repository]**: [Organization] - [Academic or scientific data access]

## VERIFICATION APPROACH RECOMMENDATIONS

1. **Primary Verification**: Contact lead institutional authorities first
2. **Cross-Reference**: Verify through multiple independent official sources  
3. **Documentation**: Request official statements or published materials
4. **Expert Consultation**: Engage with recognized subject matter experts

GUIDELINES FOR PROFESSIONAL VERIFICATION:
- Prioritize government agencies, academic institutions, and established professional organizations
- Focus on sources with direct expertise in the {domain} domain
- Include both primary sources and expert commentary options
- Provide specific, actionable contact methods when possible
- Ensure all recommendations support professional journalistic standards

Generate comprehensive, professionally appropriate source recommendations."""

            logger.info(f"Contextual analysis prompt generated successfully", 
                       extra={'session_id': session_id, 'prompt_length': len(prompt)})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate contextual analysis prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Contextual analysis prompt generation failed: {str(e)}",
                prompt_type="contextual_analysis",
                session_id=session_id
            )

    @staticmethod
    def reliability_assessment_structured(article_text: str, 
                                        recommended_sources: List[Dict[str, Any]],
                                        session_id: str = None) -> str:
        """Generate structured reliability assessment with comprehensive evaluation criteria."""
        logger = logging.getLogger(f"{__name__}.SourceRecommendationPrompts")
        
        try:
            # Input validation
            if not isinstance(recommended_sources, list):
                logger.warning("Invalid sources format, using empty list", extra={'session_id': session_id})
                recommended_sources = []

            # Prepare source list with detailed information
            source_entries = []
            for i, source in enumerate(recommended_sources[:8], 1):  # Limit to 8 for prompt efficiency
                try:
                    name = source.get('name', 'Unknown Source')
                    source_type = source.get('type', 'Unknown')
                    reliability = source.get('reliability_score', 'N/A')
                    domain = source.get('domain', 'general')
                    
                    source_entries.append(
                        f"{i}. {name}\n   Type: {source_type} | Domain: {domain} | "
                        f"Reliability: {reliability}/10"
                    )
                except Exception as source_error:
                    logger.debug(f"Error processing source {i}: {str(source_error)}", 
                               extra={'session_id': session_id})
                    continue

            sources_text = "\n".join(source_entries) if source_entries else "No specific sources provided for assessment"
            
            # Truncate article context
            article_context = article_text[:1000] if len(article_text) > 1000 else article_text

            prompt = f"""Assess the reliability and suitability of these information sources for professional fact-checking and verification.

ARTICLE CONTENT CONTEXT:
{article_context}

RECOMMENDED SOURCES FOR ASSESSMENT:
{sources_text}

COMPREHENSIVE RELIABILITY ANALYSIS REQUIRED:

## Individual Source Assessment

For each source, evaluate:

**Institutional Authority and Credibility**
- Government agency status and mandate
- Academic institution reputation and accreditation  
- Professional organization standing and membership
- Historical track record and established expertise

**Domain Expertise Relevance**
- Specific subject matter authority
- Depth of knowledge in relevant areas
- Publications and research contributions
- Recognition within professional community

**Independence and Bias Considerations**
- Potential conflicts of interest
- Funding sources and organizational independence
- Political or commercial affiliations
- Editorial standards and peer review processes

**Accessibility and Responsiveness**
- Public information availability
- Media relations and contact accessibility
- Response time expectations
- Information request procedures

## Overall Source Portfolio Analysis

**Reliability Distribution Assessment**
- Average reliability score with justification
- Range of source types and their complementary strengths
- Geographic and institutional diversity
- Coverage of different expertise areas

**Verification Strategy Recommendations**
- Primary sources to contact first
- Secondary sources for cross-verification
- Backup options if primary sources unavailable
- Timeline expectations for different source types

## Coverage Gap Analysis

**Identify Missing Source Categories**
- Additional expertise areas needed
- Underrepresented institutional types
- Geographic or jurisdictional gaps
- Specialized databases or repositories not included

**Recommendations for Source Enhancement**
- Specific additional sources to consider
- Alternative approaches if current sources insufficient
- Professional networks or associations to explore
- Database or archival resources to consult

## Verification Feasibility Assessment

**Practical Accessibility Evaluation**
- How easily can these sources be contacted?
- What type of information can each realistically provide?
- Expected response timeframes and reliability
- Cost or access barriers for different sources

**Professional Verification Protocol**
- Recommended sequence for contacting sources
- Specific questions to ask each source type
- Documentation and attribution requirements
- Quality assurance and cross-verification steps

FORMAT REQUIREMENTS: Provide structured analysis with clear sections, specific recommendations, and numerical assessments where appropriate. Focus on actionable insights for professional journalism and research applications."""

            logger.info(f"Reliability assessment prompt generated", 
                       extra={'session_id': session_id, 'sources_count': len(recommended_sources)})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate reliability assessment prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Reliability assessment prompt generation failed: {str(e)}",
                prompt_type="reliability_assessment",
                session_id=session_id
            )


class VerificationStrategyPrompts:
    """
    Enhanced verification strategy and fact-checking guidance prompts.
    
    Provides systematic approaches for claim verification with institutional
    source integration and professional journalism standards.
    """

    @staticmethod
    def verification_strategy_systematic(extracted_claims: List[Dict[str, Any]],
                                       domain_analysis: Dict[str, Any],
                                       evidence_evaluation: Dict[str, Any],
                                       session_id: str = None) -> str:
        """Generate systematic verification strategies for extracted claims."""
        logger = logging.getLogger(f"{__name__}.VerificationStrategyPrompts")
        
        try:
            # Input validation and processing
            if not isinstance(extracted_claims, list):
                logger.warning("Invalid claims format, using empty list", extra={'session_id': session_id})
                extracted_claims = []

            # Prepare priority claims with enhanced details
            priority_claims = []
            for i, claim in enumerate(extracted_claims[:6], 1):  # Limit for performance
                try:
                    claim_text = claim.get('text', 'Unknown claim')[:150]
                    priority = claim.get('priority', 3)
                    verifiability = claim.get('verifiability_score', 5)
                    claim_type = claim.get('claim_type', 'General')
                    
                    # Add priority indicator for better prompt understanding
                    priority_indicator = "HIGH" if priority <= 2 else "MEDIUM" if priority == 3 else "LOW"
                    
                    priority_claims.append(
                        f"{i}. **{claim_type} Claim** (Priority: {priority_indicator})\n"
                        f"   Content: \"{claim_text}\"\n"
                        f"   Verifiability Assessment: {verifiability}/10"
                    )
                except Exception as claim_error:
                    logger.debug(f"Error processing claim {i}: {str(claim_error)}", 
                               extra={'session_id': session_id})
                    continue

            claims_text = "\n".join(priority_claims) if priority_claims else "No priority claims identified for verification"

            # Extract domain and evidence information safely
            domain = domain_analysis.get('primary_domain', 'general') if domain_analysis else 'general'
            confidence = domain_analysis.get('confidence', 0.0) if domain_analysis else 0.0
            evidence_score = evidence_evaluation.get('overall_evidence_score', 5.0) if evidence_evaluation else 5.0

            prompt = f"""Create a comprehensive, systematic verification strategy for these claims using professional fact-checking methodologies.

CLAIMS REQUIRING SYSTEMATIC VERIFICATION:
{claims_text}

CONTEXT INFORMATION:
• Subject Domain: {domain.title()}
• Domain Classification Confidence: {confidence:.2f}
• Overall Evidence Quality: {evidence_score}/10
• Total Claims for Analysis: {len(extracted_claims)}

COMPREHENSIVE VERIFICATION STRATEGY DEVELOPMENT:

## Claim-by-Claim Verification Protocol

For each HIGH and MEDIUM priority claim, develop:

**Primary Verification Approach**
- Most authoritative source type for this specific claim
- Direct institutional contacts (government agencies, academic institutions)
- Official databases or repositories to consult
- Expected evidence type and format

**Authoritative Source Identification**
- Government agencies with relevant mandate
- Academic institutions with subject expertise
- Professional organizations with authority
- International organizations if applicable

**Search and Database Strategy**
- Specific academic databases to query
- Government information repositories
- Professional association resources
- News wire services and archives

**Evidence Collection Framework**
- Primary source documents to request
- Official statements or press releases
- Peer-reviewed research publications
- Statistical data and official reports

**Timeline and Verification Sequence**
- Estimated verification time for each claim type
- Optimal order for contacting different source types
- Parallel vs sequential verification approaches
- Milestone checkpoints and progress evaluation

## Comprehensive Source Contact Strategy

**Institutional Contact Hierarchy**
1. **Primary Authorities**: Government agencies, regulatory bodies
2. **Academic Experts**: University researchers, think tanks
3. **Professional Organizations**: Industry associations, medical societies  
4. **Official Documentation**: Published reports, statistical databases
5. **Cross-Verification**: Independent confirmation sources

**Communication Protocol**
- Specific questions tailored to each source type
- Professional introduction and context setting
- Information request format and documentation needs
- Follow-up procedures and timeline management

## Quality Assurance and Cross-Verification

**Multi-Source Verification Standards**
- Minimum number of independent confirmations required
- Source diversity requirements (geographic, institutional)
- Conflict resolution procedures for contradictory information
- Documentation standards for all source communications

**Verification Confidence Levels**
- High Confidence: 3+ independent authoritative sources
- Medium Confidence: 2 independent sources + supporting documentation
- Low Confidence: Single source + circumstantial evidence
- Insufficient: Unable to verify through reliable sources

## Risk Assessment and Contingency Planning

**Verification Challenges Identification**
- Sources that may be difficult to reach
- Information that may be classified or restricted  
- Time-sensitive claims requiring rapid verification
- Claims requiring specialized technical expertise

**Alternative Verification Pathways**
- Backup sources if primary contacts unavailable
- Alternative approaches for hard-to-verify claims
- Expert consultation options for technical content
- International sources for global claims

## Documentation and Reporting Framework

**Verification Documentation Requirements**
- Complete source contact log with dates and responses
- Direct quotes and official statements collected
- Supporting documentation and evidence files
- Confidence level assessment for each claim

**Professional Reporting Standards**
- Attribution requirements for different source types
- Transparency about verification methods used
- Clear distinction between verified facts and expert opinions
- Acknowledgment of verification limitations or uncertainties

DELIVERABLE: Comprehensive, step-by-step verification protocol suitable for professional journalism and research applications, emphasizing institutional source verification and systematic cross-referencing."""

            logger.info(f"Verification strategy prompt generated", 
                       extra={'session_id': session_id, 'claims_processed': len(priority_claims)})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate verification strategy prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Verification strategy prompt generation failed: {str(e)}",
                prompt_type="verification_strategy",
                session_id=session_id
            )

    @staticmethod
    def fact_check_guidance_comprehensive(priority_claims: List[str],
                                        available_sources: List[Dict[str, Any]],
                                        contextual_sources: List[Dict[str, Any]],
                                        session_id: str = None) -> str:
        """Generate comprehensive fact-checking guidance with professional standards."""
        logger = logging.getLogger(f"{__name__}.VerificationStrategyPrompts")
        
        try:
            # Prepare claims list with truncation for prompt efficiency
            claims_list = []
            for i, claim in enumerate(priority_claims[:5], 1):  # Limit to top 5
                truncated_claim = claim[:120] if len(claim) > 120 else claim
                claims_list.append(f"{i}. {truncated_claim}")
            
            formatted_claims = "\n".join(claims_list) if claims_list else "No priority claims provided"

            # Prepare source summaries with detailed information
            database_sources = []
            for source in available_sources[:6]:  # Limit for prompt efficiency
                try:
                    name = source.get('name', 'Unknown')
                    source_type = source.get('type', 'Unknown type')
                    reliability = source.get('reliability_score', 'N/A')
                    database_sources.append(f"• {name} ({source_type}) - Reliability: {reliability}/10")
                except Exception:
                    continue

            contextual_source_list = []
            for source in contextual_sources[:6]:  # Limit for prompt efficiency
                try:
                    name = source.get('name', 'Unknown')
                    relevance = source.get('relevance', source.get('details', 'Contextually relevant'))[:80]
                    contextual_source_list.append(f"• {name}: {relevance}")
                except Exception:
                    continue

            db_sources_text = "\n".join(database_sources) if database_sources else "No database sources available"
            ctx_sources_text = "\n".join(contextual_source_list) if contextual_source_list else "No contextual sources available"

            prompt = f"""Provide comprehensive, professional fact-checking guidance following industry-standard methodologies and best practices.

HIGH-PRIORITY CLAIMS FOR SYSTEMATIC VERIFICATION:
{formatted_claims}

INSTITUTIONAL DATABASE SOURCES AVAILABLE:
{db_sources_text}

CONTEXTUAL EXPERT SOURCES IDENTIFIED:
{ctx_sources_text}

COMPREHENSIVE FACT-CHECKING PROTOCOL:

## Professional Verification Priority Framework

**Claim Prioritization Matrix**
1. **Immediate Verification Priority**
   - Claims with potential public safety implications
   - Statements about ongoing policy decisions
   - Information affecting financial markets or health decisions
   - Time-sensitive news developments

2. **Standard Verification Priority**  
   - Claims requiring expert analysis or interpretation
   - Historical or statistical assertions
   - Attribution claims about public figures
   - Policy position statements

3. **Background Verification Priority**
   - Supporting context information
   - Historical background claims
   - General industry or sector information

## Systematic Step-by-Step Verification Process

**Phase 1: Initial Assessment and Source Selection**
- Identify claim type and required evidence standard
- Select most appropriate institutional sources from available options
- Determine expected response timeframes for different source types
- Plan verification sequence based on source accessibility

**Phase 2: Primary Source Contact Protocol**
- Begin with highest-authority institutional sources (government agencies, regulatory bodies)
- Prepare specific, professional inquiries for each source type
- Request official statements, published materials, or data access
- Document all communications with date, time, and response details

**Phase 3: Expert Consultation and Analysis**
- Engage subject matter experts from academic or professional organizations
- Request interpretation of technical or complex information  
- Seek multiple expert opinions for controversial or disputed claims
- Document expert credentials and potential conflicts of interest

**Phase 4: Cross-Reference and Corroboration**
- Compare information across multiple independent sources
- Identify consistencies and discrepancies in source responses
- Seek additional verification for contradictory information
- Assess reliability and credibility of each information source

## Quality Assurance and Professional Standards

**Source Verification Checklist**
- [ ] Minimum 2-3 independent authoritative sources consulted
- [ ] Primary institutional sources accessed when possible
- [ ] Expert opinions sought for technical or specialized claims
- [ ] All sources properly documented with contact information
- [ ] Potential conflicts of interest or bias identified and noted
- [ ] Response timeline and accessibility assessed for each source

**Evidence Quality Assessment Framework**
- **Definitive Evidence**: Official documents, published studies, statistical data
- **Strong Evidence**: Expert consensus, institutional statements, peer-reviewed analysis
- **Moderate Evidence**: Single expert opinion, preliminary data, qualified statements  
- **Weak Evidence**: Anecdotal reports, unverified claims, speculation

**Professional Documentation Standards**
- Complete source contact log with dates and methods used
- Exact quotes and official statement attributions
- Supporting documents and evidence file organization
- Clear confidence level assessment for each verified claim
- Transparent methodology description for complex verifications

## Risk Management and Ethical Considerations

**Potential Verification Challenges**
- Sources that may be unavailable or unresponsive
- Information requiring specialized technical knowledge
- Claims involving classified or proprietary information
- Time constraints affecting thorough verification process

**Professional Ethics Protocol**
- Maintain source confidentiality when requested
- Avoid leading questions that might bias responses
- Acknowledge limitations in verification process
- Distinguish clearly between verified facts and expert interpretations
- Provide fair representation of different viewpoints when applicable

## Final Verification Report Structure

**Executive Summary**
- Overall verification confidence level for each claim
- Key findings and definitive conclusions
- Significant limitations or uncertainties identified

**Detailed Findings**
- Claim-by-claim verification results with supporting evidence
- Source-by-source response summary with reliability assessment
- Cross-reference analysis and consistency evaluation

**Methodology Transparency**
- Complete list of sources contacted with response status
- Verification methods used for each type of claim
- Timeline and sequence of verification activities
- Quality assurance measures applied throughout process

OBJECTIVE: Generate detailed, actionable fact-checking procedures that meet professional journalism and research standards while ensuring transparency, accuracy, and ethical compliance."""

            logger.info(f"Comprehensive fact-checking guidance generated", 
                       extra={'session_id': session_id, 'claims_count': len(priority_claims)})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate fact-checking guidance: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Fact-checking guidance generation failed: {str(e)}",
                prompt_type="fact_check_guidance",
                session_id=session_id
            )


class DomainSpecificPrompts:
    """
    Domain-specific prompts for targeted source recommendations with professional guidance.
    """

    # Enhanced domain-specific guidance with professional standards
    DOMAIN_GUIDANCE = {
        'health': """
HEALTH DOMAIN VERIFICATION STANDARDS:

Primary Sources (Priority 1):
- Government health agencies: CDC, FDA, NIH, WHO
- Peer-reviewed medical literature: PubMed, Cochrane Library, medical journals
- Academic medical institutions: Medical schools, research hospitals
- Professional medical associations: AMA, specialty medical societies
- Clinical research databases: ClinicalTrials.gov, research registries

Verification Requirements:
- Prioritize peer-reviewed studies over preliminary reports
- Check FDA approval status for medical treatments
- Verify clinical trial registration and methodology
- Confirm expert credentials and institutional affiliations
- Cross-reference with international health organizations

Professional Standards:
- Distinguish between association and causation in studies
- Note sample sizes and study limitations
- Verify publication dates and current relevance
- Check for conflicts of interest in research funding
- Seek multiple independent medical expert opinions

Red Flags to Avoid:
- Unverified social media health claims
- Non-peer-reviewed preprint studies presented as definitive
- Commercial health websites without medical oversight
- Anecdotal reports without scientific backing
""",

        'politics': """
POLITICAL DOMAIN VERIFICATION STANDARDS:

Primary Sources (Priority 1):
- Official government sources: Press offices, official statements
- Legislative databases: Congress.gov, voting records, bill texts
- Campaign finance records: FEC filings, disclosure documents
- Nonpartisan research organizations: CBO, GAO, Pew Research
- Established fact-checking organizations with transparent methodologies

Verification Requirements:
- Confirm quotes with original transcripts or official records
- Check voting records through official legislative databases
- Verify policy positions through official campaign materials
- Cross-reference with multiple news sources across political spectrum
- Confirm event attendance and statement timing

Professional Standards:
- Present multiple political perspectives when relevant
- Distinguish between campaign promises and official policies
- Note timing and context of political statements
- Verify spokesperson authority to represent organizations
- Maintain nonpartisan approach in fact verification

Bias Awareness:
- Recognize potential political bias in all sources
- Seek diverse perspectives across political spectrum
- Verify claims through original documents when possible
- Note funding sources for political organizations and think tanks
""",

        'science': """
SCIENTIFIC DOMAIN VERIFICATION STANDARDS:

Primary Sources (Priority 1):
- Peer-reviewed journal publications: Nature, Science, Cell, discipline-specific journals
- Academic research institutions: Universities, national laboratories
- Professional scientific organizations: AAAS, discipline-specific societies
- Government research agencies: NSF, NIH, NASA, NOAA
- International research collaborations and databases

Verification Requirements:
- Confirm peer-review status and journal impact factor
- Verify study methodology and sample sizes
- Check for replication studies and scientific consensus
- Confirm researcher credentials and institutional affiliations
- Review funding sources and potential conflicts of interest

Professional Standards:
- Distinguish between preliminary findings and established science
- Note study limitations and researcher caveats
- Present uncertainty and confidence levels accurately
- Avoid oversimplification of complex scientific concepts
- Seek expert interpretation for technical content

Research Quality Indicators:
- Large sample sizes and rigorous methodology
- Independent replication of results
- Meta-analyses and systematic reviews
- Long-term studies over short-term observations
- Transparency in data sharing and methodology
""",

        'technology': """
TECHNOLOGY DOMAIN VERIFICATION STANDARDS:

Primary Sources (Priority 1):
- Technology companies' official announcements and documentation
- Academic computer science institutions: MIT, Stanford, CMU
- Professional technology organizations: IEEE, ACM
- Government technology agencies: NIST, CISA
- Independent technology research firms: Gartner, IDC

Verification Requirements:
- Confirm technical specifications through official documentation
- Verify security claims through independent testing organizations
- Check patent filings and intellectual property records
- Confirm expert credentials in relevant technology areas
- Cross-reference with multiple independent technology analysts

Professional Standards:
- Distinguish between marketing claims and verified capabilities
- Note beta vs. production technology status
- Verify timing and availability of technology releases
- Check for independent validation of performance claims
- Assess potential privacy and security implications

Technology Assessment Criteria:
- Independent third-party testing and validation
- Open source code availability for transparency
- Regulatory compliance and certification status
- Market adoption and real-world performance data
"""
    }

    @classmethod
    def get_domain_guidance(cls, domain: str) -> str:
        """Get professional guidance for specific domain verification."""
        return cls.DOMAIN_GUIDANCE.get(domain, """
GENERAL DOMAIN VERIFICATION STANDARDS:

Apply standard journalistic verification principles:
- Seek multiple independent sources
- Verify credentials and authority of experts
- Cross-reference information across different source types
- Document methodology and source contact information
- Maintain professional ethics and transparency standards
""")


class SafetyEnhancedPrompts:
    """
    Safety-enhanced prompts designed to prevent content filter issues while maintaining effectiveness.
    """

    @staticmethod
    def institutional_fallback_sources(claims_summary: str, 
                                     domain: str,
                                     session_id: str = None) -> str:
        """Generate institutional source recommendations when AI analysis faces restrictions."""
        logger = logging.getLogger(f"{__name__}.SafetyEnhancedPrompts")
        
        try:
            # Validate inputs with error handling
            if not isinstance(claims_summary, str) or not claims_summary.strip():
                claims_summary = "General information verification needed"
            
            if not isinstance(domain, str) or not domain.strip():
                domain = "general"

            logger.info(f"Generating institutional fallback sources for domain: {domain}", 
                       extra={'session_id': session_id})

            prompt = f"""Identify authoritative institutional resources for professional information verification and fact-checking.

CONTENT REQUIRING VERIFICATION:
{claims_summary[:1000]}  

SUBJECT DOMAIN: {domain.title()}

INSTITUTIONAL VERIFICATION RESOURCES:

## Government and Regulatory Agencies

**Federal Agencies and Departments**
- Agency name and relevant division or office
- Specific area of regulatory authority or expertise  
- Official information resources and databases
- Public information contact methods
- Website URLs for official statements and data

**State and Local Government Resources**
- Relevant state agencies and departments
- Local government offices with jurisdiction
- Public records and information access procedures
- Contact information for media relations or public affairs

## Academic and Research Institutions

**Universities and Research Centers**
- Institution name and relevant academic departments
- Faculty expertise directories and contact information
- Research centers and institutes with domain focus
- Published research databases and repositories
- Media relations contacts for expert commentary

**National Laboratories and Research Facilities**
- Government research facilities with relevant expertise
- Independent research organizations and institutes
- International research collaborations and databases
- Access procedures for research data and publications

## Professional and Industry Organizations

**Professional Associations and Societies**
- Organization name and specific focus area
- Professional standards and best practices resources
- Expert member directories and contact procedures
- Position statements and official guidelines
- Media relations contacts for authoritative commentary

**Industry Groups and Trade Associations**
- Organizations representing relevant industries or sectors
- Technical standards and certification bodies
- Market research and industry analysis resources
- Regulatory compliance and safety information

## Information Databases and Repositories

**Government Databases and Information Systems**
- Official data repositories and statistical databases
- Regulatory filings and compliance information
- Public records and document archives
- Search capabilities and access requirements

**Academic and Research Databases**
- Peer-reviewed publication databases
- Research data repositories and archives
- Citation indexes and academic search systems
- Access requirements and institutional subscriptions

**Professional and Technical Resources**
- Industry standards and technical documentation
- Professional certification and credentialing databases
- Best practices and guideline repositories
- Technical specifications and safety information

## Verification Contact Protocols

**Direct Contact Procedures**
- Official phone numbers and email addresses
- Media relations and public affairs contacts
- Information request procedures and requirements
- Expected response timeframes and contact hours

**Information Access Methods**
- Online database search procedures
- Document request and FOIA processes
- Public record access and retrieval methods
- Database subscription and access requirements

OBJECTIVE: Provide comprehensive, actionable institutional resources for professional verification and fact-checking applications, emphasizing official sources and established procedures."""

            logger.info(f"Institutional fallback prompt generated successfully", 
                       extra={'session_id': session_id, 'domain': domain})
            
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate institutional fallback prompt: {str(e)}", 
                        extra={'session_id': session_id})
            raise PromptGenerationError(
                f"Institutional fallback prompt generation failed: {str(e)}",
                prompt_type="institutional_fallback",
                session_id=session_id
            )


# Main prompt template access function with enhanced error handling
def get_source_prompt_template(prompt_type: str, 
                             session_id: str = None,
                             **kwargs) -> str:
    """
    Get specific source recommendation prompt template with production error handling.

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
    logger = logging.getLogger(f"{__name__}.get_source_prompt_template")
    start_time = time.time()
    
    try:
        # Enhanced prompt mapping with error handling
        prompt_mapping = {
            'contextual_analysis': SourceRecommendationPrompts.contextual_source_analysis,
            'reliability_assessment': SourceRecommendationPrompts.reliability_assessment_structured,
            'verification_strategy': VerificationStrategyPrompts.verification_strategy_systematic,
            'fact_check_guidance': VerificationStrategyPrompts.fact_check_guidance_comprehensive,
            'institutional_fallback': SafetyEnhancedPrompts.institutional_fallback_sources,
        }

        if prompt_type not in prompt_mapping:
            available_types = list(prompt_mapping.keys())
            raise_input_validation_error(
                "prompt_type",
                f"Unknown prompt type '{prompt_type}'. Available: {available_types}",
                prompt_type,
                session_id
            )

        logger.debug(f"Generating prompt of type: {prompt_type}", 
                    extra={'session_id': session_id, 'kwargs_count': len(kwargs)})

        # Generate prompt with error handling
        prompt_function = prompt_mapping[prompt_type]
        
        if prompt_type == 'contextual_analysis':
            prompt = prompt_function(
                kwargs.get('article_text', ''),
                kwargs.get('extracted_claims', []),
                kwargs.get('domain', 'general'),
                kwargs.get('evidence_score', 5.0),
                session_id
            )
        elif prompt_type == 'reliability_assessment':
            prompt = prompt_function(
                kwargs.get('article_text', ''),
                kwargs.get('recommended_sources', []),
                session_id
            )
        elif prompt_type == 'verification_strategy':
            prompt = prompt_function(
                kwargs.get('extracted_claims', []),
                kwargs.get('domain_analysis', {}),
                kwargs.get('evidence_evaluation', {}),
                session_id
            )
        elif prompt_type == 'fact_check_guidance':
            prompt = prompt_function(
                kwargs.get('priority_claims', []),
                kwargs.get('available_sources', []),
                kwargs.get('contextual_sources', []),
                session_id
            )
        elif prompt_type == 'institutional_fallback':
            prompt = prompt_function(
                kwargs.get('claims_summary', ''),
                kwargs.get('domain', 'general'),
                session_id
            )
        else:
            # Generic call for any additional prompt types
            prompt = prompt_function(session_id=session_id, **kwargs)

        generation_time = time.time() - start_time
        
        # Validate generated prompt
        if not prompt or len(prompt.strip()) < 50:
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


def get_domain_guidance(domain: str) -> str:
    """
    Get domain-specific professional guidance for source selection.

    Args:
        domain: Domain type (health, politics, science, technology, etc.)

    Returns:
        Professional guidance text with verification standards
    """
    return DomainSpecificPrompts.get_domain_guidance(domain)


# Testing functionality
if __name__ == "__main__":
    """Test credible source prompts with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    test_session_id = "prompt_test_789"
    
    print("=== CREDIBLE SOURCE PROMPTS TEST ===")
    
    try:
        # Test contextual source analysis prompt
        test_claims = [
            {
                'text': 'COVID-19 vaccines have shown 95% efficacy in clinical trials',
                'claim_type': 'Medical',
                'priority': 1,
                'verifiability_score': 8.5
            },
            {
                'text': 'FDA approved vaccine for emergency use',
                'claim_type': 'Regulatory',
                'priority': 1,
                'verifiability_score': 9.0
            }
        ]

        print("\n--- Contextual Source Analysis Prompt Test ---")
        contextual_prompt = get_source_prompt_template(
            'contextual_analysis',
            session_id=test_session_id,
            article_text="Recent clinical trial data shows promising COVID-19 vaccine efficacy rates in large-scale studies conducted by major pharmaceutical companies.",
            extracted_claims=test_claims,
            domain="health",
            evidence_score=7.8
        )
        
        print(f"✅ Contextual analysis prompt generated: {len(contextual_prompt)} characters")
        print(f"Preview: {contextual_prompt[:200]}...")

        # Test verification strategy prompt
        print("\n--- Verification Strategy Prompt Test ---")
        domain_analysis = {'primary_domain': 'health', 'confidence': 0.92}
        evidence_evaluation = {'overall_evidence_score': 7.8}
        
        strategy_prompt = get_source_prompt_template(
            'verification_strategy',
            session_id=test_session_id,
            extracted_claims=test_claims,
            domain_analysis=domain_analysis,
            evidence_evaluation=evidence_evaluation
        )
        
        print(f"✅ Verification strategy prompt generated: {len(strategy_prompt)} characters")

        # Test reliability assessment prompt
        print("\n--- Reliability Assessment Prompt Test ---")
        test_sources = [
            {'name': 'CDC', 'type': 'government', 'reliability_score': 9.5, 'domain': 'health'},
            {'name': 'Harvard Medical School', 'type': 'academic', 'reliability_score': 9.2, 'domain': 'health'}
        ]
        
        reliability_prompt = get_source_prompt_template(
            'reliability_assessment',
            session_id=test_session_id,
            article_text="Clinical trial shows vaccine efficacy",
            recommended_sources=test_sources
        )
        
        print(f"✅ Reliability assessment prompt generated: {len(reliability_prompt)} characters")

        # Test institutional fallback prompt
        print("\n--- Institutional Fallback Prompt Test ---")
        fallback_prompt = get_source_prompt_template(
            'institutional_fallback',
            session_id=test_session_id,
            claims_summary="Medical claims about vaccine efficacy requiring institutional verification",
            domain="health"
        )
        
        print(f"✅ Institutional fallback prompt generated: {len(fallback_prompt)} characters")

        # Test domain guidance
        print("\n--- Domain Guidance Test ---")
        health_guidance = get_domain_guidance('health')
        print(f"✅ Health domain guidance: {len(health_guidance)} characters")
        
        politics_guidance = get_domain_guidance('politics')  
        print(f"✅ Politics domain guidance: {len(politics_guidance)} characters")

        # Test error handling
        print("\n--- Error Handling Test ---")
        try:
            invalid_prompt = get_source_prompt_template('invalid_type', session_id=test_session_id)
            print("❌ Should have failed with invalid prompt type")
        except InputValidationError:
            print("✅ Invalid prompt type properly rejected")

        print("\n✅ Prompt generation tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
