# config/prompts_config.py

"""
Enhanced Centralized Prompt Template Management

This module provides centralized management of all prompt templates
used by the AI-powered agents in the fake news detection system.
Enhanced to fix formatting, credibility scoring, URL specificity, and source recommendations.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class PromptsConfig:
    """
    Enhanced Centralized Prompt Template Management
    
    This class manages all prompt templates used by AI agents,
    with enhanced formatting, specific URL requests, and better scoring.
    """
    
    # Agent prompt dictionaries
    llm_explanation_prompts: Dict[str, str] = field(default_factory=dict)
    credible_source_prompts: Dict[str, str] = field(default_factory=dict)
    claim_extractor_prompts: Dict[str, str] = field(default_factory=dict)
    context_analyzer_prompts: Dict[str, str] = field(default_factory=dict)
    evidence_evaluator_prompts: Dict[str, str] = field(default_factory=dict)
    
    # Enhanced prompt versioning
    prompt_version: str = "2.1.0"
    
    def __post_init__(self):
        """Initialize all enhanced prompt templates"""
        if not self.llm_explanation_prompts:
            self.llm_explanation_prompts = self._get_enhanced_llm_explanation_prompts()
        if not self.credible_source_prompts:
            self.credible_source_prompts = self._get_enhanced_credible_source_prompts()
        if not self.claim_extractor_prompts:
            self.claim_extractor_prompts = self._get_claim_extractor_prompts()
        if not self.context_analyzer_prompts:
            self.context_analyzer_prompts = self._get_context_analyzer_prompts()
        if not self.evidence_evaluator_prompts:
            self.evidence_evaluator_prompts = self._get_evidence_evaluator_prompts()
    
    def _get_enhanced_llm_explanation_prompts(self) -> Dict[str, str]:
        """Enhanced LLM explanation prompts with proper Markdown formatting"""
        return {
            "main_explanation": """
You are a professional fact-checker creating a clear, well-structured analysis report.

**ARTICLE DETAILS:**
- **Content**: {article_text}
- **Classification**: {prediction}
- **Confidence**: {confidence:.2%}
- **Source**: {source}
- **Date**: {date}
- **Subject**: {subject}

Create a comprehensive analysis using proper **Markdown formatting**:

## Classification Summary

Write 2-3 sentences explaining why this article is classified as **{prediction}**. Be clear and direct.

## Key Evidence Analysis

### Supporting Evidence

- **Evidence Point 1**: *Quote from article* → Explain significance
- **Evidence Point 2**: *Quote from article* → Explain significance
- **Evidence Point 3**: *Quote from article* → Explain significance

### Source Quality Assessment

- **Primary Sources**: List and evaluate quality
- **Expert Attribution**: Assess quoted authorities
- **Verification Potential**: How easily can claims be checked

## Credibility Factors

### ✅ Strengths

- Factor 1 that enhances credibility
- Factor 2 that supports authenticity
- Factor 3 that indicates reliability

### ⚠️ Concerns (if any)

- Issue 1 that reduces confidence
- Issue 2 that raises questions
- Issue 3 that requires verification

## Confidence Level Explanation

**Why {confidence:.2%} confidence?**

- **High certainty factors**: What made classification clear
- **Uncertainty factors**: What elements create doubt
- **Verification recommendation**: Whether readers need additional fact-checking

## Reader Guidance

**Bottom Line**: Clear recommendation for how readers should treat this information.

**Format your entire response using proper Markdown with headers (##, ###), bullet points (-), bold text (**bold**), and italics (*italic*) for maximum readability.**
""",

            "detailed_analysis": """
You are conducting a comprehensive forensic media analysis. Create a detailed Markdown report.

## FORENSIC ANALYSIS REPORT

**Article Under Investigation**: {article_text}
**Classification**: {prediction} (Confidence: {confidence:.2%})
**Analysis Date**: {metadata}

## Phase 1: Claim Decomposition

### Primary Claims Identified

1. **Claim 1**: [Extract main assertion]
   - **Verification Level**: Easy/Moderate/Difficult
   - **Impact**: High/Medium/Low

2. **Claim 2**: [Extract supporting assertion]
   - **Verification Level**: Easy/Moderate/Difficult
   - **Impact**: High/Medium/Low

### Supporting Evidence Review

- **Statistical Claims**: Numbers and percentages presented
- **Attribution Claims**: Who said what and when
- **Temporal Claims**: Dates, timelines, sequences

## Phase 2: Source Attribution Analysis

### Source Quality Matrix

| Source Type | Count | Quality | Accessibility |
|-------------|--------|---------|---------------|
| Official Documents | X | High/Med/Low | Public/Restricted |
| Expert Quotes | X | High/Med/Low | Verifiable/Unclear |
| News Reports | X | High/Med/Low | Available/Missing |

### Attribution Assessment

- **Direct Sources**: First-hand information providers
- **Secondary Sources**: Reporting based on other sources
- **Anonymous Sources**: Unidentified information providers

## Phase 3: Language & Presentation Analysis

### Writing Quality Indicators

- **Professional Standards**: ✅/❌ Meets journalistic standards
- **Technical Accuracy**: ✅/❌ Appropriate detail level
- **Emotional Language**: ✅/❌ Neutral vs. manipulative tone
- **Logical Flow**: ✅/❌ Coherent argument structure

## Phase 4: Contextual Verification

### Fact-Checking Results

- **✅ Confirmed Facts**: Information verified through reliable sources
- **❌ Contradicted Facts**: Information that conflicts with established facts
- **❓ Unverified Claims**: Information that cannot be independently confirmed

### Plausibility Assessment

- **Timeline Consistency**: Do dates and sequences align?
- **Geographic Logic**: Do locations and distances make sense?
- **Institutional Knowledge**: Do claims match how organizations actually work?

## Final Forensic Assessment

**Overall Credibility Score**: {confidence:.0%}/100%

**Evidence Quality**: High/Medium/Low
**Source Reliability**: High/Medium/Low
**Logical Consistency**: High/Medium/Low

**Recommendation**: [Clear action for readers - trust, verify, or dismiss]

**Priority for Human Review**: High/Medium/Low

Use proper Markdown formatting throughout your forensic analysis.
""",

            "confidence_analysis": """
Analyze the appropriateness of the confidence level using structured Markdown.

## CONFIDENCE LEVEL ANALYSIS

### Current AI Assessment

- **Article Classification**: {prediction}
- **Assigned Confidence**: {confidence:.2%}
- **Article Content**: {article_text}

## Confidence Appropriateness Review

### ✅ Factors Supporting Current Confidence

1. **Factor 1**: Specific reason the confidence level is justified
2. **Factor 2**: Additional evidence supporting the confidence
3. **Factor 3**: Pattern recognition that validates the assessment

### ⚠️ Uncertainty Indicators Present

1. **Ambiguous Elements**: Unclear or contradictory information
2. **Mixed Signals**: Both credible and questionable elements
3. **Knowledge Gaps**: Information that's missing or incomplete

### 🎯 Edge Case Assessment

- **Content Complexity**: Simple/Moderate/Complex
- **Novel Elements**: Standard reporting vs. unusual claims
- **Borderline Factors**: Elements that could swing classification

## Human Review Recommendation

**Priority Level**: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW

**Rationale**: Explain why human review is or isn't necessary

**Focus Areas for Human Reviewers**:
- Specific claims that need expert evaluation
- Sources that require additional verification
- Technical details that need specialist knowledge

## Confidence Calibration

**Recommended Confidence**: [X]%

**Adjustment Rationale**: Why confidence should remain same or change

**Reader Advisory**: How confident the public should be in this assessment

Format your analysis using clear Markdown headers and structured lists.
"""
        }
    
    def _get_enhanced_credible_source_prompts(self) -> Dict[str, str]:
        """Enhanced credible source prompts requesting specific, actionable URLs"""
        return {
            "source_recommendations": """
You are a fact-checking specialist providing SPECIFIC, ACTIONABLE source recommendations for this exact article.

**ARTICLE TOPIC**: {topic_summary}
**KEY CLAIMS TO VERIFY**: {extracted_claims}
**CLASSIFICATION**: {prediction}
**NEWS DOMAIN**: {news_domain}
**ORIGINAL SOURCE**: {original_source}

## SPECIFIC VERIFICATION STRATEGY

Based on the exact claims in this article, provide these specific resources:

### CLAIM-SPECIFIC SOURCES (Not Generic Institutions)

**For Each Major Claim in the Article**:

1. **Statistical Claims** (if present):
   - Direct link to: Official statistics/data pages
   - Example: https://data.gov/dataset/specific-topic NOT https://data.gov
   - Search: "exact statistic" + "official source"

2. **Expert Quotes** (if present):
   - Direct link to: Expert's original research/statements  
   - Example: https://university.edu/research/specific-study NOT https://university.edu
   - Search: "expert name" + "quoted topic" + original publication

3. **Event Claims** (if present):
   - Direct link to: Contemporary news coverage of the event
   - Example: https://reuters.com/article/specific-event-2024 NOT https://reuters.com
   - Search: "event name" + "date" + "reuters OR ap news"

4. **Policy/Legal Claims** (if present):
   - Direct link to: Actual policy documents/legal texts
   - Example: https://congress.gov/bill/specific-bill NOT https://congress.gov  
   - Search: "policy name" OR "bill number" + official government site

5. **Scientific Claims** (if present):
   - Direct link to: Peer-reviewed papers or official studies
   - Example: https://pubmed.ncbi.nlm.nih.gov/specific-study-id NOT https://pubmed.ncbi.nlm.nih.gov
   - Search: "study title" OR "doi number" + pubmed

### FACT-CHECKING CROSS-REFERENCE
Check if similar claims have been fact-checked:
- Search: "claim keywords" + site:snopes.com
- Search: "claim keywords" + site:factcheck.org  
- Search: "claim keywords" + site:politifact.com

## OUTPUT FORMAT:
For each source, provide:
- **Claim Being Verified**: [Exact quote from article]
- **Specific URL**: [Direct link to verification page]
- **What to Look For**: [Exactly what information confirms/refutes the claim]
- **Confidence**: [How certain this source can verify the claim]

**CRITICAL**: Every URL must be SPECIFIC to the article's claims, not generic institutional homepages.
""",

            "reliability_assessment": """
Conduct a comprehensive source reliability assessment with specific scoring.

## SOURCE RELIABILITY ANALYSIS

**Source Being Evaluated**: {source_name}
**Source Content**: {source_content}
**Domain Context**: {news_domain}
**Article Context**: {context}

## RELIABILITY SCORING FRAMEWORK

### 1. Authority & Expertise Assessment

**Score: ___/10**

#### Evaluation Criteria:
- **Relevant Credentials**: ✅/❌ Appropriate qualifications for topic
- **Professional Standing**: ✅/❌ Recognized expertise in field
- **Track Record**: ✅/❌ History of accurate reporting/analysis
- **Institutional Backing**: ✅/❌ Credible organizational support
- **Peer Recognition**: ✅/❌ Acknowledged by other experts

**Specific Examples**: [List concrete evidence of authority]

### 2. Transparency & Accountability

**Score: ___/10**

#### Evaluation Criteria:
- **Methodology Disclosure**: ✅/❌ Clear about how information gathered
- **Source Attribution**: ✅/❌ Properly cites original sources
- **Funding Transparency**: ✅/❌ Open about financial backing
- **Correction Policy**: ✅/❌ Acknowledges and fixes errors
- **Contact Information**: ✅/❌ Accessible for questions

**Transparency Evidence**: [Specific examples of openness]

### 3. Independence & Bias Assessment

**Score: ___/10**

#### Evaluation Criteria:
- **Financial Independence**: ✅/❌ Diverse, disclosed funding
- **Editorial Independence**: ✅/❌ Free from external pressure
- **Political Neutrality**: ✅/❌ Balanced treatment of issues
- **Commercial Conflicts**: ✅/❌ No undisclosed business interests
- **Ideological Balance**: ✅/❌ Acknowledges worldview limitations

**Independence Indicators**: [Evidence of unbiased reporting]

### 4. Consistency & Reliability

**Score: ___/10**

#### Evaluation Criteria:
- **Internal Consistency**: ✅/❌ Coherent within current content
- **External Consistency**: ✅/❌ Aligns with other credible sources
- **Temporal Consistency**: ✅/❌ Reliable reporting over time
- **Factual Accuracy**: ✅/❌ Previous claims have been verified
- **Editorial Standards**: ✅/❌ Evidence of quality control

**Consistency Examples**: [Track record evidence]

## OVERALL RELIABILITY ASSESSMENT

**Total Score: ___/40 (Average: ___/10)**

### Quality Classification:
- **9-10/10**: Extremely Reliable - Trust with high confidence
- **7-8/10**: Highly Reliable - Trust with minor verification
- **5-6/10**: Moderately Reliable - Verify through additional sources
- **3-4/10**: Low Reliability - Use only with significant caveats
- **1-2/10**: Unreliable - Do not use as credible source

## USAGE RECOMMENDATIONS

### ✅ STRENGTHS
- [Specific strength 1 with examples]
- [Specific strength 2 with examples]
- [Specific strength 3 with examples]

### ⚠️ LIMITATIONS
- [Specific limitation 1 with impact]
- [Specific limitation 2 with impact]
- [Specific limitation 3 with impact]

### 🎯 VERIFICATION STRATEGY

**For information from this source:**
1. **Always verify**: [What always needs checking]
2. **Cross-reference with**: [Specific alternative sources to check]
3. **Pay special attention to**: [Areas of particular concern]
4. **Contact directly for**: [When to reach out to source]

### 📊 COMPARISON WITH SIMILAR SOURCES

**Better than**: [Similar sources with lower reliability]
**Comparable to**: [Sources with similar reliability level]
**Not as reliable as**: [Higher-quality alternatives]

**Provide specific, actionable guidance for using this source responsibly.**
""",

            "verification_strategy": """
Design a comprehensive verification strategy with specific action steps and timelines.

## VERIFICATION STRATEGY DESIGN

**Priority Claims**: {priority_claims}
**Domain Context**: {domain_context}
**Evidence Available**: {evidence_context}
**Available Resources**: {available_resources}

## STRATEGIC VERIFICATION FRAMEWORK

### Phase 1: Rapid Assessment (15-30 minutes)

#### Primary Verification Route

1. **Official Source Check** (5 minutes)
   - Visit primary organization websites
   - Search press release sections
   - Check official social media accounts

2. **Quick Fact-Check Search** (10 minutes)
   - Search Snopes, FactCheck.org, PolitiFact
   - Look for similar claims already fact-checked
   - Check "hoax alert" databases

3. **Basic Source Verification** (15 minutes)
   - Verify quoted individuals exist and hold claimed positions
   - Check if organizations mentioned are real
   - Confirm basic biographical details

**Expected Outcome**: Initial credibility assessment - Proceed/Stop/Investigate Further

### Phase 2: Standard Verification (1-2 hours)

#### Secondary Verification Route

1. **Multi-Source Corroboration** (30 minutes)
   - Search 3-5 independent news sources
   - Compare reporting details across sources
   - Note discrepancies or consistencies

2. **Expert Consultation** (45 minutes)
   - Identify 2-3 relevant subject matter experts
   - Send brief inquiry emails with specific questions
   - Search for expert commentary on similar topics

3. **Document Verification** (30 minutes)
   - Search for original documents, studies, reports
   - Verify publication dates and authors
   - Check if documents are cited correctly

**Expected Outcome**: Confidence level assessment - Confirm/Refute/Uncertain

### Phase 3: Deep Investigation (4+ hours)

#### Comprehensive Verification Route

1. **Primary Source Contact** (2 hours)
   - Direct outreach to organizations mentioned
   - Interview with quoted individuals if possible
   - Request original documents or data

2. **Academic Literature Review** (1 hour)
   - Search PubMed, Google Scholar, institutional repositories
   - Review peer-reviewed research on topic
   - Consult subject matter academic experts

3. **Statistical/Data Verification** (1 hour)
   - Verify numbers against official databases
   - Check statistical methodologies if studies cited
   - Confirm data interpretation accuracy

**Expected Outcome**: Definitive verification status

## RESOURCE ALLOCATION STRATEGY

### High-Priority Claims (60% of resources)
**Criteria**: Central to article's main message, high public impact
**Approach**: Full Phase 1-3 verification
**Success Metrics**: Definitive confirmation or refutation

### Medium-Priority Claims (30% of resources)
**Criteria**: Supporting evidence, moderate impact
**Approach**: Phase 1-2 verification
**Success Metrics**: Reasonable confidence in assessment

### Low-Priority Claims (10% of resources)
**Criteria**: Background information, minimal impact
**Approach**: Phase 1 rapid assessment only
**Success Metrics**: Basic credibility check completed

## RISK MITIGATION STRATEGY

### Challenge: Source Inaccessibility
**Solution**:
- Maintain database of backup expert contacts
- Use archived versions of websites (Wayback Machine)
- Employ FOIA requests for government information

### Challenge: Time Sensitivity
**Solution**:
- Prioritize claims by potential impact
- Use parallel verification processes
- Publish preliminary findings with updates

### Challenge: Conflicting Information
**Solution**:
- Document all conflicting sources
- Assess relative credibility of conflicting sources
- Seek additional independent verification
- Clearly communicate uncertainty to readers

### Challenge: Incomplete Information
**Solution**:
- Clearly identify what cannot be verified
- Explain limitations in verification process
- Provide guidance on interpreting partial information
- Update findings as new information becomes available

## SUCCESS METRICS & DELIVERABLES

### Verification Confidence Levels
- **HIGH (90%+ confidence)**: Multiple independent sources confirm
- **MEDIUM (70-89% confidence)**: Preponderance of evidence supports
- **LOW (50-69% confidence)**: Some evidence but significant gaps
- **INSUFFICIENT (<50% confidence)**: Cannot verify with available resources

### Final Deliverables
1. **Verification Summary**: Clear status for each major claim
2. **Source Documentation**: Complete list of sources consulted
3. **Evidence Archive**: Key documents and screenshots preserved
4. **Uncertainty Report**: Clear identification of unresolved questions
5. **Update Protocol**: Plan for incorporating new information

**Provide specific timelines, contact strategies, and measurable outcomes for effective verification.**
""",

            "fact_check_guidance": """
You are a fact-checking editor providing detailed, actionable guidance for verifying specific claims.

## FACT-CHECKING GUIDANCE

**Claims to Verify**: {extracted_claims}
**Recommended Sources**: {recommended_sources}
**Verification Priority**: {verification_priority}
**Available Time**: {time_constraints}

## CLAIM PRIORITIZATION & TRIAGE

### 🔴 CRITICAL CLAIMS (Verify First - 60% of effort)
**Criteria**: Central to article, high public impact, easily verifiable
**Examples from article**: [List specific critical claims]
**Verification Deadline**: Within 2 hours

### 🟡 IMPORTANT CLAIMS (Verify Second - 30% of effort)
**Criteria**: Supporting main narrative, moderate impact, requires some effort
**Examples from article**: [List specific important claims]
**Verification Deadline**: Within 6 hours

### 🟢 SUPPORTING CLAIMS (Verify if Time Allows - 10% of effort)
**Criteria**: Background information, minimal impact, time-intensive
**Examples from article**: [List specific supporting claims]
**Verification Deadline**: Within 24 hours

## DETAILED VERIFICATION ACTION PLAN

### For Each Priority Claim, Follow This Protocol:

#### Step 1: Information Gathering (Time: 15-30 minutes per claim)

1. **Primary Source Identification**
   - Who originally made this claim?
   - What organization released this information?
   - When was this claim first made public?

2. **Supporting Evidence Collection**
   - Official documents that should exist
   - Expert opinions that can be solicited
   - Statistical data that can be verified

3. **Alternative Source Mapping**
   - Independent organizations that would know
   - Experts who could confirm or deny
   - Databases containing relevant information

#### Step 2: Source Contact Strategy (Time: 30-60 minutes per claim)

1. **Official Organization Outreach**
   - Call main number and ask for press office
   - Email specific questions to media relations
   - Check recent press releases and announcements

2. **Expert Consultation**
   - Identify 2-3 subject matter experts
   - Send specific questions via email
   - Provide context but don't reveal your hypothesis

3. **Peer Verification**
   - Contact other journalists who cover this beat
   - Check with fact-checking organizations
   - Search academic/professional networks

#### Step 3: Evidence Documentation (Time: 15-30 minutes per claim)

1. **Source Verification**
   - Screenshots of relevant web pages
   - PDF downloads of important documents
   - Email confirmations from official sources

2. **Quote Verification**
   - Audio/video of original statements if available
   - Official transcripts or press releases
   - Context of when and where statements were made

3. **Statistical Verification**
   - Original data sources and methodologies
   - Peer review status of cited studies
   - Potential conflicts of interest in research

## FACT-CHECKING BEST PRACTICES

### The Multiple Source Rule
✅ **GOLD STANDARD**: 3+ independent sources confirm
✅ **ACCEPTABLE**: 2 independent sources + official document
⚠️ **CAUTION**: Single source, even if authoritative
❌ **INSUFFICIENT**: No independent verification possible

### Primary Source Priority
1. **Tier 1**: Original documents, official statements, direct witnesses
2. **Tier 2**: Expert analysis, peer-reviewed research, established news reporting
3. **Tier 3**: Secondary reporting, aggregated information, unverified claims

### Expert Validation Protocol
- **Relevant Expertise**: Expert's knowledge directly relates to claim
- **Independence**: Expert has no financial/personal interest in outcome
- **Recognition**: Expert is acknowledged by peers in the field
- **Transparency**: Expert willing to be quoted and provide credentials

## QUALITY ASSURANCE CHECKLIST

### Before Publishing Fact-Check Results:
☐ **Source Independence Verified**: Confirmed sources are truly independent
☐ **Bias Assessment Completed**: Accounted for potential source biases
☐ **Context Verified**: Ensured quotes/data aren't taken out of context
☐ **Alternative Explanations Considered**: Explored other plausible interpretations
☐ **Expert Review Obtained**: At least one subject matter expert consulted
☐ **Documentation Archived**: All evidence properly saved and attributed
☐ **Confidence Level Assigned**: Clear assessment of verification certainty

## VERIFICATION STATUS REPORTING

### Status Categories:
- **✅ CONFIRMED**: Multiple reliable sources verify claim is accurate
- **❌ REFUTED**: Multiple reliable sources contradict claim
- **❓ UNCERTAIN**: Insufficient evidence to confirm or refute claim
- **🔄 ONGOING**: Verification in progress, awaiting key source responses

### For Each Verified Claim, Report:
1. **Verification Status**: [Confirmed/Refuted/Uncertain/Ongoing]
2. **Evidence Summary**: Key supporting or contradicting evidence
3. **Source Quality**: Assessment of source reliability and independence
4. **Confidence Level**: How certain the verification conclusion is
5. **Context Notes**: Important caveats or limitations
6. **Update Schedule**: When new information might become available

## COMMON VERIFICATION PITFALLS TO AVOID

### ❌ False Confirmation
- Don't stop at first source that confirms your hypothesis
- Always seek disconfirming evidence
- Be wary of circular sourcing (sources citing each other)

### ❌ Authority Bias
- Don't assume official sources are always accurate
- Verify credentials of claimed experts
- Question conflicts of interest

### ❌ Urgency Pressure
- Don't sacrifice accuracy for speed
- Clearly label preliminary findings as such
- Build in time for source responses

**Provide specific, actionable steps with realistic timelines and measurable verification standards.**
"""
        }
    
    def _get_claim_extractor_prompts(self) -> Dict[str, str]:
        """Standard claim extraction prompts with enhanced formatting"""
        return {
            "claim_extraction": """
You are an expert fact-checker extracting verifiable claims from news articles.

**Article**: {article_text}
**Classification**: {prediction} (Confidence: {confidence:.2%})
**Topic**: {topic_domain}

## CLAIM EXTRACTION GUIDELINES

Extract **5-8 specific, verifiable claims** using this format:

### Claim 1: [Priority Level]
**Text**: "Exact quote or paraphrase from article"
**Type**: Statistical/Event/Attribution/Research/Policy/Causal/Other
**Priority**: 1=Critical, 2=Important, 3=Minor
**Verifiability**: X/10 (how easily fact-checked)
**Source**: Who made this claim
**Verification Strategy**: How to check this claim
**Why Important**: Significance in the article

### Focus On:
- **Statistical Claims**: Numbers, percentages, quantities
- **Event Claims**: Things that happened or will happen
- **Attribution Claims**: Who said or did what
- **Research Claims**: Scientific findings and studies

### Avoid:
- Opinions ("This is the best policy")
- Vague statements ("Many people think")
- Predictions ("This will probably happen")

Extract the most important verifiable claims using this exact format.
""",

            "verification_analysis": """
Analyze how the extracted claims can be fact-checked:

**Claims**: {extracted_claims}

For each claim, provide:
- **Primary Sources**: Where to find original evidence
- **Expert Verification**: Which experts could confirm/deny
- **Documentation**: What documents should exist
- **Timeline**: Whether timing can be verified
- **Search Strategy**: Specific search terms and databases

Focus on actionable verification pathways.
""",

            "claim_prioritization": """
Prioritize claims by importance and verifiability:

**Claims**: {extracted_claims}

## PRIORITY RANKING

### 🔴 High Priority (Verify First)
[Most important claims that are easily verifiable]

### 🟡 Medium Priority (Verify Second)
[Important claims requiring moderate effort]

### 🟢 Low Priority (Verify If Time Allows)
[Minor claims or difficult to verify]

Explain ranking rationale for each priority level.
"""
        }
    
    def _get_context_analyzer_prompts(self) -> Dict[str, str]:
        """Context analysis prompts with enhanced bias detection"""
        return {
            "bias_detection": """
Analyze potential bias using structured assessment:

**Article**: {article_text}
**Source**: {source}
**Domain**: {topic_domain}
**Classification**: {prediction} (Confidence: {confidence:.2%})

## BIAS ANALYSIS FRAMEWORK

### Political Bias Assessment
- **Partisan Language**: Identify politically charged terms
- **Source Selection**: Are sources balanced across perspectives?
- **Issue Framing**: How is the topic presented?
- **Missing Viewpoints**: What perspectives are absent?

### Emotional Bias Assessment
- **Emotional Language**: List emotion-triggering words/phrases
- **Fear Appeals**: Identify scare tactics or threats
- **Hope Appeals**: Identify positive manipulation
- **Urgency Language**: False time pressure indicators

### Selection Bias Assessment
- **Information Emphasized**: What facts are highlighted?
- **Information Omitted**: What context is missing?
- **Source Balance**: Are opposing views included?
- **Cherry-Picking**: Selective use of evidence?

## BIAS SCORING

**Overall Bias Level**: ___/10 (10 = Highly Biased, 1 = Minimal Bias)

### Reader Advisory
- **Be aware of**: Specific bias concerns
- **Seek additional perspectives from**: Alternative source types
- **Question**: Particular claims or framings
""",

            "framing_analysis": """
Analyze how this article frames its subject matter:

**Article**: {article_text}
**Previous Analysis**: {previous_analysis}

## FRAMING ANALYSIS

### Narrative Structure
- **Primary Frame**: How is the main issue characterized?
- **Problem Definition**: What's the central problem/opportunity?
- **Causal Attribution**: Who/what is credited or blamed?
- **Solution Framing**: What actions are endorsed?

### Audience & Perspective
- **Target Audience**: Who is this written for?
- **Assumed Knowledge**: What background is assumed?
- **Missing Voices**: Whose perspectives are absent?
- **Alternative Frames**: How could this be framed differently?

Provide specific examples from the text supporting your analysis.
""",

            "emotional_manipulation": """
Identify emotional manipulation techniques:

**Article**: {article_text}
**Emotional Indicators**: {emotional_indicators}

## EMOTIONAL MANIPULATION ANALYSIS

### Fear-Based Appeals
- **Threat Amplification**: Examples of exaggerated dangers
- **Urgency Creation**: Artificial time pressure examples
- **Vulnerability Targeting**: Appeals to specific fears

### Anger & Outrage Tactics
- **Injustice Framing**: Examples of unfairness emphasis
- **Enemy Identification**: Us vs. them language
- **Moral Violations**: Appeals to violated principles

### Hope & Inspiration Manipulation
- **Solution Oversimplification**: Unrealistic easy fixes
- **Hero Worship**: Figures presented as saviors
- **Future Promises**: Unrealistic positive outcomes

**Manipulation Score**: ___/10 (10 = Highly Manipulative)

### Reader Advisory
Be aware of: [Specific manipulation techniques present]
Seek balanced perspective on: [Emotionally charged claims]
""",

            "propaganda_detection": """
Analyze propaganda techniques present in this article:

**Article**: {article_text}
**Classification**: {prediction} (Confidence: {confidence:.2%})

## PROPAGANDA TECHNIQUE ANALYSIS

### Classical Propaganda Methods
- **Name-Calling**: Negative labels for people/ideas/institutions
- **Glittering Generalities**: Emotionally appealing but vague terms
- **Transfer**: Connecting ideas with positive/negative symbols
- **Testimonial**: Inappropriate celebrity/authority endorsements
- **Plain Folks**: False appeals to common people
- **Card Stacking**: One-sided argument presentation
- **Bandwagon**: "Everyone's doing it" pressure

### Modern Propaganda Techniques
- **Astroturfing**: Fake grassroots movements
- **False Flag Operations**: Misattributed actions
- **Gaslighting**: Making readers question reality
- **Whataboutism**: Deflecting criticism with other issues

**Propaganda Intensity**: ___/10

Provide specific examples from the text for each technique identified.
"""
        }
    
    def _get_evidence_evaluator_prompts(self) -> Dict[str, str]:
        """Evidence evaluation prompts with enhanced quality assessment"""
        return {
            "evidence_evaluation": """
You are a fact-checking specialist providing SPECIFIC, ACTIONABLE evidence verification links.

**ARTICLE TO VERIFY**: {article_text}
**KEY CLAIMS**: {extracted_claims}
**CLASSIFICATION**: {prediction} ({confidence:.1%} confidence)

## EVIDENCE VERIFICATION SOURCES

Provide **EXACTLY 5 SPECIFIC VERIFICATION LINKS** that directly address claims in this article:

### 1. PRIMARY VERIFICATION SOURCE
**Specific Claim**: [Quote exact claim from article]
**Verification URL**: [Direct link to source that confirms/refutes this claim]
**What It Proves**: [Exactly what this source confirms or contradicts]
**Search Query**: "exact claim text" site:official-domain.org

### 2. STATISTICAL/DATA VERIFICATION  
**Specific Claim**: [Quote statistical claim from article]
**Verification URL**: [Direct link to official statistics/data]
**What It Proves**: [Exact numbers or data this source provides]
**Search Query**: [Exact search terms to find this data]

### 3. EXPERT VERIFICATION
**Specific Claim**: [Quote expert opinion/quote from article]  
**Verification URL**: [Direct link to expert's actual statement/research]
**What It Proves**: [What expert actually said vs. what's claimed]
**Search Query**: "expert name" + "exact quote" + original source

### 4. INSTITUTIONAL CONFIRMATION
**Specific Claim**: [Quote institutional claim from article]
**Verification URL**: [Direct link to official institutional response/data]  
**What It Proves**: [Official position vs. what article claims]
**Search Query**: site:institution.org + "specific topic"

### 5. TIMELINE/FACTUAL VERIFICATION
**Specific Claim**: [Quote date/event claim from article]
**Verification URL**: [Direct link proving/disproving timeline]
**What It Proves**: [Actual dates/events vs. claimed dates/events]
**Search Query**: "event name" + "actual date" + reliable source

## VERIFICATION INSTRUCTIONS
For each source above:
1. **Copy the exact URL** - must be clickable and specific to the claim
2. **Quote the specific text** from the article being verified  
3. **Explain exactly what** each source proves or disproves
4. **Provide search terms** that readers can use independently

**DO NOT provide generic institutional homepages - only specific verification URLs**
""",

            "source_quality": """
Evaluate source quality and credibility:

**Article**: {article_text}
**Sources**: {source_list}

## SOURCE QUALITY EVALUATION

### Individual Source Assessment
For each major source:
- **Authority Level**: Expert/Official/Institutional/Individual
- **Relevance**: How relevant to the claims?
- **Independence**: Potential conflicts of interest?
- **Accessibility**: Can readers verify this source?
- **Quality Score**: ___/10

### Overall Source Portfolio
- **Source Diversity**: Primary/Secondary/Expert/Official mix
- **Perspective Range**: Multiple viewpoints represented?
- **Independence Level**: How independent from each other?
- **Verification Potential**: How many can be checked?

**Portfolio Quality Score**: ___/10

### Source Reliability Ranking
1. **Most Reliable**: [Highest quality sources]
2. **Moderately Reliable**: [Medium quality sources]
3. **Questionable**: [Lowest quality sources]

### Verification Recommendations
- **Priority Check**: Which sources to verify first
- **Missing Sources**: What types needed
- **Red Flags**: Source concerns to investigate
""",

            "logical_consistency": """
Analyze logical consistency and reasoning quality:

**Article**: {article_text}
**Key Claims**: {key_claims}

## LOGICAL ANALYSIS FRAMEWORK

### Argument Structure Assessment
- **Premises**: Are foundational assumptions reasonable?
- **Evidence**: Does evidence actually support conclusions?
- **Logic Flow**: Do conclusions follow from premises?
- **Gap Identification**: Where are logical leaps made?

### Fallacy Detection
Common fallacies present:
- **Ad Hominem**: Attacking person vs. argument
- **Straw Man**: Misrepresenting opposing positions
- **False Dichotomy**: Only two options when more exist
- **Hasty Generalization**: Broad conclusions from limited examples
- **Appeal to Authority**: Inappropriate authority citations

### Consistency Evaluation
- **Internal**: Contradictory statements within article
- **Temporal**: Do dates and sequences align?
- **Statistical**: Do numbers add up correctly?
- **External**: Alignment with established knowledge

**Logic Quality Score**: ___/10

### Critical Thinking Advice
- **Strong Elements**: Well-reasoned parts
- **Weak Elements**: Poor reasoning areas
- **Reader Should Question**: What to be skeptical about
""",

            "evidence_gaps": """
Identify and analyze evidence gaps:

**Article**: {article_text}
**Claims**: {extracted_claims}

## EVIDENCE GAP ANALYSIS

### Missing Evidence Categories

#### Quantitative Gaps
- **Missing Statistics**: What numbers would strengthen claims?
- **Absent Comparisons**: What comparative data needed?
- **Timeline Gaps**: What chronological info missing?

#### Qualitative Gaps
- **Missing Experts**: Which experts should be consulted?
- **Absent Perspectives**: Whose viewpoints not included?
- **Context Omissions**: What background missing?

#### Verification Gaps
- **Source Documentation**: What documents should exist?
- **Cross-References**: What corroborating sources absent?
- **Independent Confirmation**: What hasn't been verified?

### Gap Priority Assessment
1. **Critical Gaps**: Must address to maintain credibility
2. **Important Gaps**: Should address to improve confidence
3. **Minor Gaps**: Could address for completeness

**Evidence Completeness Score**: ___/10

### Gap-Filling Recommendations
- **For Readers**: Where to find missing information
- **Critical Questions**: What questions gaps raise
- **Verification Steps**: How to fill gaps independently
"""
        }
    
    def get_prompt_template(self, agent_name: str, prompt_type: str) -> str:
        """Get specific prompt template"""
        agent_prompts = {
            "llm_explanation": self.llm_explanation_prompts,
            "credible_source": self.credible_source_prompts,
            "claim_extractor": self.claim_extractor_prompts,
            "context_analyzer": self.context_analyzer_prompts,
            "evidence_evaluator": self.evidence_evaluator_prompts
        }
        
        if agent_name not in agent_prompts:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        if prompt_type not in agent_prompts[agent_name]:
            raise ValueError(f"Unknown prompt type '{prompt_type}' for agent '{agent_name}'")
            
        return agent_prompts[agent_name][prompt_type]
    
    def update_prompt_template(self, agent_name: str, prompt_type: str, new_template: str):
        """Update a specific prompt template"""
        agent_prompts = {
            "llm_explanation": self.llm_explanation_prompts,
            "credible_source": self.credible_source_prompts,
            "claim_extractor": self.claim_extractor_prompts,
            "context_analyzer": self.context_analyzer_prompts,
            "evidence_evaluator": self.evidence_evaluator_prompts
        }
        
        if agent_name not in agent_prompts:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        agent_prompts[agent_name][prompt_type] = new_template
    
    def list_available_prompts(self) -> Dict[str, List[str]]:
        """List all available prompt templates by agent"""
        return {
            "llm_explanation": list(self.llm_explanation_prompts.keys()),
            "credible_source": list(self.credible_source_prompts.keys()),
            "claim_extractor": list(self.claim_extractor_prompts.keys()),
            "context_analyzer": list(self.context_analyzer_prompts.keys()),
            "evidence_evaluator": list(self.evidence_evaluator_prompts.keys())
        }
    
    def save_to_file(self, file_path: Optional[Path] = None):
        """Save all prompt templates to JSON file"""
        if file_path is None:
            from .settings import get_settings
            settings = get_settings()
            file_path = settings.project_root / "config" / "prompts.json"
        
        data = {
            "prompt_version": self.prompt_version,
            "llm_explanation_prompts": self.llm_explanation_prompts,
            "credible_source_prompts": self.credible_source_prompts,
            "claim_extractor_prompts": self.claim_extractor_prompts,
            "context_analyzer_prompts": self.context_analyzer_prompts,
            "evidence_evaluator_prompts": self.evidence_evaluator_prompts
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'PromptsConfig':
        """Load prompt templates from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# Global prompts config instance
_prompts_config_instance: Optional[PromptsConfig] = None

def get_prompt_template(agent_name: str, prompt_type: str) -> str:
    """Get specific prompt template"""
    global _prompts_config_instance
    if _prompts_config_instance is None:
        _prompts_config_instance = PromptsConfig()
    return _prompts_config_instance.get_prompt_template(agent_name, prompt_type)

def update_prompt_template(agent_name: str, prompt_type: str, new_template: str):
    """Update a specific prompt template"""
    global _prompts_config_instance
    if _prompts_config_instance is None:
        _prompts_config_instance = PromptsConfig()
    _prompts_config_instance.update_prompt_template(agent_name, prompt_type, new_template)

def list_available_prompts() -> Dict[str, List[str]]:
    """List all available prompt templates by agent"""
    global _prompts_config_instance
    if _prompts_config_instance is None:
        _prompts_config_instance = PromptsConfig()
    return _prompts_config_instance.list_available_prompts()
