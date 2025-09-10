# agents/claim_extractor/prompts.py

"""
Claim Extractor Agent Prompts

Industry-standard prompt templates for claim extraction, verification analysis,
claim prioritization, and structured claim parsing with enhanced output formatting.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class PromptResponse:
    """Structured response container for prompt outputs."""
    content: str
    metadata: Dict[str, Any]


class ClaimExtractionPrompts:
    """Core claim extraction prompts with structured output formatting."""
    
    @staticmethod
    def comprehensive_claim_extraction(article_text: str, prediction: str, 
                                     confidence: float, topic_domain: str) -> str:
        """
        Generate comprehensive claim extraction prompt with structured output.
        Enhanced for better claim identification and categorization.
        """
        return f"""You are an expert fact-checker and claim analyst specializing in identifying verifiable claims from news articles.

ARTICLE ANALYSIS CONTEXT:
- Domain: {topic_domain}
- Classification: {prediction}
- Confidence: {confidence:.2f}

ARTICLE TEXT:
{article_text}

TASK: Extract specific, verifiable claims that can be fact-checked.

EXTRACTION CRITERIA:
- Claims must be specific and testable
- Focus on factual assertions, not opinions
- Include statistical claims, attributions, and event claims
- Prioritize claims that could be misinformation if false

OUTPUT FORMAT:
For each claim, provide:

**Claim 1**: [Priority Level 1-3]
- **Text**: "[Exact claim text]"
- **Type**: [Statistical/Attribution/Event/Research/Policy/Causal]
- **Verifiability**: [Score 1-10]
- **Source**: "[Who made the claim or where it comes from]"
- **Verification Strategy**: "[How to verify this claim]"
- **Why Important**: "[Why this claim matters for fact-checking]"

**Claim 2**: [Priority Level 1-3]
[Continue with same format...]

PRIORITY LEVELS:
- Priority 1: Critical claims that could cause harm if false
- Priority 2: Important claims that significantly impact the story
- Priority 3: Supporting claims that provide additional context

Generate 3-8 claims maximum, focusing on quality over quantity."""

    @staticmethod
    def focused_claim_extraction(article_text: str, max_claims: int = 5) -> str:
        """
        Generate focused claim extraction for quick processing.
        Optimized for speed and essential claims only.
        """
        return f"""Extract the {max_claims} most important verifiable claims from this article.

ARTICLE:
{article_text}

OUTPUT FORMAT:
1. [Claim text] - Type: [Statistical/Attribution/Event/Research]
2. [Claim text] - Type: [Statistical/Attribution/Event/Research]
3. [Continue...]

Focus on:
- Claims with specific numbers, percentages, or statistics
- Direct quotes from named sources
- Specific events with dates/locations
- Research findings or study results

Keep each claim concise and factual."""

    @staticmethod
    def claim_verification_analysis(extracted_claims: str) -> str:
        """
        Generate verification analysis prompt for extracted claims.
        Provides detailed verification strategies and difficulty assessment.
        """
        return f"""You are a fact-checking expert analyzing claims for verification feasibility.

EXTRACTED CLAIMS:
{extracted_claims}

VERIFICATION ANALYSIS REQUIREMENTS:

## Individual Claim Assessment
For each claim, analyze:
- Verification difficulty (Easy/Moderate/Hard)
- Required source types (Primary/Secondary/Expert)
- Potential verification challenges
- Estimated time for verification
- Alternative verification approaches

## Overall Assessment
- Most critical claims to verify first
- Claims that require similar verification approaches
- Claims that might be interdependent
- Resource requirements for verification

## Verification Roadmap
Provide step-by-step verification process:
1. Immediate verification targets
2. Required research and sources
3. Expert consultation needs
4. Timeline for complete verification

## Red Flags
Identify claims that show signs of:
- Potential misinformation
- Unsupported assertions
- Misleading statistics
- Unreliable sourcing

Generate detailed, actionable verification guidance."""

class ClaimCategorizationPrompts:
    """Prompts for advanced claim categorization and analysis."""
    
    @staticmethod
    def claim_prioritization_analysis(extracted_claims: str, domain: str = "general") -> str:
        """
        Generate claim prioritization based on impact, verifiability, and harm potential.
        """
        return f"""You are a senior fact-checker prioritizing claims for verification in the {domain} domain.

CLAIMS TO PRIORITIZE:
{extracted_claims}

PRIORITIZATION CRITERIA:

## Impact Assessment
- Public health and safety implications
- Financial/economic impact on readers
- Political or social influence potential
- Reach and viral potential of the claim

## Verification Feasibility
- Availability of authoritative sources
- Technical complexity of verification
- Time and resources required
- Access to primary evidence

## Risk Assessment
- Potential harm if claim is false
- Difficulty for average reader to verify
- History of similar false claims
- Sensitivity of the topic

OUTPUT FORMAT:

**HIGH PRIORITY (Verify First)**
- Claim X: [Justification for high priority]
- Risk level: [Low/Medium/High]
- Verification approach: [Brief strategy]

**MEDIUM PRIORITY (Verify Second)**
- Claim Y: [Justification for medium priority]
- Risk level: [Low/Medium/High]
- Verification approach: [Brief strategy]

**LOW PRIORITY (Verify if Resources Allow)**
- Claim Z: [Justification for low priority]
- Risk level: [Low/Medium/High]
- Verification approach: [Brief strategy]

## Recommended Verification Sequence
1. [Order and rationale for verification sequence]
2. [Parallel verification opportunities]
3. [Resource allocation suggestions]

Provide clear, actionable prioritization with specific justifications."""

    @staticmethod
    def claim_type_classification(claim_text: str) -> str:
        """
        Classify individual claims into detailed categories.
        """
        return f"""Analyze this claim and provide detailed classification:

CLAIM: {claim_text}

CLASSIFICATION ANALYSIS:

## Primary Type
Choose from: Statistical, Attribution, Event, Research, Policy, Causal, Comparative, Temporal

## Secondary Characteristics
- Specificity Level: [Very Specific/Specific/General/Vague]
- Time Sensitivity: [Current/Recent/Historical/Timeless]
- Geographic Scope: [Local/National/International/Global]
- Complexity: [Simple/Moderate/Complex/Highly Complex]

## Verification Requirements
- Primary Sources Needed: [List specific types]
- Expert Consultation: [Yes/No - which experts]
- Data Access: [Public/Restricted/Proprietary/Unknown]
- Verification Timeline: [Hours/Days/Weeks/Months]

## Risk Assessment
- Misinformation Potential: [Low/Medium/High]
- Public Impact: [Minimal/Moderate/Significant/Critical]
- Harm Potential: [None/Low/Medium/High]

Provide detailed classification with justification."""

class StructuredOutputPrompts:
    """Prompts designed for consistent structured output parsing."""
    
    @staticmethod
    def json_claim_extraction(article_text: str, max_claims: int = 8) -> str:
        """
        Extract claims in JSON format for easy parsing.
        """
        return f"""Extract verifiable claims from this article and return in valid JSON format.

ARTICLE:
{article_text}

Return exactly this JSON structure with {max_claims} claims maximum:

{{
  "claims": [
    {{
      "id": 1,
      "text": "Exact claim text here",
      "type": "Statistical|Attribution|Event|Research|Policy|Causal",
      "priority": 1,
      "verifiability_score": 8,
      "source": "Who made this claim",
      "verification_strategy": "How to verify this claim",
      "importance": "Why this claim is important to fact-check"
    }}
  ],
  "metadata": {{
    "total_claims": 3,
    "high_priority_claims": 1,
    "domain": "health",
    "extraction_confidence": 0.85
  }}
}}

Ensure valid JSON syntax. Focus on verifiable, specific claims only."""

    @staticmethod
    def tabular_claim_extraction(article_text: str) -> str:
        """
        Extract claims in tabular format for structured analysis.
        """
        return f"""Extract claims from this article in table format.

ARTICLE:
{article_text}

OUTPUT FORMAT:

| ID | Claim Text | Type | Priority | Verifiability | Source | Verification Method |
|----|------------|------|----------|---------------|--------|-------------------|
| 1  | [Claim]    | Statistical | 1 | 8/10 | [Source] | [Method] |
| 2  | [Claim]    | Attribution | 2 | 7/10 | [Source] | [Method] |

COLUMN DEFINITIONS:
- ID: Sequential number
- Claim Text: Exact factual assertion
- Type: Statistical/Attribution/Event/Research/Policy/Causal
- Priority: 1 (Critical), 2 (Important), 3 (Supporting)
- Verifiability: Score 1-10 (10 = easily verifiable)
- Source: Who made the claim or where it originates
- Verification Method: Specific approach to verify

Extract 3-6 most important claims. Keep claim text concise but complete."""

    @staticmethod
    def tabular_extraction(article_text: str) -> str:
        """
        Extract claims in tabular format for structured analysis.
        """
        return f"""Extract claims from this article in table format.

ARTICLE:
{article_text}

OUTPUT FORMAT:

| ID | Claim Text | Type | Priority | Verifiability | Source | Verification Method |
|----|------------|------|----------|---------------|--------|---------------------|
| 1 | [Claim] | Statistical | 1 | 8/10 | [Source] | [Method] |
| 2 | [Claim] | Attribution | 2 | 7/10 | [Source] | [Method] |

COLUMN DEFINITIONS:

- ID: Sequential number
- Claim Text: Exact factual assertion
- Type: Statistical/Attribution/Event/Research/Policy/Causal
- Priority: 1 (Critical), 2 (Important), 3 (Supporting)
- Verifiability: Score 1-10 (10 = easily verifiable)
- Source: Who made the claim or where it originates
- Verification Method: Specific approach to verify

Extract 3-6 most important claims. Keep claim text concise but complete."""

def get_claim_prompt_template(prompt_type: str, **kwargs) -> str:
    """
    Get specific claim extraction prompt template.
    
    Args:
        prompt_type: Type of prompt needed
        **kwargs: Parameters for prompt formatting
        
    Returns:
        Formatted prompt string
    """
    prompt_mapping = {
        'comprehensive_extraction': ClaimExtractionPrompts.comprehensive_claim_extraction,
        'focused_extraction': ClaimExtractionPrompts.focused_claim_extraction,
        'verification_analysis': ClaimExtractionPrompts.claim_verification_analysis,
        'prioritization_analysis': ClaimCategorizationPrompts.claim_prioritization_analysis,
        'type_classification': ClaimCategorizationPrompts.claim_type_classification,
        'json_extraction': StructuredOutputPrompts.json_claim_extraction,
        'tabular_extraction': StructuredOutputPrompts.tabular_extraction
    }
    
    if prompt_type not in prompt_mapping:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompt_mapping[prompt_type](**kwargs)

def validate_claim_extraction_output(output: str, expected_format: str = "structured") -> bool:
    """
    Validate that claim extraction output follows expected format.
    
    Args:
        output: Generated output to validate
        expected_format: Expected format (structured, json, tabular)
        
    Returns:
        True if output format is valid
    """
    if expected_format == "structured":
        return "**Claim" in output and "Priority" in output
    elif expected_format == "json":
        try:
            import json
            json.loads(output)
            return True
        except:
            return False
    elif expected_format == "tabular":
        return "|" in output and "Claim Text" in output
    
    return False

# Testing functionality
if __name__ == "__main__":
    """Test claim extractor prompts."""
    
    # Test comprehensive claim extraction
    test_article = """
    A new study published in Nature Medicine found that 85% of patients who received
    the experimental drug showed significant improvement within 30 days. Dr. Sarah Johnson,
    lead researcher at Harvard Medical School, announced the results at yesterday's conference.
    The clinical trial included 1,200 participants across 15 hospitals worldwide.
    """
    
    comprehensive_prompt = ClaimExtractionPrompts.comprehensive_claim_extraction(
        article_text=test_article,
        prediction="REAL",
        confidence=0.78,
        topic_domain="health"
    )
    
    print("=== COMPREHENSIVE CLAIM EXTRACTION PROMPT ===")
    print(comprehensive_prompt[:500] + "...")
    
    # Test focused extraction
    focused_prompt = ClaimExtractionPrompts.focused_claim_extraction(test_article, max_claims=3)
    
    print("\n=== FOCUSED CLAIM EXTRACTION PROMPT ===")
    print(focused_prompt[:300] + "...")
    
    # Test JSON extraction
    json_prompt = StructuredOutputPrompts.json_claim_extraction(test_article, max_claims=3)
    
    print("\n=== JSON EXTRACTION PROMPT ===")
    print(json_prompt[:400] + "...")
    
    # Test prompt template function
    try:
        template_prompt = get_claim_prompt_template(
            'verification_analysis',
            extracted_claims="1. 85% improvement rate\n2. Dr. Johnson announcement"
        )
        print("\n=== TEMPLATE FUNCTION TEST ===")
        print("✓ Template function working correctly")
    except Exception as e:
        print(f"✗ Template function error: {e}")
    
    # Test validation
    test_output = "**Claim 1**: Priority 1\n- **Text**: \"Test claim\""
    is_valid = validate_claim_extraction_output(test_output, "structured")
    print(f"\n=== VALIDATION TEST ===")
    print(f"Output validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    
    print("\n=== CLAIM EXTRACTOR PROMPTS TESTING COMPLETED ===")
