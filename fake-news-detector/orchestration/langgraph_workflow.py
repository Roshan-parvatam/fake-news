"""
LangGraph Workflow for Smart Fake News Detection Pipeline
Enhanced with conditional routing to optimize processing and reduce costs
NOW WITH ALL 6 AGENTS INCLUDING CONTEXT ANALYZER
"""

from langgraph.graph import StateGraph, END
from .state import FakeNewsState
from .nodes import (
    bert_classifier_node,
    claim_extractor_node,
    context_analyzer_node,  # Added the missing 6th agent
    evidence_evaluator_node,
    credible_source_node,
    llm_explanation_node
)

def route_after_bert(state: FakeNewsState) -> str:
    """Smart routing after BERT classification"""
    confidence = state.get("confidence_scores", {}).get("bert", 0.0)
    bert_results = state.get("bert_results", {})
    prediction = bert_results.get("prediction", "UNKNOWN") if bert_results else "UNKNOWN"
    require_detailed = state.get("require_detailed_analysis", False)
    
    # Update processing path
    new_state = dict(state)
    
    # Honor explicit request for detailed analysis
    if require_detailed:
        print("🧪 [Router] Detailed analysis requested; running full pipeline")
        new_state["processing_path"] = "full_analysis_forced"
        state.update(new_state)
        return "claim_extractor"
    
    # Skip expensive processing for high-confidence REAL news
    if confidence > 0.92 and prediction == "REAL":
        print(f"🚀 [Router] Fast-tracking obvious REAL news (confidence: {confidence:.1%})")
        new_state["skip_expensive_processing"] = True
        new_state["processing_path"] = "fast_track_real"
        state.update(new_state)
        return "llm_explanation"
    
    # Continue with claim extraction for uncertain cases
    print(f"🔍 [Router] Proceeding to claim extraction (confidence: {confidence:.1%})")
    new_state["processing_path"] = "full_analysis"
    state.update(new_state)
    return "claim_extractor"

def route_after_claims(state: FakeNewsState) -> str:
    """Smart routing after claim extraction - now includes context analysis"""
    extracted_claims = state.get("extracted_claims", [])
    claims_count = len(extracted_claims)
    
    # Check if claims have richness score
    claim_richness = 0
    if extracted_claims and isinstance(extracted_claims[0], dict):
        high_priority_claims = sum(1 for claim in extracted_claims
                                 if claim.get("priority", 3) <= 2)
        verifiable_claims = sum(1 for claim in extracted_claims
                              if claim.get("verifiability_score", 0) >= 6)
        claim_richness = high_priority_claims + verifiable_claims
    else:
        claim_richness = claims_count
    
    # Always go to context analysis first (our missing 6th agent!)
    print(f"🔍 [Router] Proceeding to context analysis (claims: {claims_count}, richness: {claim_richness})")
    return "context_analyzer"

def route_after_context(state: FakeNewsState) -> str:
    """Route after context analysis based on bias and manipulation scores"""
    context_analysis = state.get("context_analysis", {})
    context_scores = context_analysis.get("context_scores", {}) if context_analysis else {}
    
    overall_context_score = context_scores.get("overall_context_score", 5.0)
    risk_level = context_scores.get("risk_level", "MEDIUM")
    
    extracted_claims = state.get("extracted_claims", [])
    claims_count = len(extracted_claims)
    
    # Skip evidence evaluation for low-risk, low-claim content
    if risk_level in ["LOW"] and claims_count < 3:
        print(f"⚡ [Router] Skipping evidence evaluation (low risk: {risk_level}, few claims: {claims_count})")
        return "credible_source"
    
    # Full evidence evaluation for high-risk or claim-rich content
    print(f"🔍 [Router] Proceeding to evidence evaluation (risk: {risk_level}, context: {overall_context_score:.1f}/10)")
    return "evidence_evaluator"

def route_after_evidence(state: FakeNewsState) -> str:
    """Smart routing after evidence evaluation based on evidence quality"""
    evidence_eval = state.get("evidence_evaluation", {})
    evidence_scores = evidence_eval.get("evidence_scores", {}) if evidence_eval else {}
    evidence_score = evidence_scores.get("overall_evidence_score", 5.0)
    
    # Skip source recommendations for very poor evidence
    if evidence_score < 2.0:
        print(f"⚠️ [Router] Skipping source recommendations (poor evidence: {evidence_score}/10)")
        return "llm_explanation"
    
    # Get source recommendations for decent evidence
    print(f"📊 [Router] Proceeding to source recommendations (evidence: {evidence_score}/10)")
    return "credible_source"

def create_smart_fake_news_workflow():
    """Create the optimized LangGraph workflow with conditional routing - ALL 6 AGENTS"""
    workflow = StateGraph(FakeNewsState)
    
    # Add all 6 agent nodes
    workflow.add_node("bert_classifier", bert_classifier_node)
    workflow.add_node("claim_extractor", claim_extractor_node)
    workflow.add_node("context_analyzer", context_analyzer_node)  # THE MISSING 6TH AGENT
    workflow.add_node("evidence_evaluator", evidence_evaluator_node)
    workflow.add_node("credible_source", credible_source_node)
    workflow.add_node("llm_explanation", llm_explanation_node)
    
    # Set entry point
    workflow.set_entry_point("bert_classifier")
    
    # Smart conditional routing
    workflow.add_conditional_edges(
        "bert_classifier",
        route_after_bert,
        {
            "llm_explanation": "llm_explanation",  # Fast track for obvious cases
            "claim_extractor": "claim_extractor"   # Normal processing
        }
    )
    
    workflow.add_conditional_edges(
        "claim_extractor",
        route_after_claims,
        {
            "context_analyzer": "context_analyzer"  # Always go to context analysis
        }
    )
    
    workflow.add_conditional_edges(
        "context_analyzer",
        route_after_context,
        {
            "credible_source": "credible_source",      # Skip evidence for low-risk
            "evidence_evaluator": "evidence_evaluator"  # Full analysis for high-risk
        }
    )
    
    workflow.add_conditional_edges(
        "evidence_evaluator",
        route_after_evidence,
        {
            "llm_explanation": "llm_explanation",    # Skip sources for poor evidence
            "credible_source": "credible_source"     # Get sources for good evidence
        }
    )
    
    # Final edges (always lead to explanation)
    workflow.add_edge("credible_source", "llm_explanation")
    workflow.add_edge("llm_explanation", END)
    
    return workflow.compile()

# Create the compiled workflow instance
fake_news_workflow = create_smart_fake_news_workflow()

# Convenience function for easy invocation
def analyze_article(text: str, url: str = "", detailed: bool = False) -> dict:
    """
    Analyze an article using the smart workflow with ALL 6 AGENTS
    
    Args:
        text: Article text to analyze
        url: Article URL (optional)
        detailed: Force detailed analysis regardless of routing
        
    Returns:
        Complete analysis results from the workflow
    """
    initial_state = FakeNewsState({
        "article_text": text,
        "article_url": url,
        "bert_results": {},
        "extracted_claims": [],
        "context_analysis": {},  # Added context analysis
        "evidence_evaluation": {},
        "source_recommendations": {},
        "final_explanation": {},
        "processing_errors": [],
        "processing_times": {},
        "confidence_scores": {},
        "total_api_cost": 0.0,
        "skip_expensive_processing": False,
        "require_detailed_analysis": detailed,
        "processing_path": "initializing"
    })
    
    try:
        result = fake_news_workflow.invoke(initial_state)
        return result
    except Exception as e:
        print(f"❌ Workflow execution failed: {str(e)}")
        # Return a safe error state
        initial_state["processing_errors"].append(f"Workflow error: {str(e)}")
        return initial_state
