"""
LangGraph State Schema for Fake News Detection Pipeline
"""

from typing import TypedDict, List, Optional, Dict, Any

class FakeNewsState(TypedDict):
    # Input data
    article_text: str
    article_url: Optional[str]
    
    # Agent results (matches your BaseAgent outputs)
    bert_results: Optional[Dict[str, Any]]
    extracted_claims: Optional[List[Dict[str, Any]]]
    context_analysis: Optional[Dict[str, Any]]  # Added missing context analysis
    evidence_evaluation: Optional[Dict[str, Any]]
    source_recommendations: Optional[Dict[str, Any]]
    final_explanation: Optional[Dict[str, Any]]
    
    # Processing metadata
    processing_errors: List[str]
    processing_times: Dict[str, float]
    confidence_scores: Dict[str, float]
    total_api_cost: float
    
    # Control flags
    skip_expensive_processing: bool
    require_detailed_analysis: bool
    processing_path: str  # Track which path was taken
