# orchestration/state.py

"""
Enhanced LangGraph State Schema for Fake News Detection Pipeline

Production-ready state management with comprehensive type safety,
validation, and enhanced metadata tracking.

Features:
- Comprehensive type definitions with runtime validation
- Enhanced metadata tracking for monitoring and debugging  
- State transition validation and consistency checking
- Performance tracking and execution metadata
- Error handling and recovery state management
- Session tracking and audit trail capabilities
- Configuration-aware state management

Version: 3.2.0 - Enhanced Production Edition
"""

from typing import TypedDict, List, Optional, Dict, Any, Union, Literal
from datetime import datetime
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingPath(Enum):
    """Enumeration of possible processing paths through the workflow."""
    INITIALIZING = "initializing"
    FAST_TRACK_REAL = "fast_track_real"
    FAST_TRACK_SHORT_REAL = "fast_track_short_real"
    FULL_ANALYSIS = "full_analysis"
    FULL_ANALYSIS_FORCED = "full_analysis_forced"
    ERROR_RECOVERY = "error_recovery"


class RiskLevel(Enum):
    """Risk level classifications."""
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PredictionType(Enum):
    """Article prediction classifications."""
    REAL = "REAL"
    FAKE = "FAKE"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ExecutionMetadata:
    """Comprehensive execution metadata for monitoring and debugging."""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    workflow_version: str = "3.2.0"
    
    # Processing tracking
    processing_path: ProcessingPath = ProcessingPath.INITIALIZING
    routing_reason: str = "initial_state"
    agents_executed: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_processing_time: float = 0.0
    agent_processing_times: Dict[str, float] = field(default_factory=dict)
    
    # Configuration context
    detailed_analysis_requested: bool = False
    fast_track_enabled: bool = True
    timeout_protection_enabled: bool = True
    
    # Quality indicators
    quality_score: float = 0.0
    confidence_level: str = "unknown"
    error_count: int = 0
    warning_count: int = 0
    
    def add_agent_execution(self, agent_name: str, processing_time: float):
        """Track agent execution."""
        self.agents_executed.append(agent_name)
        self.agent_processing_times[agent_name] = processing_time
        
    def update_processing_path(self, path: ProcessingPath, reason: str = ""):
        """Update processing path with reason."""
        self.processing_path = path
        self.routing_reason = reason
        
    def increment_error_count(self):
        """Increment error count."""
        self.error_count += 1
        
    def increment_warning_count(self):
        """Increment warning count."""
        self.warning_count += 1


@dataclass  
class BERTResults:
    """BERT classification results with enhanced metadata."""
    prediction: PredictionType = PredictionType.UNKNOWN
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=lambda: {"real": 0.5, "fake": 0.5})
    
    # Enhanced analysis metadata
    text_analysis: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "unknown"
    processing_time: float = 0.0
    mock_classification: bool = False
    
    # Quality indicators
    confidence_level: str = "unknown"  # low, medium, high
    reliability_score: float = 0.0
    
    def get_confidence_level(self) -> str:
        """Calculate confidence level based on score."""
        if self.confidence >= 0.8:
            return "high"
        elif self.confidence >= 0.6:
            return "medium"
        else:
            return "low"
            
    def validate(self) -> bool:
        """Validate BERT results structure."""
        return (
            isinstance(self.prediction, (PredictionType, str)) and
            0.0 <= self.confidence <= 1.0 and
            isinstance(self.probabilities, dict) and
            "real" in self.probabilities and
            "fake" in self.probabilities
        )


@dataclass
class ClaimData:
    """Individual claim data structure with comprehensive metadata."""
    text: str
    claim_type: str = "general"
    priority: int = 3  # 1=critical, 2=important, 3=minor
    verifiability_score: float = 0.0
    
    # Source tracking
    source_sentence: str = ""
    source_paragraph: int = 0
    
    # Analysis metadata
    confidence: float = 0.0
    extracted_by: str = "claim_extractor"
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Validation flags
    requires_verification: bool = True
    has_supporting_evidence: bool = False
    
    def is_high_priority(self) -> bool:
        """Check if claim is high priority."""
        return self.priority <= 2 and self.verifiability_score >= 6.0


@dataclass
class ContextScores:
    """Context analysis scores with comprehensive metrics."""
    overall_context_score: float = 5.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    bias_score: float = 0.0
    credibility_score: float = 50.0
    manipulation_score: float = 0.0
    
    # Enhanced LLM scores (if available)
    llm_bias: Optional[float] = None
    llm_credibility: Optional[float] = None  
    llm_risk: Optional[float] = None
    
    # Consistency indicators
    scoring_consistency: bool = True
    consistency_deviation: float = 0.0
    
    def get_risk_classification(self) -> str:
        """Get textual risk classification."""
        if self.overall_context_score >= 8.0:
            return "LOW"
        elif self.overall_context_score >= 6.0:
            return "MEDIUM"
        else:
            return "HIGH"
            
    def has_llm_scores(self) -> bool:
        """Check if LLM scores are available."""
        return all([
            self.llm_bias is not None,
            self.llm_credibility is not None,
            self.llm_risk is not None
        ])


@dataclass
class EvidenceScores:
    """Evidence evaluation scores with quality metrics."""
    overall_evidence_score: float = 5.0
    quality_level: str = "UNKNOWN"
    source_quality_score: float = 5.0
    verification_links_quality_score: float = 5.0
    
    # Verification metadata
    verification_links_count: int = 0
    institutional_sources_count: int = 0
    specific_sources_count: int = 0
    
    # Quality indicators
    evidence_completeness: float = 0.0
    source_diversity: float = 0.0
    
    def get_quality_classification(self) -> str:
        """Get evidence quality classification."""
        if self.overall_evidence_score >= 8.0:
            return "EXCELLENT"
        elif self.overall_evidence_score >= 6.0:
            return "GOOD"
        elif self.overall_evidence_score >= 4.0:
            return "ACCEPTABLE"
        else:
            return "POOR"


@dataclass
class VerificationLink:
    """Individual verification link with quality metrics."""
    url: str
    title: str = ""
    source_type: str = "general"  # institutional, academic, government, media
    quality_score: float = 0.0
    relevance_score: float = 0.0
    
    # Metadata
    description: str = ""
    accessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    verification_method: str = "automatic"
    
    # Quality indicators
    is_institutional: bool = False
    is_specific: bool = False
    confidence_level: str = "unknown"
    
    def is_high_quality(self) -> bool:
        """Check if verification link is high quality."""
        return self.quality_score >= 7.0 and self.relevance_score >= 6.0


@dataclass
class SourceRecommendation:
    """Individual source recommendation with contextual metadata."""
    name: str
    url: str = ""
    source_type: str = "general"
    reliability_score: float = 0.0
    
    # Contextual information
    description: str = ""
    relevance_to_claims: List[str] = field(default_factory=list)
    expert_type: str = ""
    
    # Quality metrics
    domain_expertise: float = 0.0
    institutional_backing: bool = False
    verification_record: float = 0.0
    
    # Metadata
    recommended_by: str = "credible_source_agent"
    recommendation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def is_expert_source(self) -> bool:
        """Check if source is an expert source."""
        return "expert" in self.source_type and self.domain_expertise >= 7.0


class FakeNewsState(TypedDict):
    """
    Enhanced state schema for the fake news detection pipeline.
    
    Provides comprehensive state management with type safety,
    validation, and enhanced metadata tracking for production use.
    """
    
    # Core input data
    article_text: str
    article_url: Optional[str]
    
    # Agent results with enhanced structure
    bert_results: Optional[Dict[str, Any]]
    extracted_claims: Optional[List[Dict[str, Any]]]  
    context_analysis: Optional[Dict[str, Any]]
    evidence_evaluation: Optional[Dict[str, Any]]
    source_recommendations: Optional[Dict[str, Any]]
    final_explanation: Optional[Dict[str, Any]]
    
    # Processing metadata and tracking
    processing_errors: List[str]
    processing_times: Dict[str, float]
    confidence_scores: Dict[str, float]
    
    # Enhanced execution tracking  
    execution_metadata: Optional[Dict[str, Any]]
    session_id: Optional[str]
    routing_decisions: Optional[List[Dict[str, Any]]]
    
    # Performance and cost tracking
    total_api_cost: float
    api_call_count: Optional[Dict[str, int]]
    
    # Control flags and configuration
    skip_expensive_processing: bool
    require_detailed_analysis: bool
    processing_path: str
    routing_reason: Optional[str]
    
    # Quality and validation metadata
    quality_validation: Optional[Dict[str, Any]]
    state_validation_errors: Optional[List[str]]
    
    # Timestamps and audit trail
    created_at: Optional[str]
    last_updated: Optional[str]
    completed_at: Optional[str]


# State validation functions
def validate_state_structure(state: FakeNewsState) -> List[str]:
    """
    Validate state structure and return list of validation errors.
    
    Args:
        state: State to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    try:
        # Validate required fields
        if not state.get("article_text"):
            errors.append("article_text is required and cannot be empty")
            
        if state.get("article_text") and len(state["article_text"]) < 50:
            errors.append("article_text too short (minimum 50 characters)")
            
        # Validate confidence scores
        confidence_scores = state.get("confidence_scores", {})
        for agent, score in confidence_scores.items():
            if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
                errors.append(f"Invalid confidence score for {agent}: {score}")
                
        # Validate processing times
        processing_times = state.get("processing_times", {})
        for agent, time_val in processing_times.items():
            if not isinstance(time_val, (int, float)) or time_val < 0:
                errors.append(f"Invalid processing time for {agent}: {time_val}")
                
        # Validate BERT results if present
        bert_results = state.get("bert_results")
        if bert_results:
            prediction = bert_results.get("prediction")
            confidence = bert_results.get("confidence", 0.0)
            
            if prediction not in ["REAL", "FAKE", "UNKNOWN"]:
                errors.append(f"Invalid BERT prediction: {prediction}")
                
            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                errors.append(f"Invalid BERT confidence: {confidence}")
                
        # Validate extracted claims if present
        extracted_claims = state.get("extracted_claims", [])
        if extracted_claims and not isinstance(extracted_claims, list):
            errors.append("extracted_claims must be a list")
            
    except Exception as e:
        errors.append(f"State validation error: {str(e)}")
        
    return errors


def create_initial_state(
    article_text: str,
    article_url: Optional[str] = None,
    detailed_analysis: bool = False,
    session_id: Optional[str] = None
) -> FakeNewsState:
    """
    Create initial state with proper defaults and validation.
    
    Args:
        article_text: Article text to analyze
        article_url: Optional article URL
        detailed_analysis: Whether to force detailed analysis
        session_id: Optional session ID for tracking
        
    Returns:
        Initialized FakeNewsState
        
    Raises:
        ValueError: If article_text is invalid
    """
    if not article_text or len(article_text.strip()) < 50:
        raise ValueError("Article text must be at least 50 characters long")
        
    current_time = datetime.now().isoformat()
    session_id = session_id or str(uuid.uuid4())[:8]
    
    return FakeNewsState({
        # Core input
        "article_text": article_text.strip(),
        "article_url": article_url or "",
        
        # Agent results (initialized as empty)
        "bert_results": {},
        "extracted_claims": [],
        "context_analysis": {},
        "evidence_evaluation": {},
        "source_recommendations": {},
        "final_explanation": {},
        
        # Processing metadata
        "processing_errors": [],
        "processing_times": {},
        "confidence_scores": {},
        
        # Execution tracking
        "execution_metadata": {
            "execution_id": str(uuid.uuid4())[:8],
            "session_id": session_id,
            "created_at": current_time,
            "workflow_version": "3.2.0"
        },
        "session_id": session_id,
        "routing_decisions": [],
        
        # Performance tracking
        "total_api_cost": 0.0,
        "api_call_count": {},
        
        # Control flags
        "skip_expensive_processing": False,
        "require_detailed_analysis": detailed_analysis,
        "processing_path": ProcessingPath.INITIALIZING.value,
        "routing_reason": "initial_state",
        
        # Quality tracking
        "quality_validation": {},
        "state_validation_errors": [],
        
        # Timestamps
        "created_at": current_time,
        "last_updated": current_time,
        "completed_at": None
    })


def update_state_metadata(state: FakeNewsState, agent_name: str, processing_time: float) -> FakeNewsState:
    """
    Update state metadata after agent processing.
    
    Args:
        state: Current state
        agent_name: Name of agent that processed
        processing_time: Time taken for processing
        
    Returns:
        Updated state with new metadata
    """
    new_state = dict(state)
    current_time = datetime.now().isoformat()
    
    # Update processing times
    new_state.setdefault("processing_times", {})[agent_name] = processing_time
    
    # Update execution metadata
    execution_metadata = new_state.setdefault("execution_metadata", {})
    execution_metadata.setdefault("agents_executed", []).append(agent_name)
    execution_metadata.setdefault("agent_processing_times", {})[agent_name] = processing_time
    
    # Update timestamps
    new_state["last_updated"] = current_time
    
    return FakeNewsState(new_state)


def add_routing_decision(
    state: FakeNewsState, 
    from_node: str, 
    to_node: str, 
    reason: str,
    metadata: Optional[Dict[str, Any]] = None
) -> FakeNewsState:
    """
    Add routing decision to state for audit trail.
    
    Args:
        state: Current state
        from_node: Source node
        to_node: Target node  
        reason: Routing reason
        metadata: Optional additional metadata
        
    Returns:
        Updated state with routing decision
    """
    new_state = dict(state)
    
    routing_decision = {
        "from_node": from_node,
        "to_node": to_node,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    new_state.setdefault("routing_decisions", []).append(routing_decision)
    new_state["routing_reason"] = reason
    
    return FakeNewsState(new_state)


def finalize_state(state: FakeNewsState) -> FakeNewsState:
    """
    Finalize state processing with completion metadata.
    
    Args:
        state: State to finalize
        
    Returns:
        Finalized state with completion metadata
    """
    new_state = dict(state)
    completion_time = datetime.now().isoformat()
    
    # Set completion timestamp
    new_state["completed_at"] = completion_time
    new_state["last_updated"] = completion_time
    
    # Calculate total processing time
    processing_times = new_state.get("processing_times", {})
    total_processing_time = sum(processing_times.values())
    
    # Update execution metadata
    execution_metadata = new_state.setdefault("execution_metadata", {})
    execution_metadata["total_processing_time"] = total_processing_time
    execution_metadata["completed_at"] = completion_time
    execution_metadata["agents_executed_count"] = len(processing_times)
    
    # Calculate overall confidence
    confidence_scores = new_state.get("confidence_scores", {})
    if confidence_scores:
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        execution_metadata["average_confidence"] = avg_confidence
        
    # Final validation
    validation_errors = validate_state_structure(new_state)
    new_state["state_validation_errors"] = validation_errors
    
    if not validation_errors:
        execution_metadata["final_validation"] = "passed"
    else:
        execution_metadata["final_validation"] = "failed"
        execution_metadata["validation_error_count"] = len(validation_errors)
        
    return FakeNewsState(new_state)


def get_state_summary(state: FakeNewsState) -> Dict[str, Any]:
    """
    Get comprehensive state summary for monitoring and debugging.
    
    Args:
        state: State to summarize
        
    Returns:
        State summary dictionary
    """
    try:
        bert_results = state.get("bert_results", {})
        extracted_claims = state.get("extracted_claims", [])
        confidence_scores = state.get("confidence_scores", {})
        processing_times = state.get("processing_times", {})
        
        return {
            "session_info": {
                "session_id": state.get("session_id", "unknown"),
                "processing_path": state.get("processing_path", "unknown"),
                "created_at": state.get("created_at"),
                "completed_at": state.get("completed_at")
            },
            "content_info": {
                "article_length": len(state.get("article_text", "")),
                "article_url": state.get("article_url", ""),
                "claims_count": len(extracted_claims),
                "has_url": bool(state.get("article_url"))
            },
            "results_summary": {
                "prediction": bert_results.get("prediction", "UNKNOWN"),
                "confidence": bert_results.get("confidence", 0.0),
                "high_priority_claims": sum(1 for claim in extracted_claims 
                                          if isinstance(claim, dict) and claim.get("priority", 3) <= 2),
                "average_confidence": sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
            },
            "performance_summary": {
                "total_processing_time": sum(processing_times.values()),
                "agents_processed": len(processing_times),
                "error_count": len(state.get("processing_errors", [])),
                "validation_passed": len(state.get("state_validation_errors", [])) == 0
            },
            "execution_metadata": state.get("execution_metadata", {}),
            "routing_decisions_count": len(state.get("routing_decisions", []))
        }
        
    except Exception as e:
        logger.error(f"Error generating state summary: {str(e)}")
        return {
            "error": str(e),
            "session_id": state.get("session_id", "unknown"),
            "state_keys": list(state.keys())
        }


# Export all public interfaces
__all__ = [
    'FakeNewsState',
    'ProcessingPath', 
    'RiskLevel',
    'PredictionType',
    'ExecutionMetadata',
    'BERTResults',
    'ClaimData',
    'ContextScores',
    'EvidenceScores', 
    'VerificationLink',
    'SourceRecommendation',
    'validate_state_structure',
    'create_initial_state',
    'update_state_metadata', 
    'add_routing_decision',
    'finalize_state',
    'get_state_summary'
]

# Log successful initialization
logger.info("🎯 Enhanced FakeNewsState schema loaded successfully")
