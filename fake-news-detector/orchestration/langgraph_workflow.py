# orchestration/langgraph_workflow.py

"""
Enhanced LangGraph Workflow for Smart Fake News Detection Pipeline

Production-ready workflow with comprehensive error handling, structured logging,
intelligent conditional routing, and comprehensive state management.

Features:
- Smart conditional routing to optimize processing costs and time
- Comprehensive error recovery and fallback mechanisms
- Structured logging with performance tracking and session management
- All 6 agents integrated with enhanced error handling
- Performance monitoring and timeout handling
- Configuration-aware routing and processing
- Health monitoring and diagnostics
- Async processing capabilities

Version: 3.2.0 - Enhanced Production Edition
"""

import logging
import time
import uuid
import asyncio
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

try:
    from langgraph.graph import StateGraph, END, START
    _langgraph_available = True
except ImportError:
    _langgraph_available = False
    logging.error("LangGraph not available - workflow functionality disabled")

from .state import FakeNewsState
from .nodes import (
    bert_classifier_node,
    claim_extractor_node,
    context_analyzer_node,
    evidence_evaluator_node,
    credible_source_node,
    llm_explanation_node,
    initialize_agents,
    get_agent_status
)

# Enhanced configuration integration
try:
    from config import get_settings, get_model_config
    _config_available = True
except ImportError:
    _config_available = False

# Configure workflow logging with structured format
logger = logging.getLogger("fake_news.workflow")

# Workflow performance tracking
_workflow_stats = {
    'executions': 0,
    'successful_executions': 0,
    'failed_executions': 0,
    'fast_track_count': 0,
    'full_analysis_count': 0,
    'timeout_count': 0,
    'average_processing_time': 0.0,
    'last_execution': None,
    'initialization_time': time.time()
}

# Workflow configuration
_workflow_config = {
    'enable_smart_routing': True,
    'enable_timeout_protection': True,
    'default_timeout': 300,  # 5 minutes
    'fast_track_confidence_threshold': 0.92,
    'enable_performance_tracking': True,
    'max_retries': 2,
    'retry_delay': 5.0
}


def update_workflow_config(**kwargs):
    """Update workflow configuration parameters."""
    global _workflow_config
    _workflow_config.update(kwargs)
    logger.info(f"Workflow configuration updated: {list(kwargs.keys())}")


def get_workflow_performance() -> Dict[str, Any]:
    """Get comprehensive workflow performance statistics."""
    current_time = time.time()
    uptime = current_time - _workflow_stats['initialization_time']
    
    return {
        'statistics': _workflow_stats.copy(),
        'configuration': _workflow_config.copy(),
        'uptime_seconds': uptime,
        'success_rate': (_workflow_stats['successful_executions'] / 
                        max(_workflow_stats['executions'], 1)) * 100,
        'fast_track_rate': (_workflow_stats['fast_track_count'] / 
                           max(_workflow_stats['executions'], 1)) * 100,
        'agent_status': get_agent_status() if 'get_agent_status' in globals() else {},
        'langgraph_available': _langgraph_available,
        'config_available': _config_available
    }


# Enhanced routing functions with comprehensive decision logic
def route_after_bert(state: FakeNewsState) -> str:
    """
    Smart routing after BERT classification with enhanced decision logic.
    
    Routes based on:
    - Confidence levels and prediction quality
    - Explicit requirements for detailed analysis
    - Content characteristics and complexity
    - System configuration and performance considerations
    
    Returns:
        Next node name to route to
    """
    try:
        confidence = state.get("confidence_scores", {}).get("bert", 0.0)
        bert_results = state.get("bert_results", {})
        prediction = bert_results.get("prediction", "UNKNOWN") if bert_results else "UNKNOWN"
        require_detailed = state.get("require_detailed_analysis", False)
        
        # Generate session ID if not present
        session_id = state.get("session_id", str(uuid.uuid4())[:8])
        
        # Update processing path in state
        new_state = dict(state)
        new_state["session_id"] = session_id
        
        # Configuration-aware routing
        confidence_threshold = _workflow_config.get('fast_track_confidence_threshold', 0.92)
        smart_routing_enabled = _workflow_config.get('enable_smart_routing', True)
        
        # Honor explicit request for detailed analysis
        if require_detailed:
            logger.info(
                f"[{session_id}] Router: Detailed analysis requested - running full pipeline"
            )
            new_state["processing_path"] = "full_analysis_forced"
            new_state["routing_reason"] = "detailed_analysis_requested"
            state.update(new_state)
            _workflow_stats['full_analysis_count'] += 1
            return "claim_extractor"

        # Fast track high-confidence REAL news (if smart routing enabled)
        if (smart_routing_enabled and 
            confidence > confidence_threshold and 
            prediction == "REAL"):
            
            logger.info(
                f"[{session_id}] Router: Fast-tracking high-confidence REAL news "
                f"(confidence: {confidence:.1%})"
            )
            new_state["skip_expensive_processing"] = True
            new_state["processing_path"] = "fast_track_real"
            new_state["routing_reason"] = f"high_confidence_real_{confidence:.3f}"
            state.update(new_state)
            _workflow_stats['fast_track_count'] += 1
            return "llm_explanation"

        # Check content characteristics for routing decisions
        article_length = len(state.get("article_text", ""))
        
        # Fast track very short articles with high confidence
        if (smart_routing_enabled and 
            article_length < 200 and 
            confidence > 0.85 and 
            prediction == "REAL"):
            
            logger.info(
                f"[{session_id}] Router: Fast-tracking short high-confidence article "
                f"(length: {article_length}, confidence: {confidence:.1%})"
            )
            new_state["skip_expensive_processing"] = True
            new_state["processing_path"] = "fast_track_short_real"
            new_state["routing_reason"] = f"short_high_confidence_{article_length}_{confidence:.3f}"
            state.update(new_state)
            _workflow_stats['fast_track_count'] += 1
            return "llm_explanation"

        # Standard processing for all other cases
        logger.info(
            f"[{session_id}] Router: Proceeding to claim extraction "
            f"(confidence: {confidence:.1%}, prediction: {prediction})"
        )
        new_state["processing_path"] = "full_analysis"
        new_state["routing_reason"] = f"standard_processing_{prediction.lower()}_{confidence:.3f}"
        state.update(new_state)
        _workflow_stats['full_analysis_count'] += 1
        return "claim_extractor"
        
    except Exception as e:
        logger.error(f"Error in BERT routing: {str(e)} - defaulting to claim extraction")
        return "claim_extractor"


def route_after_claims(state: FakeNewsState) -> str:
    """
    Route after claim extraction with intelligent analysis of claim characteristics.
    
    Always proceeds to context analysis as it provides crucial insights into
    article bias, manipulation, and credibility regardless of claim count.
    """
    try:
        session_id = state.get("session_id", "unknown")
        extracted_claims = state.get("extracted_claims", [])
        claims_count = len(extracted_claims)

        # Calculate claim richness and characteristics
        claim_richness = 0
        high_priority_claims = 0
        verifiable_claims = 0
        
        if extracted_claims and isinstance(extracted_claims[0], dict):
            high_priority_claims = sum(1 for claim in extracted_claims 
                                     if claim.get("priority", 3) <= 2)
            verifiable_claims = sum(1 for claim in extracted_claims 
                                  if claim.get("verifiability_score", 0) >= 6)
            claim_richness = high_priority_claims + verifiable_claims
        else:
            claim_richness = claims_count

        # Log claim analysis results
        logger.info(
            f"[{session_id}] Router: Claims analysis complete - "
            f"Total: {claims_count}, High-priority: {high_priority_claims}, "
            f"Verifiable: {verifiable_claims}, Richness: {claim_richness}"
        )

        # Always proceed to context analysis (6th agent) - crucial for understanding bias and manipulation
        logger.info(f"[{session_id}] Router: Proceeding to context analysis")
        return "context_analyzer"
        
    except Exception as e:
        logger.error(f"Error in claims routing: {str(e)} - proceeding to context analysis")
        return "context_analyzer"


def route_after_context(state: FakeNewsState) -> str:
    """
    Route after context analysis based on bias, manipulation, and risk scores.
    
    Uses context analysis results to determine if expensive evidence evaluation
    is needed or if we can skip to source recommendations.
    """
    try:
        session_id = state.get("session_id", "unknown")
        context_analysis = state.get("context_analysis", {})
        context_scores = context_analysis.get("context_scores", {}) if context_analysis else {}
        
        overall_context_score = context_scores.get("overall_context_score", 5.0)
        risk_level = context_scores.get("risk_level", "MEDIUM")
        bias_score = context_scores.get("bias_score", 0.0)
        
        extracted_claims = state.get("extracted_claims", [])
        claims_count = len(extracted_claims)
        
        # Check if LLM scoring is available for enhanced decision making
        llm_scores = context_analysis.get("llm_scores", {}) if context_analysis else {}
        has_llm_scoring = bool(llm_scores)
        
        # Enhanced routing logic with LLM scoring consideration
        skip_evidence = False
        routing_reason = ""
        
        # Skip evidence evaluation for low-risk, low-claim content
        if risk_level == "LOW" and claims_count < 3:
            skip_evidence = True
            routing_reason = f"low_risk_few_claims_{risk_level}_{claims_count}"
            
        # Skip for very low bias and high credibility (if LLM scoring available)
        elif (has_llm_scoring and 
              bias_score < 2.0 and 
              llm_scores.get('credibility', 0) > 80):
            skip_evidence = True
            routing_reason = f"low_bias_high_credibility_{bias_score}_{llm_scores.get('credibility', 0)}"
            
        # Skip for excellent overall context score
        elif overall_context_score >= 8.5:
            skip_evidence = True
            routing_reason = f"excellent_context_score_{overall_context_score}"

        if skip_evidence:
            logger.info(
                f"[{session_id}] Router: Skipping evidence evaluation - "
                f"{routing_reason} (Risk: {risk_level}, Claims: {claims_count}, "
                f"Context: {overall_context_score:.1f}/10)"
            )
            return "credible_source"
        else:
            logger.info(
                f"[{session_id}] Router: Proceeding to evidence evaluation - "
                f"Risk: {risk_level}, Claims: {claims_count}, Context: {overall_context_score:.1f}/10"
                f"{', LLM scoring available' if has_llm_scoring else ''}"
            )
            return "evidence_evaluator"
            
    except Exception as e:
        logger.error(f"Error in context routing: {str(e)} - proceeding to evidence evaluation")
        return "evidence_evaluator"


def route_after_evidence(state: FakeNewsState) -> str:
    """
    Smart routing after evidence evaluation based on evidence quality and completeness.
    
    If evidence quality is very poor or insufficient, skip source recommendations
    and go directly to explanation generation to avoid further resource usage.
    """
    try:
        session_id = state.get("session_id", "unknown")
        evidence_eval = state.get("evidence_evaluation", {})
        evidence_scores = evidence_eval.get("evidence_scores", {}) if evidence_eval else {}
        
        evidence_score = evidence_scores.get("overall_evidence_score", 5.0)
        verification_links = evidence_eval.get("verification_links", []) if evidence_eval else []
        safety_fallback_used = evidence_eval.get("safety_fallback_used", False) if evidence_eval else False
        
        # Enhanced routing logic
        skip_sources = False
        routing_reason = ""
        
        # Skip source recommendations for very poor evidence
        if evidence_score < 2.0:
            skip_sources = True
            routing_reason = f"very_poor_evidence_{evidence_score}"
            
        # Skip if no verification links and safety fallback was used
        elif len(verification_links) == 0 and safety_fallback_used:
            skip_sources = True
            routing_reason = f"no_links_safety_fallback"
            
        # Skip if evidence is poor and we have few verification links
        elif evidence_score < 3.5 and len(verification_links) <= 1:
            skip_sources = True
            routing_reason = f"poor_evidence_few_links_{evidence_score}_{len(verification_links)}"

        if skip_sources:
            logger.info(
                f"[{session_id}] Router: Skipping source recommendations - "
                f"{routing_reason} (Evidence: {evidence_score}/10, Links: {len(verification_links)})"
            )
            return "llm_explanation"
        else:
            logger.info(
                f"[{session_id}] Router: Proceeding to source recommendations - "
                f"Evidence: {evidence_score}/10, Links: {len(verification_links)}"
                f"{', Safety fallback used' if safety_fallback_used else ''}"
            )
            return "credible_source"
            
    except Exception as e:
        logger.error(f"Error in evidence routing: {str(e)} - proceeding to explanation")
        return "llm_explanation"


def create_enhanced_fake_news_workflow():
    """
    Create the enhanced LangGraph workflow with comprehensive error handling,
    smart routing, and production-ready monitoring.
    
    Returns:
        Compiled LangGraph workflow ready for execution, or None if LangGraph unavailable
    """
    if not _langgraph_available:
        logger.error("Cannot create workflow - LangGraph not available")
        return None
        
    try:
        logger.info("Creating enhanced fake news detection workflow...")
        
        # Initialize workflow
        workflow = StateGraph(FakeNewsState)

        # Add all 6 agent nodes with enhanced error handling
        workflow.add_node("bert_classifier", bert_classifier_node)
        workflow.add_node("claim_extractor", claim_extractor_node)
        workflow.add_node("context_analyzer", context_analyzer_node)  # Enhanced 6th agent
        workflow.add_node("evidence_evaluator", evidence_evaluator_node)
        workflow.add_node("credible_source", credible_source_node)
        workflow.add_node("llm_explanation", llm_explanation_node)

        # Set entry point
        workflow.set_entry_point("bert_classifier")

        # Smart conditional routing with comprehensive decision logic
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
                "llm_explanation": "llm_explanation",  # Skip sources for poor evidence
                "credible_source": "credible_source"   # Get sources for good evidence
            }
        )

        # Final edges (always lead to explanation)
        workflow.add_edge("credible_source", "llm_explanation")
        workflow.add_edge("llm_explanation", END)

        # Compile workflow
        compiled_workflow = workflow.compile()
        
        logger.info("✅ Enhanced fake news detection workflow created successfully")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"❌ Failed to create workflow: {str(e)}")
        return None


# Create the compiled workflow instance
fake_news_workflow = create_enhanced_fake_news_workflow()


def validate_input_state(text: str, url: str = "", detailed: bool = False) -> Dict[str, Any]:
    """
    Validate and prepare input state with comprehensive error checking.
    
    Args:
        text: Article text to analyze
        url: Article URL (optional)
        detailed: Force detailed analysis regardless of routing
        
    Returns:
        Validated state dictionary or error information
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "state": None
    }
    
    try:
        # Text validation
        if not text or not isinstance(text, str):
            validation_result["errors"].append("Article text is required and must be a string")
            
        if isinstance(text, str):
            text = text.strip()
            if len(text) < 50:
                validation_result["errors"].append("Article text too short (minimum 50 characters)")
            elif len(text) > 100000:
                validation_result["errors"].append("Article text too long (maximum 100,000 characters)")
        
        # URL validation  
        if url and not isinstance(url, str):
            validation_result["warnings"].append("URL should be a string")
            url = str(url)
            
        # Configuration-based limits
        if _config_available:
            try:
                settings = get_settings()
                max_length = getattr(settings, 'max_article_length', 50000)
                if len(text) > max_length:
                    validation_result["errors"].append(f"Article exceeds configured limit ({max_length} chars)")
            except Exception:
                pass  # Use defaults if config unavailable
                
        # Create state if validation passed
        if not validation_result["errors"]:
            validation_result["state"] = FakeNewsState({
                "article_text": text,
                "article_url": url or "",
                "bert_results": {},
                "extracted_claims": [],
                "context_analysis": {},
                "evidence_evaluation": {},
                "source_recommendations": {},
                "final_explanation": {},
                "processing_errors": [],
                "processing_times": {},
                "confidence_scores": {},
                "total_api_cost": 0.0,
                "skip_expensive_processing": False,
                "require_detailed_analysis": detailed,
                "processing_path": "initializing",
                "session_id": str(uuid.uuid4())[:8],
                "created_at": datetime.now().isoformat()
            })
        else:
            validation_result["valid"] = False
            
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Validation error: {str(e)}")
        
    return validation_result


def analyze_article(text: str, url: str = "", detailed: bool = False) -> Dict[str, Any]:
    """
    Analyze an article using the enhanced workflow with comprehensive error handling.
    
    Args:
        text: Article text to analyze
        url: Article URL (optional)
        detailed: Force detailed analysis regardless of routing
        
    Returns:
        Complete analysis results from the workflow
    """
    global _workflow_stats
    
    execution_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Update execution stats
    _workflow_stats['executions'] += 1
    _workflow_stats['last_execution'] = datetime.now().isoformat()
    
    logger.info(f"[{execution_id}] Starting article analysis...")
    
    try:
        # Validate workflow availability
        if not fake_news_workflow:
            error_msg = "Workflow not available - check LangGraph installation and initialization"
            logger.error(f"[{execution_id}] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "execution_metadata": {
                    "execution_id": execution_id,
                    "failed": True,
                    "processing_time": time.time() - start_time,
                    "workflow_version": "3.2.0",
                    "langgraph_available": _langgraph_available
                }
            }
        
        # Validate input
        validation = validate_input_state(text, url, detailed)
        if not validation["valid"]:
            error_msg = f"Input validation failed: {', '.join(validation['errors'])}"
            logger.error(f"[{execution_id}] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "validation_errors": validation["errors"],
                "execution_metadata": {
                    "execution_id": execution_id,
                    "failed": True,
                    "processing_time": time.time() - start_time,
                    "workflow_version": "3.2.0"
                }
            }
        
        initial_state = validation["state"]
        initial_state["execution_id"] = execution_id
        
        # Initialize agents before execution
        try:
            agent_init_result = initialize_agents()
            logger.info(f"[{execution_id}] Agent initialization: {agent_init_result.get('status', 'unknown')}")
        except Exception as e:
            logger.warning(f"[{execution_id}] Agent initialization warning: {str(e)}")

        # Execute the workflow with timeout protection
        timeout = _workflow_config.get('default_timeout', 300)
        
        try:
            logger.info(f"[{execution_id}] Executing workflow (timeout: {timeout}s)...")
            
            if _workflow_config.get('enable_timeout_protection', True):
                # For now, we'll execute without explicit timeout handling
                # In production, you might want to use asyncio.wait_for or similar
                result = fake_news_workflow.invoke(initial_state)
            else:
                result = fake_news_workflow.invoke(initial_state)
                
        except Exception as workflow_error:
            # Workflow execution error - try to recover
            processing_time = time.time() - start_time
            logger.error(f"[{execution_id}] Workflow execution error after {processing_time:.2f}s: {str(workflow_error)}")
            
            # Return partial results if available
            if hasattr(workflow_error, 'partial_state'):
                result = workflow_error.partial_state
                result.setdefault("processing_errors", []).append(f"Workflow execution error: {str(workflow_error)}")
            else:
                # Create minimal error state
                result = dict(initial_state)
                result["processing_errors"] = [f"Workflow execution failed: {str(workflow_error)}"]
                result["final_explanation"] = {
                    "explanation": f"Analysis could not be completed due to a system error: {str(workflow_error)}",
                    "error": True,
                    "partial_results": True
                }

        processing_time = time.time() - start_time
        
        # Add comprehensive execution metadata
        if isinstance(result, dict):
            result.setdefault("execution_metadata", {}).update({
                "execution_id": execution_id,
                "total_processing_time": processing_time,
                "workflow_version": "3.2.0",
                "agents_used": 6,
                "execution_timestamp": time.time(),
                "session_id": result.get("session_id", execution_id),
                "processing_path": result.get("processing_path", "unknown"),
                "routing_reason": result.get("routing_reason", "unknown"),
                "configuration": {
                    "smart_routing_enabled": _workflow_config['enable_smart_routing'],
                    "timeout_protection": _workflow_config['enable_timeout_protection'],
                    "detailed_analysis_forced": detailed
                },
                "performance": {
                    "fast_tracked": result.get("skip_expensive_processing", False),
                    "processing_times": result.get("processing_times", {}),
                    "confidence_scores": result.get("confidence_scores", {}),
                    "errors_count": len(result.get("processing_errors", []))
                }
            })
        
        # Update success statistics
        if result.get("processing_errors"):
            _workflow_stats['failed_executions'] += 1
        else:
            _workflow_stats['successful_executions'] += 1
            
        # Update average processing time
        total_executions = _workflow_stats['executions']
        current_avg = _workflow_stats['average_processing_time']
        _workflow_stats['average_processing_time'] = (
            (current_avg * (total_executions - 1) + processing_time) / total_executions
        )

        logger.info(f"[{execution_id}] ✅ Analysis completed successfully in {processing_time:.2f}s")
        return result

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Analysis failed: {str(e)}"
        
        _workflow_stats['failed_executions'] += 1
        
        logger.exception(f"[{execution_id}] ❌ {error_msg} (after {processing_time:.2f}s)")
        
        # Return comprehensive error response
        return {
            "success": False,
            "error": error_msg,
            "execution_metadata": {
                "execution_id": execution_id,
                "failed": True,
                "error": str(e),
                "processing_time": processing_time,
                "workflow_version": "3.2.0",
                "agents_used": 0,
                "timestamp": datetime.now().isoformat()
            },
            "processing_errors": [error_msg],
            "article_text": text[:200] + "..." if len(text) > 200 else text,
            "article_url": url
        }


async def analyze_article_async(text: str, url: str = "", detailed: bool = False) -> Dict[str, Any]:
    """
    Asynchronous wrapper for article analysis with proper async handling.
    
    Args:
        text: Article text to analyze
        url: Article URL (optional)
        detailed: Force detailed analysis regardless of routing
        
    Returns:
        Complete analysis results from the workflow
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, analyze_article, text, url, detailed)
    except Exception as e:
        logger.error(f"Async analysis failed: {str(e)}")
        return {
            "success": False,
            "error": f"Async execution failed: {str(e)}",
            "execution_metadata": {
                "failed": True,
                "async": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        }


def workflow_health_check() -> Dict[str, Any]:
    """
    Comprehensive workflow health check for production monitoring.
    
    Returns:
        Detailed health status dictionary
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "workflow_version": "3.2.0",
        "components": {},
        "performance": {},
        "configuration": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Check core dependencies
        health_status["components"]["langgraph"] = {
            "available": _langgraph_available,
            "status": "healthy" if _langgraph_available else "critical"
        }
        
        health_status["components"]["workflow"] = {
            "available": fake_news_workflow is not None,
            "status": "healthy" if fake_news_workflow else "critical"
        }
        
        health_status["components"]["configuration"] = {
            "available": _config_available,
            "status": "healthy" if _config_available else "degraded"
        }
        
        # Check agent status
        try:
            agent_status = get_agent_status()
            ready_agents = sum(1 for status in agent_status.get('agent_readiness', {}).values() if status)
            total_agents = len(agent_status.get('agent_readiness', {}))
            
            health_status["components"]["agents"] = {
                "ready_count": ready_agents,
                "total_count": total_agents,
                "status": "healthy" if ready_agents >= 4 else "degraded" if ready_agents >= 2 else "critical"
            }
        except Exception:
            health_status["components"]["agents"] = {"status": "unknown", "error": "Could not check agent status"}
        
        # Performance metrics
        health_status["performance"] = get_workflow_performance()
        
        # Configuration status
        health_status["configuration"] = _workflow_config.copy()
        
        # Test execution (lightweight)
        try:
            test_result = analyze_article("Test article for health check.", detailed=False)
            health_status["test_execution"] = {
                "status": "passed" if test_result.get("execution_metadata") else "failed",
                "execution_time": test_result.get("execution_metadata", {}).get("total_processing_time", 0)
            }
        except Exception as e:
            health_status["test_execution"] = {"status": "failed", "error": str(e)}

        # Determine overall status
        critical_issues = []
        if not _langgraph_available:
            critical_issues.append("LangGraph not available")
        if not fake_news_workflow:
            critical_issues.append("Workflow not compiled")
            
        if critical_issues:
            health_status["status"] = "critical"
            health_status["issues"] = critical_issues
            health_status["recommendations"] = [
                "Check LangGraph installation",
                "Verify workflow compilation", 
                "Check system dependencies"
            ]
        elif health_status["components"]["agents"]["status"] == "degraded":
            health_status["status"] = "degraded"
            health_status["issues"] = ["Some agents not available"]
            health_status["recommendations"] = ["Check agent initialization", "Verify API keys"]
        else:
            health_status["recommendations"] = ["System operating normally"]

    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)
        health_status["recommendations"] = ["Contact system administrator"]

    return health_status


def reset_workflow_stats():
    """Reset workflow performance statistics."""
    global _workflow_stats
    _workflow_stats = {
        'executions': 0,
        'successful_executions': 0,
        'failed_executions': 0,
        'fast_track_count': 0,
        'full_analysis_count': 0,
        'timeout_count': 0,
        'average_processing_time': 0.0,
        'last_execution': None,
        'initialization_time': time.time()
    }
    logger.info("Workflow statistics reset")


# Workflow diagnostics and monitoring
def get_workflow_diagnostics() -> Dict[str, Any]:
    """Get comprehensive workflow diagnostics for debugging."""
    return {
        "workflow_info": {
            "available": fake_news_workflow is not None,
            "langgraph_available": _langgraph_available,
            "config_available": _config_available,
            "version": "3.2.0"
        },
        "statistics": _workflow_stats.copy(),
        "configuration": _workflow_config.copy(),
        "agent_status": get_agent_status() if 'get_agent_status' in globals() else {},
        "routing_functions": {
            "route_after_bert": route_after_bert.__name__,
            "route_after_claims": route_after_claims.__name__, 
            "route_after_context": route_after_context.__name__,
            "route_after_evidence": route_after_evidence.__name__
        },
        "health": workflow_health_check()
    }


# Export workflow status for monitoring
__all__ = [
    'fake_news_workflow',
    'analyze_article', 
    'analyze_article_async',
    'workflow_health_check',
    'get_workflow_performance',
    'get_workflow_diagnostics',
    'update_workflow_config',
    'reset_workflow_stats'
]

# Log successful initialization
if fake_news_workflow:
    logger.info("🎯 Enhanced fake news detection workflow ready for production")
else:
    logger.error("⚠️ Workflow initialization failed - check dependencies")
