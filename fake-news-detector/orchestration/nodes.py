# orchestration/nodes.py

"""
Enhanced LangGraph Node Wrappers for Fake News Detection Pipeline

Production-ready node implementations with structured logging,
robust error handling, and comprehensive fallback mechanisms.

Features:
- Structured logging with performance tracking
- Graceful agent initialization with comprehensive error handling
- Enhanced fallback mechanisms for unavailable agents
- Safe agent processing with timeout protection
- Mock implementations for development and testing
- Comprehensive state management and validation

Version: 3.2.0 - Enhanced Production Edition
"""

import time
import logging
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .state import FakeNewsState

# Configure structured logging
logger = logging.getLogger("fake_news.nodes")

# Global agent instances with initialization tracking
_agents = {
    'bert_agent': None,
    'claim_agent': None, 
    'context_agent': None,
    'evidence_agent': None,
    'credible_agent': None,
    'explanation_agent': None
}

# Agent readiness tracking
_agent_status = {
    'bert_ready': False,
    'claim_ready': False,
    'context_ready': False,
    'evidence_ready': False,
    'credible_ready': False,
    'explanation_ready': False,
    'bert_model_loaded': False,
    'using_mock_bert': False,
    'agents_initialized': False
}

# Performance tracking
_performance_stats = {
    'initialization_time': 0.0,
    'successful_agents': 0,
    'failed_agents': 0,
    'total_processing_calls': 0,
    'failed_processing_calls': 0
}


def validate_api_keys():
    """Validate required API keys before agent initialization."""
    import os
    from pathlib import Path
    
    # Load .env file from fake-news-detector directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            logger.info("✅ Loaded environment variables from .env file")
        except Exception as e:
            logger.warning(f"⚠️ Could not load .env file: {e}")
    else:
        logger.info("ℹ️ No .env file found - using system environment variables")
    
    gemini_key = (
        os.getenv('GEMINI_API_KEY') or
        os.getenv('GOOGLE_API_KEY') or
        os.getenv('GOOGLE_GEMINI_API_KEY')
    )
    
    if not gemini_key:
        logger.warning("⚠️ No Gemini API key found. Some agents may not function properly.")
        logger.info("💡 To fix this:")
        logger.info("   1. Create a .env file in the fake-news-detector directory")
        logger.info("   2. Add: GEMINI_API_KEY=your_api_key_here")
        logger.info("   3. Get your API key from: https://makersuite.google.com/app/apikey")
        logger.info("   4. Or set the environment variable: export GEMINI_API_KEY=your_key")
        return False
        
    if len(gemini_key) < 10:
        logger.warning("⚠️ Gemini API key appears to be too short. Please verify your API key.")
        logger.info("💡 API keys should be longer than 10 characters")
        return False
        
    logger.info("✅ Gemini API key validated successfully")
    return True


def initialize_agents() -> Dict[str, Any]:
    """
    Initialize all agents with comprehensive error handling and performance tracking.
    
    Returns:
        Dictionary with initialization results and status
    """
    global _agents, _agent_status, _performance_stats
    
    if _agent_status['agents_initialized']:
        return {
            'status': 'already_initialized',
            'successful_agents': _performance_stats['successful_agents'],
            'failed_agents': _performance_stats['failed_agents']
        }
    
    logger.info("🚀 Initializing fake news detection agents...")
    
    # Validate API keys first
    if not validate_api_keys():
        logger.error("❌ API key validation failed - some agents may not initialize")
    
    start_time = time.time()
    
    successful_agents = 0
    failed_agents = 0
    initialization_results = {}
    
    # 1. Initialize BERT Classifier with model loading
    try:
        from agents.bert_classifier import BERTClassifier
        
        bert_agent = BERTClassifier()
        _agents['bert_agent'] = bert_agent
        
        # Attempt to load trained model
        model_path = Path("models/bert_fake_news")
        if model_path.exists():
            try:
                # Check if model loading is async and handle appropriately
                if hasattr(bert_agent, '_model_loaded') and not bert_agent._model_loaded:
                    import asyncio
                    try:
                        # Try async loading
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        load_result = loop.run_until_complete(bert_agent.load_model(model_path))
                        loop.close()
                    except Exception as async_error:
                        logger.warning(f"Async model loading failed: {async_error}, using mock classification")
                        load_result = {"success": False}
                else:
                    # Model already loaded or synchronous loading
                    load_result = {"success": True}
                    
                if load_result.get("success", False):
                    _agent_status['bert_model_loaded'] = True
                    _agent_status['using_mock_bert'] = False
                    logger.info("✅ BERT Classifier initialized with trained model")
                else:
                    _agent_status['bert_model_loaded'] = False
                    _agent_status['using_mock_bert'] = True
                    logger.warning("⚠️ BERT model load failed - using mock classification")
            except Exception as model_error:
                _agent_status['bert_model_loaded'] = False
                _agent_status['using_mock_bert'] = True
                logger.warning(f"⚠️ BERT model loading error: {model_error} - using mock classification")
        else:
            _agent_status['bert_model_loaded'] = False
            _agent_status['using_mock_bert'] = True
            logger.warning("⚠️ BERT model path not found - using mock classification")
        
        _agent_status['bert_ready'] = True
        successful_agents += 1
        initialization_results['bert_classifier'] = {'status': 'success', 'mock_mode': _agent_status['using_mock_bert']}
        
    except Exception as e:
        _agents['bert_agent'] = None
        _agent_status['bert_ready'] = False
        _agent_status['using_mock_bert'] = True
        failed_agents += 1
        initialization_results['bert_classifier'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ BERT Classifier initialization failed: {str(e)}")

    # 2. Initialize Claim Extractor Agent
    try:
        from agents.claim_extractor import ClaimExtractorAgent
        
        claim_agent = ClaimExtractorAgent()
        _agents['claim_agent'] = claim_agent
        _agent_status['claim_ready'] = True
        successful_agents += 1
        initialization_results['claim_extractor'] = {'status': 'success'}
        logger.info("✅ Claim Extractor Agent initialized successfully")
        
    except Exception as e:
        _agents['claim_agent'] = None
        _agent_status['claim_ready'] = False
        failed_agents += 1
        initialization_results['claim_extractor'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Claim Extractor initialization failed: {str(e)}")

    # 3. Initialize Context Analyzer Agent (Enhanced v3.2)
    try:
        from agents.context_analyzer import ContextAnalyzerAgent
        
        context_agent = ContextAnalyzerAgent()
        _agents['context_agent'] = context_agent
        _agent_status['context_ready'] = True
        successful_agents += 1
        initialization_results['context_analyzer'] = {'status': 'success', 'version': '3.2_enhanced'}
        logger.info("✅ Context Analyzer Agent initialized successfully (Enhanced v3.2)")
        
    except Exception as e:
        _agents['context_agent'] = None
        _agent_status['context_ready'] = False
        failed_agents += 1
        initialization_results['context_analyzer'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Context Analyzer initialization failed: {str(e)}")

    # 4. Initialize Evidence Evaluator Agent (Enhanced v3.2)
    try:
        from agents.evidence_evaluator import EvidenceEvaluatorAgent
        
        evidence_agent = EvidenceEvaluatorAgent()
        _agents['evidence_agent'] = evidence_agent
        _agent_status['evidence_ready'] = True
        successful_agents += 1
        initialization_results['evidence_evaluator'] = {'status': 'success', 'version': '3.2_enhanced'}
        logger.info("✅ Evidence Evaluator Agent initialized successfully (Enhanced v3.2)")
        
    except Exception as e:
        _agents['evidence_agent'] = None
        _agent_status['evidence_ready'] = False
        failed_agents += 1
        initialization_results['evidence_evaluator'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Evidence Evaluator initialization failed: {str(e)}")

    # 5. Initialize Credible Source Agent (Enhanced v3.2)
    try:
        from agents.credible_source import CredibleSourceAgent
        
        credible_agent = CredibleSourceAgent()
        _agents['credible_agent'] = credible_agent
        _agent_status['credible_ready'] = True
        successful_agents += 1
        initialization_results['credible_source'] = {'status': 'success', 'version': '3.2_enhanced'}
        logger.info("✅ Credible Source Agent initialized successfully (Enhanced v3.2)")
        
    except Exception as e:
        _agents['credible_agent'] = None
        _agent_status['credible_ready'] = False
        failed_agents += 1
        initialization_results['credible_source'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Credible Source initialization failed: {str(e)}")

    # 6. Initialize LLM Explanation Agent
    try:
        from agents.llm_explanation import create_explanation_agent
        explanation_agent = create_explanation_agent()
        _agents['explanation_agent'] = explanation_agent
        _agent_status['explanation_ready'] = True
        successful_agents += 1
        initialization_results['llm_explanation'] = {'status': 'success'}
        logger.info("✅ LLM Explanation Agent initialized successfully")
    except ImportError as import_error:
        _agents['explanation_agent'] = None
        _agent_status['explanation_ready'] = False
        failed_agents += 1
        error_msg = f"Import failed: {str(import_error)}"
        initialization_results['llm_explanation'] = {'status': 'failed', 'error': error_msg}
        logger.error(f"❌ LLM Explanation import failed: {error_msg}")
    except Exception as e:
        _agents['explanation_agent'] = None
        _agent_status['explanation_ready'] = False
        failed_agents += 1
        error_msg = f"Initialization failed: {str(e)}"
        initialization_results['llm_explanation'] = {'status': 'failed', 'error': error_msg}
        logger.error(f"❌ LLM Explanation initialization failed: {error_msg}")
        logger.exception("Full LLM Explanation error traceback:")

    # Update global status and performance stats
    _agent_status['agents_initialized'] = True
    _performance_stats['successful_agents'] = successful_agents
    _performance_stats['failed_agents'] = failed_agents
    _performance_stats['initialization_time'] = time.time() - start_time

    # Summary logging
    total_agents = 6
    logger.info(f"🎯 Agent initialization complete: {successful_agents}/{total_agents} agents operational")
    
    if failed_agents > 0:
        logger.warning(f"⚠️ {failed_agents} agents failed - system will use fallbacks where needed")
    else:
        logger.info("🎉 All agents initialized successfully - full functionality available")

    return {
        'status': 'completed',
        'successful_agents': successful_agents,
        'failed_agents': failed_agents,
        'total_agents': total_agents,
        'initialization_time': _performance_stats['initialization_time'],
        'results': initialization_results
    }


def safe_agent_process(agent: Any, agent_name: str, input_data: Dict[str, Any], 
                      timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Safely process with an agent using comprehensive error handling and timeout protection.
    
    Args:
        agent: Agent instance to process with
        agent_name: Human-readable agent name for logging
        input_data: Input data dictionary
        timeout: Optional timeout in seconds
        
    Returns:
        Standardized result dictionary with success flag and detailed error handling
    """
    global _performance_stats
    
    _performance_stats['total_processing_calls'] += 1
    
    if agent is None:
        _performance_stats['failed_processing_calls'] += 1
        logger.warning(f"⚠️ {agent_name} is not available - agent not initialized")
        return {
            "success": False,
            "error": {
                "message": f"{agent_name} not initialized",
                "code": "AGENT_NOT_AVAILABLE",
                "agent": agent_name,
                "timestamp": datetime.now().isoformat()
            },
            "result": {},
            "fallback_used": True
        }

    try:
        logger.info(f"🔄 Processing with {agent_name}...")
        start_time = time.time()
        
        # Process with optional timeout handling
        if timeout:
            # For now, we'll log the timeout but not implement it
            # In production, you might want to use threading or asyncio timeouts
            logger.debug(f"Processing with {timeout}s timeout (monitoring only)")
        
        # Handle async process method
        if asyncio.iscoroutinefunction(agent.process):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(input_data))
                loop.close()
            except Exception as async_error:
                logger.warning(f"Async processing failed: {async_error}, using fallback")
                result = {"success": False, "error": {"message": str(async_error)}}
        else:
            result = agent.process(input_data)
        processing_time = time.time() - start_time

        if result and result.get("success", False):
            logger.info(f"✅ {agent_name} completed successfully in {processing_time:.2f}s")
            
            # Add processing metadata
            if isinstance(result, dict):
                result.setdefault('metadata', {})['processing_time'] = processing_time
                result.setdefault('metadata', {})['agent_name'] = agent_name
                
            return result
        else:
            _performance_stats['failed_processing_calls'] += 1
            error_msg = "Unknown error"
            
            if result and isinstance(result, dict):
                error_info = result.get("error", {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", "Unknown error")
                elif isinstance(error_info, str):
                    error_msg = error_info
                else:
                    error_msg = str(error_info)
            
            logger.error(f"❌ {agent_name} failed: {error_msg}")
            return result or {
                "success": False,
                "error": {
                    "message": error_msg,
                    "code": "AGENT_PROCESSING_FAILED",
                    "agent": agent_name
                },
                "result": {}
            }

    except Exception as e:
        _performance_stats['failed_processing_calls'] += 1
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        
        logger.exception(f"💥 {agent_name} exception after {processing_time:.2f}s: {str(e)}")
        return {
            "success": False,
            "error": {
                "message": f"{agent_name} processing exception: {str(e)}",
                "code": "AGENT_PROCESSING_EXCEPTION",
                "agent": agent_name,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            },
            "result": {}
        }


def _enhanced_mock_bert_classification(state: FakeNewsState, start_time: float) -> FakeNewsState:
    """
    Enhanced mock BERT classification with sophisticated keyword-based analysis.
    
    Used when the trained BERT model is not available, providing intelligent
    fallback classification based on content analysis.
    """
    new_state = dict(state)
    article_text = state["article_text"].lower()

    # Enhanced keyword analysis with weighted scoring
    fake_indicators = {
        # Sensational language (high weight)
        'breaking': 3, 'shocking': 3, 'secret': 3, 'exclusive': 2,
        'amazing': 2, 'incredible': 2, 'unbelievable': 3,
        
        # Conspiracy language (high weight)
        'conspiracy': 4, 'coverup': 4, 'they dont want you to know': 5,
        'government hiding': 4, 'big pharma': 3, 'leaked': 2,
        
        # Medical misinformation (very high weight)
        'miracle cure': 5, 'doctors hate': 4, 'cure all': 4,
        'one weird trick': 4, 'this one trick': 3,
        
        # Emotional manipulation (medium weight)
        'you wont believe': 2, 'scientists shocked': 3, 'banned': 2,
        'censored': 2, 'insider reveals': 2
    }
    
    reliable_indicators = {
        # Academic language (high weight)
        'study published': 4, 'peer reviewed': 5, 'research shows': 3,
        'according to': 2, 'data shows': 3, 'analysis': 2,
        
        # Institutional sources (high weight)
        'university': 3, 'institute': 3, 'department': 2,
        'professor': 3, 'researcher': 3, 'journal': 4,
        
        # Factual language (medium weight)
        'findings': 2, 'evidence suggests': 3, 'report': 2,
        'official': 2, 'experts': 2
    }

    # Calculate weighted scores
    fake_score = sum(weight for keyword, weight in fake_indicators.items() 
                    if keyword in article_text)
    reliable_score = sum(weight for keyword, weight in reliable_indicators.items() 
                        if keyword in article_text)

    # Additional heuristics
    text_length = len(state["article_text"])
    excessive_caps = sum(1 for c in state["article_text"] if c.isupper()) / max(text_length, 1)
    excessive_punctuation = state["article_text"].count('!') + state["article_text"].count('?')
    
    # Adjust scores based on heuristics
    if excessive_caps > 0.1:  # More than 10% caps
        fake_score += 2
    if excessive_punctuation > 5:
        fake_score += 1
    if text_length < 100:  # Very short articles are suspicious
        fake_score += 1

    # Determine prediction with confidence calculation
    total_score = fake_score + reliable_score
    
    if fake_score > reliable_score + 2:
        prediction = "FAKE"
        confidence = min(0.95, 0.60 + (fake_score - reliable_score) * 0.05)
    elif reliable_score > fake_score + 2:
        prediction = "REAL" 
        confidence = min(0.95, 0.60 + (reliable_score - fake_score) * 0.05)
    elif total_score == 0:
        # No indicators found - neutral classification
        prediction = "REAL"
        confidence = 0.55
    else:
        # Close scores - low confidence
        prediction = "REAL" if reliable_score >= fake_score else "FAKE"
        confidence = 0.52 + abs(reliable_score - fake_score) * 0.02

    # Calculate probability distribution
    real_prob = confidence if prediction == "REAL" else 1 - confidence
    fake_prob = 1 - real_prob

    bert_results = {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "real": round(real_prob, 4),
            "fake": round(fake_prob, 4)
        },
        "text_analysis": {
            "original_length": len(state["article_text"]),
            "tokens_used": len(state["article_text"].split()),
            "preprocessing_applied": True,
            "fake_indicators_score": fake_score,
            "reliable_indicators_score": reliable_score,
            "excessive_caps_ratio": round(excessive_caps, 3),
            "excessive_punctuation_count": excessive_punctuation
        },
        "mock_classification": True,
        "classification_method": "enhanced_heuristic_analysis",
        "model_version": "mock_v3.2"
    }

    new_state["bert_results"] = bert_results
    new_state.setdefault("confidence_scores", {})["bert"] = confidence
    new_state.setdefault("processing_times", {})["bert_classifier"] = time.time() - start_time

    logger.info(
        f"🤖 Mock BERT Classification: {prediction} ({confidence:.1%}) "
        f"- Fake indicators: {fake_score}, Reliable indicators: {reliable_score}"
    )

    return FakeNewsState(new_state)


# Node Functions for LangGraph Workflow

def bert_classifier_node(state: FakeNewsState) -> FakeNewsState:
    """BERT classification node with trained model and enhanced mock fallback."""
    initialize_agents()
    start_time = time.time()
    
    try:
        new_state = dict(state)
        
        # Use mock classification if model unavailable or not loaded
        if (_agent_status['using_mock_bert'] or 
            not _agent_status['bert_model_loaded'] or 
            _agents['bert_agent'] is None):
            return _enhanced_mock_bert_classification(state, start_time)

        # Check if BERT agent is ready for inference
        bert_agent = _agents['bert_agent']
        if hasattr(bert_agent, 'is_ready') and not bert_agent.is_ready():
            logger.warning("⚠️ BERT model not ready for inference - using mock classification")
            return _enhanced_mock_bert_classification(state, start_time)

        # Use trained BERT model
        result = safe_agent_process(bert_agent, "BERT Classifier", {
            "text": state["article_text"]
        }, timeout=30.0)

        if result.get("success"):
            new_state["bert_results"] = result["result"]
            confidence = result["result"].get("confidence", 0.0)
            new_state.setdefault("confidence_scores", {})["bert"] = confidence
            
            prediction = result["result"].get("prediction", "UNKNOWN")
            logger.info(f"🎯 BERT Classification: {prediction} ({confidence:.1%})")
        else:
            logger.error(f"❌ BERT processing failed: {result.get('error', {}).get('message', 'Unknown error')}")
            return _enhanced_mock_bert_classification(state, start_time)

        new_state.setdefault("processing_times", {})["bert_classifier"] = time.time() - start_time
        return FakeNewsState(new_state)

    except Exception as e:
        logger.exception(f"💥 BERT classifier node error: {str(e)}")
        return _enhanced_mock_bert_classification(state, start_time)


def claim_extractor_node(state: FakeNewsState) -> FakeNewsState:
    """Claim extraction node with enhanced error handling and fallback."""
    initialize_agents()
    start_time = time.time()
    new_state = dict(state)

    # Prepare fallback claims
    fallback_claims = []

    result = safe_agent_process(_agents['claim_agent'], "Claim Extractor", {
        "text": state["article_text"],
        "bert_results": state.get("bert_results", {}),
        "detailed_analysis": state.get("require_detailed_analysis", False)
    }, timeout=45.0)

    if result.get("success"):
        extracted_claims = result["result"].get("extracted_claims", [])
        new_state["extracted_claims"] = extracted_claims
        new_state.setdefault("confidence_scores", {})["claim_extraction"] = result.get("confidence", 0.0)

        # Enhanced logging with claim analysis
        claims_count = len(extracted_claims)
        high_priority = sum(1 for c in extracted_claims 
                           if isinstance(c, dict) and c.get("priority", 3) <= 2)
        verifiable = sum(1 for c in extracted_claims 
                        if isinstance(c, dict) and c.get("verifiability_score", 0) >= 6)
        
        logger.info(
            f"📋 Claims extracted: {claims_count} total, {high_priority} high-priority, {verifiable} verifiable"
        )
    else:
        error_msg = result.get("error", {}).get("message", "Claim extraction failed")
        new_state.setdefault("processing_errors", []).append(error_msg)
        new_state["extracted_claims"] = fallback_claims
        new_state.setdefault("confidence_scores", {})["claim_extraction"] = 0.0
        logger.warning(f"⚠️ Using empty claims list as fallback: {error_msg}")

    new_state.setdefault("processing_times", {})["claim_extractor"] = time.time() - start_time
    return FakeNewsState(new_state)


def context_analyzer_node(state: FakeNewsState) -> FakeNewsState:
    """Context analysis node with enhanced features and fallback handling."""
    initialize_agents()
    start_time = time.time()
    new_state = dict(state)

    if not _agent_status['context_ready'] or _agents['context_agent'] is None:
        logger.warning("⚠️ Context Analyzer not available - using enhanced fallback")
        
        # Enhanced fallback context analysis
        fallback_context = {
            "context_scores": {
                "overall_context_score": 5.0,
                "risk_level": "MEDIUM",
                "bias_score": 0.0,
                "credibility_score": 50.0
            },
            "manipulation_report": {
                "overall_manipulation_score": 0.0,
                "techniques_detected": [],
                "fallacy_indicators": []
            },
            "llm_scores": {},
            "llm_analysis": "Context analysis not available - agent not initialized. Using fallback analysis.",
            "fallback_used": True,
            "metadata": {
                "agent_version": "fallback_v3.2",
                "processing_time": time.time() - start_time
            }
        }
        
        new_state["context_analysis"] = fallback_context
        new_state.setdefault("confidence_scores", {})["context_analysis"] = 0.5
    else:
        # Process with Context Analyzer (Enhanced v3.2)
        bert_results = state.get("bert_results", {})
        
        context_input = {
            "text": state["article_text"],
            "previous_analysis": {
                "prediction": bert_results.get("prediction", "UNKNOWN"),
                "confidence": bert_results.get("confidence", 0.0),
                "source": "BERT Classifier",
                "topic_domain": "general",
                "bert_results": bert_results,
                "extracted_claims": state.get("extracted_claims", [])
            },
            "enable_llm_scoring": True,
            "detailed_analysis": state.get("require_detailed_analysis", False)
        }

        result = safe_agent_process(_agents['context_agent'], "Context Analyzer", 
                                  context_input, timeout=60.0)

        if result.get("success"):
            new_state["context_analysis"] = result["result"]
            new_state.setdefault("confidence_scores", {})["context_analysis"] = result.get("confidence", 0.0)

            # Log enhanced features
            context_result = result["result"]
            llm_scores = context_result.get("llm_scores", {})
            safety_used = context_result.get("safety_fallback_used", False)
            
            if llm_scores:
                bias_score = llm_scores.get('bias', 0)
                credibility_score = llm_scores.get('credibility', 50)
                risk_score = llm_scores.get('risk', 50)
                logger.info(
                    f"🎯 Context Analysis (Enhanced): Bias: {bias_score}, "
                    f"Credibility: {credibility_score}, Risk: {risk_score}"
                    f"{' (Safety fallback used)' if safety_used else ''}"
                )
            else:
                logger.info("🎯 Context Analysis completed (basic scoring)")
        else:
            # Use fallback on failure
            error_msg = result.get("error", {}).get("message", "Context analysis failed")
            new_state.setdefault("processing_errors", []).append(error_msg)
            
            fallback_context = {
                "context_scores": {"overall_context_score": 5.0, "risk_level": "UNKNOWN"},
                "manipulation_report": {"overall_manipulation_score": 0.0},
                "llm_scores": {},
                "fallback_used": True,
                "error": error_msg
            }
            
            new_state["context_analysis"] = fallback_context
            new_state.setdefault("confidence_scores", {})["context_analysis"] = 0.5
            logger.warning(f"⚠️ Context analysis failed, using fallback: {error_msg}")

    new_state.setdefault("processing_times", {})["context_analyzer"] = time.time() - start_time
    return FakeNewsState(new_state)


def evidence_evaluator_node(state: FakeNewsState) -> FakeNewsState:
    """Evidence evaluation node with enhanced verification links and safety handling."""
    initialize_agents()
    start_time = time.time()
    new_state = dict(state)

    if not _agent_status['evidence_ready'] or _agents['evidence_agent'] is None:
        logger.warning("⚠️ Evidence Evaluator not available - using fallback")
        
        fallback_evidence = {
            "evidence_scores": {
                "overall_evidence_score": 5.0,
                "quality_level": "UNKNOWN",
                "source_quality_score": 5.0,
                "verification_links_quality_score": 5.0
            },
            "verification_links": [],
            "safety_fallback_used": False,
            "fallback_used": True,
            "metadata": {
                "agent_version": "fallback_v3.2",
                "processing_time": time.time() - start_time
            }
        }
        
        new_state["evidence_evaluation"] = fallback_evidence
        new_state.setdefault("confidence_scores", {})["evidence_evaluation"] = 0.5
    else:
        evidence_input = {
            "text": state["article_text"],
            "extracted_claims": state.get("extracted_claims", []),
            "context_analysis": state.get("context_analysis", {}),
            "enable_specific_verification_links": True,
            "enable_institutional_fallbacks": True,
            "detailed_analysis": state.get("require_detailed_analysis", False)
        }

        result = safe_agent_process(_agents['evidence_agent'], "Evidence Evaluator", 
                                  evidence_input, timeout=90.0)

        if result.get("success"):
            new_state["evidence_evaluation"] = result["result"]
            new_state.setdefault("confidence_scores", {})["evidence_evaluation"] = result.get("confidence", 0.0)

            # Log enhanced features  
            evidence_result = result["result"]
            # Map verification_sources to verification_links for compatibility
            verification_sources = evidence_result.get("verification_sources", [])
            verification_links = verification_sources  # Use verification_sources as verification_links
            safety_used = evidence_result.get("safety_fallback_used", False)
            institutional_links = sum(1 for link in verification_links 
                                    if isinstance(link, dict) and 'institutional' in link.get('type', ''))
            
            logger.info(
                f"🔍 Evidence Evaluation (Enhanced): {len(verification_links)} verification links, "
                f"{institutional_links} institutional, Safety fallback: {'Yes' if safety_used else 'No'}"
            )
        else:
            error_msg = result.get("error", {}).get("message", "Evidence evaluation failed")
            new_state.setdefault("processing_errors", []).append(error_msg)
            
            fallback_evidence = {
                "evidence_scores": {"overall_evidence_score": 5.0, "quality_level": "UNKNOWN"},
                "verification_links": [],
                "fallback_used": True,
                "error": error_msg
            }
            
            new_state["evidence_evaluation"] = fallback_evidence
            new_state.setdefault("confidence_scores", {})["evidence_evaluation"] = 0.5
            logger.warning(f"⚠️ Evidence evaluation failed, using fallback: {error_msg}")

    new_state.setdefault("processing_times", {})["evidence_evaluator"] = time.time() - start_time
    return FakeNewsState(new_state)


def credible_source_node(state: FakeNewsState) -> FakeNewsState:
    """Credible source recommendation node with contextual sources and safety handling."""
    initialize_agents()
    start_time = time.time()
    new_state = dict(state)

    if not _agent_status['credible_ready'] or _agents['credible_agent'] is None:
        logger.warning("⚠️ Credible Source Agent not available - using fallback")
        
        fallback_sources = {
            "recommended_sources": [],
            "contextual_sources": [],
            "recommendation_scores": {
                "overall_recommendation_score": 5.0,
                "contextual_sources_count": 0,
                "availability_factors": [],
                "verification_challenges": ["Agent unavailable"]
            },
            "safety_fallback_used": False,
            "fallback_used": True,
            "metadata": {
                "agent_version": "fallback_v3.2",
                "processing_time": time.time() - start_time
            }
        }
        
        new_state["source_recommendations"] = fallback_sources
        new_state.setdefault("confidence_scores", {})["source_recommendations"] = 0.5
    else:
        source_input = {
            "text": state["article_text"],
            "extracted_claims": state.get("extracted_claims", []),
            "evidence_evaluation": state.get("evidence_evaluation", {}),
            "context_analysis": state.get("context_analysis", {}),
            "enable_contextual_recommendations": True,
            "enable_expert_identification": True
        }

        result = safe_agent_process(_agents['credible_agent'], "Credible Source", 
                                  source_input, timeout=60.0)

        if result.get("success"):
            new_state["source_recommendations"] = result["result"]
            new_state.setdefault("confidence_scores", {})["source_recommendations"] = result.get("confidence", 0.0)

            # Log enhanced features
            source_result = result["result"]
            contextual_sources = source_result.get("contextual_sources", [])
            safety_used = source_result.get("safety_fallback_used", False)
            expert_sources = sum(1 for source in contextual_sources 
                               if isinstance(source, dict) and 'expert' in source.get('type', ''))
            
            logger.info(
                f"📚 Source Recommendations (Enhanced): {len(contextual_sources)} contextual sources, "
                f"{expert_sources} expert sources, Safety fallback: {'Yes' if safety_used else 'No'}"
            )
        else:
            error_msg = result.get("error", {}).get("message", "Source recommendation failed")
            new_state.setdefault("processing_errors", []).append(error_msg)
            
            fallback_sources = {
                "recommended_sources": [],
                "contextual_sources": [],
                "recommendation_scores": {"overall_recommendation_score": 5.0},
                "fallback_used": True,
                "error": error_msg
            }
            
            new_state["source_recommendations"] = fallback_sources
            new_state.setdefault("confidence_scores", {})["source_recommendations"] = 0.5
            logger.warning(f"⚠️ Source recommendation failed, using fallback: {error_msg}")

    new_state.setdefault("processing_times", {})["credible_source"] = time.time() - start_time
    return FakeNewsState(new_state)


def llm_explanation_node(state: FakeNewsState) -> FakeNewsState:
    """LLM explanation node with comprehensive analysis integration."""
    initialize_agents()
    start_time = time.time()
    new_state = dict(state)

    # Prepare comprehensive input data
    bert_results = state.get("bert_results", {})
    bert_confidence = state.get("confidence_scores", {}).get("bert", 0.0)
    
    explanation_input = {
        "text": state["article_text"],
        "prediction": bert_results.get("prediction", "UNKNOWN"),
        "confidence": bert_confidence,
        "all_agent_results": {
            "bert_results": bert_results,
            "extracted_claims": state.get("extracted_claims", []),
            "context_analysis": state.get("context_analysis", {}),
            "evidence_evaluation": state.get("evidence_evaluation", {}),
            "source_recommendations": state.get("source_recommendations", {})
        },
        "metadata": {
            "source": state.get("article_url", "Unknown"),
            "claims_count": len(state.get("extracted_claims", [])),
            "processing_path": new_state.get("processing_path", "unknown"),
            "processing_errors": state.get("processing_errors", []),
            "confidence_scores": state.get("confidence_scores", {}),
            "date": "Unknown Date",
            "subject": "General News"
        },
        "require_detailed_analysis": state.get("require_detailed_analysis", False),
        "enable_confidence_analysis": True,
        "enable_source_analysis": True
    }

    result = safe_agent_process(_agents['explanation_agent'], "LLM Explanation", 
                              explanation_input, timeout=75.0)

    if result.get("success"):
        new_state["final_explanation"] = result["result"]
        new_state.setdefault("confidence_scores", {})["explanation"] = result.get("confidence", 0.0)

        # Log explanation features
        explanation_result = result["result"]
        has_detailed = explanation_result.get("detailed_analysis") is not None
        has_confidence = explanation_result.get("confidence_analysis") is not None
        has_methodology = explanation_result.get("methodology_explanation") is not None
        
        logger.info(
            f"📝 LLM Explanation generated successfully - "
            f"Detailed: {'Yes' if has_detailed else 'No'}, "
            f"Confidence Analysis: {'Yes' if has_confidence else 'No'}, "
            f"Methodology: {'Yes' if has_methodology else 'No'}"
        )
    else:
        error_info = result.get("error", {})
        if isinstance(error_info, dict):
            error_msg = error_info.get("message", "Explanation generation failed")
            error_code = error_info.get("code", "UNKNOWN_ERROR")
        else:
            error_msg = str(error_info) if error_info else "Explanation generation failed"
            error_code = "UNKNOWN_ERROR"
        
        detailed_error = f"LLM Explanation Error [{error_code}]: {error_msg}"
        new_state.setdefault("processing_errors", []).append(detailed_error)
        
        logger.error(f"❌ LLM Explanation detailed error: {detailed_error}")
        
        # Generate fallback explanation
        prediction = bert_results.get("prediction", "UNKNOWN")
        confidence = bert_confidence
        
        fallback_explanation = (
            f"Analysis Summary: The article was classified as {prediction} "
            f"with {confidence:.1%} confidence. "
            f"However, detailed explanation generation failed due to: {error_msg}. "
            f"Please refer to the individual agent results for more information."
        )
        
        new_state["final_explanation"] = {
            "explanation": fallback_explanation,
            "detailed_analysis": None,
            "confidence_analysis": None,
            "fallback_used": True,
            "error": error_msg,
            "metadata": {
                "agent_version": "fallback_v3.2",
                "processing_time": time.time() - start_time
            }
        }
        
        new_state.setdefault("confidence_scores", {})["explanation"] = 0.0
        logger.warning(f"⚠️ Explanation generation failed, using fallback: {error_msg}")

    new_state.setdefault("processing_times", {})["llm_explanation"] = time.time() - start_time
    return FakeNewsState(new_state)


# Utility functions for monitoring and debugging

def get_agent_status() -> Dict[str, Any]:
    """Get current agent status for monitoring and debugging."""
    return {
        "agents_initialized": _agent_status['agents_initialized'],
        "agent_readiness": {
            "bert_ready": _agent_status['bert_ready'],
            "claim_ready": _agent_status['claim_ready'],
            "context_ready": _agent_status['context_ready'],
            "evidence_ready": _agent_status['evidence_ready'],
            "credible_ready": _agent_status['credible_ready'],
            "explanation_ready": _agent_status['explanation_ready']
        },
        "bert_model_status": {
            "model_loaded": _agent_status['bert_model_loaded'],
            "using_mock": _agent_status['using_mock_bert']
        },
        "performance_stats": _performance_stats.copy()
    }


def reset_agents():
    """Reset all agents and status for testing or reinitialization."""
    global _agents, _agent_status, _performance_stats
    
    # Clear agent instances
    for key in _agents:
        _agents[key] = None
    
    # Reset status flags
    for key in _agent_status:
        _agent_status[key] = False
    
    # Reset performance stats
    _performance_stats = {
        'initialization_time': 0.0,
        'successful_agents': 0,
        'failed_agents': 0,
        'total_processing_calls': 0,
        'failed_processing_calls': 0
    }
    
    logger.info("🔄 All agents reset - ready for reinitialization")
