"""
Enhanced LangGraph Node Wrappers with Smart Routing Signals
FIXED: All 6 agents with proper error handling and debugging
"""

import time
from typing import Dict, Any
from pathlib import Path
from .state import FakeNewsState

# Initialize agents (reuse instances for efficiency)
bert_agent = None
claim_agent = None
context_agent = None
evidence_agent = None
source_agent = None
explanation_agent = None
bert_model_loaded = False
using_mock_bert = False

def initialize_agents():
    """Initialize all agents with comprehensive debugging and error handling"""
    global bert_agent, claim_agent, context_agent, evidence_agent, source_agent, explanation_agent
    global bert_model_loaded, using_mock_bert
    
    if bert_agent is None:
        print("🔄 Initializing all 6 agents with debugging...")
        
        # 1. Initialize BERT Classifier (working)
        try:
            from agents.bert_classifier.classifier import BERTClassifier
            bert_agent = BERTClassifier()
            print("✅ BERT Classifier initialized")
            
            # Try to load your trained model
            model_path = Path("models/bert_fake_news")
            if model_path.exists():
                load_result = bert_agent.load_model(model_path)
                if load_result["success"]:
                    print("🎯 Real BERT model loaded successfully!")
                    bert_model_loaded = True
                    using_mock_bert = False
                else:
                    print("⚠️ BERT model load failed, using mock")
                    bert_model_loaded = False
                    using_mock_bert = True
            else:
                print("⚠️ Model path not found, using mock")
                bert_model_loaded = False
                using_mock_bert = True
                
        except Exception as e:
            print(f"❌ BERT Classifier init error: {str(e)}")
            bert_agent = None
        
        # 2. Initialize Claim Extractor (working)
        try:
            from agents.claim_extractor.extractor_agent import ClaimExtractorAgent
            claim_agent = ClaimExtractorAgent()
            print("✅ Claim Extractor initialized")
        except Exception as e:
            print(f"❌ Claim Extractor init error: {str(e)}")
            claim_agent = None
        
        # 3. Initialize Context Analyzer (FIXING)
        try:
            from agents.context_analyzer.analyzer_agent import ContextAnalyzerAgent
            context_agent = ContextAnalyzerAgent()
            print(f"✅ Context Analyzer initialized: {type(context_agent)}")
        except ImportError as e:
            print(f"❌ Context Analyzer import error: {str(e)}")
            context_agent = None
        except Exception as e:
            print(f"❌ Context Analyzer init error: {str(e)}")
            context_agent = None
        
        # 4. Initialize Evidence Evaluator (FIXING)
        try:
            from agents.evidence_evaluator.evaluator_agent import EvidenceEvaluatorAgent
            evidence_agent = EvidenceEvaluatorAgent()
            print(f"✅ Evidence Evaluator initialized: {type(evidence_agent)}")
        except ImportError as e:
            print(f"❌ Evidence Evaluator import error: {str(e)}")
            evidence_agent = None
        except Exception as e:
            print(f"❌ Evidence Evaluator init error: {str(e)}")
            evidence_agent = None
        
        # 5. Initialize Credible Source (FIXING)
        try:
            from agents.credible_source.source_agent import CredibleSourceAgent
            source_agent = CredibleSourceAgent()
            print(f"✅ Credible Source initialized: {type(source_agent)}")
        except ImportError as e:
            print(f"❌ Credible Source import error: {str(e)}")
            source_agent = None
        except Exception as e:
            print(f"❌ Credible Source init error: {str(e)}")
            source_agent = None
        
        # 6. Initialize LLM Explanation (working)
        try:
            from agents.llm_explanation.explanation_agent import LLMExplanationAgent
            explanation_agent = LLMExplanationAgent()
            print("✅ LLM Explanation initialized")
        except Exception as e:
            print(f"❌ LLM Explanation init error: {str(e)}")
            explanation_agent = None
        
        # Summary
        working_agents = sum([
            bert_agent is not None,
            claim_agent is not None,
            context_agent is not None,
            evidence_agent is not None,
            source_agent is not None,
            explanation_agent is not None
        ])
        
        print(f"🎯 Agent Initialization Complete: {working_agents}/6 agents working")
        
        if working_agents < 6:
            print("⚠️ Some agents failed to initialize - system will work with partial functionality")

def safe_agent_process(agent, agent_name, input_data):
    """Safely process with an agent, providing detailed error info"""
    if agent is None:
        print(f"⚠️ {agent_name} is None - agent not initialized")
        return {
            "success": False, 
            "error": {"message": f"{agent_name} not initialized"},
            "result": {}
        }
    
    try:
        print(f"🔄 Processing with {agent_name}...")
        result = agent.process(input_data)
        
        if result.get("success"):
            print(f"✅ {agent_name} completed successfully")
            return result
        else:
            error_msg = result.get("error", {}).get("message", "Unknown error")
            print(f"❌ {agent_name} failed: {error_msg}")
            return result
            
    except Exception as e:
        print(f"❌ {agent_name} exception: {str(e)}")
        return {
            "success": False,
            "error": {"message": f"{agent_name} exception: {str(e)}"},
            "result": {}
        }

def _mock_bert_classification(state: FakeNewsState, start_time: float) -> FakeNewsState:
    """Mock BERT classification for when model isn't available"""
    new_state = dict(state)
    
    article_text = state["article_text"].lower()
    
    # Enhanced keyword-based mock classification
    fake_keywords = [
        'breaking', 'secret', 'shocking', 'they dont want you to know', 
        'anonymous', 'leaked', 'big pharma', 'government hiding',
        'cure all', 'doctors hate', 'miracle', 'conspiracy',
        'amazing discovery', 'scientists shocked', 'this one trick',
        'you wont believe', 'exclusive', 'insider reveals',
        'banned', 'censored', 'coverup', 'hidden truth'
    ]
    
    real_keywords = [
        'according to', 'study published', 'research shows', 'peer reviewed',
        'university', 'journal', 'official', 'report', 'data shows',
        'experts', 'analysis', 'findings', 'evidence suggests'
    ]
    
    fake_score = sum(1 for keyword in fake_keywords if keyword in article_text)
    real_score = sum(1 for keyword in real_keywords if keyword in article_text)
    
    # Calculate confidence based on keyword presence
    if fake_score > real_score + 1:
        prediction = "FAKE"
        confidence = min(0.92, 0.65 + (fake_score - real_score) * 0.08)
    elif real_score > fake_score + 1:
        prediction = "REAL"
        confidence = min(0.92, 0.65 + (real_score - fake_score) * 0.08)
    else:
        prediction = "REAL"
        confidence = 0.58
    
    real_prob = confidence if prediction == "REAL" else 1 - confidence
    fake_prob = 1 - real_prob
    
    # Mock BERT results
    bert_results = {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "real": real_prob,
            "fake": fake_prob
        },
        "text_analysis": {
            "original_length": len(state["article_text"]),
            "processed_length": len(state["article_text"]),
            "tokens_used": len(state["article_text"].split()),
            "preprocessing_applied": True
        },
        "mock_classification": True,
        "fake_keywords_found": fake_score,
        "real_keywords_found": real_score
    }
    
    new_state["bert_results"] = bert_results
    new_state["confidence_scores"]["bert"] = confidence
    
    print(f"🎭 [MOCK BERT] Classified as {prediction} with {confidence:.1%} confidence (fake_kw:{fake_score}, real_kw:{real_score})")
    
    new_state["processing_times"]["bert_classifier"] = time.time() - start_time
    return FakeNewsState(new_state)

# ✅ MISSING FUNCTION 1: BERT CLASSIFIER NODE
def bert_classifier_node(state: FakeNewsState) -> FakeNewsState:
    """Enhanced BERT classification node with your trained model"""
    initialize_agents()
    start_time = time.time()
    
    try:
        new_state = dict(state)
        
        # Check if we should use mock BERT or real BERT
        if using_mock_bert or not bert_model_loaded:
            return _mock_bert_classification(state, start_time)
        
        # Check if model is ready
        if not bert_agent.is_ready():
            print("⚠️ BERT model not ready, falling back to mock classification")
            return _mock_bert_classification(state, start_time)
        
        # Use your trained BERT model
        result = bert_agent.process({
            "text": state["article_text"]
        })
        
        if result.get("success"):
            new_state["bert_results"] = result["result"]
            confidence = result["result"].get("confidence", 0.0)
            new_state["confidence_scores"]["bert"] = confidence
            print(f"🎯 [YOUR BERT MODEL] Classified as {result['result'].get('prediction', 'UNKNOWN')} with {confidence:.1%} confidence")
        else:
            error_msg = result.get("error", {}).get("message", "BERT classification failed")
            new_state["processing_errors"].append(error_msg)
            print(f"❌ BERT trained model failed: {error_msg}, using mock instead")
            return _mock_bert_classification(state, start_time)
            
    except Exception as e:
        print(f"❌ BERT classifier error: {str(e)}, using mock instead")
        return _mock_bert_classification(state, start_time)
    
    new_state["processing_times"]["bert_classifier"] = time.time() - start_time
    return FakeNewsState(new_state)

# ✅ MISSING FUNCTION 2: CLAIM EXTRACTOR NODE
def claim_extractor_node(state: FakeNewsState) -> FakeNewsState:
    """Enhanced claim extraction node with safe processing"""
    initialize_agents()
    start_time = time.time()
    
    new_state = dict(state)
    
    result = safe_agent_process(claim_agent, "Claim Extractor", {
        "text": state["article_text"],
        "bert_results": state.get("bert_results", {})
    })
    
    if result.get("success"):
        extracted_claims = result["result"]["extracted_claims"]
        new_state["extracted_claims"] = extracted_claims
        new_state["confidence_scores"]["claim_extraction"] = result.get("confidence", 0.0)
        
        # Enhanced routing signals based on claim analysis
        claims_count = len(extracted_claims)
        high_priority_claims = sum(1 for claim in extracted_claims
                                 if isinstance(claim, dict) and claim.get("priority", 3) <= 2)
        
        # Set processing hints for routing
        new_state["skip_expensive_processing"] = claims_count < 2 and high_priority_claims == 0
        
        print(f"📋 [Claims] Found {claims_count} claims, {high_priority_claims} high-priority")
    else:
        error_msg = result.get("error", {}).get("message", "Claim extraction failed")
        new_state["processing_errors"].append(error_msg)
        new_state["skip_expensive_processing"] = True
        new_state["extracted_claims"] = []
        
    new_state["processing_times"]["claim_extractor"] = time.time() - start_time
    return FakeNewsState(new_state)

def context_analyzer_node(state: FakeNewsState) -> FakeNewsState:
    """Context analysis node with enhanced error handling"""
    initialize_agents()
    start_time = time.time()
    
    new_state = dict(state)
    
    if context_agent is None:
        print("⚠️ Context Analyzer not available, using fallback")
        # Provide fallback context analysis
        new_state["context_analysis"] = {
            "context_scores": {
                "overall_context_score": 5.0,
                "risk_level": "UNKNOWN",
                "bias_score": 0.0
            },
            "manipulation_report": {
                "overall_manipulation_score": 0.0
            }
        }
        new_state.setdefault("confidence_scores", {})["context_analysis"] = 0.5
    else:
        result = safe_agent_process(context_agent, "Context Analyzer", {
            "text": state["article_text"],
            "previous_analysis": {
                "bert_results": state.get("bert_results", {}),
                "extracted_claims": state.get("extracted_claims", [])
            }
        })
        
        if result.get("success"):
            new_state["context_analysis"] = result["result"]
            new_state.setdefault("confidence_scores", {})["context_analysis"] = result.get("confidence", 0.0)
        else:
            # Fallback
            new_state["context_analysis"] = {
                "context_scores": {
                    "overall_context_score": 5.0,
                    "risk_level": "UNKNOWN",
                    "bias_score": 0.0
                },
                "manipulation_report": {
                    "overall_manipulation_score": 0.0
                }
            }
            error_msg = result.get("error", {}).get("message", "Context analysis failed")
            new_state.setdefault("processing_errors", []).append(error_msg)
    
    new_state.setdefault("processing_times", {})["context_analyzer"] = time.time() - start_time
    return FakeNewsState(new_state)

def evidence_evaluator_node(state: FakeNewsState) -> FakeNewsState:
    """Evidence evaluation node with enhanced error handling"""
    initialize_agents()
    start_time = time.time()
    
    new_state = dict(state)
    
    if evidence_agent is None:
        print("⚠️ Evidence Evaluator not available, using fallback")
        # Provide fallback evidence evaluation
        new_state["evidence_evaluation"] = {
            "evidence_scores": {
                "overall_evidence_score": 5.0,
                "quality_level": "UNKNOWN",
                "source_quality_score": 5.0
            }
        }
        new_state.setdefault("confidence_scores", {})["evidence_evaluation"] = 0.5
    else:
        result = safe_agent_process(evidence_agent, "Evidence Evaluator", {
            "text": state["article_text"],
            "extracted_claims": state.get("extracted_claims", []),
            "context_analysis": state.get("context_analysis", {}),
            "include_detailed_analysis": state.get("require_detailed_analysis", False)
        })
        
        if result.get("success"):
            new_state["evidence_evaluation"] = result["result"]
            new_state.setdefault("confidence_scores", {})["evidence_evaluation"] = result.get("confidence", 0.0)
        else:
            # Fallback
            new_state["evidence_evaluation"] = {
                "evidence_scores": {
                    "overall_evidence_score": 5.0,
                    "quality_level": "UNKNOWN",
                    "source_quality_score": 5.0
                }
            }
            error_msg = result.get("error", {}).get("message", "Evidence evaluation failed")
            new_state.setdefault("processing_errors", []).append(error_msg)
    
    new_state.setdefault("processing_times", {})["evidence_evaluator"] = time.time() - start_time
    return FakeNewsState(new_state)

def credible_source_node(state: FakeNewsState) -> FakeNewsState:
    """Credible source recommendation node with enhanced error handling"""
    initialize_agents()
    start_time = time.time()
    
    new_state = dict(state)
    
    if source_agent is None:
        print("⚠️ Source Recommender not available, using fallback")
        # Provide fallback source recommendations
        new_state["source_recommendations"] = {
            "recommended_sources": [],
            "recommendation_scores": {
                "overall_recommendation_score": 5.0,
                "availability_factors": [],
                "verification_challenges": ["Agent unavailable"]
            }
        }
        new_state.setdefault("confidence_scores", {})["source_recommendations"] = 0.5
    else:
        result = safe_agent_process(source_agent, "Source Recommender", {
            "text": state["article_text"],
            "extracted_claims": state.get("extracted_claims", []),
            "evidence_evaluation": state.get("evidence_evaluation", {})
        })
        
        if result.get("success"):
            new_state["source_recommendations"] = result["result"]
            new_state.setdefault("confidence_scores", {})["source_recommendations"] = result.get("confidence", 0.0)
        else:
            # Fallback
            new_state["source_recommendations"] = {
                "recommended_sources": [],
                "recommendation_scores": {
                    "overall_recommendation_score": 5.0,
                    "availability_factors": [],
                    "verification_challenges": ["Source recommendation failed"]
                }
            }
            error_msg = result.get("error", {}).get("message", "Source recommendation failed")
            new_state.setdefault("processing_errors", []).append(error_msg)
    
    new_state.setdefault("processing_times", {})["credible_source"] = time.time() - start_time
    return FakeNewsState(new_state)

def llm_explanation_node(state: FakeNewsState) -> FakeNewsState:
    """LLM explanation node with safe processing"""
    initialize_agents()
    start_time = time.time()
    
    new_state = dict(state)
    
    # Your existing working code here...
    bert_results = state.get("bert_results", {})
    bert_confidence = state.get("confidence_scores", {}).get("bert", 0.0)
    
    result = safe_agent_process(explanation_agent, "LLM Explanation", {
        "text": state["article_text"],
        "prediction": bert_results.get("prediction", "UNKNOWN"),
        "confidence": bert_confidence,
        "metadata": {
            "source": state.get("article_url", "Unknown"),
            "claims_count": len(state.get("extracted_claims", [])),
            "processing_path": new_state.get("processing_path", "unknown")
        },
        "require_detailed_analysis": True
    })
    
    if result.get("success"):
        new_state["final_explanation"] = result["result"]
        new_state.setdefault("confidence_scores", {})["explanation"] = result.get("confidence", 0.0)
    else:
        error_msg = result.get("error", {}).get("message", "Explanation generation failed")
        new_state.setdefault("processing_errors", []).append(error_msg)
    
    new_state.setdefault("processing_times", {})["llm_explanation"] = time.time() - start_time
    return FakeNewsState(new_state)
