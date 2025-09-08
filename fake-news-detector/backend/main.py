"""
FastAPI Server for Fake News Detection with Enhanced Safety Handling

Enhanced with safety filter handling, improved quality validation,
and robust error handling for all enhanced agents.

Version 3.1.0 - Safety Enhanced Edition
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables first
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware

# Import your configuration system
from config.settings import get_settings
from config.model_configs import get_model_config

# Import LangGraph workflow and state
from orchestration.langgraph_workflow import fake_news_workflow, analyze_article

# Import URL scraper
from utils.url_scraper import NewsArticleScraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global settings
settings = get_settings()

# Create FastAPI app with dynamic configuration
app = FastAPI(
    title="Enhanced Fake News Detection API with Safety Handling",
    description="Multi-agent fake news detection with safety filter handling, LLM scoring, and contextual recommendations",
    version="3.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize URL scraper
scraper = NewsArticleScraper()

# Request/Response models
class ArticleRequest(BaseModel):
    text: Optional[str] = None  # Raw article text
    url: Optional[str] = None   # Article URL
    detailed: bool = False

    @validator('*', pre=True)
    def validate_input(cls, v, values):
        """Ensure either text or url is provided"""
        if 'text' in values and 'url' in values:
            if not values.get('text') and not values.get('url'):
                raise ValueError("Either 'text' or 'url' must be provided")
        return v

class AnalysisResponse(BaseModel):
    success: bool
    results: Optional[Dict[str, Any]] = None
    errors: list = []
    metadata: Dict[str, Any] = {}

class ConfigResponse(BaseModel):
    agent_configs: Dict[str, Any]
    api_key_configured: bool
    current_models: Dict[str, str]

def is_url(text: str) -> bool:
    """Simple URL detection"""
    if not text:
        return False
    return text.strip().lower().startswith(('http://', 'https://'))

def validate_analysis_quality(result: dict) -> dict:
    """
    ✅ ENHANCED QUALITY VALIDATION WITH SAFETY AWARENESS
    
    Validates the quality of analysis results with awareness of safety fallbacks
    """
    issues = []
    
    # ✅ CHECK LLM SCORING CONSISTENCY
    context_analysis = result.get('context_analysis', {})
    llm_scores = context_analysis.get('llm_scores', {})
    
    if llm_scores:
        # Check for consistent LLM scoring
        bias_score = llm_scores.get('bias', 50)
        credibility_score = llm_scores.get('credibility', 50)
        risk_score = llm_scores.get('risk', 50)
        
        # High bias + low credibility should be consistent
        if bias_score > 70 and credibility_score > 70:
            issues.append("inconsistent_llm_scoring")
        
        # Very high risk should correlate with low credibility
        if risk_score > 80 and credibility_score > 60:
            issues.append("risk_credibility_mismatch")
    else:
        issues.append("missing_llm_scores")

    # ✅ CHECK VERIFICATION LINK QUALITY WITH SAFETY AWARENESS
    evidence_eval = result.get('evidence_evaluation', {})
    verification_links = evidence_eval.get('verification_links', [])
    
    if len(verification_links) == 0:
        issues.append("no_verification_links")
    elif len(verification_links) < 2:
        issues.append("insufficient_verification_links")
    
    # Check for specific vs generic links (safety fallbacks are acceptable)
    specific_links = sum(1 for link in verification_links
                        if isinstance(link, dict) and
                        link.get('quality_score', 0) > 0.7)
    
    institutional_fallbacks = sum(1 for link in verification_links
                                 if isinstance(link, dict) and
                                 'institutional' in link.get('type', ''))
    
    # Safety fallbacks are acceptable - don't flag as issue if present
    if specific_links == 0 and institutional_fallbacks == 0 and len(verification_links) > 0:
        issues.append("only_generic_verification_links")
    
    # ✅ CHECK FOR SAFETY FALLBACK USAGE (NOT AN ISSUE, JUST INFO)
    evidence_safety = evidence_eval.get('safety_fallback_used', False)
    
    # ✅ CHECK CONTEXTUAL SOURCE RECOMMENDATIONS WITH SAFETY AWARENESS
    source_recommendations = result.get('source_recommendations', {})
    contextual_sources = source_recommendations.get('contextual_sources', [])
    
    if len(contextual_sources) == 0:
        issues.append("no_contextual_sources")
    
    # Check for contextual vs generic sources (safety fallbacks acceptable)
    contextual_count = sum(1 for source in contextual_sources
                          if isinstance(source, dict) and
                          source.get('type', '').startswith('contextual'))
    
    source_safety = source_recommendations.get('safety_fallback_used', False)
    
    # Don't flag as issue if safety fallbacks provided institutional sources
    if contextual_count == 0 and not source_safety and len(contextual_sources) > 0:
        issues.append("only_generic_source_recommendations")

    # ✅ CHECK EVIDENCE QUALITY
    evidence_scores = evidence_eval.get('evidence_scores', {})
    source_quality_score = evidence_scores.get('source_quality_score', 5.0)
    link_quality_score = evidence_scores.get('verification_links_quality_score', 5.0)
    
    if source_quality_score < 4.0:
        issues.append("poor_evidence_sources")
    if link_quality_score < 4.0 and not evidence_safety:
        # Only flag if not using safety fallbacks
        issues.append("poor_verification_link_quality")

    # ✅ CHECK EXPLANATION QUALITY
    explanation = result.get('final_explanation', {}).get('explanation', '')
    llm_analysis = context_analysis.get('llm_analysis', '')
    
    # Check for formatted explanation
    if not ('##' in explanation or '**' in explanation or '###' in explanation):
        issues.append("unformatted_explanation")
    
    # Check for LLM analysis presence
    if not llm_analysis or len(llm_analysis) < 100:
        issues.append("insufficient_llm_analysis")

    # ✅ CALCULATE ENHANCED QUALITY SCORE WITH SAFETY AWARENESS
    base_score = 100
    penalty_per_issue = 12  # Reduced penalty since some issues are mitigated by safety fallbacks
    
    quality_score = max(0, base_score - len(issues) * penalty_per_issue)
    
    # Bonus for enhanced features
    if llm_scores:
        quality_score += 5
    if specific_links > 0 or institutional_fallbacks > 0:
        quality_score += 5  # Credit for both specific links and institutional fallbacks
    if contextual_count > 0:
        quality_score += 5
    
    # ✅ BONUS FOR SUCCESSFUL SAFETY HANDLING
    if evidence_safety and len(verification_links) > 0:
        quality_score += 3  # Bonus for successful safety fallback handling
    if source_safety and len(contextual_sources) > 0:
        quality_score += 3  # Bonus for successful source fallback handling
    
    quality_score = min(100, quality_score)

    # ✅ COUNT SAFETY BLOCKS ENCOUNTERED
    safety_blocks_total = 0
    if evidence_eval.get('metadata', {}).get('safety_blocks_encountered', 0):
        safety_blocks_total += evidence_eval['metadata']['safety_blocks_encountered']
    if source_recommendations.get('metadata', {}).get('safety_blocks_encountered', 0):
        safety_blocks_total += source_recommendations['metadata']['safety_blocks_encountered']

    # Add enhanced quality metadata
    result['quality_validation'] = {
        'issues_detected': issues,
        'quality_score': quality_score,
        'requires_human_review': len(issues) > 4 or safety_blocks_total > 2,  # Adjusted threshold
        'enhanced_features': {
            'llm_scoring_enabled': bool(llm_scores),
            'specific_links_available': specific_links > 0,
            'institutional_fallbacks_used': institutional_fallbacks > 0,  # ✅ New
            'contextual_sources_available': contextual_count > 0,
            'verification_links_count': len(verification_links),
            'contextual_sources_count': contextual_count
        },
        'safety_analysis': {  # ✅ New safety analysis section
            'safety_blocks_encountered': safety_blocks_total,
            'evidence_safety_fallback_used': evidence_safety,
            'source_safety_fallback_used': source_safety,
            'institutional_sources_provided': institutional_fallbacks,
            'safety_handling_successful': (evidence_safety and len(verification_links) > 0) or 
                                        (source_safety and len(contextual_sources) > 0)
        },
        'credibility_assessment': {
            'llm_bias_score': llm_scores.get('bias', 0),
            'llm_credibility_score': llm_scores.get('credibility', 50),
            'llm_risk_score': llm_scores.get('risk', 50),
            'source_quality_score': source_quality_score,
            'link_quality_score': link_quality_score
        }
    }

    return result

@app.on_event("startup")
async def startup_event():
    """Validate system configuration on startup"""
    logger.info("🚀 Starting Enhanced Fake News Detection API v3.1.0")
    logger.info("🛡️ NEW: Enhanced with safety filter handling and robust error recovery")
    
    # Validate API keys
    if not settings.validate_api_keys():
        logger.warning("⚠️ GEMINI_API_KEY not configured properly")
        logger.info("Set it with: export GEMINI_API_KEY='your_actual_api_key'")
    else:
        logger.info("✅ API keys validated successfully")

    # Log current model configurations
    logger.info("🤖 Enhanced Agent Configurations:")
    try:
        for agent_name in ["bert_classifier", "claim_extractor", "context_analyzer",
                          "evidence_evaluator", "credible_source", "llm_explanation"]:
            config = get_model_config(agent_name)
            model_name = config.get("model_name", "Unknown")
            version = "3.1" if agent_name in ["context_analyzer", "evidence_evaluator", "credible_source"] else "2.0"
            logger.info(f" {agent_name}: {model_name} (v{version})")
    except Exception as e:
        logger.error(f"❌ Error loading model configs: {str(e)}")

    logger.info("🎯 Enhanced Features Enabled:")
    logger.info(" • LLM-driven consistent scoring with safety handling")
    logger.info(" • Specific verification links with institutional fallbacks") 
    logger.info(" • Contextual source recommendations with safety awareness")
    logger.info(" • Smart conditional routing")
    logger.info(" • URL scraping support")
    logger.info(" • Quality validation system with safety metrics")

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Enhanced Fake News Detection API with Safety Handling",
        "status": "running",
        "version": "3.1.0",
        "agents": [
            "bert_classifier",
            "claim_extractor", 
            "context_analyzer",  # Enhanced with LLM scoring + safety
            "evidence_evaluator",  # Enhanced with specific links + safety
            "credible_source",  # Enhanced with contextual recommendations + safety
            "llm_explanation"
        ],
        "enhanced_features": [
            "llm_driven_scoring_with_safety_handling",
            "specific_verification_links_with_fallbacks", 
            "contextual_source_recommendations_with_safety",
            "multi_agent_orchestration",
            "smart_conditional_routing",
            "url_scraping_support", 
            "safety_aware_quality_validation"
        ],
        "supported_inputs": ["raw_text", "news_urls"],
        "api_key_configured": settings.validate_api_keys(),
        "improvements": [
            "Consistent numerical scores that match text explanations",
            "Actual verification URLs with institutional fallbacks when AI is restricted",
            "Context-specific source recommendations with safety handling",
            "Graceful degradation when content triggers safety filters"
        ]
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with enhanced features status"""
    try:
        # Check if workflow is loaded
        workflow_loaded = fake_news_workflow is not None
        
        # Get model configurations for ALL 6 AGENTS
        current_models = {}
        enhanced_agents = {}
        try:
            for agent_name in ["bert_classifier", "claim_extractor", "context_analyzer",
                              "evidence_evaluator", "credible_source", "llm_explanation"]:
                config = get_model_config(agent_name)
                current_models[agent_name] = config.get("model_name", "Unknown")
                # Mark enhanced agents with safety features
                enhanced_agents[agent_name] = agent_name in ["context_analyzer", "evidence_evaluator", "credible_source"]
        except Exception as e:
            current_models = {"error": str(e)}
            enhanced_agents = {}

        return {
            "status": "healthy" if settings.validate_api_keys() and workflow_loaded else "configuration_required",
            "langgraph_loaded": workflow_loaded,
            "url_scraper_loaded": scraper is not None,
            "agents": [
                "bert_classifier",
                "claim_extractor",
                "context_analyzer",  # Enhanced with LLM scoring + safety
                "evidence_evaluator",  # Enhanced with specific links + safety
                "credible_source",  # Enhanced with contextual recommendations + safety
                "llm_explanation"
            ],
            "agents_count": 6,
            "enhanced_agents": enhanced_agents,
            "api_key_configured": settings.validate_api_keys(),
            "current_models": current_models,
            "enhanced_features_enabled": {
                "llm_scoring": True,
                "specific_verification_links": True,
                "contextual_source_recommendations": True,
                "smart_routing": True,
                "url_support": True,
                "quality_validation": True,
                "safety_filter_handling": True,  # ✅ New
                "institutional_fallbacks": True  # ✅ New
            },
            "config_version": "3.1.0"
        }
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_article_endpoint(request: ArticleRequest):
    """
    ✅ ENHANCED ARTICLE ANALYSIS WITH SAFETY-AWARE PROCESSING
    
    Analyzes articles using enhanced agents with:
    - LLM-driven consistent scoring with safety handling
    - Specific verification links with institutional fallbacks
    - Contextual source recommendations with safety awareness
    - Quality validation with safety metrics
    """
    # Validate API configuration
    if not settings.validate_api_keys():
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Please set environment variable."
        )

    start_time = datetime.now()
    try:
        article_text = ""
        article_title = ""
        article_url = ""
        scraping_info = {}

        # Determine input type and extract text
        if request.url:
            # URL provided - scrape the content
            logger.info(f"🔗 Scraping article from URL: {request.url}")
            scrape_result = scraper.scrape_article(request.url)
            
            if not scrape_result['success']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to scrape URL: {scrape_result['error']}"
                )

            article_text = scrape_result['text']
            article_title = scrape_result['title']
            article_url = scrape_result['url']
            scraping_info = {
                "scraping_method": scrape_result.get('method', 'unknown'),
                "scraped_title": article_title,
                "scraped_author": scrape_result.get('author', ''),
                "scraped_date": scrape_result.get('publish_date'),
                "original_url": request.url,
                "scraping_successful": True
            }

            logger.info(f"✅ Successfully scraped: {article_title[:100]}...")

        elif request.text:
            # Check if text field contains a URL
            if is_url(request.text):
                # Auto-detect URL in text field
                logger.info(f"🔗 Auto-detected URL in text field: {request.text}")
                scrape_result = scraper.scrape_article(request.text)
                
                if not scrape_result['success']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to scrape URL: {scrape_result['error']}"
                    )

                article_text = scrape_result['text']
                article_title = scrape_result['title'] 
                article_url = scrape_result['url']
                scraping_info = {
                    "scraping_method": scrape_result.get('method', 'unknown'),
                    "auto_detected_url": True,
                    "scraping_successful": True
                }

            else:
                # Plain text provided
                article_text = request.text
                article_title = "User Provided Text"
                article_url = "N/A"
                scraping_info = {"input_type": "raw_text"}
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'text' or 'url' must be provided"
            )

        # Validate article length
        if len(article_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Article text too short (minimum 50 characters required)"
            )

        if len(article_text) > settings.max_article_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Article text too long. Maximum length: {settings.max_article_length} characters"
            )

        logger.info(f"🔍 Starting enhanced analysis for: {article_title[:100]}...")
        logger.info(f"📄 Article length: {len(article_text)} characters")

        # ✅ USE ENHANCED LANGGRAPH WORKFLOW WITH SAFETY HANDLING
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            analyze_article,
            article_text,
            article_url,
            request.detailed
        )

        # ✅ APPLY ENHANCED QUALITY VALIDATION WITH SAFETY AWARENESS
        result = validate_analysis_quality(result)

        # Calculate processing time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # ✅ SAFELY EXTRACT ENHANCED RESULTS FROM LANGGRAPH STATE
        bert_results = result.get("bert_results") or {}
        extracted_claims = result.get("extracted_claims") or []
        context_analysis = result.get("context_analysis") or {}
        evidence_evaluation = result.get("evidence_evaluation") or {}
        source_recommendations = result.get("source_recommendations") or {}
        final_explanation = result.get("final_explanation") or {}
        quality_validation = result.get("quality_validation", {})

        # ✅ FORMAT ENHANCED RESPONSE WITH SAFETY AWARENESS
        response = AnalysisResponse(
            success=True,
            results={
                "classification": {
                    "prediction": bert_results.get("prediction", "UNKNOWN"),
                    "confidence": bert_results.get("confidence", 0.0),
                    "probabilities": bert_results.get("probabilities", {"real": 0.5, "fake": 0.5})
                },
                "claims": {
                    "extracted_claims": extracted_claims,
                    "total_claims": len(extracted_claims),
                    "high_priority_claims": sum(1 for c in extracted_claims
                                               if isinstance(c, dict) and c.get("priority", 3) <= 2)
                },
                # ✅ ENHANCED CONTEXT ANALYSIS WITH LLM SCORES AND SAFETY
                "context_analysis": {
                    "llm_scores": context_analysis.get("llm_scores", {}),  # LLM-driven scores
                    "llm_analysis": context_analysis.get("llm_analysis", ""),  # LLM analysis text
                    "overall_score": context_analysis.get("context_scores", {}).get("overall_context_score", 5.0),
                    "risk_level": context_analysis.get("context_scores", {}).get("risk_level", "UNKNOWN"), 
                    "bias_detected": context_analysis.get("context_scores", {}).get("bias_score", 0.0),
                    "manipulation_score": context_analysis.get("manipulation_report", {}).get("overall_manipulation_score", 0.0),
                    "credibility": context_analysis.get("context_scores", {}).get("credibility", 50),
                    "scoring_method": context_analysis.get("metadata", {}).get("scoring_method", "traditional"),
                    "safety_handled": context_analysis.get("safety_fallback_used", False)  # ✅ New
                },
                # ✅ ENHANCED EVIDENCE WITH VERIFICATION LINKS AND SAFETY
                "evidence": {
                    "overall_evidence_score": evidence_evaluation.get("evidence_scores", {}).get("overall_evidence_score", 5.0),
                    "quality_level": evidence_evaluation.get("evidence_scores", {}).get("quality_level", "UNKNOWN"),
                    "source_quality_score": evidence_evaluation.get("evidence_scores", {}).get("source_quality_score", 5.0),
                    "verification_links_quality_score": evidence_evaluation.get("evidence_scores", {}).get("verification_links_quality_score", 5.0),
                    "verification_links": evidence_evaluation.get("verification_links", []),  # Specific verification links
                    "safety_fallback_used": evidence_evaluation.get("safety_fallback_used", False),  # ✅ New
                    "sources": [
                        (s.get("name") or s.get("url") or "Unknown Source")
                        for s in source_recommendations.get("recommended_sources", [])
                    ]
                },
                # ✅ ENHANCED SOURCES WITH CONTEXTUAL RECOMMENDATIONS AND SAFETY
                "sources": {
                    "contextual_sources": source_recommendations.get("contextual_sources", []),  # Contextual sources
                    "recommended_count": len(source_recommendations.get("recommended_sources", [])),
                    "recommendation_score": source_recommendations.get("recommendation_scores", {}).get("overall_recommendation_score", 5.0),
                    "contextual_sources_count": source_recommendations.get("recommendation_scores", {}).get("contextual_sources_count", 0),
                    "safety_fallback_used": source_recommendations.get("safety_fallback_used", False),  # ✅ New
                    "top_sources": [
                        {
                            "name": s.get("name") or s.get("source") or s.get("url") or "Source",
                            "url": s.get("url") if isinstance(s, dict) else (s if isinstance(s, str) else None),
                            "type": s.get("type", "unknown"),
                            "relevance_score": s.get("relevance_score", 5),
                            "quality_score": s.get("quality_score", 0.5)  # ✅ New
                        }
                        for s in (source_recommendations.get("contextual_sources", [])[:5])
                    ]
                },
                "explanation": {
                    "text": final_explanation.get("explanation", "No explanation available"),
                    "detailed_analysis": final_explanation.get("detailed_analysis") is not None,
                    "confidence_analysis": final_explanation.get("confidence_analysis") is not None
                },
                # ✅ ENHANCED QUALITY VALIDATION WITH SAFETY METRICS
                "quality_validation": quality_validation
            },
            errors=result.get("processing_errors", []),
            metadata={
                "processing_time_seconds": round(total_time, 2),
                "processing_path": result.get("processing_path", "unknown"),
                "article_length": len(article_text),
                "article_title": article_title,
                "article_url": article_url,
                "detailed_analysis": request.detailed,
                "agent_times": result.get("processing_times", {}),
                "confidence_scores": result.get("confidence_scores", {}),
                "agents_used": 6,  # All 6 agents available
                "scraping_info": scraping_info,
                "input_type": "url" if (request.url or is_url(request.text or "")) else "text",
                "api_version": "3.1.0",
                # ✅ ENHANCED FEATURES METADATA WITH SAFETY
                "enhanced_features_used": {
                    "llm_scoring": bool(context_analysis.get("llm_scores")),
                    "specific_verification_links": len(evidence_evaluation.get("verification_links", [])) > 0,
                    "contextual_source_recommendations": len(source_recommendations.get("contextual_sources", [])) > 0,
                    "safety_fallbacks_used": quality_validation.get("safety_analysis", {}).get("safety_handling_successful", False)  # ✅ New
                },
                "smart_routing_enabled": True,
                "url_support_enabled": True,
                "quality_validation_enabled": True,
                "safety_handling_enabled": True,  # ✅ New
                "config_version": "enhanced_v3.1",
                "timestamp": start_time.isoformat(),
                # ✅ ENHANCED QUALITY METRICS WITH SAFETY
                "quality_score": quality_validation.get("quality_score", 100),
                "requires_human_review": quality_validation.get("requires_human_review", False),
                "quality_issues": quality_validation.get("issues_detected", []),
                "safety_blocks_encountered": quality_validation.get("safety_analysis", {}).get("safety_blocks_encountered", 0),  # ✅ New
                "enhanced_features_available": quality_validation.get("enhanced_features", {})
            }
        )

        # ✅ ENHANCED LOGGING WITH SAFETY INFORMATION
        logger.info(f"✅ Enhanced analysis completed in {total_time:.2f} seconds")
        logger.info(f"🎯 Processing path: {result.get('processing_path', 'unknown')}")
        logger.info(f"🤖 Enhanced agents used: {list(quality_validation.get('enhanced_features', {}).keys())}")
        logger.info(f"🔍 Quality score: {quality_validation.get('quality_score', 100)}/100")
        logger.info(f"🔗 Verification links: {len(evidence_evaluation.get('verification_links', []))}")
        logger.info(f"🎯 Contextual sources: {len(source_recommendations.get('contextual_sources', []))}")
        logger.info(f"🛡️ Safety blocks encountered: {quality_validation.get('safety_analysis', {}).get('safety_blocks_encountered', 0)}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Enhanced analysis failed after {error_time:.2f} seconds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced analysis failed: {str(e)}"
        )

@app.get("/config/models", response_model=ConfigResponse) 
async def get_model_configurations():
    """Get current model configurations for ALL 6 enhanced agents with safety features"""
    try:
        # Get all agent configurations
        agent_configs = {}
        current_models = {}
        
        for agent_name in ["bert_classifier", "claim_extractor", "context_analyzer",
                          "evidence_evaluator", "credible_source", "llm_explanation"]:
            config = get_model_config(agent_name)
            
            # Mark enhanced features including safety
            enhanced_features = []
            if agent_name == "context_analyzer":
                enhanced_features = ["llm_driven_scoring", "consistent_scoring_validation", "safety_handling"]
            elif agent_name == "evidence_evaluator":
                enhanced_features = ["specific_verification_links", "link_quality_validation", "institutional_fallbacks", "safety_handling"]
            elif agent_name == "credible_source":
                enhanced_features = ["contextual_source_recommendations", "claim_specific_guidance", "safety_aware_sources", "safety_handling"]

            agent_configs[agent_name] = {
                "model_name": config.get("model_name", "Unknown"),
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens"),
                "enabled_features": [k for k, v in config.items() if isinstance(v, bool) and v],
                "enhanced_features": enhanced_features,
                "version": "3.1" if enhanced_features else "2.0",
                "safety_enhanced": "safety_handling" in enhanced_features  # ✅ New
            }

            current_models[agent_name] = config.get("model_name", "Unknown")

        return ConfigResponse(
            agent_configs=agent_configs,
            api_key_configured=settings.validate_api_keys(),
            current_models=current_models
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configurations: {str(e)}")

@app.post("/config/update-model")
async def update_model_version(agent_name: str, model_name: str):
    """Update model version for specific agent"""
    try:
        from config.model_configs import update_model_config

        # Validate agent name - ALL 6 AGENTS
        valid_agents = ["bert_classifier", "claim_extractor", "context_analyzer",
                       "evidence_evaluator", "credible_source", "llm_explanation"]
        
        if agent_name not in valid_agents:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent name. Valid agents: {valid_agents}"
            )

        # Update configuration
        update_model_config(agent_name, model_name=model_name)
        logger.info(f"📝 Updated {agent_name} model to {model_name}")

        return {
            "success": True,
            "message": f"Updated {agent_name} to use {model_name}",
            "agent_name": agent_name,
            "new_model": model_name,
            "enhanced_features": agent_name in ["context_analyzer", "evidence_evaluator", "credible_source"],
            "safety_enhanced": agent_name in ["context_analyzer", "evidence_evaluator", "credible_source"],  # ✅ New
            "restart_required": True,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Model update failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/test-scraping")
async def test_url_scraping(url: str):
    """Test endpoint to verify URL scraping functionality"""
    try:
        logger.info(f"🧪 Testing URL scraping: {url}")
        scrape_result = scraper.scrape_article(url)
        
        return {
            "success": scrape_result['success'],
            "url": url,
            "title": scrape_result.get('title', '')[:200],
            "text_preview": scrape_result.get('text', '')[:500] + "..." if scrape_result.get('text') else "",
            "method": scrape_result.get('method', 'unknown'),
            "error": scrape_result.get('error') if not scrape_result['success'] else None,
            "text_length": len(scrape_result.get('text', '')),
            "author": scrape_result.get('author', ''),
            "publish_date": scrape_result.get('publish_date')
        }

    except Exception as e:
        logger.error(f"❌ Scraping test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scraping test failed: {str(e)}")

@app.get("/metrics")
async def get_system_metrics():
    """Get enhanced system performance metrics with safety information"""
    return {
        "system_status": "operational",
        "agents_count": 6,
        "agents": [
            "bert_classifier",
            "claim_extractor",
            "context_analyzer",  # Enhanced with LLM scoring + safety
            "evidence_evaluator",  # Enhanced with specific links + safety
            "credible_source",  # Enhanced with contextual recommendations + safety
            "llm_explanation"
        ],
        "enhanced_features": [
            "llm_driven_consistent_scoring_with_safety",
            "specific_verification_links_with_institutional_fallbacks",
            "contextual_source_recommendations_with_safety_awareness",
            "smart_routing_enabled",
            "url_scraping_support",
            "safety_aware_quality_validation"
        ],
        "safety_features": [  # ✅ New section
            "gemini_safety_filter_detection",
            "institutional_fallback_generation", 
            "graceful_degradation_on_content_restrictions",
            "safety_aware_quality_scoring",
            "transparent_safety_reporting"
        ],
        "supported_sites": [
            "timesofindia.indiatimes.com",
            "cnn.com", 
            "bbc.com",
            "reuters.com",
            "Most news websites"
        ],
        "processing_modes": ["fast_track", "full_enhanced_analysis_with_safety"],
        "average_processing_time": "45-75 seconds",
        "enhancement_benefits": [
            "Consistent scoring (no more contradictions)",
            "Actual verification URLs with institutional fallbacks when restricted",
            "Contextual source recommendations with safety handling",
            "Graceful degradation when content triggers safety filters"
        ],
        "quality_validation_enabled": True,
        "safety_handling_enabled": True,  # ✅ New
        "api_version": "3.1.0",
        "uptime": "operational"
    }

@app.get("/features")
async def get_enhanced_features():
    """Get detailed information about enhanced features including safety handling"""
    return {
        "version": "3.1.0",
        "enhanced_agents": {
            "context_analyzer": {
                "enhancement": "LLM-driven scoring with safety handling",
                "benefits": [
                    "Consistent numerical scores that match text explanations",
                    "No more contradictions between analysis text and scores",
                    "0-100 scale scoring for better interpretation",
                    "Graceful handling when content triggers safety filters"
                ],
                "safety_features": ["fallback_scoring", "content_sensitivity_detection"],
                "before": "Manual score estimation often contradicted LLM analysis text",
                "after": "LLM generates both analysis text AND numerical scores consistently with safety awareness"
            },
            "evidence_evaluator": {
                "enhancement": "Specific verification links with institutional fallbacks",
                "benefits": [
                    "Actual verification URLs instead of generic Google searches",
                    "Quality-validated links with relevance scores",
                    "Claim-specific verification sources",
                    "High-quality institutional fallbacks when AI is restricted"
                ],
                "safety_features": ["institutional_fallbacks", "domain_aware_sources", "quality_preservation"],
                "before": "Generic 'google.com/search?q=...' links that were often broken",
                "after": "Specific URLs like 'https://www.nejm.org/doi/full/10.1056/...' with institutional fallbacks"
            },
            "credible_source": {
                "enhancement": "Contextual source recommendations with safety awareness", 
                "benefits": [
                    "Sources relevant to specific claims in the article",
                    "Expert contacts and institutional sources",
                    "Specific search strategies for each claim type",
                    "Domain-appropriate institutional fallbacks when needed"
                ],
                "safety_features": ["contextual_fallbacks", "institutional_sources", "domain_classification"],
                "before": "Generic 'check FactCheck.org' recommendations",
                "after": "Contextual recommendations with institutional fallbacks when AI analysis is restricted"
            }
        },
        "safety_handling": {  # ✅ New section
            "detection": "Automatic detection of Gemini safety filter blocks (finish_reason: 2)",
            "fallback_strategies": [
                "Institutional verification sources for evidence evaluation",
                "Domain-appropriate source recommendations", 
                "High-quality fact-checking organizations",
                "Government and academic institution contacts"
            ],
            "quality_preservation": "Safety fallbacks maintain system functionality and quality",
            "transparency": "Clear indication when safety fallbacks are used"
        },
        "quality_improvements": [
            "Score consistency validation with safety awareness",
            "Link quality assessment with fallback scoring",
            "Source relevance scoring with safety handling",
            "Comprehensive quality metrics including safety information"
        ],
        "backward_compatibility": "Fully compatible with existing frontend code",
        "new_in_3_1": [
            "Safety filter detection and handling",
            "Institutional fallback generation",
            "Safety-aware quality scoring",
            "Transparent safety reporting in API responses"
        ]
    }

# ✅ NEW SAFETY STATUS ENDPOINT
@app.get("/safety-status")
async def get_safety_status():
    """Get current safety handling status and statistics"""
    return {
        "safety_handling_enabled": True,
        "safety_features": {
            "gemini_safety_filter_detection": True,
            "institutional_fallback_generation": True,
            "graceful_degradation": True,
            "transparent_reporting": True
        },
        "fallback_types": [
            "institutional_verification_sources",
            "domain_specific_recommendations", 
            "government_agency_contacts",
            "academic_institution_sources",
            "professional_fact_checkers"
        ],
        "safety_thresholds": {
            "max_safety_blocks_per_analysis": 3,
            "quality_score_adjustment_for_fallbacks": "+3 bonus points",
            "human_review_trigger": "safety_blocks > 2 OR quality_issues > 4"
        },
        "supported_content_types": [
            "general_news_articles",
            "health_medical_claims",
            "scientific_research_claims", 
            "government_policy_claims",
            "statistical_data_claims"
        ],
        "api_version": "3.1.0"
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors"""
    logger.error(f"🚨 Unexpected error in enhanced API with safety handling: {str(exc)}")
    return {
        "success": False,
        "error": "Internal server error",
        "message": "An unexpected error occurred in the enhanced analysis system. Safety handling is active.",
        "timestamp": datetime.now().isoformat(),
        "api_version": "3.1.0",
        "safety_handling_enabled": True
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
