"""
FastAPI Server for Fake News Detection with LangGraph Orchestration

Enhanced with centralized configuration, smart routing, and URL scraping support

NOW WITH ALL 6 AGENTS INCLUDING URL PROCESSING + QUALITY VALIDATION
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
    title="Smart Fake News Detection API with URL Support + Quality Validation",
    description="Multi-agent fake news detection using LangGraph orchestration - ALL 6 AGENTS + URL SCRAPING + QUALITY VALIDATION",
    version="2.1.1"
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
    """Validate analysis quality before returning to frontend"""
    issues = []
    
    # Check credibility scoring - High context score means MORE bias/manipulation (bad)
    context_scores = result.get('context_analysis', {}).get('context_scores', {})
    overall_context_score = context_scores.get('overall_context_score', 5.0)
    credibility_score = context_scores.get('credibility', 50)  # Default middle value
    
    # High context score (>7) indicates high bias/manipulation = low credibility
    if overall_context_score > 7.0 or credibility_score < 30:
        issues.append("low_credibility_detected")
    
    # Check evidence specificity
    evidence_scores = result.get('evidence_evaluation', {}).get('evidence_scores', {})
    source_quality_score = evidence_scores.get('source_quality_score', 5.0)
    
    if source_quality_score < 4.0:
        issues.append("poor_evidence_sources")
    
    # Check explanation formatting
    explanation = result.get('final_explanation', {}).get('explanation', '')
    if not ('##' in explanation or '**' in explanation):
        issues.append("unformatted_explanation")
    
    # Check for verification links
    evidence_eval = result.get('evidence_evaluation', {})
    verification_links = evidence_eval.get('verification_links', [])
    if len(verification_links) < 2:
        issues.append("insufficient_verification_links")
    
    # Add quality metadata
    result['quality_validation'] = {
        'issues_detected': issues,
        'quality_score': max(0, 100 - len(issues) * 20),
        'requires_human_review': len(issues) > 2,
        'credibility_assessment': {
            'overall_context_score': overall_context_score,
            'credibility_score': credibility_score,
            'source_quality_score': source_quality_score,
            'verification_links_count': len(verification_links)
        }
    }
    
    return result

@app.on_event("startup")
async def startup_event():
    """Validate system configuration on startup"""
    logger.info("🚀 Starting Fake News Detection API v2.1.1 with ALL 6 AGENTS + URL SUPPORT + QUALITY VALIDATION")
    
    # Validate API keys
    if not settings.validate_api_keys():
        logger.warning("⚠️ GEMINI_API_KEY not configured properly")
        logger.info("Set it with: export GEMINI_API_KEY='your_actual_api_key'")
    else:
        logger.info("✅ API keys validated successfully")
    
    # Log current model configurations
    logger.info("🤖 Current Model Configurations (ALL 6 AGENTS):")
    try:
        for agent_name in ["bert_classifier", "claim_extractor", "context_analyzer",
                          "evidence_evaluator", "credible_source", "llm_explanation"]:
            config = get_model_config(agent_name)
            model_name = config.get("model_name", "Unknown")
            logger.info(f"  {agent_name}: {model_name}")
    except Exception as e:
        logger.error(f"❌ Error loading model configs: {str(e)}")
    
    logger.info("🎯 Smart conditional routing enabled for cost optimization")
    logger.info("🔍 Context analyzer (6th agent) integrated into pipeline")
    logger.info("🔗 URL scraping support enabled for major news sites")
    logger.info("✅ Quality validation system enabled")

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Smart Fake News Detection API with LangGraph - ALL 6 AGENTS + URL SUPPORT + QUALITY VALIDATION",
        "status": "running",
        "version": "2.1.1",
        "agents": [
            "bert_classifier",
            "claim_extractor", 
            "context_analyzer",  # The 6th agent
            "evidence_evaluator",
            "credible_source",
            "llm_explanation"
        ],
        "features": [
            "multi_agent_orchestration",
            "smart_conditional_routing",
            "url_scraping_support",
            "times_of_india_support",
            "cost_optimization",
            "centralized_configuration",
            "context_analysis_integration",
            "quality_validation_system"  # NEW
        ],
        "supported_inputs": ["raw_text", "news_urls", "times_of_india_urls"],
        "api_key_configured": settings.validate_api_keys()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with configuration status"""
    try:
        # Check if workflow is loaded
        workflow_loaded = fake_news_workflow is not None
        
        # Get model configurations for ALL 6 AGENTS
        current_models = {}
        try:
            for agent_name in ["bert_classifier", "claim_extractor", "context_analyzer",
                              "evidence_evaluator", "credible_source", "llm_explanation"]:
                config = get_model_config(agent_name)
                current_models[agent_name] = config.get("model_name", "Unknown")
        except Exception as e:
            current_models = {"error": str(e)}
        
        return {
            "status": "healthy" if settings.validate_api_keys() and workflow_loaded else "configuration_required",
            "langgraph_loaded": workflow_loaded,
            "url_scraper_loaded": scraper is not None,
            "agents": [
                "bert_classifier",
                "claim_extractor",
                "context_analyzer",  # The 6th agent
                "evidence_evaluator", 
                "credible_source",
                "llm_explanation"
            ],
            "agents_count": 6,  # Now correctly showing 6 agents
            "api_key_configured": settings.validate_api_keys(),
            "current_models": current_models,
            "smart_routing_enabled": True,
            "url_support_enabled": True,
            "quality_validation_enabled": True,  # NEW
            "config_version": "2.1.1"
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
    Analyze article for fake news with smart LangGraph orchestration
    
    NOW SUPPORTS BOTH TEXT AND URL INPUTS - INCLUDING TIMES OF INDIA + QUALITY VALIDATION
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
        
        logger.info(f"🔍 Starting analysis for: {article_title[:100]}...")
        logger.info(f"📄 Article length: {len(article_text)} characters")
        
        # Use smart LangGraph workflow with ALL 6 AGENTS
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            analyze_article,
            article_text,
            article_url,
            request.detailed
        )
        
        # ✅ APPLY QUALITY VALIDATION
        result = validate_analysis_quality(result)
        
        # Calculate processing time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Safely extract results from LangGraph state
        bert_results = result.get("bert_results") or {}
        extracted_claims = result.get("extracted_claims") or []
        context_analysis = result.get("context_analysis") or {}
        evidence_evaluation = result.get("evidence_evaluation") or {}
        source_recommendations = result.get("source_recommendations") or {}
        final_explanation = result.get("final_explanation") or {}
        quality_validation = result.get("quality_validation", {})
        
        # Format response with safe extraction + quality validation
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
                "context_analysis": {
                    "overall_score": context_analysis.get("context_scores", {}).get("overall_context_score", 5.0),
                    "risk_level": context_analysis.get("context_scores", {}).get("risk_level", "UNKNOWN"),
                    "bias_detected": context_analysis.get("context_scores", {}).get("bias_score", 0.0),
                    "manipulation_score": context_analysis.get("manipulation_report", {}).get("overall_manipulation_score", 0.0),
                    "credibility": context_analysis.get("context_scores", {}).get("credibility", 50)
                },
                "evidence": {
                    "overall_score": evidence_evaluation.get("evidence_scores", {}).get("overall_evidence_score", 5.0),
                    "quality_level": evidence_evaluation.get("evidence_scores", {}).get("quality_level", "UNKNOWN"),
                    "source_quality": evidence_evaluation.get("evidence_scores", {}).get("source_quality_score", 5.0),
                    "verification_links": evidence_evaluation.get("verification_links", []),
                    "sources": [
                        (s.get("name") or s.get("url") or "Unknown Source")
                        for s in source_recommendations.get("recommended_sources", [])
                    ]
                },
                "sources": {
                    "recommended_count": len(source_recommendations.get("recommended_sources", [])),
                    "recommendation_score": source_recommendations.get("recommendation_scores", {}).get("overall_recommendation_score", 5.0),
                    "top_sources": [
                        {
                            "name": s.get("name") or s.get("source") or s.get("url") or "Source",
                            "url": s.get("url") if isinstance(s, dict) else (s if isinstance(s, str) else None)
                        }
                        for s in (source_recommendations.get("recommended_sources", [])[:3])
                    ]
                },
                "explanation": {
                    "text": final_explanation.get("explanation", "No explanation available"),
                    "detailed_analysis": final_explanation.get("detailed_analysis") is not None,
                    "confidence_analysis": final_explanation.get("confidence_analysis") is not None
                },
                # ✅ ADD QUALITY VALIDATION RESULTS
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
                "api_version": "2.1.1",
                "smart_routing_enabled": True,
                "url_support_enabled": True,
                "quality_validation_enabled": True,  # NEW
                "config_version": "centralized",
                "timestamp": start_time.isoformat(),
                # ✅ ADD QUALITY METRICS TO METADATA
                "quality_score": quality_validation.get("quality_score", 100),
                "requires_human_review": quality_validation.get("requires_human_review", False),
                "quality_issues": quality_validation.get("issues_detected", [])
            }
        )
        
        logger.info(f"✅ Analysis completed in {total_time:.2f} seconds")
        logger.info(f"🎯 Processing path: {result.get('processing_path', 'unknown')}")
        logger.info(f"🤖 Agents used: 6 (including context_analyzer)")
        logger.info(f"🔍 Quality score: {quality_validation.get('quality_score', 100)}/100")
        logger.info(f"⚠️ Quality issues: {len(quality_validation.get('issues_detected', []))}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Analysis failed after {error_time:.2f} seconds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/config/models", response_model=ConfigResponse)
async def get_model_configurations():
    """Get current model configurations for ALL 6 agents"""
    try:
        # Get all agent configurations
        agent_configs = {}
        current_models = {}
        
        for agent_name in ["bert_classifier", "claim_extractor", "context_analyzer",
                          "evidence_evaluator", "credible_source", "llm_explanation"]:
            config = get_model_config(agent_name)
            agent_configs[agent_name] = {
                "model_name": config.get("model_name", "Unknown"),
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens"),
                "enabled_features": [k for k, v in config.items() if isinstance(v, bool) and v]
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
        
        # Validate agent name - NOW INCLUDING CONTEXT ANALYZER
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
    """Get system performance metrics"""
    return {
        "system_status": "operational",
        "agents_count": 6,  # All 6 agents
        "agents": [
            "bert_classifier",
            "claim_extractor",
            "context_analyzer",  # The 6th agent
            "evidence_evaluator",
            "credible_source",
            "llm_explanation"
        ],
        "features": [
            "smart_routing_enabled",
            "url_scraping_support",
            "times_of_india_support",
            "cost_optimization_active",
            "quality_validation_system"  # NEW
        ],
        "supported_sites": [
            "timesofindia.indiatimes.com",
            "cnn.com",
            "bbc.com",
            "reuters.com",
            "Most news websites"
        ],
        "processing_modes": ["fast_track", "full_analysis"],
        "average_processing_time": "40-60 seconds",
        "cost_savings_estimate": "60-80% for obvious cases",
        "quality_validation_enabled": True,
        "api_version": "2.1.1",
        "uptime": "operational"
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors"""
    logger.error(f"🚨 Unexpected error: {str(exc)}")
    return {
        "success": False,
        "error": "Internal server error", 
        "message": "An unexpected error occurred. Please try again.",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
