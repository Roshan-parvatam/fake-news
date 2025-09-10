"""
FastAPI Server for Fake News Detection with Enhanced Safety Handling

Production-ready version with security improvements, proper error handling,
and integration with refactored agents.

Version 3.2.0 - Production Enhanced Edition
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables first
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware

# Import your configuration system
from config.settings import get_settings
from config.model_configs import get_model_config

# Import LangGraph workflow and state
from orchestration.langgraph_workflow import fake_news_workflow, analyze_article

# Import URL scraper
from utils.url_scraper import ProductionNewsScraper

# Import enhanced exception classes for better error handling
try:
    from agents.claim_extractor.exceptions import ClaimExtractorError
    from agents.llm_explanation.exceptions import LLMExplanationError
    from agents.context_analyzer.exceptions import ContextAnalyzerError
except ImportError as e:
    logging.warning(f"Could not import enhanced exceptions: {e}")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Initialize global settings
settings = get_settings()

# Create FastAPI app with dynamic configuration
app = FastAPI(
    title="Enhanced Fake News Detection API",
    description="Production-ready multi-agent fake news detection with safety handling",
    version="3.2.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,  # Hide docs in production
    redoc_url="/redoc" if ENVIRONMENT == "development" else None
)

# Production-safe CORS configuration
def setup_cors_middleware():
    """Configure CORS middleware based on environment"""
    if ENVIRONMENT == "production":
        # Production - restrict to specific origins
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        if not allowed_origins or allowed_origins == [""]:
            logger.warning("No ALLOWED_ORIGINS set for production environment")
            allowed_origins = []
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
        logger.info(f"Production CORS configured for origins: {allowed_origins}")
    else:
        # Development - more permissive but still restricted
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",  # React dev server
                "http://localhost:8080",  # Vue dev server
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080",
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
        logger.info("Development CORS configured")

# Setup CORS
setup_cors_middleware()

# Initialize URL scraper
scraper = ProductionNewsScraper()

# Enhanced Request/Response models
class ArticleRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
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

class HealthResponse(BaseModel):
    status: str
    agents_health: Dict[str, Any]
    api_key_configured: bool
    environment: str
    version: str

# Enhanced Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code
            },
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Enhanced exception handlers for refactored agents
try:
    @app.exception_handler(ClaimExtractorError)
    async def claim_extractor_exception_handler(request: Request, exc: ClaimExtractorError):
        logger.error(f"Claim extraction error: {exc.message}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "type": "claim_extraction_error",
                    "message": exc.message,
                    "error_code": exc.error_code,
                    "details": exc.details
                },
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.exception_handler(LLMExplanationError)
    async def llm_explanation_exception_handler(request: Request, exc: LLMExplanationError):
        logger.error(f"LLM explanation error: {exc.message}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "type": "llm_explanation_error",
                    "message": exc.message,
                    "error_code": exc.error_code,
                    "details": exc.details
                },
                "timestamp": datetime.now().isoformat()
            }
        )
except NameError:
    logger.info("Enhanced exception handlers not available - using fallback")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors"""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "internal_server_error",
                "message": "An unexpected error occurred",
                "details": str(exc) if ENVIRONMENT == "development" else "Contact support"
            },
            "timestamp": datetime.now().isoformat(),
            "environment": ENVIRONMENT
        }
    )

# Utility functions
def is_url(text: str) -> bool:
    """Simple URL detection"""
    if not text:
        return False
    return text.strip().lower().startswith(('http://', 'https://'))

def validate_analysis_quality(result: dict) -> dict:
    """Enhanced quality validation with safety awareness"""
    issues = []

    # Check LLM scoring consistency
    context_analysis = result.get('context_analysis', {})
    llm_scores = context_analysis.get('llm_scores', {})
    
    if llm_scores:
        bias_score = llm_scores.get('bias', 50)
        credibility_score = llm_scores.get('credibility', 50)
        risk_score = llm_scores.get('risk', 50)
        
        if bias_score > 70 and credibility_score > 70:
            issues.append("inconsistent_llm_scoring")
        if risk_score > 80 and credibility_score > 60:
            issues.append("risk_credibility_mismatch")
    else:
        issues.append("missing_llm_scores")

    # Check verification link quality
    evidence_eval = result.get('evidence_evaluation', {})
    verification_links = evidence_eval.get('verification_links', [])
    
    if len(verification_links) == 0:
        issues.append("no_verification_links")
    elif len(verification_links) < 2:
        issues.append("insufficient_verification_links")

    specific_links = sum(1 for link in verification_links
                        if isinstance(link, dict) and link.get('quality_score', 0) > 0.7)
    institutional_fallbacks = sum(1 for link in verification_links
                                 if isinstance(link, dict) and 'institutional' in link.get('type', ''))

    if specific_links == 0 and institutional_fallbacks == 0 and len(verification_links) > 0:
        issues.append("only_generic_verification_links")

    evidence_safety = evidence_eval.get('safety_fallback_used', False)

    # Check contextual source recommendations
    source_recommendations = result.get('source_recommendations', {})
    contextual_sources = source_recommendations.get('contextual_sources', [])
    
    if len(contextual_sources) == 0:
        issues.append("no_contextual_sources")

    contextual_count = sum(1 for source in contextual_sources
                          if isinstance(source, dict) and source.get('type', '').startswith('contextual'))
    source_safety = source_recommendations.get('safety_fallback_used', False)

    if contextual_count == 0 and not source_safety and len(contextual_sources) > 0:
        issues.append("only_generic_source_recommendations")

    # Calculate quality score
    base_score = 100
    penalty_per_issue = 12
    quality_score = max(0, base_score - len(issues) * penalty_per_issue)

    # Bonus for enhanced features
    if llm_scores:
        quality_score += 5
    if specific_links > 0 or institutional_fallbacks > 0:
        quality_score += 5
    if contextual_count > 0:
        quality_score += 5

    # Safety handling bonuses
    if evidence_safety and len(verification_links) > 0:
        quality_score += 3
    if source_safety and len(contextual_sources) > 0:
        quality_score += 3

    quality_score = min(100, quality_score)

    # Count safety blocks
    safety_blocks_total = 0
    if evidence_eval.get('metadata', {}).get('safety_blocks_encountered', 0):
        safety_blocks_total += evidence_eval['metadata']['safety_blocks_encountered']
    if source_recommendations.get('metadata', {}).get('safety_blocks_encountered', 0):
        safety_blocks_total += source_recommendations['metadata']['safety_blocks_encountered']

    # Add quality metadata
    result['quality_validation'] = {
        'issues_detected': issues,
        'quality_score': quality_score,
        'requires_human_review': len(issues) > 4 or safety_blocks_total > 2,
        'enhanced_features': {
            'llm_scoring_enabled': bool(llm_scores),
            'specific_links_available': specific_links > 0,
            'institutional_fallbacks_used': institutional_fallbacks > 0,
            'contextual_sources_available': contextual_count > 0,
            'verification_links_count': len(verification_links),
            'contextual_sources_count': contextual_count
        },
        'safety_analysis': {
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
            'source_quality_score': evidence_eval.get('evidence_scores', {}).get('source_quality_score', 5.0),
            'link_quality_score': evidence_eval.get('evidence_scores', {}).get('verification_links_quality_score', 5.0)
        }
    }

    return result

# Startup event
@app.on_event("startup")
async def startup_event():
    """Enhanced startup validation"""
    logger.info("🚀 Starting Enhanced Fake News Detection API v3.2.0")
    logger.info(f"🌍 Environment: {ENVIRONMENT}")
    
    # Validate API keys
    if not settings.validate_api_keys():
        logger.warning("⚠️ GEMINI_API_KEY not configured properly")
        logger.info("Set it with: export GEMINI_API_KEY='your_actual_api_key'")
    else:
        logger.info("✅ API keys validated successfully")

    # Test agent imports and initialization
    agent_status = {}
    agent_names = ["bert_classifier", "claim_extractor", "context_analyzer",
                   "evidence_evaluator", "credible_source", "llm_explanation"]
    
    for agent_name in agent_names:
        try:
            config = get_model_config(agent_name)
            model_name = config.get("model_name", "Unknown")
            version = "3.2" if agent_name in ["context_analyzer", "evidence_evaluator", "credible_source"] else "2.0"
            agent_status[agent_name] = {"status": "configured", "model": model_name, "version": version}
            logger.info(f"  {agent_name}: {model_name} (v{version})")
        except Exception as e:
            agent_status[agent_name] = {"status": "error", "error": str(e)}
            logger.error(f"  {agent_name}: Configuration error - {str(e)}")

    logger.info("🎯 Enhanced Features Enabled:")
    logger.info("  • LLM-driven consistent scoring with safety handling")
    logger.info("  • Specific verification links with institutional fallbacks")
    logger.info("  • Contextual source recommendations with safety awareness")
    logger.info("  • Production-ready error handling and logging")
    logger.info("  • Environment-aware CORS configuration")
    logger.info("  • Enhanced quality validation with safety metrics")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Enhanced Fake News Detection API",
        "status": "running",
        "version": "3.2.0",
        "environment": ENVIRONMENT,
        "agents": [
            "bert_classifier",
            "claim_extractor", 
            "context_analyzer",
            "evidence_evaluator",
            "credible_source",
            "llm_explanation"
        ],
        "enhanced_features": [
            "production_ready_architecture",
            "enhanced_error_handling",
            "environment_aware_configuration",
            "security_hardened_cors",
            "structured_logging",
            "safety_aware_quality_validation"
        ],
        "supported_inputs": ["raw_text", "news_urls"],
        "api_key_configured": settings.validate_api_keys()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with agent validation"""
    try:
        # Test workflow availability
        workflow_loaded = fake_news_workflow is not None
        
        # Test agent configurations
        agent_health = {}
        for agent_name in ["bert_classifier", "claim_extractor", "context_analyzer",
                          "evidence_evaluator", "credible_source", "llm_explanation"]:
            try:
                config = get_model_config(agent_name)
                agent_health[agent_name] = {
                    "status": "healthy",
                    "model": config.get("model_name", "Unknown"),
                    "enhanced": agent_name in ["context_analyzer", "evidence_evaluator", "credible_source"]
                }
            except Exception as e:
                agent_health[agent_name] = {"status": "error", "error": str(e)}

        overall_status = "healthy" if (
            settings.validate_api_keys() and 
            workflow_loaded and 
            all(agent.get("status") == "healthy" for agent in agent_health.values())
        ) else "degraded"

        return HealthResponse(
            status=overall_status,
            agents_health=agent_health,
            api_key_configured=settings.validate_api_keys(),
            environment=ENVIRONMENT,
            version="3.2.0"
        )

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_article_endpoint(request: ArticleRequest):
    """Enhanced article analysis with comprehensive error handling"""
    
    # Validate API configuration
    if not settings.validate_api_keys():
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Please set environment variable."
        )

    start_time = datetime.now()
    
    try:
        # Process input and extract text
        article_text = ""
        article_title = ""
        article_url = ""
        scraping_info = {}

        if request.url:
            logger.info(f"Scraping article from URL: {request.url}")
            scrape_result = scraper.scrape_article(request.url)
            
            if not scrape_result['success']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to scrape URL: {scrape_result.get('error', 'Unknown error')}"
                )

            article_text = scrape_result['text']
            article_title = scrape_result['title']
            article_url = scrape_result['url']
            scraping_info = {
                "scraping_method": scrape_result.get('method', 'unknown'),
                "scraped_title": article_title,
                "original_url": request.url,
                "scraping_successful": True
            }

        elif request.text:
            if is_url(request.text):
                logger.info(f"Auto-detected URL in text field: {request.text}")
                scrape_result = scraper.scrape_article(request.text)
                
                if not scrape_result['success']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to scrape URL: {scrape_result.get('error', 'Unknown error')}"
                    )

                article_text = scrape_result['text']
                article_title = scrape_result['title']
                article_url = scrape_result['url']
                scraping_info = {"auto_detected_url": True, "scraping_successful": True}
            else:
                article_text = request.text
                article_title = "User Provided Text"
                article_url = "N/A"
                scraping_info = {"input_type": "raw_text"}

        # Validate article content
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

        logger.info(f"Starting analysis for: {article_title[:100]}...")
        
        # Run LangGraph workflow
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            analyze_article,
            article_text,
            article_url,
            request.detailed
        )

        # Apply quality validation
        result = validate_analysis_quality(result)

        # Calculate processing time
        total_time = (datetime.now() - start_time).total_seconds()

        # Extract results safely
        bert_results = result.get("bert_results") or {}
        extracted_claims = result.get("extracted_claims") or []
        context_analysis = result.get("context_analysis") or {}
        evidence_evaluation = result.get("evidence_evaluation") or {}
        source_recommendations = result.get("source_recommendations") or {}
        final_explanation = result.get("final_explanation") or {}
        quality_validation = result.get("quality_validation", {})

        # Format response
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
                    "llm_scores": context_analysis.get("llm_scores", {}),
                    "llm_analysis": context_analysis.get("llm_analysis", ""),
                    "overall_score": context_analysis.get("context_scores", {}).get("overall_context_score", 5.0),
                    "risk_level": context_analysis.get("context_scores", {}).get("risk_level", "UNKNOWN"),
                    "safety_handled": context_analysis.get("safety_fallback_used", False)
                },
                "evidence": {
                    "overall_evidence_score": evidence_evaluation.get("evidence_scores", {}).get("overall_evidence_score", 5.0),
                    "verification_links": evidence_evaluation.get("verification_links", []),
                    "safety_fallback_used": evidence_evaluation.get("safety_fallback_used", False)
                },
                "sources": {
                    "contextual_sources": source_recommendations.get("contextual_sources", []),
                    "safety_fallback_used": source_recommendations.get("safety_fallback_used", False)
                },
                "explanation": {
                    "text": final_explanation.get("explanation", "No explanation available"),
                    "detailed_analysis": final_explanation.get("detailed_analysis") is not None
                },
                "quality_validation": quality_validation
            },
            errors=result.get("processing_errors", []),
            metadata={
                "processing_time_seconds": round(total_time, 2),
                "article_length": len(article_text),
                "article_title": article_title,
                "article_url": article_url,
                "environment": ENVIRONMENT,
                "api_version": "3.2.0",
                "scraping_info": scraping_info,
                "quality_score": quality_validation.get("quality_score", 100),
                "safety_blocks_encountered": quality_validation.get("safety_analysis", {}).get("safety_blocks_encountered", 0)
            }
        )

        logger.info(f"Analysis completed in {total_time:.2f} seconds")
        return response

    except HTTPException:
        raise
    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Analysis failed after {error_time:.2f} seconds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/config/models", response_model=ConfigResponse)
async def get_model_configurations():
    """Get current model configurations"""
    try:
        agent_configs = {}
        current_models = {}
        
        for agent_name in ["bert_classifier", "claim_extractor", "context_analyzer",
                          "evidence_evaluator", "credible_source", "llm_explanation"]:
            config = get_model_config(agent_name)
            
            enhanced_features = []
            if agent_name == "context_analyzer":
                enhanced_features = ["llm_driven_scoring", "safety_handling"]
            elif agent_name == "evidence_evaluator":
                enhanced_features = ["specific_verification_links", "institutional_fallbacks"]
            elif agent_name == "credible_source":
                enhanced_features = ["contextual_recommendations", "safety_awareness"]

            agent_configs[agent_name] = {
                "model_name": config.get("model_name", "Unknown"),
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens"),
                "enhanced_features": enhanced_features,
                "version": "3.2" if enhanced_features else "2.0"
            }
            
            current_models[agent_name] = config.get("model_name", "Unknown")

        return ConfigResponse(
            agent_configs=agent_configs,
            api_key_configured=settings.validate_api_keys(),
            current_models=current_models
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configurations: {str(e)}")

# Additional endpoints for testing and monitoring
@app.get("/test-scraping")
async def test_url_scraping(url: str):
    """Test endpoint to verify URL scraping functionality"""
    try:
        logger.info(f"Testing URL scraping: {url}")
        scrape_result = scraper.scrape_article(url)
        
        return {
            "success": scrape_result['success'],
            "url": url,
            "title": scrape_result.get('title', '')[:200],
            "text_preview": scrape_result.get('text', '')[:500] + "..." if scrape_result.get('text') else "",
            "method": scrape_result.get('method', 'unknown'),
            "error": scrape_result.get('error') if not scrape_result['success'] else None,
            "text_length": len(scrape_result.get('text', ''))
        }

    except Exception as e:
        logger.error(f"Scraping test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scraping test failed: {str(e)}")

@app.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    return {
        "system_status": "operational",
        "environment": ENVIRONMENT,
        "version": "3.2.0",
        "agents_count": 6,
        "enhanced_features": [
            "production_ready_architecture",
            "enhanced_error_handling", 
            "environment_aware_configuration",
            "security_hardened_cors",
            "safety_aware_processing"
        ],
        "uptime": "operational"
    }

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if ENVIRONMENT == "development" else False,
        log_level="info"
    )
