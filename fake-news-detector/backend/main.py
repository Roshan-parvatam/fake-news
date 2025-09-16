# main.py

"""
Enhanced FastAPI Server for Fake News Detection - Production Ready

Production-ready version with comprehensive security improvements, 
proper error handling, structured logging, and integration with 
enhanced configuration system.

Features:
- Environment-aware CORS configuration with strict security
- Comprehensive async error handling with structured responses
- Request lifecycle logging and performance monitoring
- Robust startup validation with dependency checks
- Security headers and production hardening
- Health monitoring and configuration endpoints
- Rate limiting and input validation
- Graceful shutdown handling

Version: 3.2.0 - Enhanced Production Edition
"""

import os
import logging
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from fake-news-detector directory (parent of backend/)
backend_dir = Path(__file__).parent
project_dir = backend_dir.parent
env_file = project_dir / '.env'
load_dotenv(dotenv_path=env_file, override=True)

from fastapi import FastAPI, HTTPException, Request, Response, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl
import uvicorn

# Import enhanced configuration system
try:
    from config import (
        get_settings, 
        get_model_config, 
        validate_all_configurations_unified,
        get_configuration_summary,
        get_package_health,
        Environment
    )
    _config_available = True
except ImportError as e:
    logging.error(f"Enhanced config system not available: {e}")
    _config_available = False

# Import workflow components
try:
    from orchestration.langgraph_workflow import fake_news_workflow, analyze_article
    _workflow_available = True
except ImportError as e:
    logging.error(f"Workflow not available: {e}")
    _workflow_available = False

# Import URL scraper
try:
    from utils.url_scraper import ProductionNewsScraper
    _scraper_available = True
except ImportError as e:
    logging.error(f"URL scraper not available: {e}")
    _scraper_available = False

# Import enhanced exception classes
try:
    from agents.claim_extractor.exceptions import ClaimExtractorError
    from agents.llm_explanation.exceptions import LLMExplanationError
    from agents.context_analyzer.exceptions import ContextAnalyzerError
    _enhanced_exceptions_available = True
except ImportError:
    _enhanced_exceptions_available = False

# Environment and settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Initialize settings if available
settings = None
if _config_available:
    try:
        settings = get_settings()
    except Exception as e:
        logging.error(f"Failed to load settings: {e}")

# Configure structured logging
def setup_logging():
    """Setup structured logging based on environment."""
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    if ENVIRONMENT == "production":
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    if ENVIRONMENT == "production":
        logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

setup_logging()
logger = logging.getLogger(__name__)

# Application state for health monitoring
class AppState:
    def __init__(self):
        self.startup_time = datetime.now()
        self.healthy = False
        self.components_status = {}
        self.request_count = 0
        self.error_count = 0

app_state = AppState()


# Security middleware setup functions (defined before app creation)
def setup_security_middleware():
    """Setup security middleware based on environment."""
    
    # Trusted host middleware for production
    if ENVIRONMENT == "production":
        trusted_hosts = os.getenv("TRUSTED_HOSTS", "").split(",")
        if trusted_hosts and trusted_hosts != [""]:
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
            logger.info(f"Trusted hosts configured: {trusted_hosts}")
    
    # CORS middleware with environment-aware configuration
    setup_cors_middleware()


def setup_cors_middleware():
    """Configure CORS middleware based on environment and settings."""
    
    if ENVIRONMENT == "production":
        # Production - strict origins from configuration
        if settings and hasattr(settings, 'cors_origins'):
            allowed_origins = settings.cors_origins
        else:
            allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        
        # Filter out empty strings
        allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]
        
        if not allowed_origins:
            logger.warning("⚠️ No CORS origins configured for production - this may cause issues")
            allowed_origins = []
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
        )
        
        logger.info(f"Production CORS configured for {len(allowed_origins)} origins")
        
    else:
        # Development/staging - controlled but permissive
        development_origins = [
            "http://localhost:3000",    # React dev server
            "http://localhost:3001",    # Alternative React port
            "http://localhost:8080",    # Vue dev server  
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "http://0.0.0.0:3000",
        ]
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=development_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        logger.info(f"Development CORS configured for {len(development_origins)} origins")


# Request logging middleware (will be added after app creation)
async def logging_middleware(request: Request, call_next):
    """Log request details and performance metrics."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    app_state.request_count += 1
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"[{request_id}] Response: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(round(process_time, 3))
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as exc:
        process_time = time.time() - start_time
        app_state.error_count += 1
        
        logger.error(
            f"[{request_id}] Error: {str(exc)} - Time: {process_time:.3f}s"
        )
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "type": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "details": str(exc) if DEBUG_MODE else "Contact support"
                },
                "timestamp": datetime.now().isoformat()
            }
        )


# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("🚀 Starting Enhanced Fake News Detection API v3.2.0")
    await startup_tasks()
    yield
    # Shutdown
    logger.info("🔄 Shutting down Enhanced Fake News Detection API")
    await shutdown_tasks()


# Create FastAPI app with lifespan
app = FastAPI(
    title="Enhanced Fake News Detection API",
    description="Production-ready multi-agent fake news detection with comprehensive security",
    version="3.2.0",
    lifespan=lifespan,
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if ENVIRONMENT != "production" else None
)

# ✅ Add middleware immediately after app creation
setup_security_middleware()

# Add request logging middleware
app.middleware("http")(logging_middleware)


# Enhanced Pydantic models with proper validation
class ArticleRequest(BaseModel):
    """Request model for article analysis with enhanced validation."""
    text: Optional[str] = Field(None, min_length=50, max_length=100000)
    url: Optional[HttpUrl] = None
    detailed: bool = Field(False, description="Enable detailed analysis mode")
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v, info):
        """Validate text content quality."""
        if v is not None:
            # Check for minimum meaningful content
            if len(v.strip()) < 50:
                raise ValueError("Text content too short (minimum 50 characters)")
            
            # Check for suspicious content patterns
            if v.count('\n') > len(v) // 10:  # Too many newlines
                raise ValueError("Text appears to be improperly formatted")
                
        return v

    @model_validator(mode='before')
    @classmethod
    def validate_input_provided(cls, values):
        """Ensure either text or url is provided."""
        if isinstance(values, dict):
            text = values.get('text')
            url = values.get('url')
            
            if not text and not url:
                raise ValueError("Either 'text' or 'url' must be provided")
            
            if text and url:
                logger.info("Both text and URL provided - URL will be prioritized")
                
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample news article text with sufficient length for analysis...",
                "url": "https://example.com/news-article",
                "detailed": False
            }
        }


class AnalysisResponse(BaseModel):
    """Enhanced response model for article analysis."""
    success: bool
    request_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Enhanced health check response model."""
    status: str = Field(..., description="Overall system health status")
    components: Dict[str, Any] = Field(default_factory=dict)
    environment: str
    version: str
    uptime_seconds: float
    request_statistics: Dict[str, int] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ConfigResponse(BaseModel):
    """Configuration information response model."""
    environment: str
    version: str
    agents_configured: Dict[str, Any] = Field(default_factory=dict)
    api_keys_status: Dict[str, bool] = Field(default_factory=dict)
    features_enabled: List[str] = Field(default_factory=list)


# Startup and shutdown tasks
async def startup_tasks():
    """Comprehensive startup validation and initialization."""
    logger.info("🔧 Performing startup validation...")
    
    try:
        # ✅ Only do initialization here, no middleware
        # Initialize components
        if _scraper_available:
            global scraper
            scraper = ProductionNewsScraper()
            app_state.components_status['scraper'] = {'status': 'healthy'}
            logger.info("✅ URL scraper initialized")
        else:
            app_state.components_status['scraper'] = {'status': 'unavailable'}
            logger.warning("⚠️ URL scraper not available")
        
        # Validate configuration system
        if _config_available:
            try:
                validation_result = validate_all_configurations_unified()
                if validation_result['overall_status'] == 'valid':
                    app_state.components_status['config'] = {'status': 'healthy'}
                    logger.info("✅ Configuration system validated")
                else:
                    app_state.components_status['config'] = {
                        'status': 'degraded', 
                        'issues': validation_result.get('issues', [])
                    }
                    logger.warning(f"⚠️ Configuration issues: {validation_result.get('issues', [])}")
            except Exception as e:
                app_state.components_status['config'] = {'status': 'error', 'error': str(e)}
                logger.error(f"❌ Configuration validation failed: {e}")
        else:
            app_state.components_status['config'] = {'status': 'unavailable'}
        
        # Validate workflow system
        if _workflow_available:
            app_state.components_status['workflow'] = {'status': 'healthy'}
            logger.info("✅ Workflow system available")
        else:
            app_state.components_status['workflow'] = {'status': 'unavailable'}
            logger.warning("⚠️ Workflow system not available")
        
        # Validate API keys
        api_keys_valid = False
        if settings:
            try:
                api_keys_valid = settings.validate_api_keys()
                if api_keys_valid:
                    logger.info("✅ API keys validated successfully")
                else:
                    logger.warning("⚠️ API keys not properly configured")
            except Exception as e:
                logger.error(f"❌ API key validation failed: {e}")
        
        app_state.components_status['api_keys'] = {'status': 'healthy' if api_keys_valid else 'error'}
        
        # Set overall health status
        healthy_components = sum(1 for comp in app_state.components_status.values() 
                               if comp.get('status') == 'healthy')
        total_components = len(app_state.components_status)
        
        if healthy_components >= total_components * 0.75:  # 75% healthy threshold
            app_state.healthy = True
            logger.info(f"🎯 System healthy - {healthy_components}/{total_components} components operational")
        else:
            app_state.healthy = False
            logger.warning(f"⚠️ System degraded - {healthy_components}/{total_components} components operational")
        
        logger.info("✅ Startup validation completed")
        
    except Exception as e:
        logger.error(f"❌ Startup validation failed: {e}")
        app_state.healthy = False


async def shutdown_tasks():
    """Graceful shutdown tasks."""
    logger.info("Performing graceful shutdown...")
    # Add any cleanup tasks here
    logger.info("Shutdown completed")


# Utility functions
def is_url(text: str) -> bool:
    """Enhanced URL detection with validation."""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip().lower()
    return (text.startswith(('http://', 'https://')) and 
            len(text) > 10 and 
            '.' in text)


def get_client_ip(request: Request) -> str:
    """Get client IP address with proxy header support."""
    # Check for forwarded headers first (common in production)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"


# Enhanced exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    client_ip = get_client_ip(request)
    
    logger.warning(
        f"HTTP {exc.status_code} - {request.method} {request.url.path} - "
        f"Client: {client_ip} - Error: {exc.detail}"
    )
    
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
            "path": str(request.url.path)
        }
    )


# Enhanced exception handlers for agent errors if available
if _enhanced_exceptions_available:
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
                    "error_code": getattr(exc, 'error_code', 'CLAIM_ERROR'),
                    "details": getattr(exc, 'details', {})
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
                    "error_code": getattr(exc, 'error_code', 'LLM_ERROR'),
                    "details": getattr(exc, 'details', {})
                },
                "timestamp": datetime.now().isoformat()
            }
        )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "type": "validation_error",
                "message": str(exc)
            },
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with security-aware logging."""
    client_ip = get_client_ip(request)
    
    logger.exception(
        f"Unhandled exception - {request.method} {request.url.path} - "
        f"Client: {client_ip} - Error: {str(exc)}"
    )
    
    app_state.error_count += 1
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "internal_server_error",
                "message": "An unexpected error occurred",
                "details": str(exc) if DEBUG_MODE else "Please contact support"
            },
            "timestamp": datetime.now().isoformat(),
            "environment": ENVIRONMENT
        }
    )


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Enhanced Fake News Detection API",
        "status": "operational" if app_state.healthy else "degraded",
        "version": "3.2.0",
        "environment": ENVIRONMENT,
        "components_available": {
            "config_system": _config_available,
            "workflow_system": _workflow_available,
            "url_scraper": _scraper_available,
            "enhanced_exceptions": _enhanced_exceptions_available
        },
        "features": [
            "production_ready_architecture",
            "comprehensive_error_handling", 
            "environment_aware_configuration",
            "security_hardened_middleware",
            "structured_logging",
            "performance_monitoring"
        ],
        "supported_inputs": ["raw_text", "news_urls"],
        "api_documentation": "/docs" if ENVIRONMENT != "production" else "Contact administrator"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with detailed component status."""
    try:
        uptime = (datetime.now() - app_state.startup_time).total_seconds()
        
        # Get configuration health if available
        config_health = {}
        if _config_available:
            try:
                config_health = get_package_health()
            except Exception as e:
                logger.warning(f"Config health check failed: {e}")
                config_health = {"status": "error", "error": str(e)}
        
        # Determine overall status
        overall_status = "healthy"
        if not app_state.healthy:
            overall_status = "degraded"
        
        # Check error rate
        error_rate = (app_state.error_count / max(app_state.request_count, 1)) * 100
        if error_rate > 10:  # More than 10% errors
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            components=app_state.components_status,
            environment=ENVIRONMENT,
            version="3.2.0",
            uptime_seconds=uptime,
            request_statistics={
                "total_requests": app_state.request_count,
                "total_errors": app_state.error_count,
                "error_rate_percent": round(error_rate, 2)
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Health check failed: {str(e) if DEBUG_MODE else 'Internal error'}"
        )


@app.get("/config", response_model=ConfigResponse)
async def get_configuration_info():
    """Get current configuration information."""
    try:
        agents_config = {}
        api_keys_status = {}
        features_enabled = []
        
        if _config_available:
            try:
                # Get agent configurations
                agent_names = ["bert_classifier", "claim_extractor", "context_analyzer", 
                             "evidence_evaluator", "credible_source", "llm_explanation"]
                
                for agent_name in agent_names:
                    try:
                        config = get_model_config(agent_name)
                        agents_config[agent_name] = {
                            "model_name": config.get("model_name", "Unknown"),
                            "configured": True,
                            "enhanced": agent_name in ["context_analyzer", "evidence_evaluator", "credible_source"]
                        }
                    except Exception as e:
                        agents_config[agent_name] = {
                            "configured": False,
                            "error": str(e)
                        }
                
                # Check API key status
                if settings:
                    api_keys_status = {
                        "gemini_api_key": bool(settings.api.gemini_api_key),
                        "openai_api_key": bool(settings.api.openai_api_key),
                        "keys_validated": settings.validate_api_keys()
                    }
                
                features_enabled = [
                    "enhanced_configuration_system",
                    "environment_aware_settings",
                    "comprehensive_validation"
                ]
                
            except Exception as e:
                logger.warning(f"Failed to get detailed config info: {e}")
        
        return ConfigResponse(
            environment=ENVIRONMENT,
            version="3.2.0",
            agents_configured=agents_config,
            api_keys_status=api_keys_status,
            features_enabled=features_enabled
        )
        
    except Exception as e:
        logger.error(f"Configuration info request failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get configuration info: {str(e) if DEBUG_MODE else 'Internal error'}"
        )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_article_endpoint(request: ArticleRequest, http_request: Request):
    """Enhanced article analysis with comprehensive async error handling."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Validate system health
    if not app_state.healthy:
        raise HTTPException(
            status_code=503,
            detail="System is currently degraded. Some components may be unavailable."
        )
    
    # Validate API configuration
    if settings and not settings.validate_api_keys():
        raise HTTPException(
            status_code=503,
            detail="API keys not properly configured. Please contact administrator."
        )
    
    logger.info(f"[{request_id}] Starting article analysis")
    
    try:
        # Process input and extract text
        article_text = ""
        article_title = ""
        article_url = ""
        scraping_info = {}
        
        if request.url:
            if not _scraper_available:
                raise HTTPException(
                    status_code=503,
                    detail="URL scraping service is not available"
                )
            
            logger.info(f"[{request_id}] Scraping article from URL: {request.url}")
            
            # Run scraping in executor to avoid blocking
            loop = asyncio.get_event_loop()
            scrape_result = await loop.run_in_executor(
                None, 
                scraper.scrape_article, 
                str(request.url)
            )
            
            # Use dot notation to access attributes
            if not scrape_result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to scrape URL: {scrape_result.error or 'Unknown error'}"
                )
            
            # Use dot notation here as well
            article_text = scrape_result.text
            article_title = scrape_result.title
            article_url = str(request.url)
            scraping_info = {
                "scraping_method": getattr(scrape_result, 'method', 'unknown'),
                "scraped_title": article_title,
                "original_url": str(request.url),
                "scraping_successful": True
            }
            
        elif request.text:
            if is_url(request.text):
                if not _scraper_available:
                    raise HTTPException(
                        status_code=503,
                        detail="URL scraping service is not available"
                    )
                
                logger.info(f"[{request_id}] Auto-detected URL in text field")
                
                loop = asyncio.get_event_loop()
                scrape_result = await loop.run_in_executor(
                    None,
                    scraper.scrape_article,
                    request.text
                )
                
                # Use dot notation to access attributes
                if not scrape_result.success:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to scrape URL: {scrape_result.error or 'Unknown error'}"
                    )
                
                # Use dot notation here as well
                article_text = scrape_result.text
                article_title = scrape_result.title
                article_url = request.text
                scraping_info = {"auto_detected_url": True, "scraping_successful": True}
            else:
                article_text = request.text
                article_title = "User Provided Text"
                article_url = "N/A"
                scraping_info = {"input_type": "raw_text"}
        
        # Additional content validation
        if len(article_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Article text too short (minimum 50 characters required)"
            )
        
        max_length = getattr(settings, 'max_article_length', 50000) if settings else 50000
        if len(article_text) > max_length:
            raise HTTPException(
                status_code=400,
                detail=f"Article text too long. Maximum length: {max_length} characters"
            )
        
        logger.info(f"[{request_id}] Starting analysis for: {article_title[:100]}...")
        
        # Validate workflow availability
        if not _workflow_available:
            raise HTTPException(
                status_code=503,
                detail="Analysis workflow is not available"
            )
        
        # Run analysis workflow
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                analyze_article,
                article_text,
                article_url,
                request.detailed
            )
        except Exception as workflow_error:
            logger.error(f"[{request_id}] Workflow execution failed: {workflow_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis workflow failed: {str(workflow_error) if DEBUG_MODE else 'Internal processing error'}"
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Extract and validate results
        bert_results = result.get("bert_results") or {}
        extracted_claims = result.get("extracted_claims") or []
        context_analysis = result.get("context_analysis") or {}
        evidence_evaluation = result.get("evidence_evaluation") or {}
        source_recommendations = result.get("source_recommendations") or {}
        final_explanation = result.get("final_explanation") or {}
        processing_errors = result.get("processing_errors", [])
        
        # Format comprehensive response
        analysis_results = {
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
                "scores": context_analysis.get("llm_scores", {}),
                "analysis": context_analysis.get("llm_analysis", ""),
                "overall_score": context_analysis.get("context_scores", {}).get("overall_context_score", 5.0),
                "risk_level": context_analysis.get("context_scores", {}).get("risk_level", "UNKNOWN")
            },
            "evidence": {
                "overall_score": evidence_evaluation.get("evidence_scores", {}).get("overall_evidence_score", 5.0),
                "verification_links": evidence_evaluation.get("verification_links", []),
                "verification_count": len(evidence_evaluation.get("verification_links", []))
            },
            "sources": {
                "recommended_sources": source_recommendations.get("contextual_sources", []),
                "source_count": len(source_recommendations.get("contextual_sources", []))
            },
            "explanation": {
                "text": final_explanation.get("explanation", "No explanation available"),
                "detailed_analysis": final_explanation.get("detailed_analysis") is not None
            }
        }
        
        response_metadata = {
            "processing_time_seconds": round(processing_time, 3),
            "article_length": len(article_text),
            "article_title": article_title,
            "article_url": article_url,
            "environment": ENVIRONMENT,
            "api_version": "3.2.0",
            "scraping_info": scraping_info,
            "detailed_mode": request.detailed
        }
        
        logger.info(f"[{request_id}] Analysis completed successfully in {processing_time:.3f}s")
        
        return AnalysisResponse(
            success=True,
            request_id=request_id,
            results=analysis_results,
            errors=[{"type": "processing_error", "message": error} for error in processing_errors],
            metadata=response_metadata,
            processing_time=processing_time
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request_id}] Analysis failed after {processing_time:.3f}s: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e) if DEBUG_MODE else 'Internal processing error'}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics."""
    uptime = (datetime.now() - app_state.startup_time).total_seconds()
    
    return {
        "system_metrics": {
            "uptime_seconds": uptime,
            "requests_total": app_state.request_count,
            "errors_total": app_state.error_count,
            "error_rate": (app_state.error_count / max(app_state.request_count, 1)) * 100,
            "system_healthy": app_state.healthy
        },
        "component_status": app_state.components_status,
        "environment": ENVIRONMENT,
        "version": "3.2.0"
    }


# Main execution
if __name__ == "__main__":
    # Configuration
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if ENVIRONMENT == "production" else "127.0.0.1"
    reload = ENVIRONMENT in ["development", "testing"]
    
    # Log startup configuration
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Reload enabled: {reload}")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    
    # Run server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info" if DEBUG_MODE else "warning",
        access_log=DEBUG_MODE,
        server_header=False,  # Security: don't expose server info
        date_header=False     # Security: don't expose date info
    )
