# agents/base_agent.py

"""
Enhanced Base Agent for Production Fake News Detection System

Production-ready foundational class that all fake news detection agents inherit from.
Provides comprehensive functionality for LangGraph orchestration with robust error
handling, structured logging, performance monitoring, and full compatibility with
enhanced agent ecosystem.

Key Features:
- Abstract base class for consistent agent interfaces
- Configuration-driven approach with environment support
- Standardized input/output formats for LangGraph compatibility
- Comprehensive error handling with enhanced exception integration
- Structured logging with session tracking and production file handlers
- Advanced performance tracking and health monitoring
- Async support for web applications and concurrent processing
- Memory-efficient processing with automated cleanup
- Integration with enhanced exception system for consistent error handling

Version: 3.2.0 - Enhanced Production Edition
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import time
import os
import asyncio
from datetime import datetime
import json
import hashlib
from pathlib import Path
import traceback
import gc
import psutil

# Enhanced exception integration with comprehensive error handling
from agents.llm_explanation.exceptions import (
    handle_llm_explanation_exception,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    log_exception_with_context,
    is_recoverable_error,
    get_retry_delay,
    get_error_recovery_suggestion
)


class BaseAgent(ABC):
    """
    Enhanced Base Agent Class for Production Fake News Detection

    Provides comprehensive foundation for all detection agents with:
    - Robust error handling and recovery mechanisms with enhanced exception integration
    - Performance monitoring and optimization with detailed metrics
    - LangGraph state management compatibility
    - Production logging and debugging support with session tracking
    - Memory management and automated cleanup
    - Async processing capabilities with concurrent session support
    - Health monitoring and status reporting for production environments
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced base agent with production-ready configuration.

        Args:
            config: Optional configuration dictionary with agent settings
        """
        # Setup enhanced configuration with environment awareness
        self.config = self._setup_production_config(config)

        # Initialize structured logging with session tracking
        self.logger = self._setup_enhanced_logging()

        # Agent identification and metadata
        self.agent_name = self.__class__.__name__
        self.agent_type = self._determine_agent_type()
        self.agent_version = "3.2.0"
        self.initialized_at = datetime.now().isoformat()

        # Performance and monitoring setup with enhanced metrics
        self.metrics_enabled = self.config.get("enable_metrics", True)
        self.performance_metrics = self._initialize_enhanced_performance_tracking()
        self._start_time = time.time()

        # State management for LangGraph with session tracking
        self.processing_state = {
            "last_input": None,
            "last_output": None,
            "processing_history": [],
            "current_session_id": None,
            "active_sessions": {}
        }

        # Enhanced error handling and recovery with new exception system
        self.error_recovery = {
            "max_retries": self.config.get("max_retries", 3),
            "retry_delay": self.config.get("retry_delay", 1.0),
            "circuit_breaker_threshold": self.config.get("circuit_breaker_threshold", 5),
            "recovery_strategies": {},
            "error_patterns": {}
        }

        # Resource management with enhanced monitoring
        self._processing_start_time = None
        self._memory_threshold = self.config.get("memory_threshold_mb", 512)
        self._health_status = "INITIALIZING"
        self._last_health_check = time.time()

        self.logger.info(f"Enhanced {self.agent_name} v{self.agent_version} initialized successfully")

    def _setup_production_config(self, user_config: Optional[Dict]) -> Dict[str, Any]:
        """
        Setup enhanced production-ready configuration with environment variable support.

        Args:
            user_config: User-provided configuration

        Returns:
            Complete configuration with production defaults and enhanced features
        """
        # Environment-aware defaults with enhanced configuration
        environment = os.getenv("ENVIRONMENT", "development")
        default_config = {
            # Core settings
            "environment": environment,
            "agent_version": "3.2.0",
            "debug_mode": environment == "development",

            # Model configuration (agent-specific)
            "model_name": None,
            "temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.3")),
            "max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", "2048")),
            "timeout": int(os.getenv("DEFAULT_TIMEOUT", "30")),

            # Performance and monitoring with enhanced metrics
            "enable_metrics": self._parse_bool(os.getenv("ENABLE_METRICS", "true")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "enable_caching": self._parse_bool(os.getenv("ENABLE_CACHING", "false")),
            "memory_threshold_mb": int(os.getenv("MEMORY_THRESHOLD_MB", "512")),
            "enable_health_monitoring": True,
            "health_check_interval": 60.0,

            # Enhanced error handling with new exception system
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "retry_delay": float(os.getenv("RETRY_DELAY", "1.0")),
            "circuit_breaker_threshold": int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
            "enable_error_recovery": True,
            "detailed_error_logging": True,

            # LangGraph integration with enhanced state management
            "state_key": None,
            "next_agents": [],
            "parallel_enabled": False,
            "async_enabled": self._parse_bool(os.getenv("ASYNC_ENABLED", "true")),
            "session_tracking_enabled": True,

            # Security and validation with enhanced checks
            "max_input_length": int(os.getenv("MAX_INPUT_LENGTH", "50000")),
            "sanitize_input": True,
            "validate_output": True,
            "enable_security_checks": True,

            # Production features
            "enable_file_logging": environment == "production",
            "log_rotation": True,
            "performance_optimization": True,
            "quality_validation_enabled": True,

            # Custom parameters
            "custom_params": {}
        }

        # Merge with user configuration with deep update support
        if user_config:
            for key, value in user_config.items():
                if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value

        return default_config

    def _parse_bool(self, value: str) -> bool:
        """Parse boolean values from environment variables."""
        return str(value).lower() in ('true', '1', 'yes', 'on', 'enabled')

    def _setup_enhanced_logging(self) -> logging.Logger:
        """
        Setup enhanced structured logging with session tracking and production features.

        Returns:
            Configured logger instance with enhanced handlers and formatting
        """
        logger_name = f"agents.{self.agent_name.lower()}"
        logger = logging.getLogger(logger_name)

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # Enhanced structured formatter with session tracking
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - '
            '[%(session_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add SessionIDFilter to handle missing session_id
        class SessionIDFilter(logging.Filter):
            def filter(self, record):
                if not hasattr(record, 'session_id'):
                    record.session_id = 'main'
                return True
        
        session_filter = SessionIDFilter()

        # Console handler with enhanced formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(session_filter)
        logger.addHandler(console_handler)

        # Enhanced file handler for production with rotation
        if self.config.get("enable_file_logging", False):
            try:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                
                if self.config.get("log_rotation", True):
                    from logging.handlers import RotatingFileHandler
                    file_handler = RotatingFileHandler(
                        log_dir / f"{self.agent_name.lower()}.log",
                        maxBytes=10*1024*1024,  # 10MB
                        backupCount=5
                    )
                else:
                    file_handler = logging.FileHandler(
                        log_dir / f"{self.agent_name.lower()}.log"
                    )
                
                file_handler.setFormatter(formatter)
                file_handler.addFilter(session_filter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")

        # Set logging level with enhanced configuration
        log_level = self.config.get('log_level', 'INFO').upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        return logger

    def _determine_agent_type(self) -> str:
        """
        Determine agent type based on class name for enhanced routing optimization.

        Returns:
            String indicating agent category with enhanced type mapping
        """
        name_lower = self.agent_name.lower()
        enhanced_type_mapping = {
            "classifier": ["classifier", "bert", "classification"],
            "generator": ["explanation", "llm", "generator", "generation"],
            "recommender": ["source", "credible", "recommender", "recommendation"],
            "extractor": ["claim", "extractor", "extract", "extraction"],
            "analyzer": ["context", "analyzer", "analyse", "analysis"],
            "evaluator": ["evidence", "evaluator", "evaluate", "evaluation"],
            "validator": ["validator", "validation", "verify"],
            "monitor": ["monitor", "monitoring", "health", "status"]
        }

        for agent_type, keywords in enhanced_type_mapping.items():
            if any(keyword in name_lower for keyword in keywords):
                return agent_type

        return "generic"

    def _initialize_enhanced_performance_tracking(self) -> Dict[str, Any]:
        """
        Initialize comprehensive performance tracking system with enhanced metrics.

        Returns:
            Performance metrics dictionary with detailed tracking capabilities
        """
        return {
            # Enhanced call statistics
            "total_calls": 0,
            "successful_calls": 0,
            "error_calls": 0,
            "retry_calls": 0,
            "recovered_calls": 0,

            # Enhanced timing metrics with percentiles
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "min_processing_time": float('inf'),
            "max_processing_time": 0.0,
            "p50_processing_time": 0.0,
            "p95_processing_time": 0.0,
            "p99_processing_time": 0.0,
            "processing_time_samples": [],

            # Enhanced quality metrics
            "confidence_scores": [],
            "average_confidence": 0.0,
            "quality_threshold_met": 0,
            "quality_scores": [],
            "validation_pass_rate": 100.0,

            # Enhanced resource metrics
            "memory_usage_samples": [],
            "peak_memory_usage": 0.0,
            "current_memory_usage": 0.0,
            "cpu_usage_samples": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_effectiveness": 0.0,

            # Enhanced error tracking with new exception integration
            "error_types": {},
            "error_recovery_attempts": 0,
            "successful_recoveries": 0,
            "circuit_breaker_trips": 0,
            "last_error": None,
            "error_patterns": {},

            # Enhanced session tracking
            "session_count": 0,
            "concurrent_sessions": 0,
            "active_sessions": {},
            "session_duration_samples": [],
            "average_session_duration": 0.0,

            # Enhanced health and status tracking
            "health_checks_performed": 0,
            "health_status_history": [],
            "uptime_seconds": 0.0,
            "component_health": {},

            # Enhanced timestamps
            "first_call_time": None,
            "last_call_time": None,
            "last_reset_time": datetime.now().isoformat(),
            "last_health_check_time": datetime.now().isoformat()
        }

    async def process(self, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced process method with comprehensive error handling, session tracking, and performance monitoring.

        Args:
            input_data: Input data dictionary to process
            session_id: Optional session ID for tracking and context

        Returns:
            Processed result with enhanced metadata and error handling
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = self._generate_session_id()

        # Update performance metrics
        self.performance_metrics["total_calls"] += 1
        if self.performance_metrics["first_call_time"] is None:
            self.performance_metrics["first_call_time"] = datetime.now().isoformat()

        # Start session tracking
        session_start_time = time.time()
        self._start_processing_session(input_data, session_id)

        try:
            # Enhanced input validation with security checks
            is_valid, validation_error = self.validate_input(input_data)
            if not is_valid:
                raise ValueError(f"Input validation failed: {validation_error}")

            # Call abstract processing method (implemented by subclasses)
            result = await self._process_internal(input_data, session_id)

            # Enhanced output validation
            if self.config.get("validate_output", True):
                self._validate_output(result, session_id)

            # Update success metrics
            self.performance_metrics["successful_calls"] += 1
            self._update_quality_metrics(result)

            return self.format_output(result, session_id)

        except Exception as e:
            # Enhanced error handling with new exception system
            self.performance_metrics["error_calls"] += 1
            
            # Create error context for enhanced exception handling
            error_context = ErrorContext(
                session_id=session_id,
                operation="processing",
                model_used=self.config.get("model_name"),
                processing_time=time.time() - session_start_time,
                input_size=len(str(input_data)) if input_data else 0
            )

            # Convert to standardized exception
            standardized_error = handle_llm_explanation_exception(e, error_context)
            
            # Log with enhanced context
            log_exception_with_context(
                standardized_error, 
                session_id, 
                {
                    'agent_name': self.agent_name,
                    'agent_type': self.agent_type,
                    'processing_time': time.time() - session_start_time
                }
            )

            # Attempt error recovery if recoverable
            if is_recoverable_error(standardized_error) and self.config.get("enable_error_recovery", True):
                recovery_result = await self._attempt_error_recovery(standardized_error, input_data, session_id)
                if recovery_result:
                    return recovery_result

            # Update error metrics
            self._update_error_metrics(standardized_error)

            return self.format_error_output(standardized_error, input_data, session_id)

        finally:
            # Complete session tracking
            session_duration = time.time() - session_start_time
            self._end_processing_session(session_id, session_duration)
            
            # Update performance metrics
            self._update_performance_metrics(session_duration)
            
            # Memory cleanup
            self._cleanup_resources()

    @abstractmethod
    async def _process_internal(self, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Abstract method for internal processing - must be implemented by subclasses.

        Args:
            input_data: Validated input data
            session_id: Session identifier for tracking

        Returns:
            Processing results dictionary
        """
        raise NotImplementedError("Subclasses must implement _process_internal method")

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Enhanced input validation with security and safety checks.

        Args:
            input_data: Input data dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic type validation
            if not isinstance(input_data, dict):
                return False, "Input must be a dictionary"
            
            if not input_data:
                return False, "Input data cannot be empty"

            # Enhanced text validation if present
            if 'text' in input_data:
                text = input_data['text']
                if not isinstance(text, str):
                    return False, "Text field must be a string"
                
                text = text.strip()
                if len(text) == 0:
                    return False, "Text content cannot be empty"
                
                max_length = self.config.get("max_input_length", 50000)
                if len(text) > max_length:
                    return False, f"Text too long (max {max_length} characters)"

                # Enhanced security validation
                if self.config.get("enable_security_checks", True):
                    security_issues = self._check_security_patterns(text)
                    if security_issues:
                        return False, f"Security validation failed: {security_issues}"

            # Additional field validations
            if 'confidence' in input_data:
                confidence = input_data['confidence']
                if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                    return False, "Confidence must be a number between 0 and 1"

            if 'prediction' in input_data:
                prediction = input_data['prediction']
                if not isinstance(prediction, str):
                    return False, "Prediction must be a string"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _check_security_patterns(self, text: str) -> Optional[str]:
        """Check for security-related patterns in text input."""
        if not self.config.get("enable_security_checks", True):
            return None

        # Basic security pattern detection
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
            r'union\s+select',
            r'drop\s+table'
        ]

        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return f"Suspicious pattern detected: {pattern}"

        return None

    def _validate_output(self, result: Dict[str, Any], session_id: str) -> bool:
        """Enhanced output validation with quality checks."""
        try:
            if not isinstance(result, dict):
                raise ValueError("Output must be a dictionary")

            # Check for required fields based on agent type
            if self.agent_type == "generator" and 'explanation' in result:
                explanation = result['explanation']
                if not isinstance(explanation, str) or len(explanation.strip()) < 10:
                    raise ValueError("Explanation must be a meaningful string")

            return True

        except Exception as e:
            self.logger.warning(f"Output validation failed: {str(e)}", extra={'session_id': session_id})
            return False

    async def _attempt_error_recovery(self, error: Exception, input_data: Dict[str, Any], session_id: str) -> Optional[Dict[str, Any]]:
        """
        Attempt error recovery using enhanced exception system.

        Args:
            error: Standardized exception to recover from
            input_data: Original input data
            session_id: Session identifier

        Returns:
            Recovery result if successful, None otherwise
        """
        self.performance_metrics["error_recovery_attempts"] += 1
        
        try:
            retry_delay = get_retry_delay(error)
            if retry_delay and retry_delay > 0:
                self.logger.info(f"Attempting error recovery with {retry_delay}s delay", extra={'session_id': session_id})
                await asyncio.sleep(retry_delay)
                
                # Retry the operation
                result = await self._process_internal(input_data, session_id)
                
                self.performance_metrics["successful_recoveries"] += 1
                self.performance_metrics["recovered_calls"] += 1
                
                return self.format_output(result, session_id)

        except Exception as recovery_error:
            self.logger.warning(f"Error recovery failed: {str(recovery_error)}", extra={'session_id': session_id})

        return None

    def _generate_session_id(self) -> str:
        """Generate unique session ID for tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_hash = hashlib.md5(f"{self.agent_name}_{timestamp}_{time.time()}".encode()).hexdigest()[:8]
        return f"{self.agent_name}_{timestamp}_{unique_hash}"

    def _start_processing_session(self, input_data: Dict[str, Any], session_id: str) -> None:
        """Start processing session with enhanced tracking."""
        session_info = {
            'session_id': session_id,
            'start_time': time.time(),
            'input_size': len(str(input_data)) if input_data else 0,
            'agent_name': self.agent_name
        }
        
        self.processing_state["active_sessions"][session_id] = session_info
        self.processing_state["current_session_id"] = session_id
        self.performance_metrics["concurrent_sessions"] = len(self.processing_state["active_sessions"])

    def _end_processing_session(self, session_id: str, duration: float) -> None:
        """End processing session and update metrics."""
        if session_id in self.processing_state["active_sessions"]:
            del self.processing_state["active_sessions"][session_id]
        
        self.performance_metrics["concurrent_sessions"] = len(self.processing_state["active_sessions"])
        self.performance_metrics["session_duration_samples"].append(duration)
        
        # Calculate average session duration (keep last 100 samples)
        samples = self.performance_metrics["session_duration_samples"][-100:]
        self.performance_metrics["average_session_duration"] = sum(samples) / len(samples) if samples else 0.0

    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update enhanced performance metrics with processing time."""
        metrics = self.performance_metrics
        
        # Update timing metrics
        metrics["total_processing_time"] += processing_time
        metrics["average_processing_time"] = metrics["total_processing_time"] / metrics["total_calls"]
        metrics["min_processing_time"] = min(metrics["min_processing_time"], processing_time)
        metrics["max_processing_time"] = max(metrics["max_processing_time"], processing_time)
        metrics["last_call_time"] = datetime.now().isoformat()
        
        # Update processing time samples for percentile calculation
        metrics["processing_time_samples"].append(processing_time)
        
        # Keep only last 1000 samples for percentile calculation
        if len(metrics["processing_time_samples"]) > 1000:
            metrics["processing_time_samples"] = metrics["processing_time_samples"][-1000:]
        
        # Calculate percentiles
        samples = sorted(metrics["processing_time_samples"])
        if samples:
            n = len(samples)
            metrics["p50_processing_time"] = samples[int(n * 0.5)]
            metrics["p95_processing_time"] = samples[int(n * 0.95)]
            metrics["p99_processing_time"] = samples[int(n * 0.99)]

    def _update_quality_metrics(self, result: Dict[str, Any]) -> None:
        """Update quality metrics based on processing result."""
        if 'confidence' in result:
            confidence = result['confidence']
            if isinstance(confidence, (int, float)):
                self.performance_metrics["confidence_scores"].append(confidence)
                
                # Keep only last 100 samples
                if len(self.performance_metrics["confidence_scores"]) > 100:
                    self.performance_metrics["confidence_scores"] = self.performance_metrics["confidence_scores"][-100:]
                
                # Calculate average confidence
                scores = self.performance_metrics["confidence_scores"]
                self.performance_metrics["average_confidence"] = sum(scores) / len(scores)

    def _update_error_metrics(self, error: Exception) -> None:
        """Update error metrics with enhanced tracking."""
        error_type = type(error).__name__
        self.performance_metrics["error_types"][error_type] = \
            self.performance_metrics["error_types"].get(error_type, 0) + 1
        self.performance_metrics["last_error"] = {
            'type': error_type,
            'message': str(error),
            'timestamp': datetime.now().isoformat()
        }

    def _cleanup_resources(self) -> None:
        """Enhanced resource cleanup with memory monitoring."""
        try:
            # Update memory usage
            self._update_memory_metrics()
            
            # Garbage collection
            gc.collect()
            
            # Clear old processing history (keep last 50 entries)
            if len(self.processing_state["processing_history"]) > 50:
                self.processing_state["processing_history"] = self.processing_state["processing_history"][-50:]
                
        except Exception as e:
            self.logger.warning(f"Resource cleanup warning: {str(e)}")

    def _update_memory_metrics(self) -> None:
        """Update memory usage metrics."""
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.performance_metrics["current_memory_usage"] = memory_mb
            self.performance_metrics["memory_usage_samples"].append(memory_mb)
            self.performance_metrics["peak_memory_usage"] = max(
                self.performance_metrics["peak_memory_usage"], memory_mb
            )
            
            # Keep only last 100 samples
            if len(self.performance_metrics["memory_usage_samples"]) > 100:
                self.performance_metrics["memory_usage_samples"] = \
                    self.performance_metrics["memory_usage_samples"][-100:]
                    
        except (ImportError, Exception):
            # psutil not available or other error - use fallback
            self.performance_metrics["current_memory_usage"] = 0.0

    def format_output(self, result: Any, session_id: str, confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format successful processing output with enhanced metadata.

        Args:
            result: Processing result
            session_id: Session identifier
            confidence: Optional confidence score
            metadata: Optional additional metadata

        Returns:
            Formatted output dictionary
        """
        output = {
            "success": True,
            "result": result,
            "agent_info": {
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "agent_version": self.agent_version
            },
            "session_info": {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "processing_time": self.performance_metrics.get("average_processing_time", 0.0)
            },
            "metadata": metadata or {}
        }
        
        if confidence is not None:
            output["confidence"] = confidence
            
        return output

    def format_error_output(self, error: Exception, input_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Format error output with enhanced error information.

        Args:
            error: Exception that occurred
            input_data: Original input data
            session_id: Session identifier

        Returns:
            Formatted error output dictionary
        """
        return {
            "success": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "recoverable": is_recoverable_error(error),
                "suggestion": get_error_recovery_suggestion(error)
            },
            "agent_info": {
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "agent_version": self.agent_version
            },
            "session_info": {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            },
            "input_data_summary": {
                "size": len(str(input_data)) if input_data else 0,
                "keys": list(input_data.keys()) if isinstance(input_data, dict) else []
            }
        }

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status with enhanced health monitoring.

        Returns:
            Complete status dictionary with detailed metrics
        """
        total_calls = self.performance_metrics.get("total_calls", 0)
        successful_calls = self.performance_metrics.get("successful_calls", 0)
        error_calls = self.performance_metrics.get("error_calls", 0)
        
        # Calculate health metrics
        success_rate = (successful_calls / max(total_calls, 1)) * 100
        error_rate = (error_calls / max(total_calls, 1)) * 100
        
        # Determine health status
        if success_rate >= 95 and error_rate <= 5:
            health_status = "healthy"
        elif success_rate >= 85 and error_rate <= 15:
            health_status = "warning"
        elif success_rate >= 70 and error_rate <= 30:
            health_status = "degraded"
        else:
            health_status = "critical"

        return {
            "agent_info": {
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "agent_version": self.agent_version,
                "initialized_at": self.initialized_at,
                "uptime_seconds": time.time() - self._start_time
            },
            "health_status": {
                "status": health_status,
                "success_rate": round(success_rate, 2),
                "error_rate": round(error_rate, 2),
                "last_health_check": datetime.now().isoformat()
            },
            "performance_summary": {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "error_calls": error_calls,
                "average_processing_time": round(self.performance_metrics.get("average_processing_time", 0), 3),
                "p95_processing_time": round(self.performance_metrics.get("p95_processing_time", 0), 3),
                "current_memory_mb": round(self.performance_metrics.get("current_memory_usage", 0), 1),
                "peak_memory_mb": round(self.performance_metrics.get("peak_memory_usage", 0), 1)
            },
            "session_info": {
                "active_sessions": self.performance_metrics.get("concurrent_sessions", 0),
                "total_sessions": self.performance_metrics.get("session_count", 0),
                "average_session_duration": round(self.performance_metrics.get("average_session_duration", 0), 3)
            },
            "error_summary": {
                "error_types": self.performance_metrics.get("error_types", {}),
                "recovery_attempts": self.performance_metrics.get("error_recovery_attempts", 0),
                "successful_recoveries": self.performance_metrics.get("successful_recoveries", 0),
                "last_error": self.performance_metrics.get("last_error")
            },
            "configuration": {
                "environment": self.config.get("environment"),
                "debug_mode": self.config.get("debug_mode"),
                "metrics_enabled": self.metrics_enabled,
                "async_enabled": self.config.get("async_enabled"),
                "health_monitoring": self.config.get("enable_health_monitoring")
            }
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics for fresh monitoring period."""
        self.performance_metrics = self._initialize_enhanced_performance_tracking()
        self.logger.info(f"Performance metrics reset for {self.agent_name}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status for monitoring systems.

        Returns:
            Health status dictionary for external monitoring
        """
        status = self.get_comprehensive_status()
        return {
            "agent_name": self.agent_name,
            "status": status["health_status"]["status"],
            "uptime": status["agent_info"]["uptime_seconds"],
            "success_rate": status["health_status"]["success_rate"],
            "error_rate": status["health_status"]["error_rate"],
            "avg_response_time": status["performance_summary"]["average_processing_time"],
            "memory_usage": status["performance_summary"]["current_memory_mb"],
            "active_sessions": status["session_info"]["active_sessions"],
            "last_check": status["health_status"]["last_health_check"]
        }

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.agent_name}(type={self.agent_type}, version={self.agent_version}, status={self._health_status})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Enhanced {self.agent_name} v{self.agent_version} - {self.agent_type} agent"
