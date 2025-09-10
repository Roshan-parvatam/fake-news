"""
Enhanced Base Agent Class for Production Fake News Detection System

Production-ready foundational class that all fake news detection agents inherit from.
Provides comprehensive functionality for LangGraph orchestration with robust error
handling, structured logging, and performance monitoring.

Key Features:
- Abstract base class for consistent agent interfaces
- Configuration-driven approach with environment support
- Standardized input/output formats for LangGraph compatibility
- Comprehensive error handling with custom exceptions
- Structured logging and performance tracking
- Async support for web applications
- Memory-efficient processing with cleanup
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


class BaseAgent(ABC):
    """
    Enhanced Base Agent Class for Production Fake News Detection
    
    Provides comprehensive foundation for all detection agents with:
    - Robust error handling and recovery mechanisms
    - Performance monitoring and optimization
    - LangGraph state management compatibility
    - Production logging and debugging support
    - Memory management and cleanup
    - Async processing capabilities
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent with production-ready configuration.
        
        Args:
            config: Optional configuration dictionary with agent settings
        """
        # Setup configuration with environment awareness
        self.config = self._setup_production_config(config)
        
        # Initialize structured logging
        self.logger = self._setup_structured_logging()
        
        # Agent identification and metadata
        self.agent_name = self.__class__.__name__
        self.agent_type = self._determine_agent_type()
        self.agent_version = "3.2.0"
        self.initialized_at = datetime.now().isoformat()
        
        # Performance and monitoring setup
        self.metrics_enabled = self.config.get("enable_metrics", True)
        self.performance_metrics = self._initialize_performance_tracking()
        
        # State management for LangGraph
        self.processing_state = {
            "last_input": None,
            "last_output": None,
            "processing_history": [],
            "current_session_id": None
        }
        
        # Error handling and recovery
        self.error_recovery = {
            "max_retries": self.config.get("max_retries", 3),
            "retry_delay": self.config.get("retry_delay", 1.0),
            "circuit_breaker_threshold": self.config.get("circuit_breaker_threshold", 5)
        }
        
        # Resource management
        self._processing_start_time = None
        self._memory_threshold = self.config.get("memory_threshold_mb", 512)
        
        self.logger.info(f"Enhanced {self.agent_name} v{self.agent_version} initialized successfully")

    def _setup_production_config(self, user_config: Optional[Dict]) -> Dict[str, Any]:
        """
        Setup production-ready configuration with environment variable support.
        
        Args:
            user_config: User-provided configuration
            
        Returns:
            Complete configuration with production defaults
        """
        # Environment-aware defaults
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
            
            # Performance and monitoring
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "enable_caching": os.getenv("ENABLE_CACHING", "false").lower() == "true",
            "memory_threshold_mb": int(os.getenv("MEMORY_THRESHOLD_MB", "512")),
            
            # Error handling
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "retry_delay": float(os.getenv("RETRY_DELAY", "1.0")),
            "circuit_breaker_threshold": int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
            
            # LangGraph integration
            "state_key": None,
            "next_agents": [],
            "parallel_enabled": False,
            "async_enabled": os.getenv("ASYNC_ENABLED", "true").lower() == "true",
            
            # Security and validation
            "max_input_length": int(os.getenv("MAX_INPUT_LENGTH", "50000")),
            "sanitize_input": True,
            "validate_output": True,
            
            # Custom parameters
            "custom_params": {}
        }
        
        # Merge with user configuration
        if user_config:
            for key, value in user_config.items():
                if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        return default_config

    def _setup_structured_logging(self) -> logging.Logger:
        """
        Setup structured logging with production-ready formatting.
        
        Returns:
            Configured logger instance with appropriate handlers
        """
        logger_name = f"agents.{self.agent_name.lower()}"
        logger = logging.getLogger(logger_name)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for production
        if self.config.get("environment") == "production":
            try:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                
                file_handler = logging.FileHandler(
                    log_dir / f"{self.agent_name.lower()}.log"
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")
        
        # Set logging level
        log_level = self.config.get('log_level', 'INFO').upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        return logger

    def _determine_agent_type(self) -> str:
        """
        Determine agent type based on class name for routing optimization.
        
        Returns:
            String indicating agent category
        """
        name_lower = self.agent_name.lower()
        
        type_mapping = {
            "classifier": ["classifier", "bert"],
            "generator": ["explanation", "llm", "generator"],
            "recommender": ["source", "credible", "recommender"],
            "extractor": ["claim", "extractor", "extract"],
            "analyzer": ["context", "analyzer", "analyse"],
            "evaluator": ["evidence", "evaluator", "evaluate"]
        }
        
        for agent_type, keywords in type_mapping.items():
            if any(keyword in name_lower for keyword in keywords):
                return agent_type
        
        return "generic"

    def _initialize_performance_tracking(self) -> Dict[str, Any]:
        """
        Initialize comprehensive performance tracking system.
        
        Returns:
            Performance metrics dictionary
        """
        return {
            # Call statistics
            "total_calls": 0,
            "successful_calls": 0,
            "error_calls": 0,
            "retry_calls": 0,
            
            # Timing metrics
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "min_processing_time": float('inf'),
            "max_processing_time": 0.0,
            "p95_processing_time": 0.0,
            
            # Quality metrics
            "confidence_scores": [],
            "average_confidence": 0.0,
            "quality_threshold_met": 0,
            
            # Resource metrics
            "memory_usage_samples": [],
            "peak_memory_usage": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            
            # Error tracking
            "error_types": {},
            "error_recovery_attempts": 0,
            "circuit_breaker_trips": 0,
            "last_error": None,
            
            # Session tracking
            "session_count": 0,
            "concurrent_sessions": 0,
            
            # Timestamps
            "first_call_time": None,
            "last_call_time": None,
            "last_reset_time": datetime.now().isoformat()
        }

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
            
            # Text validation if present
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
                
                # Basic security validation
                if self.config.get("sanitize_input", True):
                    suspicious_patterns = ['<script>', '<?php', '${', 'javascript:', 'data:']
                    text_lower = text.lower()
                    if any(pattern in text_lower for pattern in suspicious_patterns):
                        return False, "Input contains potentially unsafe content"
            
            # Validate nested structures
            for key, value in input_data.items():
                if isinstance(value, dict) and len(str(value)) > 10000:
                    return False, f"Nested object '{key}' too large"
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Input validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def format_output(self, result: Any, confidence: float = None, 
                     metadata: Dict = None) -> Dict[str, Any]:
        """
        Format standardized output for LangGraph compatibility.
        
        Args:
            result: Main processing result
            confidence: Confidence score (0.0-1.0)
            metadata: Additional processing metadata
            
        Returns:
            Standardized output dictionary
        """
        # Calculate processing time if available
        processing_time = getattr(self, '_last_processing_time', None)
        
        # Generate output hash for caching/deduplication
        output_hash = self._generate_output_hash(result)
        
        output = {
            # Standard identification fields
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "agent_version": self.agent_version,
            
            # Core result data
            "result": result,
            "success": True,
            "confidence": confidence,
            
            # Processing metadata
            "metadata": {
                **(metadata or {}),
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.processing_state.get("current_session_id"),
                "output_hash": output_hash,
                "memory_usage_mb": self._get_memory_usage(),
                "environment": self.config.get("environment")
            },
            
            # LangGraph routing
            "next_agents": self.config.get("next_agents", []),
            "state_key": self.config.get("state_key"),
            "parallel_ready": self.config.get("parallel_enabled", False),
            
            # Quality indicators
            "quality_indicators": {
                "confidence_above_threshold": confidence > 0.7 if confidence else None,
                "processing_time_acceptable": processing_time < 10.0 if processing_time else None,
                "memory_usage_acceptable": self._get_memory_usage() < self._memory_threshold
            }
        }
        
        # Add validation if enabled
        if self.config.get("validate_output", True):
            validation_result = self._validate_output_structure(output)
            if not validation_result["is_valid"]:
                output["warnings"] = validation_result["warnings"]
        
        return output

    def format_error_output(self, error: Exception, input_data: Dict = None) -> Dict[str, Any]:
        """
        Format comprehensive error output with recovery information.
        
        Args:
            error: Exception that occurred
            input_data: Input data that caused the error
            
        Returns:
            Standardized error output
        """
        error_id = self._generate_error_id()
        
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "agent_version": self.agent_version,
            
            "result": None,
            "success": False,
            
            "error": {
                "id": error_id,
                "type": error.__class__.__name__,
                "message": str(error),
                "timestamp": datetime.now().isoformat(),
                "traceback": traceback.format_exc() if self.config.get("debug_mode") else None,
                "recovery_suggestions": self._get_error_recovery_suggestions(error)
            },
            
            "metadata": {
                "input_received": input_data is not None,
                "retry_recommended": self._should_retry_error(error),
                "retry_attempts_remaining": self._get_retry_attempts_remaining(),
                "circuit_breaker_active": self._is_circuit_breaker_active(),
                "memory_usage_mb": self._get_memory_usage(),
                "environment": self.config.get("environment")
            },
            
            "recovery_actions": {
                "immediate_retry": self._should_immediate_retry(error),
                "delayed_retry": self._should_delayed_retry(error),
                "fallback_available": self._has_fallback_mechanism(),
                "manual_intervention_required": self._requires_manual_intervention(error)
            }
        }

    def _should_retry_error(self, error: Exception) -> bool:
        """Determine if error is retryable based on type and context."""
        retryable_indicators = [
            "timeout", "connection", "rate limit", "service unavailable",
            "temporary", "network", "503", "502", "429", "quota"
        ]
        
        error_str = str(error).lower()
        error_type = error.__class__.__name__.lower()
        
        return any(indicator in error_str or indicator in error_type 
                  for indicator in retryable_indicators)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _generate_output_hash(self, result: Any) -> str:
        """Generate hash of output for caching and deduplication."""
        try:
            result_str = json.dumps(result, sort_keys=True, default=str)
            return hashlib.sha256(result_str.encode()).hexdigest()[:16]
        except:
            return hashlib.sha256(str(result).encode()).hexdigest()[:16]

    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        timestamp = str(int(time.time() * 1000))
        return f"{self.agent_name[:3].upper()}-{timestamp[-8:]}"

    def process_with_recovery(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input with built-in error recovery and retry logic.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processing result with recovery handling
        """
        max_retries = self.error_recovery["max_retries"]
        retry_delay = self.error_recovery["retry_delay"]
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Update session tracking
                self._start_processing_session(input_data)
                
                # Process with timing
                result = self.process(input_data)
                
                # Update success metrics
                self._update_success_metrics(result.get("confidence"))
                self._cleanup_processing_session()
                
                return result
                
            except Exception as e:
                last_error = e
                self._update_error_metrics(e)
                
                if attempt < max_retries and self._should_retry_error(e):
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    self.logger.error(f"All retry attempts failed: {str(e)}")
                    break
        
        # All retries failed
        self._cleanup_processing_session()
        return self.format_error_output(last_error, input_data)

    async def process_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous processing wrapper for web applications.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processing result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_with_recovery, input_data)

    def _start_processing_session(self, input_data: Dict[str, Any]):
        """Initialize processing session with tracking."""
        self._processing_start_time = time.time()
        self.processing_state["last_input"] = input_data
        self.processing_state["current_session_id"] = self._generate_session_id()
        
        if self.metrics_enabled:
            self.performance_metrics["session_count"] += 1
            self.performance_metrics["concurrent_sessions"] += 1

    def _cleanup_processing_session(self):
        """Clean up processing session and update metrics."""
        if self._processing_start_time:
            processing_time = time.time() - self._processing_start_time
            self._last_processing_time = processing_time
            
            if self.metrics_enabled:
                self._update_timing_metrics(processing_time)
                self.performance_metrics["concurrent_sessions"] = max(0, 
                    self.performance_metrics["concurrent_sessions"] - 1)
        
        # Memory cleanup
        if self._get_memory_usage() > self._memory_threshold:
            gc.collect()

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = str(int(time.time() * 1000000))
        return f"{self.agent_name[:3]}-{timestamp[-12:]}"

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status for monitoring and debugging.
        
        Returns:
            Detailed status information
        """
        if not self.metrics_enabled:
            return {
                "agent_name": self.agent_name,
                "status": "metrics_disabled",
                "version": self.agent_version
            }
        
        metrics = self.performance_metrics
        
        # Calculate health indicators
        error_rate = (metrics["error_calls"] / max(metrics["total_calls"], 1)) * 100
        avg_time = metrics["average_processing_time"]
        memory_usage = self._get_memory_usage()
        
        # Determine overall status
        if error_rate > 25:
            status = "critical"
        elif error_rate > 10 or avg_time > 15:
            status = "degraded"
        elif memory_usage > self._memory_threshold:
            status = "resource_constrained"
        else:
            status = "healthy"
        
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "agent_version": self.agent_version,
            "status": status,
            "environment": self.config.get("environment"),
            
            "performance_summary": {
                "total_calls": metrics["total_calls"],
                "success_rate": ((metrics["successful_calls"] / max(metrics["total_calls"], 1)) * 100),
                "error_rate": error_rate,
                "average_processing_time": avg_time,
                "current_memory_usage_mb": memory_usage
            },
            
            "health_indicators": {
                "error_rate_acceptable": error_rate < 10,
                "response_time_acceptable": avg_time < 10,
                "memory_usage_acceptable": memory_usage < self._memory_threshold,
                "circuit_breaker_healthy": metrics["circuit_breaker_trips"] == 0
            },
            
            "operational_info": {
                "initialized_at": self.initialized_at,
                "last_call_time": metrics.get("last_call_time"),
                "concurrent_sessions": metrics["concurrent_sessions"],
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "uptime_hours": self._calculate_uptime_hours()
            }
        }

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method that each agent must implement.
        
        This method should contain the core processing logic specific to each agent.
        It should follow this pattern:
        
        1. Validate input using self.validate_input()
        2. Extract required data from input_data
        3. Perform agent-specific processing
        4. Calculate confidence score if applicable
        5. Return formatted output using self.format_output()
        
        Args:
            input_data: Standardized input dictionary
            
        Returns:
            Standardized output dictionary
        """
        pass

    def cleanup(self):
        """Clean up resources and prepare for shutdown."""
        self.logger.info(f"Cleaning up {self.agent_name}")
        
        # Force garbage collection
        gc.collect()
        
        # Log final metrics
        if self.metrics_enabled:
            final_status = self.get_comprehensive_status()
            self.logger.info(f"Final status: {final_status['status']} "
                           f"({final_status['performance_summary']['total_calls']} total calls)")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"{self.agent_name}(type={self.agent_type}, v{self.agent_version})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"{self.__class__.__name__}("
                f"type='{self.agent_type}', "
                f"version='{self.agent_version}', "
                f"environment='{self.config.get('environment')}', "
                f"metrics_enabled={self.metrics_enabled})")

    # Helper methods for internal use
    def _validate_output_structure(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output structure meets requirements."""
        required_fields = ["agent_name", "success", "result"]
        warnings = []
        
        for field in required_fields:
            if field not in output:
                warnings.append(f"Missing required field: {field}")
        
        return {
            "is_valid": len(warnings) == 0,
            "warnings": warnings
        }

    def _update_timing_metrics(self, processing_time: float):
        """Update timing-related metrics."""
        if not self.metrics_enabled:
            return
            
        metrics = self.performance_metrics
        metrics["total_processing_time"] += processing_time
        metrics["min_processing_time"] = min(metrics["min_processing_time"], processing_time)
        metrics["max_processing_time"] = max(metrics["max_processing_time"], processing_time)
        
        if metrics["total_calls"] > 0:
            metrics["average_processing_time"] = metrics["total_processing_time"] / metrics["total_calls"]
        
        metrics["last_call_time"] = datetime.now().isoformat()

    def _update_success_metrics(self, confidence: float = None):
        """Update metrics for successful processing."""
        if not self.metrics_enabled:
            return
            
        metrics = self.performance_metrics
        metrics["total_calls"] += 1
        metrics["successful_calls"] += 1
        
        if confidence is not None:
            metrics["confidence_scores"].append(confidence)
            metrics["average_confidence"] = sum(metrics["confidence_scores"]) / len(metrics["confidence_scores"])
            
            if confidence > 0.7:
                metrics["quality_threshold_met"] += 1

    def _update_error_metrics(self, error: Exception):
        """Update metrics for error tracking."""
        if not self.metrics_enabled:
            return
            
        metrics = self.performance_metrics
        metrics["total_calls"] += 1
        metrics["error_calls"] += 1
        
        error_type = error.__class__.__name__
        metrics["error_types"][error_type] = metrics["error_types"].get(error_type, 0) + 1
        
        metrics["last_error"] = {
            "type": error_type,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if not self.metrics_enabled:
            return 0.0
            
        metrics = self.performance_metrics
        total_cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
        
        if total_cache_requests == 0:
            return 0.0
            
        return (metrics["cache_hits"] / total_cache_requests) * 100

    def _calculate_uptime_hours(self) -> float:
        """Calculate agent uptime in hours."""
        try:
            init_time = datetime.fromisoformat(self.initialized_at)
            uptime = datetime.now() - init_time
            return uptime.total_seconds() / 3600
        except:
            return 0.0

    def _get_error_recovery_suggestions(self, error: Exception) -> List[str]:
        """Get contextual error recovery suggestions."""
        suggestions = []
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            suggestions.append("Increase timeout value or optimize processing")
        if "memory" in error_str:
            suggestions.append("Reduce input size or restart agent")
        if "rate limit" in error_str:
            suggestions.append("Implement exponential backoff or reduce request frequency")
        if "connection" in error_str:
            suggestions.append("Check network connectivity and service availability")
        
        return suggestions if suggestions else ["Contact system administrator"]

    def _get_retry_attempts_remaining(self) -> int:
        """Get number of retry attempts remaining."""
        return max(0, self.error_recovery["max_retries"] - 
                  self.performance_metrics.get("retry_calls", 0))

    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        return (self.performance_metrics.get("circuit_breaker_trips", 0) > 
                self.error_recovery["circuit_breaker_threshold"])

    def _should_immediate_retry(self, error: Exception) -> bool:
        """Determine if immediate retry is recommended."""
        transient_errors = ["ConnectionError", "TimeoutError", "TemporaryFailure"]
        return error.__class__.__name__ in transient_errors

    def _should_delayed_retry(self, error: Exception) -> bool:
        """Determine if delayed retry is recommended."""
        return "rate limit" in str(error).lower() or "quota" in str(error).lower()

    def _has_fallback_mechanism(self) -> bool:
        """Check if agent has fallback processing capability."""
        return hasattr(self, '_fallback_process') and callable(self._fallback_process)

    def _requires_manual_intervention(self, error: Exception) -> bool:
        """Determine if error requires manual intervention."""
        critical_errors = ["ConfigurationError", "AuthenticationError", "PermissionError"]
        return error.__class__.__name__ in critical_errors
