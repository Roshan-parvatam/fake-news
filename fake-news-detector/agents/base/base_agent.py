# agents/base/base_agent.py
"""
Base Agent Class for Modular Fake News Detection System

This module provides the foundational BaseAgent class that all fake news detection agents inherit from.
It provides common functionality while avoiding hardcoded models and parameters, making it perfect
for LangGraph orchestration.

Key Features:
- Abstract base class for consistent agent interfaces
- Configuration-driven approach (no hardcoding)
- Standardized input/output formats for LangGraph
- Common logging and error handling
- Performance tracking utilities
- Flexible model calling interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
import time
import os
from datetime import datetime
import json
from pathlib import Path


class BaseAgent(ABC):
    """
    🤖 BASE AGENT CLASS FOR MODULAR ARCHITECTURE
    
    This is the foundational class that all fake news detection agents inherit from.
    It provides common functionality while maintaining flexibility for LangGraph integration:
    
    - **Configuration-Driven**: No hardcoded models or parameters
    - **Standardized Interface**: Consistent input/output formats
    - **LangGraph Ready**: Compatible with state management and orchestration
    - **Modular Design**: Each agent can be developed and tested independently
    - **Performance Tracking**: Built-in metrics and monitoring
    - **Error Handling**: Graceful error handling and recovery
    
    USAGE EXAMPLE:
    ```
    class MyAgent(BaseAgent):
        def __init__(self, config: Dict = None):
            super().__init__(config)
        
        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            # Agent-specific processing logic
            result = self._analyze_content(input_data['text'])
            return self._format_output(result)
    ```
    
    LANGGRAPH INTEGRATION:
    Each agent returns standardized format compatible with LangGraph state management:
    {
        "agent_name": "AgentName",
        "result": {...},
        "confidence": 0.85,
        "metadata": {...},
        "success": True
    }
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        🏗️ INITIALIZE BASE AGENT WITH FLEXIBLE CONFIGURATION
        
        Args:
            config: Optional configuration dictionary. Can include:
                {
                    "model_name": "model_identifier",      # For LLM agents
                    "temperature": 0.3,                    # LLM parameters
                    "max_tokens": 2048,                    # Token limits
                    "timeout": 30,                         # API timeouts
                    "log_level": "INFO",                   # Logging level
                    "enable_metrics": True,                # Performance tracking
                    "custom_params": {...}                 # Agent-specific parameters
                }
        
        CONFIGURATION PRINCIPLES:
        - No hardcoded values - everything comes from config
        - Sensible defaults for all parameters
        - Easy to override for different environments
        - Compatible with LangGraph configuration management
        """
        # Set up configuration with defaults
        self.config = self._setup_configuration(config)
        
        # Set up logging for this agent
        self.logger = self._setup_logger()
        
        # Agent identification
        self.agent_name = self.__class__.__name__
        self.agent_type = self._determine_agent_type()
        self.initialized_at = datetime.now().isoformat()
        
        # Performance tracking (if enabled)
        self.metrics_enabled = self.config.get("enable_metrics", True)
        if self.metrics_enabled:
            self.performance_metrics = self._initialize_metrics()
        
        # State management for LangGraph compatibility
        self.last_input = None
        self.last_output = None
        self.processing_history = []
        
        self.logger.info(f"✅ {self.agent_name} initialized with modular configuration")
    
    def _setup_configuration(self, user_config: Optional[Dict]) -> Dict[str, Any]:
        """
        ⚙️ SETUP CONFIGURATION WITH DEFAULTS
        
        Creates a complete configuration by merging user config with sensible defaults.
        This approach avoids hardcoding while ensuring all agents have what they need.
        
        Args:
            user_config: User-provided configuration (can be None)
            
        Returns:
            Complete configuration dictionary with all required keys
        """
        # Default configuration that works for most agents
        default_config = {
            # Model configuration (for LLM agents)
            "model_name": None,                    # No default model - must be specified by agent
            "temperature": 0.3,                    # Balanced creativity/consistency
            "max_tokens": 2048,                    # Reasonable response length
            "timeout": 30,                         # 30 second timeout
            
            # System configuration
            "log_level": "INFO",                   # Logging verbosity
            "enable_metrics": True,                # Performance tracking
            "enable_caching": False,               # Response caching
            "retry_attempts": 2,                   # Number of retries on failure
            
            # LangGraph integration
            "state_key": None,                     # Key for storing results in LangGraph state
            "next_agents": [],                     # Suggested next agents in workflow
            "parallel_enabled": False,             # Whether this agent can run in parallel
            
            # Agent-specific customization
            "custom_params": {},                   # Agent-specific parameters
            "preprocessing_enabled": True,         # Whether to preprocess inputs
            "postprocessing_enabled": True,        # Whether to postprocess outputs
            
            # Development and debugging
            "debug_mode": False,                   # Enhanced logging and validation
            "test_mode": False,                    # Use test configurations
            "mock_responses": False                # Use mock responses for testing
        }
        
        # Merge user config with defaults (user config takes precedence)
        if user_config:
            # Deep merge for nested dictionaries
            for key, value in user_config.items():
                if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        return default_config
    
    def _setup_logger(self) -> logging.Logger:
        """
        📝 SET UP LOGGING FOR THIS AGENT
        
        Creates a logger with appropriate formatting and level based on configuration.
        Each agent gets its own logger namespace for easier debugging.
        
        Returns:
            Configured logger instance
        """
        logger_name = f"agents.{self.agent_name}"
        logger = logging.getLogger(logger_name)
        
        # Only add handler if it doesn't already exist (prevents duplicate logs)
        if not logger.handlers:
            # Create console handler with formatting
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Set logging level from configuration
            log_level = self.config.get('log_level', 'INFO').upper()
            logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        return logger
    
    def _determine_agent_type(self) -> str:
        """
        🏷️ DETERMINE AGENT TYPE BASED ON CLASS NAME
        
        Categorizes agents for easier management and routing in LangGraph.
        
        Returns:
            String indicating agent type (e.g., "classifier", "analyzer", "evaluator")
        """
        name_lower = self.agent_name.lower()
        
        if 'classifier' in name_lower or 'bert' in name_lower:
            return "classifier"
        elif 'explanation' in name_lower or 'llm' in name_lower:
            return "generator"
        elif 'source' in name_lower or 'credible' in name_lower:
            return "recommender"
        elif 'claim' in name_lower or 'extractor' in name_lower:
            return "extractor"
        elif 'context' in name_lower or 'analyzer' in name_lower:
            return "analyzer"
        elif 'evidence' in name_lower or 'evaluator' in name_lower:
            return "evaluator"
        else:
            return "generic"
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """
        📊 INITIALIZE PERFORMANCE METRICS TRACKING
        
        Sets up performance tracking for monitoring and optimization.
        
        Returns:
            Dictionary containing metric counters and accumulators
        """
        return {
            # Basic counters
            "total_calls": 0,                      # Total number of process() calls
            "successful_calls": 0,                 # Successful completions
            "error_calls": 0,                      # Failed calls
            
            # Timing metrics
            "total_processing_time": 0.0,          # Total time spent processing
            "average_processing_time": 0.0,        # Average time per call
            "min_processing_time": float('inf'),   # Fastest processing time
            "max_processing_time": 0.0,            # Slowest processing time
            
            # Quality metrics (agent-specific)
            "confidence_scores": [],               # List of confidence scores
            "average_confidence": 0.0,             # Average confidence across calls
            
            # Resource usage
            "memory_usage": [],                    # Memory usage samples
            "cache_hits": 0,                       # Cache hit count (if caching enabled)
            "cache_misses": 0,                     # Cache miss count
            
            # Error tracking
            "error_types": {},                     # Count of different error types
            "last_error": None,                    # Details of most recent error
            "error_rate": 0.0                      # Percentage of calls that failed
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        ✅ VALIDATE INPUT DATA
        
        Performs common input validation that all agents need.
        Individual agents can override this for specific validation.
        
        Args:
            input_data: Input data dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(input_data, dict):
            return False, "Input must be a dictionary"
        
        # Check for required common fields
        if not input_data:
            return False, "Input data cannot be empty"
        
        # Validate text content if present
        if 'text' in input_data:
            text = input_data['text']
            if not isinstance(text, str):
                return False, "Text field must be a string"
            if len(text.strip()) == 0:
                return False, "Text content cannot be empty"
            if len(text) > 50000:  # Reasonable upper limit
                return False, "Text content too long (max 50,000 characters)"
        
        return True, None
    
    def format_output(self, result: Any, confidence: float = None, metadata: Dict = None) -> Dict[str, Any]:
        """
        📤 FORMAT OUTPUT FOR LANGGRAPH COMPATIBILITY
        
        Standardizes output format across all agents for consistent LangGraph integration.
        
        Args:
            result: The main result from agent processing
            confidence: Confidence score (0.0-1.0) if applicable
            metadata: Additional metadata about the processing
            
        Returns:
            Standardized output dictionary compatible with LangGraph state management
        """
        output = {
            # Standard fields for all agents
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "result": result,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            
            # Optional fields
            "confidence": confidence,
            "metadata": metadata or {},
            
            # LangGraph routing information
            "next_agents": self.config.get("next_agents", []),
            "state_key": self.config.get("state_key"),
            
            # Processing information
            "processing_time": getattr(self, '_last_processing_time', None),
            "input_hash": self._hash_input(self.last_input) if self.last_input else None
        }
        
        # Add agent-specific output fields if configured
        custom_fields = self.config.get("custom_output_fields", {})
        output.update(custom_fields)
        
        return output
    
    def format_error_output(self, error: Exception, input_data: Dict = None) -> Dict[str, Any]:
        """
        ❌ FORMAT ERROR OUTPUT FOR CONSISTENT ERROR HANDLING
        
        Creates standardized error output that LangGraph can handle gracefully.
        
        Args:
            error: The exception that occurred
            input_data: The input that caused the error (optional)
            
        Returns:
            Standardized error output dictionary
        """
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "result": None,
            "success": False,
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat(),
            "input_received": input_data is not None,
            "retry_recommended": self._should_retry_error(error)
        }
    
    def _should_retry_error(self, error: Exception) -> bool:
        """
        🔄 DETERMINE IF ERROR IS RETRYABLE
        
        Analyzes errors to determine if retrying might succeed.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if retry is recommended, False otherwise
        """
        retryable_errors = [
            "timeout", "connection", "rate limit", "service unavailable",
            "temporary", "network", "503", "502", "429"
        ]
        
        error_str = str(error).lower()
        return any(retryable in error_str for retryable in retryable_errors)
    
    def _hash_input(self, input_data: Dict[str, Any]) -> str:
        """
        🔐 CREATE HASH OF INPUT DATA
        
        Creates a hash of input data for caching and deduplication.
        
        Args:
            input_data: Input data to hash
            
        Returns:
            Hash string of the input
        """
        import hashlib
        
        try:
            # Convert to JSON string and hash
            input_str = json.dumps(input_data, sort_keys=True, default=str)
            return hashlib.md5(input_str.encode()).hexdigest()
        except:
            # Fallback for unhashable content
            return hashlib.md5(str(input_data).encode()).hexdigest()
    
    def _start_processing_timer(self):
        """⏱️ Start timing for performance metrics"""
        self._processing_start_time = time.time()
    
    def _end_processing_timer(self):
        """⏱️ End timing and update metrics"""
        if hasattr(self, '_processing_start_time'):
            processing_time = time.time() - self._processing_start_time
            self._last_processing_time = processing_time
            
            if self.metrics_enabled:
                self._update_timing_metrics(processing_time)
    
    def _update_timing_metrics(self, processing_time: float):
        """📊 Update timing-related performance metrics"""
        metrics = self.performance_metrics
        
        metrics["total_processing_time"] += processing_time
        metrics["min_processing_time"] = min(metrics["min_processing_time"], processing_time)
        metrics["max_processing_time"] = max(metrics["max_processing_time"], processing_time)
        
        # Update average
        total_calls = metrics["total_calls"]
        if total_calls > 0:
            metrics["average_processing_time"] = metrics["total_processing_time"] / total_calls
    
    def _update_success_metrics(self, confidence: float = None):
        """📈 Update metrics for successful processing"""
        if not self.metrics_enabled:
            return
        
        metrics = self.performance_metrics
        metrics["total_calls"] += 1
        metrics["successful_calls"] += 1
        
        if confidence is not None:
            metrics["confidence_scores"].append(confidence)
            metrics["average_confidence"] = sum(metrics["confidence_scores"]) / len(metrics["confidence_scores"])
    
    def _update_error_metrics(self, error: Exception):
        """📉 Update metrics for failed processing"""
        if not self.metrics_enabled:
            return
        
        metrics = self.performance_metrics
        metrics["total_calls"] += 1
        metrics["error_calls"] += 1
        
        # Track error types
        error_type = error.__class__.__name__
        metrics["error_types"][error_type] = metrics["error_types"].get(error_type, 0) + 1
        
        # Store last error details
        metrics["last_error"] = {
            "type": error_type,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        # Update error rate
        metrics["error_rate"] = (metrics["error_calls"] / metrics["total_calls"]) * 100
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        📊 GET COMPREHENSIVE PERFORMANCE METRICS
        
        Returns all performance metrics for monitoring and optimization.
        
        Returns:
            Dictionary containing all performance data
        """
        if not self.metrics_enabled:
            return {"metrics_enabled": False}
        
        metrics = self.performance_metrics.copy()
        
        # Add derived metrics
        total_calls = metrics["total_calls"]
        if total_calls > 0:
            metrics["success_rate"] = (metrics["successful_calls"] / total_calls) * 100
        else:
            metrics["success_rate"] = 0.0
        
        # Add agent information
        metrics.update({
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "initialized_at": self.initialized_at,
            "config_summary": {
                "model_name": self.config.get("model_name"),
                "temperature": self.config.get("temperature"),
                "debug_mode": self.config.get("debug_mode", False)
            }
        })
        
        return metrics
    
    def reset_metrics(self):
        """🔄 Reset all performance metrics"""
        if self.metrics_enabled:
            self.performance_metrics = self._initialize_metrics()
            self.logger.info("📊 Performance metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """
        ❤️ GET AGENT HEALTH STATUS
        
        Returns current status and health indicators for monitoring.
        
        Returns:
            Dictionary containing status information
        """
        status = {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": "healthy",
            "initialized_at": self.initialized_at,
            "last_activity": getattr(self, '_last_activity', None),
            "config_valid": True,
            "issues": []
        }
        
        if self.metrics_enabled:
            metrics = self.performance_metrics
            
            # Check for health issues
            if metrics["error_rate"] > 50:
                status["status"] = "degraded"
                status["issues"].append("High error rate")
            
            if metrics["total_calls"] > 0 and metrics["average_processing_time"] > 30:
                status["status"] = "slow"
                status["issues"].append("Slow response times")
            
            # Add performance summary
            status["performance_summary"] = {
                "total_calls": metrics["total_calls"],
                "success_rate": metrics.get("success_rate", 0),
                "average_time": metrics["average_processing_time"],
                "error_rate": metrics["error_rate"]
            }
        
        return status
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 ABSTRACT PROCESS METHOD
        
        Each agent must implement this method to define its core processing logic.
        This ensures all agents have a consistent interface while allowing
        each to implement its specific functionality.
        
        Args:
            input_data: Standardized input dictionary containing:
                - text: Article text (for text-processing agents)
                - previous_results: Results from previous agents
                - config_override: Runtime configuration overrides
                - metadata: Additional context information
                
        Returns:
            Standardized output dictionary for LangGraph compatibility:
                {
                    "agent_name": "AgentName",
                    "result": agent_specific_result,
                    "confidence": 0.85,
                    "success": True,
                    "metadata": {...},
                    "next_agents": [...]
                }
        
        Implementation Pattern:
        ```
        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            # 1. Validate input
            is_valid, error_msg = self.validate_input(input_data)
            if not is_valid:
                return self.format_error_output(ValueError(error_msg), input_data)
            
            # 2. Start timing
            self._start_processing_timer()
            
            try:
                # 3. Extract required data
                text = input_data.get('text', '')
                previous_results = input_data.get('previous_results', {})
                
                # 4. Perform agent-specific processing
                result = self._perform_analysis(text, previous_results)
                confidence = self._calculate_confidence(result)
                
                # 5. Format successful output
                self._end_processing_timer()
                self._update_success_metrics(confidence)
                
                return self.format_output(
                    result=result,
                    confidence=confidence,
                    metadata={"processing_info": "..."}
                )
                
            except Exception as e:
                self._end_processing_timer()
                self._update_error_metrics(e)
                return self.format_error_output(e, input_data)
        ```
        """
        pass
    
    def __str__(self) -> str:
        """📝 String representation for debugging"""
        return f"{self.agent_name}(type={self.agent_type}, config={bool(self.config)})"
    
    def __repr__(self) -> str:
        """🔧 Detailed representation for debugging"""
        return (f"{self.__class__.__name__}("
                f"agent_type='{self.agent_type}', "
                f"config_keys={list(self.config.keys())}, "
                f"metrics_enabled={self.metrics_enabled})")


# 🧪 TESTING AND EXAMPLE USAGE
if __name__ == "__main__":
    """
    Example usage and testing of the BaseAgent class.
    
    This demonstrates how to:
    1. Create a simple agent that inherits from BaseAgent
    2. Use the standardized input/output format
    3. Handle configuration and metrics
    4. Prepare for LangGraph integration
    """
    
    print("🧪 Testing Modular BaseAgent for LangGraph Integration")
    print("=" * 60)
    
    # Example agent implementation
    class TestAgent(BaseAgent):
        def __init__(self, config: Dict = None):
            super().__init__(config)
        
        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Example implementation of the process method"""
            # Validate input
            is_valid, error_msg = self.validate_input(input_data)
            if not is_valid:
                return self.format_error_output(ValueError(error_msg), input_data)
            
            # Start timing
            self._start_processing_timer()
            
            try:
                # Simulate processing
                text = input_data.get('text', '')
                result = f"Processed {len(text)} characters"
                confidence = 0.85
                
                # Update metrics and format output
                self._end_processing_timer()
                self._update_success_metrics(confidence)
                
                return self.format_output(
                    result=result,
                    confidence=confidence,
                    metadata={"characters_processed": len(text)}
                )
                
            except Exception as e:
                self._end_processing_timer()
                self._update_error_metrics(e)
                return self.format_error_output(e, input_data)
    
    try:
        # Test 1: Basic agent creation
        print("\n📝 Test 1: Creating test agent...")
        config = {
            "model_name": "test-model",
            "temperature": 0.5,
            "enable_metrics": True,
            "debug_mode": True
        }
        agent = TestAgent(config)
        print(f"✅ Agent created: {agent}")
        
        # Test 2: Process valid input
        print("\n📝 Test 2: Processing valid input...")
        input_data = {
            "text": "This is a test article for processing.",
            "metadata": {"test": True}
        }
        result = agent.process(input_data)
        print(f"✅ Processing successful: {result['success']}")
        print(f"   Result: {result['result']}")
        print(f"   Confidence: {result['confidence']}")
        
        # Test 3: Process invalid input
        print("\n📝 Test 3: Processing invalid input...")
        invalid_input = {"invalid": "data"}
        error_result = agent.process(invalid_input)
        print(f"❌ Processing failed as expected: {not error_result['success']}")
        print(f"   Error: {error_result['error']['message']}")
        
        # Test 4: Check metrics
        print("\n📝 Test 4: Checking performance metrics...")
        metrics = agent.get_performance_metrics()
        print(f"✅ Total calls: {metrics['total_calls']}")
        print(f"   Success rate: {metrics['success_rate']:.1f}%")
        print(f"   Average time: {metrics['average_processing_time']:.3f}s")
        
        # Test 5: Check status
        print("\n📝 Test 5: Checking agent status...")
        status = agent.get_status()
        print(f"✅ Agent status: {status['status']}")
        print(f"   Agent type: {status['agent_type']}")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"🚀 BaseAgent is ready for LangGraph integration!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
