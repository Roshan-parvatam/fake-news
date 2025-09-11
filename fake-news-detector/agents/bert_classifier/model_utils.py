# agents/bert_classifier/model_utils.py

"""
Enhanced Model Management Utilities for BERT Classifier - Production Ready

Production-grade device management and model loading with comprehensive error handling,
performance monitoring, and integration with enhanced agent architecture.

Features:
- Dynamic device detection with configuration override capabilities
- Comprehensive model loading with validation and error recovery
- Enhanced error handling with recovery strategies
- Performance monitoring and resource tracking
- Memory management and optimization
- Detailed logging with session tracking
- Health monitoring and status reporting
- Integration with enhanced exception system

Version: 3.2.0 - Enhanced Production Edition
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging
import json
import time
from datetime import datetime
import psutil
import gc

# Enhanced exception integration
try:
    from agents.llm_explanation.exceptions import (
        handle_llm_explanation_exception,
        ErrorContext,
        log_exception_with_context
    )
    _enhanced_exceptions_available = True
except ImportError:
    _enhanced_exceptions_available = False


class DeviceManager:
    """
    Enhanced Device Management with Production Features

    Automatically detects and manages the best available processing device with
    comprehensive configuration support, performance monitoring, and health tracking.

    Features:
    - Intelligent device detection with fallback strategies
    - Configuration override capabilities with validation
    - Comprehensive device information and capabilities reporting
    - Performance monitoring and resource tracking
    - Memory usage monitoring and optimization
    - Health status reporting for production monitoring
    - Enhanced error handling with recovery strategies
    """

    def __init__(self, device_preference: str = 'auto', config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced device manager with comprehensive configuration.

        Args:
            device_preference: Device preference ('auto', 'cpu', 'cuda', 'mps', or specific device)
            config: Optional configuration dictionary for advanced settings
        """
        self.device_preference = device_preference
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Configuration parameters
        self.enable_metrics = self.config.get('enable_metrics', True)
        self.memory_threshold_gb = self.config.get('memory_threshold_gb', 2.0)
        self.enable_optimization = self.config.get('enable_optimization', True)

        # Performance tracking
        self.device_metrics = {
            'device_switches': 0,
            'memory_allocations': 0,
            'optimization_calls': 0,
            'health_checks': 0,
            'last_health_check': None,
            'initialization_time': time.time()
        }

        # Detect and set device with comprehensive validation
        self.device = self._detect_and_set_device()
        
        # Cache device info for performance
        self._device_info = None
        self._last_memory_check = 0
        self._cached_memory_info = {}

        # Initialize device optimization if enabled
        if self.enable_optimization:
            self._initialize_device_optimization()

        self.logger.info(f"Enhanced DeviceManager initialized: {self.device}")

    def _detect_and_set_device(self) -> torch.device:
        """
        Enhanced device detection with comprehensive fallback strategies and validation.

        Returns:
            torch.device: Best available device based on preference and capabilities
        """
        try:
            if self.device_preference != 'auto':
                # Try to use specified device with validation
                try:
                    device = torch.device(self.device_preference)
                    if self._validate_device(device):
                        self.logger.info(f"Using specified device: {device}")
                        return device
                    else:
                        self.logger.warning(f"Specified device '{self.device_preference}' validation failed")
                except Exception as e:
                    self.logger.warning(f"Specified device '{self.device_preference}' not available: {e}")
                
                self.logger.info("Falling back to auto-detection...")

            # Enhanced auto-detection logic with comprehensive checks
            available_devices = []

            # Check CUDA availability and capabilities
            if torch.cuda.is_available():
                try:
                    cuda_device = torch.device("cuda:0")
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    
                    if gpu_memory_gb >= self.memory_threshold_gb:
                        available_devices.append({
                            'device': cuda_device,
                            'type': 'cuda',
                            'name': gpu_name,
                            'memory_gb': gpu_memory_gb,
                            'priority': 1,
                            'performance_score': gpu_memory_gb * 100
                        })
                        self.logger.info(f"CUDA available: {gpu_name} ({gpu_memory_gb:.1f}GB)")
                    else:
                        self.logger.warning(f"CUDA GPU memory insufficient: {gpu_memory_gb:.1f}GB < {self.memory_threshold_gb}GB")
                except Exception as e:
                    self.logger.warning(f"CUDA detection failed: {e}")

            # Check MPS (Apple Silicon) availability
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    mps_device = torch.device("mps")
                    available_devices.append({
                        'device': mps_device,
                        'type': 'mps',
                        'name': 'Apple Silicon GPU',
                        'memory_gb': 0.0,  # MPS shares system memory
                        'priority': 2,
                        'performance_score': 80
                    })
                    self.logger.info("Apple Metal Performance Shaders (MPS) available")
                except Exception as e:
                    self.logger.warning(f"MPS detection failed: {e}")

            # CPU is always available as fallback
            try:
                cpu_device = torch.device("cpu")
                cpu_info = self._get_cpu_info()
                available_devices.append({
                    'device': cpu_device,
                    'type': 'cpu',
                    'name': cpu_info.get('name', 'CPU'),
                    'memory_gb': cpu_info.get('memory_gb', 0.0),
                    'priority': 3,
                    'performance_score': cpu_info.get('cores', 1) * 10
                })
            except Exception as e:
                self.logger.error(f"CPU device creation failed: {e}")
                # Absolute fallback
                return torch.device("cpu")

            # Select best available device
            if available_devices:
                # Sort by priority (lower is better) then by performance score (higher is better)
                best_device = min(available_devices, key=lambda x: (x['priority'], -x['performance_score']))
                
                self.logger.info(f"Selected device: {best_device['name']} ({best_device['type']})")
                if best_device['memory_gb'] > 0:
                    self.logger.info(f"Available memory: {best_device['memory_gb']:.1f}GB")
                
                return best_device['device']
            else:
                # Ultimate fallback
                self.logger.warning("No suitable devices found, using CPU")
                return torch.device("cpu")

        except Exception as e:
            error_msg = f"Device detection failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Enhanced exception handling
            if _enhanced_exceptions_available:
                context = ErrorContext(operation="device_detection")
                standardized_error = handle_llm_explanation_exception(e, context)
                log_exception_with_context(standardized_error, None, {'component': 'DeviceManager'})
            
            # Absolute fallback to CPU
            return torch.device("cpu")

    def _validate_device(self, device: torch.device) -> bool:
        """
        Validate device availability and capabilities.

        Args:
            device: Device to validate

        Returns:
            bool: True if device is valid and available
        """
        try:
            # Basic availability check
            if device.type == 'cuda':
                if not torch.cuda.is_available():
                    return False
                # Try to allocate a small tensor
                test_tensor = torch.tensor([1.0], device=device)
                del test_tensor
                torch.cuda.empty_cache()
                return True
            elif device.type == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    return False
                # Try to allocate a small tensor
                test_tensor = torch.tensor([1.0], device=device)
                del test_tensor
                return True
            elif device.type == 'cpu':
                # CPU is always available
                return True
            else:
                return False

        except Exception as e:
            self.logger.warning(f"Device validation failed for {device}: {e}")
            return False

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get comprehensive CPU information."""
        try:
            cpu_info = {
                'name': 'CPU',
                'cores': psutil.cpu_count(logical=True),
                'physical_cores': psutil.cpu_count(logical=False),
                'memory_gb': psutil.virtual_memory().total / 1e9,
                'available_memory_gb': psutil.virtual_memory().available / 1e9
            }
            return cpu_info
        except Exception as e:
            self.logger.warning(f"Failed to get CPU info: {e}")
            return {'name': 'CPU', 'cores': 1, 'memory_gb': 4.0}

    def _initialize_device_optimization(self):
        """Initialize device-specific optimizations."""
        try:
            if self.device.type == 'cuda':
                # CUDA-specific optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.logger.debug("CUDA optimizations enabled")
            elif self.device.type == 'mps':
                # MPS-specific optimizations
                self.logger.debug("MPS optimizations enabled")
            
            self.device_metrics['optimization_calls'] += 1
            
        except Exception as e:
            self.logger.warning(f"Device optimization failed: {e}")

    def get_device(self) -> torch.device:
        """
        Get the current device.

        Returns:
            torch.device: Current device instance
        """
        return self.device

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get comprehensive device information with caching for performance.

        Returns:
            Dictionary with detailed device information and capabilities
        """
        if self._device_info is None:
            self._device_info = self._build_device_info()
        
        return self._device_info.copy()

    def _build_device_info(self) -> Dict[str, Any]:
        """Build comprehensive device information dictionary."""
        try:
            base_info = {
                "device_type": self.device.type,
                "device_preference_set": self.device_preference,
                "device_string": str(self.device),
                "is_available": True,
                "enhanced_version": "3.2.0",
                "metrics_enabled": self.enable_metrics
            }

            if self.device.type == "cuda":
                cuda_info = self._get_cuda_info()
                base_info.update(cuda_info)
            elif self.device.type == "mps":
                mps_info = self._get_mps_info()
                base_info.update(mps_info)
            else:  # CPU
                cpu_info = self._get_cpu_device_info()
                base_info.update(cpu_info)

            return base_info

        except Exception as e:
            self.logger.error(f"Failed to build device info: {e}")
            return {
                "device_type": "unknown",
                "error": str(e),
                "device_string": str(self.device)
            }

    def _get_cuda_info(self) -> Dict[str, Any]:
        """Get comprehensive CUDA device information."""
        try:
            device_id = self.device.index or 0
            props = torch.cuda.get_device_properties(device_id)
            
            return {
                "device_name": torch.cuda.get_device_name(device_id),
                "memory_total_gb": round(props.total_memory / 1e9, 2),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(device_id) / 1e9, 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(device_id) / 1e9, 2),
                "memory_free_gb": round((props.total_memory - torch.cuda.memory_reserved(device_id)) / 1e9, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multiprocessor_count": props.multi_processor_count,
                "max_threads_per_multiprocessor": props.max_threads_per_multi_processor,
                "optimization_notes": "GPU available - excellent performance expected",
                "driver_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "unknown"
            }
        except Exception as e:
            self.logger.warning(f"Failed to get CUDA info: {e}")
            return {"device_name": "CUDA GPU", "error": str(e)}

    def _get_mps_info(self) -> Dict[str, Any]:
        """Get MPS device information."""
        try:
            return {
                "device_name": "Apple Silicon GPU",
                "backend": "Metal Performance Shaders",
                "shared_memory": "System memory shared with CPU",
                "optimization_notes": "MPS available - good performance on Apple Silicon",
                "pytorch_version": torch.__version__
            }
        except Exception as e:
            return {"device_name": "Apple MPS", "error": str(e)}

    def _get_cpu_device_info(self) -> Dict[str, Any]:
        """Get CPU device information."""
        try:
            cpu_info = self._get_cpu_info()
            return {
                **cpu_info,
                "device_name": "CPU",
                "optimization_notes": "Using CPU - consider GPU for better performance",
                "threading": f"{torch.get_num_threads()} threads available"
            }
        except Exception as e:
            return {"device_name": "CPU", "error": str(e)}

    def get_memory_usage(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get current memory usage information with caching.

        Args:
            detailed: Whether to include detailed memory breakdown

        Returns:
            Dictionary with memory usage information
        """
        current_time = time.time()
        
        # Use cached info if recent (within 5 seconds)
        if current_time - self._last_memory_check < 5.0 and self._cached_memory_info:
            return self._cached_memory_info.copy()

        try:
            memory_info = {"device_type": self.device.type}

            if self.device.type == "cuda":
                device_id = self.device.index or 0
                memory_info.update({
                    "memory_allocated_mb": round(torch.cuda.memory_allocated(device_id) / 1024**2, 2),
                    "memory_reserved_mb": round(torch.cuda.memory_reserved(device_id) / 1024**2, 2),
                    "memory_free_mb": round((torch.cuda.get_device_properties(device_id).total_memory - 
                                           torch.cuda.memory_reserved(device_id)) / 1024**2, 2),
                    "memory_total_mb": round(torch.cuda.get_device_properties(device_id).total_memory / 1024**2, 2)
                })
                
                if detailed:
                    memory_info.update({
                        "memory_stats": torch.cuda.memory_stats(device_id),
                        "memory_summary": torch.cuda.memory_summary(device_id)
                    })

            elif self.device.type == "mps":
                # MPS shares system memory
                vm = psutil.virtual_memory()
                memory_info.update({
                    "shared_memory_mb": round(vm.total / 1024**2, 2),
                    "available_memory_mb": round(vm.available / 1024**2, 2),
                    "memory_percent_used": vm.percent,
                    "note": "MPS uses shared system memory"
                })

            else:  # CPU
                vm = psutil.virtual_memory()
                memory_info.update({
                    "system_memory_mb": round(vm.total / 1024**2, 2),
                    "available_memory_mb": round(vm.available / 1024**2, 2),
                    "memory_percent_used": vm.percent
                })

            # Cache the result
            self._cached_memory_info = memory_info
            self._last_memory_check = current_time
            
            return memory_info

        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return {
                "device_type": self.device.type,
                "error": str(e),
                "note": "Memory tracking not available"
            }

    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Optimize device memory usage.

        Args:
            aggressive: Whether to use aggressive optimization strategies

        Returns:
            Dictionary with optimization results
        """
        try:
            optimization_results = {
                "device_type": self.device.type,
                "optimization_performed": False,
                "memory_freed_mb": 0.0,
                "timestamp": datetime.now().isoformat()
            }

            if self.device.type == "cuda":
                # Get memory before optimization
                memory_before = torch.cuda.memory_allocated()
                
                # Standard CUDA memory optimization
                torch.cuda.empty_cache()
                
                if aggressive:
                    # Aggressive optimization
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Calculate memory freed
                memory_after = torch.cuda.memory_allocated()
                memory_freed = max(0, memory_before - memory_after)
                
                optimization_results.update({
                    "optimization_performed": True,
                    "memory_freed_mb": round(memory_freed / 1024**2, 2),
                    "cuda_cache_cleared": True
                })

            else:
                # CPU/MPS memory optimization
                gc.collect()
                optimization_results.update({
                    "optimization_performed": True,
                    "garbage_collection_performed": True
                })

            self.device_metrics['optimization_calls'] += 1
            self.logger.info(f"Memory optimization completed: {optimization_results}")
            
            return optimization_results

        except Exception as e:
            error_msg = f"Memory optimization failed: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "device_type": self.device.type,
                "optimization_performed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive device health status for monitoring."""
        try:
            self.device_metrics['health_checks'] += 1
            self.device_metrics['last_health_check'] = datetime.now().isoformat()
            
            health_info = {
                "device_type": self.device.type,
                "device_available": True,
                "status": "healthy",
                "last_check": self.device_metrics['last_health_check']
            }

            # Device-specific health checks
            if self.device.type == "cuda":
                try:
                    # Check CUDA availability
                    if not torch.cuda.is_available():
                        health_info["status"] = "unhealthy"
                        health_info["issues"] = ["CUDA not available"]
                    else:
                        # Check memory usage
                        memory_info = self.get_memory_usage()
                        total_memory = memory_info.get("memory_total_mb", 0)
                        used_memory = memory_info.get("memory_allocated_mb", 0)
                        
                        if total_memory > 0:
                            memory_usage_percent = (used_memory / total_memory) * 100
                            health_info["memory_usage_percent"] = round(memory_usage_percent, 2)
                            
                            if memory_usage_percent > 90:
                                health_info["status"] = "warning"
                                health_info["issues"] = ["High memory usage"]
                            elif memory_usage_percent > 95:
                                health_info["status"] = "critical"
                                health_info["issues"] = ["Very high memory usage"]

                except Exception as e:
                    health_info["status"] = "degraded"
                    health_info["cuda_check_error"] = str(e)

            elif self.device.type == "mps":
                try:
                    # Basic MPS availability check
                    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                        health_info["status"] = "unhealthy"
                        health_info["issues"] = ["MPS not available"]
                except Exception as e:
                    health_info["status"] = "degraded"
                    health_info["mps_check_error"] = str(e)

            # Add performance metrics
            health_info["metrics"] = {
                "device_switches": self.device_metrics["device_switches"],
                "optimization_calls": self.device_metrics["optimization_calls"],
                "health_checks": self.device_metrics["health_checks"],
                "uptime_seconds": round(time.time() - self.device_metrics["initialization_time"], 2)
            }

            return health_info

        except Exception as e:
            self.logger.error(f"Health status check failed: {e}")
            return {
                "device_type": self.device.type,
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }


class ModelManager:
    """
    Enhanced Model Management with Production Features

    Handles BERT model loading, validation, and management with comprehensive
    error handling, performance monitoring, and health tracking.

    Features:
    - Comprehensive model loading with validation and integrity checks
    - Enhanced error handling with recovery strategies and detailed reporting
    - Performance monitoring and resource usage tracking
    - Model metadata management and version tracking
    - Memory usage optimization and monitoring
    - Health status reporting for production monitoring
    - Integration with enhanced exception system
    """

    def __init__(self, device_manager: DeviceManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced model manager with comprehensive configuration.

        Args:
            device_manager: DeviceManager instance for device handling
            config: Optional configuration dictionary for advanced settings
        """
        self.device_manager = device_manager
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Configuration parameters
        self.enable_metrics = self.config.get('enable_metrics', True)
        self.validate_model_output = self.config.get('validate_model_output', True)
        self.enable_model_optimization = self.config.get('enable_model_optimization', True)
        self.max_model_size_gb = self.config.get('max_model_size_gb', 5.0)

        # Model storage
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.model_name = None

        # Enhanced model metadata
        self.model_info = {}
        self.training_metadata = {}
        self.load_timestamp = None

        # Performance tracking
        self.model_metrics = {
            'load_attempts': 0,
            'successful_loads': 0,
            'validation_checks': 0,
            'inference_calls': 0,
            'optimization_calls': 0,
            'memory_optimizations': 0,
            'health_checks': 0,
            'last_health_check': None,
            'initialization_time': time.time()
        }

        self.logger.info("Enhanced ModelManager initialized")

    def load_model(self, model_path: Path, model_name: str, session_id: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load BERT model with comprehensive validation and enhanced error handling.

        Args:
            model_path: Path to the saved model directory
            model_name: Name/type of the BERT model
            session_id: Optional session ID for tracking

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            FileNotFoundError: If model files are missing
            ValueError: If model validation fails
            RuntimeError: If model loading fails critically
        """
        self.model_metrics['load_attempts'] += 1
        load_start_time = time.time()

        try:
            self.logger.info(f"Loading BERT model from: {model_path}", extra={'session_id': session_id})
            
            self.model_path = model_path
            self.model_name = model_name

            # Step 1: Comprehensive model directory validation
            self._validate_model_files(model_path, session_id)

            # Step 2: Load and validate tokenizer
            tokenizer = self._load_tokenizer(model_path, session_id)

            # Step 3: Load and validate BERT model
            model = self._load_bert_model(model_path, session_id)

            # Step 4: Load training metadata if available
            self._load_training_metadata(model_path, session_id)

            # Step 5: Move model to appropriate device with optimization
            model = self._setup_model_device(model, session_id)

            # Step 6: Comprehensive model functionality verification
            self._verify_model_functionality(model, tokenizer, session_id)

            # Step 7: Apply model optimizations if enabled
            if self.enable_model_optimization:
                model = self._optimize_model(model, session_id)

            # Step 8: Store references and build comprehensive info
            self.model = model
            self.tokenizer = tokenizer
            self.load_timestamp = datetime.now().isoformat()
            self._build_comprehensive_model_info()

            # Update success metrics
            self.model_metrics['successful_loads'] += 1
            load_time = time.time() - load_start_time

            self.logger.info(
                f"Model loaded successfully in {load_time:.3f}s",
                extra={'session_id': session_id, 'load_time': load_time}
            )

            return model, tokenizer

        except Exception as e:
            load_time = time.time() - load_start_time
            error_msg = f"Model loading failed after {load_time:.3f}s: {str(e)}"
            self.logger.error(error_msg, extra={'session_id': session_id})

            # Enhanced exception handling
            if _enhanced_exceptions_available:
                context = ErrorContext(
                    session_id=session_id,
                    operation="model_loading",
                    model_used=model_name,
                    processing_time=load_time
                )
                standardized_error = handle_llm_explanation_exception(e, context)
                log_exception_with_context(standardized_error, session_id, {'component': 'ModelManager'})

            raise

    def _validate_model_files(self, model_path: Path, session_id: Optional[str] = None):
        """Comprehensive model file validation with detailed error reporting."""
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {model_path}")

            if not model_path.is_dir():
                raise ValueError(f"Model path is not a directory: {model_path}")

            # Check for model files (multiple formats supported)
            model_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]
            has_model_file = any((model_path / file_name).exists() for file_name in model_files)

            # Required files check
            required_files = ["config.json"]
            tokenizer_files = ["vocab.txt", "tokenizer.json", "tokenizer_config.json"]
            has_tokenizer = any((model_path / file_name).exists() for file_name in tokenizer_files)

            missing_files = []
            
            if not has_model_file:
                missing_files.append(f"Model file ({', '.join(model_files)})")
            
            if not has_tokenizer:
                missing_files.append(f"Tokenizer files ({', '.join(tokenizer_files)})")
            
            for file_name in required_files:
                if not (model_path / file_name).exists():
                    missing_files.append(file_name)

            if missing_files:
                raise FileNotFoundError(f"Missing required model files: {missing_files}")

            # Validate config.json
            config_path = model_path / "config.json"
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                if 'model_type' not in config_data:
                    self.logger.warning("Model config missing 'model_type' field")
                    
                if config_data.get('model_type', '').lower() not in ['bert', 'roberta', 'distilbert']:
                    self.logger.warning(f"Unexpected model type: {config_data.get('model_type')}")
                    
            except Exception as e:
                self.logger.warning(f"Config validation failed: {e}")

            # Check model size
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            size_gb = total_size / 1e9
            
            if size_gb > self.max_model_size_gb:
                self.logger.warning(f"Large model detected: {size_gb:.2f}GB > {self.max_model_size_gb}GB")

            self.logger.debug(f"Model validation passed: {size_gb:.2f}GB", extra={'session_id': session_id})

        except Exception as e:
            error_msg = f"Model file validation failed: {str(e)}"
            self.logger.error(error_msg, extra={'session_id': session_id})
            raise

    def _load_tokenizer(self, model_path: Path, session_id: Optional[str] = None):
        """Load and validate BERT tokenizer with comprehensive error handling."""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                use_fast=True,
                trust_remote_code=False  # Security consideration
            )

            # Validate tokenizer
            vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size
            
            if vocab_size < 1000:  # Sanity check
                self.logger.warning(f"Unusually small vocabulary: {vocab_size}")
            
            self.logger.info(f"Tokenizer loaded successfully: {vocab_size} vocab size", extra={'session_id': session_id})
            
            return tokenizer

        except Exception as e:
            error_msg = f"Tokenizer loading failed: {str(e)}"
            self.logger.error(error_msg, extra={'session_id': session_id})
            raise ValueError(error_msg)

    def _load_bert_model(self, model_path: Path, session_id: Optional[str] = None):
        """Load and validate BERT model with comprehensive error handling."""
        try:
            from transformers import AutoModelForSequenceClassification
            
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True,
                num_labels=2,  # Binary classification: REAL (0) vs FAKE (1)
                trust_remote_code=False  # Security consideration
            )

            # Model validation
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            if total_params < 1000:  # Sanity check
                raise ValueError(f"Suspiciously small model: {total_params} parameters")

            self.logger.info(
                f"BERT model loaded: {total_params:,} total params, {trainable_params:,} trainable",
                extra={'session_id': session_id}
            )

            return model

        except Exception as e:
            error_msg = f"BERT model loading failed: {str(e)}"
            self.logger.error(error_msg, extra={'session_id': session_id})
            raise ValueError(error_msg)

    def _load_training_metadata(self, model_path: Path, session_id: Optional[str] = None):
        """Load training metadata with enhanced error handling."""
        metadata_files = ["training_metadata.json", "training_args.json", "trainer_state.json"]
        
        for metadata_file in metadata_files:
            file_path = model_path / metadata_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                        self.training_metadata.update(metadata)
                    
                    self.logger.debug(f"Loaded metadata from {metadata_file}", extra={'session_id': session_id})
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load {metadata_file}: {e}")

        if not self.training_metadata:
            self.training_metadata = {
                "note": "No training metadata available",
                "loaded_at": datetime.now().isoformat()
            }
        else:
            self.logger.info("Training metadata loaded successfully", extra={'session_id': session_id})

    def _setup_model_device(self, model, session_id: Optional[str] = None):
        """Setup model on appropriate device with optimization."""
        try:
            device = self.device_manager.get_device()
            
            # Move model to device
            model = model.to(device)
            
            # Set to evaluation mode
            model.eval()
            
            # Device-specific optimizations
            if device.type == "cuda" and self.enable_model_optimization:
                # Enable CUDA optimizations
                model = model.half() if self.config.get('use_fp16', False) else model
                
            self.logger.info(f"Model setup on device: {device}", extra={'session_id': session_id})
            
            return model

        except Exception as e:
            error_msg = f"Model device setup failed: {str(e)}"
            self.logger.error(error_msg, extra={'session_id': session_id})
            raise RuntimeError(error_msg)

    def _verify_model_functionality(self, model, tokenizer, session_id: Optional[str] = None):
        """Comprehensive model functionality verification."""
        self.model_metrics['validation_checks'] += 1
        
        try:
            test_texts = [
                "This is a test sentence for model verification.",
                "Another test to ensure model works correctly.",
                ""  # Test empty input handling
            ]

            for i, test_text in enumerate(test_texts):
                try:
                    if not test_text:  # Skip empty text test for now
                        continue
                        
                    # Tokenize
                    inputs = tokenizer(
                        test_text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    )

                    # Move to device
                    device = self.device_manager.get_device()
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Run inference
                    model.eval()
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probabilities = torch.softmax(outputs.logits, dim=1)
                        prediction = torch.argmax(probabilities, dim=1)

                    # Validate output format
                    batch_size = inputs['input_ids'].shape[0]
                    
                    if probabilities.shape != (batch_size, 2):
                        raise RuntimeError(f"Unexpected output shape: {probabilities.shape}, expected ({batch_size}, 2)")
                    
                    if prediction.item() not in [0, 1]:
                        raise RuntimeError(f"Invalid prediction: {prediction.item()}, expected 0 or 1")

                    # Check probability values
                    prob_sum = probabilities.sum(dim=1).item()
                    if abs(prob_sum - 1.0) > 1e-5:
                        raise RuntimeError(f"Probabilities don't sum to 1: {prob_sum}")

                except Exception as e:
                    raise RuntimeError(f"Model verification failed on test {i+1}: {str(e)}")

            self.logger.info("Model functionality verification passed", extra={'session_id': session_id})

        except Exception as e:
            error_msg = f"Model functionality verification failed: {str(e)}"
            self.logger.error(error_msg, extra={'session_id': session_id})
            raise RuntimeError(error_msg)

    def _optimize_model(self, model, session_id: Optional[str] = None):
        """Apply model optimizations for better performance."""
        try:
            self.model_metrics['optimization_calls'] += 1
            
            # Compile model for better performance (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.config.get('use_torch_compile', False):
                try:
                    model = torch.compile(model)
                    self.logger.info("Model compiled with torch.compile", extra={'session_id': session_id})
                except Exception as e:
                    self.logger.warning(f"torch.compile failed: {e}")

            # Memory optimization
            if self.config.get('optimize_memory', True):
                self.device_manager.optimize_memory()
                self.model_metrics['memory_optimizations'] += 1

            return model

        except Exception as e:
            self.logger.warning(f"Model optimization failed: {e}", extra={'session_id': session_id})
            return model  # Return unoptimized model

    def _build_comprehensive_model_info(self):
        """Build comprehensive model information dictionary."""
        if self.model is None:
            return

        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024**2)  # Approximate size in MB

            self.model_info = {
                # Basic model information
                "model_name": self.model_name,
                "model_path": str(self.model_path) if self.model_path else None,
                "model_type": "AutoModelForSequenceClassification",
                "num_labels": 2,
                
                # Model parameters
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": round(model_size_mb, 2),
                
                # Device and performance
                "device": str(self.device_manager.get_device()),
                "device_info": self.device_manager.get_device_info(),
                
                # Metadata
                "training_metadata": self.training_metadata,
                "loaded_at": self.load_timestamp,
                
                # Configuration
                "config_enhanced": True,
                "enhanced_version": "3.2.0",
                "optimization_enabled": self.enable_model_optimization,
                
                # Validation status
                "validation_passed": True,
                "functionality_verified": True
            }

        except Exception as e:
            self.logger.error(f"Failed to build model info: {e}")
            self.model_info = {
                "error": str(e),
                "model_name": self.model_name,
                "build_failed": True
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return self.model_info.copy() if self.model_info else {"model_loaded": False}

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.model is not None and self.tokenizer is not None

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get model-specific memory usage information."""
        if not self.is_model_loaded():
            return {"model_loaded": False}

        try:
            base_memory = self.device_manager.get_memory_usage(detailed=True)
            
            # Add model-specific information
            model_memory = {
                "model_loaded": True,
                "model_parameters": self.model_info.get("total_parameters", 0),
                "estimated_model_size_mb": self.model_info.get("model_size_mb", 0)
            }
            
            return {**base_memory, **model_memory}

        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {"error": str(e), "model_loaded": self.is_model_loaded()}

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive model health status for monitoring."""
        try:
            self.model_metrics['health_checks'] += 1
            self.model_metrics['last_health_check'] = datetime.now().isoformat()
            
            health_status = {
                "model_loaded": self.is_model_loaded(),
                "status": "healthy",
                "last_check": self.model_metrics['last_health_check'],
                "device_status": self.device_manager.get_health_status()
            }

            if self.is_model_loaded():
                # Perform basic health checks
                try:
                    # Check if model is in eval mode
                    if self.model.training:
                        health_status["warnings"] = ["Model not in eval mode"]
                    
                    # Check device consistency
                    model_device = next(self.model.parameters()).device
                    expected_device = self.device_manager.get_device()
                    
                    if model_device != expected_device:
                        health_status["status"] = "degraded"
                        health_status["issues"] = [f"Device mismatch: model on {model_device}, expected {expected_device}"]
                    
                    # Add performance metrics
                    health_status["performance_metrics"] = {
                        "load_attempts": self.model_metrics["load_attempts"],
                        "successful_loads": self.model_metrics["successful_loads"],
                        "validation_checks": self.model_metrics["validation_checks"],
                        "inference_calls": self.model_metrics["inference_calls"],
                        "uptime_seconds": round(time.time() - self.model_metrics["initialization_time"], 2)
                    }

                except Exception as e:
                    health_status["status"] = "degraded"
                    health_status["health_check_error"] = str(e)
            else:
                health_status["status"] = "unloaded"
                health_status["message"] = "Model not loaded"

            return health_status

        except Exception as e:
            self.logger.error(f"Health status check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat(),
                "model_loaded": self.is_model_loaded()
            }

    def cleanup(self):
        """Cleanup model resources and memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Optimize device memory
            self.device_manager.optimize_memory(aggressive=True)
            
            self.logger.info("Model cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Model cleanup failed: {e}")


# Testing functionality with comprehensive validation
if __name__ == "__main__":
    """Test enhanced model utilities with comprehensive scenarios."""
    import time
    from pprint import pprint

    print("=== Testing Enhanced Model Utilities ===")
    print("=" * 60)

    # Test DeviceManager
    print("🔧 Testing Enhanced DeviceManager...")
    device_manager = DeviceManager('auto')
    
    print(f"✅ Device selected: {device_manager.get_device()}")
    
    device_info = device_manager.get_device_info()
    print("📊 Device Information:")
    for key, value in device_info.items():
        if isinstance(value, dict):
            print(f"  {key}: [nested dict]")
        else:
            print(f"  {key}: {value}")

    # Test memory operations
    print(f"\n💾 Testing Memory Operations...")
    memory_info = device_manager.get_memory_usage()
    print("Memory Usage:")
    pprint(memory_info)
    
    optimization_result = device_manager.optimize_memory()
    print("Memory Optimization:")
    pprint(optimization_result)

    # Test health monitoring
    print(f"\n🏥 Testing Health Monitoring...")
    health_status = device_manager.get_health_status()
    print("Device Health:")
    pprint(health_status)

    # Test ModelManager
    print(f"\n🤖 Testing Enhanced ModelManager...")
    model_manager = ModelManager(device_manager)
    
    print(f"Model manager ready: {model_manager.is_model_loaded()}")
    
    # Test health status
    model_health = model_manager.get_health_status()
    print("Model Manager Health:")
    pprint(model_health)

    # Test memory usage
    model_memory = model_manager.get_memory_usage()
    print("Model Memory Usage:")
    pprint(model_memory)

    print(f"\n✅ Enhanced Model Utilities testing completed!")
