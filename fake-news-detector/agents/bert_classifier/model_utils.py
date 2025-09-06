# agents/bert_classifier/model_utils.py
"""
Model Management Utilities for BERT Classifier - Config Enhanced

Enhanced device management and model loading with configuration support.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import json

class DeviceManager:
    """
    🖥️ ENHANCED DEVICE MANAGEMENT WITH CONFIG SUPPORT
    
    Automatically detects and manages the best available processing device
    with configuration override capabilities.
    """
    
    def __init__(self, device_preference: str = 'auto'):
        """
        Initialize device manager with config preference
        
        Args:
            device_preference: 'auto', 'cpu', 'cuda', 'mps', or specific device
        """
        self.device_preference = device_preference
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Detect and set device
        self.device = self._detect_and_set_device()
        
        # Cache device info for performance
        self._device_info = None
    
    def _detect_and_set_device(self) -> torch.device:
        """Detect best device based on preference and availability"""
        
        if self.device_preference != 'auto':
            # Try to use specified device
            try:
                device = torch.device(self.device_preference)
                self.logger.info(f"🎯 Using specified device: {device}")
                return device
            except Exception as e:
                self.logger.warning(f"⚠️ Specified device '{self.device_preference}' not available: {e}")
                self.logger.info("🔄 Falling back to auto-detection...")
        
        # Auto-detection logic
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"🚀 Using NVIDIA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return device
        
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info("🍎 Using Apple Metal Performance Shaders (MPS)")
            return device
        
        else:
            device = torch.device("cpu")
            self.logger.info("💻 Using CPU (consider GPU for better performance)")
            return device
    
    def get_device(self) -> torch.device:
        """Get the current device"""
        return self.device
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information with config context
        
        Returns:
            Dictionary with comprehensive device information
        """
        if self._device_info is None:
            info = {
                "device_type": self.device.type,
                "device_preference_set": self.device_preference,
                "is_available": True,
                "device_string": str(self.device)
            }
            
            if self.device.type == "cuda":
                info.update({
                    "device_name": torch.cuda.get_device_name(self.device.index or 0),
                    "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                    "memory_available_gb": round((torch.cuda.get_device_properties(0).total_memory - 
                                                torch.cuda.memory_reserved(0)) / 1e9, 2),
                    "compute_capability": f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}",
                    "optimization_notes": "GPU available - excellent performance expected"
                })
                
            elif self.device.type == "mps":
                info.update({
                    "device_name": "Apple Silicon GPU",
                    "optimization_notes": "MPS available - good performance on Apple Silicon"
                })
                
            else:  # CPU
                info.update({
                    "device_name": "CPU",
                    "optimization_notes": "Using CPU - consider GPU for better performance"
                })
            
            self._device_info = info
        
        return self._device_info

class ModelManager:
    """
    🤖 ENHANCED MODEL MANAGEMENT WITH CONFIG SUPPORT
    
    Handles BERT model loading, validation, and management with
    configuration integration and comprehensive error handling.
    """
    
    def __init__(self, device_manager: DeviceManager):
        """
        Initialize model manager with device manager
        
        Args:
            device_manager: DeviceManager instance for device handling
        """
        self.device_manager = device_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model storage
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.model_name = None
        
        # Model metadata
        self.model_info = {}
        self.training_metadata = {}
    
    def load_model(self, model_path: Path, model_name: str) -> Tuple[Any, Any]:
        """
        Load BERT model with comprehensive validation and config support
        
        Args:
            model_path: Path to the saved model directory
            model_name: Name/type of the BERT model
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            FileNotFoundError: If model files are missing
            ValueError: If model validation fails
        """
        try:
            self.logger.info(f"📂 Loading BERT model from: {model_path}")
            self.model_path = model_path
            self.model_name = model_name
            
            # Validate model directory and files
            self._validate_model_files(model_path)
            
            # Load tokenizer
            tokenizer = self._load_tokenizer(model_path)
            
            # Load BERT model
            model = self._load_bert_model(model_path)
            
            # Load training metadata if available
            self._load_training_metadata(model_path)
            
            # Move model to appropriate device
            model = model.to(self.device_manager.get_device())
            
            # Set to evaluation mode
            model.eval()
            
            # Verify model functionality
            self._verify_model_functionality(model, tokenizer)
            
            # Store references
            self.model = model
            self.tokenizer = tokenizer
            
            # Build model info
            self._build_model_info()
            
            self.logger.info("✅ Model loaded and verified successfully")
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {str(e)}")
            raise
    
    def _validate_model_files(self, model_path: Path):
        """Validate that required model files exist"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check for either pytorch_model.bin or model.safetensors
        model_files = ["pytorch_model.bin", "model.safetensors"]
        has_model_file = any((model_path / file_name).exists() for file_name in model_files)
        
        required_files = ["config.json", "vocab.txt"]
        missing_files = []
        
        if not has_model_file:
            missing_files.append("pytorch_model.bin or model.safetensors")
        
        for file_name in required_files:
            if not (model_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required model files: {missing_files}")
    
    def _load_tokenizer(self, model_path: Path):
        """Load BERT tokenizer"""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                use_fast=True
            )
            
            self.logger.info(f"✅ Tokenizer loaded: {len(tokenizer.vocab)} vocab size")
            return tokenizer
            
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer: {str(e)}")
    
    def _load_bert_model(self, model_path: Path):
        """Load BERT model"""
        try:
            from transformers import AutoModelForSequenceClassification
            
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True,
                num_labels=2  # Binary classification: REAL (0) vs FAKE (1)
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"✅ BERT model loaded: {total_params:,} total params, {trainable_params:,} trainable")
            
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to load BERT model: {str(e)}")
    
    def _load_training_metadata(self, model_path: Path):
        """Load training metadata if available"""
        metadata_file = model_path / "training_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    self.training_metadata = json.load(f)
                self.logger.info("✅ Training metadata loaded")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load training metadata: {e}")
        else:
            self.training_metadata = {"note": "No training metadata available"}
    
    def _verify_model_functionality(self, model, tokenizer):
        """Verify model can perform inference"""
        test_text = "This is a test sentence for model verification."
        
        try:
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
            assert probabilities.shape == (1, 2), f"Expected (1, 2) output, got {probabilities.shape}"
            assert prediction.item() in [0, 1], f"Expected 0 or 1 prediction, got {prediction.item()}"
            
            self.logger.info("✅ Model functionality verified successfully")
            
        except Exception as e:
            raise RuntimeError(f"Model functionality verification failed: {str(e)}")
    
    def _build_model_info(self):
        """Build comprehensive model information"""
        if self.model is None:
            return
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024**2)  # Approximate size in MB
        
        self.model_info = {
            "model_name": self.model_name,
            "model_path": str(self.model_path) if self.model_path else None,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "device": str(self.device_manager.get_device()),
            "training_metadata": self.training_metadata,
            "model_type": "AutoModelForSequenceClassification",
            "num_labels": 2,
            "loaded_at": torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else "unknown",
            "config_enhanced": True
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return self.model_info.copy()
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and self.tokenizer is not None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        device = self.device_manager.get_device()
        
        if device.type == "cuda":
            return {
                "device_type": "cuda",
                "memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
                "memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
                "memory_free_mb": round((torch.cuda.get_device_properties(0).total_memory - 
                                       torch.cuda.memory_reserved()) / 1024**2, 2)
            }
        else:
            return {
                "device_type": device.type,
                "note": "Memory tracking not available for this device type"
            }

# Testing
if __name__ == "__main__":
    """Test device and model management with config"""
    print("🧪 Testing Enhanced Model Utils")
    
    # Test device manager
    device_manager = DeviceManager('auto')
    device_info = device_manager.get_device_info()
    print(f"Device: {device_info}")
    
    # Test model manager
    model_manager = ModelManager(device_manager)
    print(f"Model manager ready: {model_manager.is_model_loaded()}")
