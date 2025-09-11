# agents/bert_classifier/classifier.py

"""
Enhanced BERT Classification Agent - Production Ready with Full Integration

Production-ready BERT classifier with comprehensive enhancements:
- Dynamic configuration management with fallbacks
- Async processing with proper session handling
- Enhanced exception handling and error management
- Performance metrics tracking with detailed analytics
- Batch processing support with optimized DataLoader
- Logging integration with structured format
- Modular architecture with separated concerns
- LangGraph compatibility with state management

Version: 3.2.0 - Enhanced Production Edition
"""

import torch
from torch.utils.data import Dataset, DataLoader
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Enhanced BaseAgent import with correct path
from agents.base import BaseAgent
from .preprocessing import TextPreprocessor
from .model_utils import DeviceManager, ModelManager

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

# Optional config imports with fallback handling
try:
    from config import get_model_config, get_settings
except ImportError:
    get_model_config = None
    get_settings = None

# Enhanced text sanitization with fallback
def basic_sanitize_text(text: str) -> str:
    """Basic text sanitization fallback function."""
    if not isinstance(text, str):
        return ""
    return text.strip().replace('\x00', '').replace('\r\n', '\n').replace('\t', ' ')

sanitize_text = basic_sanitize_text

try:
    from utils import sanitize_text as utils_sanitize_text
    sanitize_text = utils_sanitize_text
except ImportError:
    pass


class FakeNewsDataset(Dataset):
    """
    Enhanced Dataset class for efficient batch processing with preprocessing support.
    
    Features:
    - Optional text preprocessing integration
    - Proper tensor handling for batch operations
    - Memory-efficient data loading
    """

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize dataset with optional preprocessing.

        Args:
            texts: List of text strings to process
            tokenizer: BERT tokenizer instance
            max_length: Maximum sequence length for tokenization
            preprocessor: Optional TextPreprocessor instance
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor

        # Apply preprocessing if available
        if self.preprocessor:
            self.texts = [self.preprocessor.preprocess_text(text) for text in texts]
        else:
            self.texts = [str(text) for text in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class BERTClassifier(BaseAgent):
    """
    Enhanced BERT Classification Agent with full production integration.

    Features:
    - Dynamic configuration with environment awareness
    - Async processing with proper session management
    - Enhanced error handling with recovery strategies
    - Comprehensive performance metrics tracking
    - Batch processing with optimized throughput
    - Modular architecture with component separation
    - LangGraph state management compatibility
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced BERT classifier with comprehensive configuration.

        Args:
            config: Optional configuration dictionary with runtime overrides
        """
        # Load configuration with fallback handling
        bert_config = {}
        if get_model_config:
            try:
                bert_config = get_model_config('bert_classifier') or {}
            except Exception as e:
                self.logger.warning(f"Failed to load model config: {e}")

        # Merge with runtime overrides
        if config:
            bert_config.update(config)

        # Set agent name before parent initialization
        self.agent_name = "bert_classifier"
        
        # Initialize enhanced base agent
        super().__init__(bert_config)

        # Configuration parameters with defaults
        self.model_name = self.config.get('model_name', 'bert-base-uncased')
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 16)
        self.device_setting = self.config.get('device', 'auto')
        self.enable_preprocessing = self.config.get('enable_preprocessing', True)
        preprocessing_config = self.config.get('preprocessing_config', {})

        # Enhanced models directory handling
        models_dir_default = Path('./models')
        try:
            if get_settings:
                system_settings = get_settings()
                self.models_dir = Path(getattr(system_settings, 'models_dir', models_dir_default))
            else:
                self.models_dir = models_dir_default
        except Exception:
            self.models_dir = models_dir_default

        self.enable_metrics = self.config.get('enable_metrics', True)

        # Initialize modular components with enhanced error handling
        try:
            self.device_manager = DeviceManager(self.device_setting)
            # Fixed: Initialize ModelManager with device_manager instance, not DeviceManager class
            self.model_manager = ModelManager(self.device_manager)
        except Exception as e:
            self.logger.error(f"Failed to initialize device/model managers: {e}")
            raise

        # Initialize preprocessor with configuration
        if self.enable_preprocessing:
            try:
                preprocessing_config['max_length'] = self.max_length
                self.preprocessor = TextPreprocessor(preprocessing_config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize preprocessor: {e}")
                self.preprocessor = None
        else:
            self.preprocessor = None

        # Model components (initialized during load_model)
        self.model = None
        self.tokenizer = None

        # Enhanced performance metrics tracking
        self.bert_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'error_count': 0,
            'average_inference_time': 0.0,
            'min_inference_time': float('inf'),
            'max_inference_time': 0.0,
            'batch_predictions': 0,
            'preprocessing_enabled': self.enable_preprocessing,
            'config_loaded': True,
            'model_loaded': False
        }

        # Don't call async methods in __init__
        self._model_loaded = False
        
        self.logger.info(f"Enhanced BERT Classifier initialized")
        self.logger.info(f"Model: {self.model_name}, Device: {self.device_manager.get_device()}")
        self.logger.info(f"Preprocessing: {'Enabled' if self.preprocessor else 'Disabled'}")
        self.logger.info(f"Max Length: {self.max_length}, Batch Size: {self.batch_size}")

    @classmethod
    async def create(cls, config: Optional[Dict[str, Any]] = None) -> 'BERTClassifier':
        """
        Create BERTClassifier instance and load model asynchronously.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            BERTClassifier instance with loaded model
        """
        instance = cls(config)
        await instance.load_model()
        return instance

    async def load_model(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load BERT model with comprehensive error handling and validation.

        Args:
            model_path: Optional path to model directory

        Returns:
            Dictionary with load result and model information
        """
        try:
            # Determine model path from config if not provided
            if model_path is None:
                model_path = self.models_dir / self.config.get('model_path', 'bert_fake_news_classifier')

            self.logger.info(f"Loading BERT model from: {model_path}")

            # Load model using model manager with enhanced error handling
            self.model, self.tokenizer = self.model_manager.load_model(model_path, self.model_name)

            # Update metrics and flags
            self.bert_metrics['model_loaded'] = True
            self._model_loaded = True

            # Get comprehensive model information
            model_info = self.model_manager.get_model_info()
            device_info = self.device_manager.get_device_info()

            self.logger.info("BERT model loaded and ready for fake news detection")

            return {
                "success": True,
                "model_info": model_info,
                "device_info": device_info,
                "preprocessing_enabled": self.preprocessor is not None,
                "config_integrated": True,
                "model_path": str(model_path)
            }

        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Enhanced exception handling
            if _enhanced_exceptions_available:
                context = ErrorContext(
                    operation="model_loading",
                    model_used=self.model_name
                )
                standardized_error = handle_llm_explanation_exception(e, context)
                log_exception_with_context(standardized_error, None, {'agent': self.agent_name})

            return {
                "success": False,
                "error": error_msg,
                "model_loaded": False,
                "config_integrated": True
            }

    async def _process_internal(self, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Internal processing method for BaseAgent compatibility.

        Args:
            input_data: Input data dictionary containing text and parameters
            session_id: Optional session identifier for tracking

        Returns:
            Processing result dictionary
        """
        try:
            # Extract parameters with defaults
            article_text = input_data.get('text', '')
            skip_preprocessing = input_data.get('skip_preprocessing', False)
            confidence_threshold = self.config.get('high_confidence_threshold', 0.8)

            # Validate model is loaded
            if not self._model_loaded or not self.model_manager.is_model_loaded():
                raise ValueError("Model not loaded. Please call load_model() first.")

            # Perform prediction
            prediction_result = await self.predict(article_text, skip_preprocessing)

            # Check for prediction errors
            if prediction_result.get('prediction') == 'ERROR':
                raise Exception(prediction_result.get('error', 'Unknown prediction error'))

            # Format comprehensive result
            result = {
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'probabilities': {
                    'real': prediction_result['real_probability'],
                    'fake': prediction_result['fake_probability']
                },
                'text_analysis': {
                    'original_length': len(article_text),
                    'processed_length': prediction_result.get('processed_length', 0),
                    'tokens_used': prediction_result.get('tokens_used', 0),
                    'preprocessing_applied': prediction_result.get('preprocessing_applied', False)
                },
                'quality_flags': {
                    'high_confidence': prediction_result['confidence'] >= confidence_threshold,
                    'config_applied': True
                },
                'performance': {
                    'inference_time': prediction_result.get('inference_time', 0.0),
                    'total_time': prediction_result.get('total_time_seconds', 0.0)
                }
            }

            # Update success metrics
            self.bert_metrics['successful_predictions'] += 1

            return result

        except Exception as e:
            # Update error metrics
            self.bert_metrics['error_count'] += 1
            raise

    async def predict(self, article_text: str, skip_preprocessing: bool = False) -> Dict[str, Any]:
        """
        Predict single article with comprehensive analysis and error handling.

        Args:
            article_text: Text content to classify
            skip_preprocessing: Whether to skip text preprocessing

        Returns:
            Dictionary with prediction results and metadata
        """
        start_time = time.time()

        # Validate model is loaded
        if not self.model_manager.is_model_loaded():
            return {
                'prediction': 'ERROR',
                'error': 'Model not loaded. Please call load_model() first.',
                'confidence': 0.0,
                'total_time_seconds': time.time() - start_time
            }

        try:
            # Step 1: Text preprocessing with configuration
            preprocessing_applied = False
            processed_text = article_text

            if self.enable_preprocessing and not skip_preprocessing and self.preprocessor:
                try:
                    processed_text = self.preprocessor.preprocess_text(article_text)
                    preprocessing_applied = True
                    self.logger.debug(f"Text preprocessed: {len(article_text)} → {len(processed_text)} chars")
                except Exception as e:
                    self.logger.warning(f"Preprocessing failed, using original text: {e}")
                    processed_text = sanitize_text(article_text)
            else:
                processed_text = sanitize_text(article_text)

            # Validate processed text
            min_length = self.config.get('min_text_length', 5)
            if not processed_text or len(processed_text.strip()) < min_length:
                raise ValueError(f"Article text is too short (< {min_length} chars) after preprocessing")

            # Step 2: Tokenization with configuration
            encoding = self.tokenizer(
                processed_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to appropriate device
            device = self.device_manager.get_device()
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Step 3: Model inference with timing
            inference_start = time.time()
            self.model.eval()
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                real_prob = probabilities[0][0].item()
                fake_prob = probabilities[0][1].item()
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = max(real_prob, fake_prob)

            inference_time = time.time() - inference_start
            total_time = time.time() - start_time

            # Step 4: Format comprehensive results
            result = {
                # Main prediction results
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'prediction_numeric': prediction,
                'confidence': confidence,
                'real_probability': real_prob,
                'fake_probability': fake_prob,

                # Text analysis
                'original_text_length': len(article_text),
                'processed_length': len(processed_text),
                'tokens_used': int(attention_mask.sum().item()),
                'preprocessing_applied': preprocessing_applied,

                # System information
                'model_name': self.model_name,
                'device_used': str(device),
                'predicted_at': datetime.now().isoformat(),
                'max_length_used': self.max_length,

                # Performance metrics
                'total_time_seconds': round(total_time, 3),
                'inference_time_seconds': round(inference_time, 3),

                # Agent integration
                'agent_name': 'BERT_Classifier',
                'agent_version': '3.2.0_enhanced',
                'config_integrated': True
            }

            # Update comprehensive metrics
            self.bert_metrics['total_predictions'] += 1
            self._update_inference_metrics(inference_time)

            return result

        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"Prediction failed: {str(e)}"
            self.logger.error(error_msg)

            # Enhanced exception handling
            if _enhanced_exceptions_available:
                context = ErrorContext(
                    operation="prediction",
                    model_used=self.model_name,
                    processing_time=error_time,
                    input_size=len(article_text) if article_text else 0
                )
                standardized_error = handle_llm_explanation_exception(e, context)
                log_exception_with_context(standardized_error, None, {'agent': self.agent_name})

            return {
                'prediction': 'ERROR',
                'error': error_msg,
                'confidence': 0.0,
                'predicted_at': datetime.now().isoformat(),
                'agent_name': 'BERT_Classifier',
                'total_time_seconds': error_time,
                'config_integrated': True
            }

    async def predict_batch(self, texts: List[str], batch_size: Optional[int] = None,
                           skip_preprocessing: bool = False) -> List[Dict[str, Any]]:
        """
        Batch prediction with optimized processing and comprehensive error handling.

        Args:
            texts: List of text strings to classify
            batch_size: Optional batch size override
            skip_preprocessing: Whether to skip text preprocessing

        Returns:
            List of prediction result dictionaries
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("Model not loaded. Please call load_model() first.")

        if not texts:
            return []

        # Use configured batch size if not specified
        batch_size = batch_size or self.batch_size

        self.logger.info(f"Running batch prediction on {len(texts)} articles (batch_size={batch_size})")

        try:
            # Create dataset with optional preprocessing
            preprocessor = None if skip_preprocessing else self.preprocessor
            dataset = FakeNewsDataset(texts, self.tokenizer, self.max_length, preprocessor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            self.model.eval()
            all_results = []
            device = self.device_manager.get_device()

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    batch_start = time.time()
                    
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    # Model inference
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)

                    batch_time = time.time() - batch_start
                    batch_idx_start = batch_idx * batch_size

                    # Process each item in batch
                    for j in range(len(input_ids)):
                        text_idx = batch_idx_start + j
                        if text_idx >= len(texts):  # Handle last batch
                            break

                        real_prob = probabilities[j][0].item()
                        fake_prob = probabilities[j][1].item()
                        pred = predictions[j].item()
                        confidence = max(real_prob, fake_prob)

                        result = {
                            'prediction': 'FAKE' if pred == 1 else 'REAL',
                            'prediction_numeric': pred,
                            'confidence': confidence,
                            'real_probability': real_prob,
                            'fake_probability': fake_prob,
                            'text_length': len(texts[text_idx]),
                            'tokens_used': int(attention_mask[j].sum().item()),
                            'batch_index': text_idx,
                            'batch_time_seconds': round(batch_time / len(input_ids), 3),
                            'model_name': self.model_name,
                            'predicted_at': datetime.now().isoformat(),
                            'agent_name': 'BERT_Classifier',
                            'agent_version': '3.2.0_enhanced',
                            'config_integrated': True,
                            'batch_size_used': batch_size
                        }

                        all_results.append(result)

            # Update batch metrics
            self.bert_metrics['batch_predictions'] += len(all_results)
            self.logger.info(f"Batch prediction completed: {len(all_results)} results")

            return all_results

        except Exception as e:
            error_msg = f"Batch prediction failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Enhanced exception handling
            if _enhanced_exceptions_available:
                context = ErrorContext(
                    operation="batch_prediction",
                    model_used=self.model_name,
                    input_size=len(texts)
                )
                standardized_error = handle_llm_explanation_exception(e, context)
                log_exception_with_context(standardized_error, None, {'agent': self.agent_name})
            
            raise RuntimeError(error_msg)

    def _update_inference_metrics(self, inference_time: float):
        """Update inference-specific metrics with enhanced tracking."""
        total_predictions = self.bert_metrics['total_predictions']
        
        if total_predictions == 1:
            self.bert_metrics['average_inference_time'] = inference_time
            self.bert_metrics['min_inference_time'] = inference_time
            self.bert_metrics['max_inference_time'] = inference_time
        else:
            current_avg = self.bert_metrics['average_inference_time']
            self.bert_metrics['average_inference_time'] = (
                (current_avg * (total_predictions - 1) + inference_time) / total_predictions
            )
            self.bert_metrics['min_inference_time'] = min(self.bert_metrics['min_inference_time'], inference_time)
            self.bert_metrics['max_inference_time'] = max(self.bert_metrics['max_inference_time'], inference_time)

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics with configuration info."""
        # Get base metrics from enhanced BaseAgent
        base_metrics = self.get_comprehensive_status()

        # Get component metrics
        component_metrics = {}
        if self.preprocessor:
            component_metrics['preprocessing'] = self.preprocessor.get_preprocessing_stats()

        if self.model_manager.is_model_loaded():
            component_metrics['model'] = self.model_manager.get_model_info()
            component_metrics['device'] = self.device_manager.get_device_info()

        # Configuration metrics
        config_metrics = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'preprocessing_enabled': self.enable_preprocessing,
            'device_setting': self.device_setting,
            'config_version': '3.2.0_enhanced'
        }

        # Combine all metrics
        return {
            **base_metrics,
            'bert_specific_metrics': self.bert_metrics,
            'component_metrics': component_metrics,
            'config_metrics': config_metrics,
            'agent_type': 'bert_classifier',
            'modular_architecture': True,
            'config_integrated': True,
            'enhanced_exceptions_available': _enhanced_exceptions_available
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information with configuration context."""
        if not self.model_manager.is_model_loaded():
            return {
                'model_loaded': False,
                'config_model_name': self.model_name,
                'config_integrated': True
            }

        base_info = self.model_manager.get_model_info()
        base_info.update({
            'config_integrated': True,
            'config_model_name': self.model_name,
            'enhanced_version': '3.2.0'
        })
        return base_info

    def is_ready(self) -> bool:
        """Check if the classifier is ready for predictions."""
        return self.model_manager.is_model_loaded()

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'preprocessing_enabled': self.enable_preprocessing,
            'device_setting': self.device_setting,
            'metrics_enabled': self.enable_metrics,
            'config_source': 'config_files_with_fallback',
            'models_dir': str(self.models_dir),
            'enhanced_exceptions_available': _enhanced_exceptions_available
        }


# Testing functionality with enhanced error handling
if __name__ == "__main__":
    """Test the enhanced BERT classifier with comprehensive validation."""
    import asyncio
    from pprint import pprint

    print("=== Testing Enhanced BERT Classifier ===")
    print("=" * 60)

    async def run_comprehensive_test():
        try:
            # Initialize classifier
            print("🔧 Initializing Enhanced BERT Classifier...")
            classifier = BERTClassifier()
            
            print("✅ Classifier initialized successfully")
            print(f"Agent Name: {classifier.agent_name}")
            print(f"Model Name: {classifier.model_name}")
            print(f"Device: {classifier.device_manager.get_device()}")
            
            # Show configuration summary
            config_summary = classifier.get_config_summary()
            print(f"\n⚙️ Configuration Summary:")
            for key, value in config_summary.items():
                print(f"  {key}: {value}")

            # Test model loading (will fail without actual model files)
            print(f"\n📂 Testing model loading...")
            model_result = await classifier.load_model()
            
            if model_result['success']:
                print("✅ Model loaded successfully")
                
                # Test single prediction
                print(f"\n🎯 Testing single prediction...")
                test_input = {
                    "text": "This is a test article for the enhanced BERT classifier with comprehensive error handling and async processing."
                }
                
                result = await classifier.process(test_input)
                print("Single prediction result:")
                pprint(result)
                
                # Test batch prediction
                print(f"\n🔄 Testing batch prediction...")
                batch_texts = [
                    "First test article for batch processing.",
                    "Second test article with different content.",
                    "Third article to validate batch functionality."
                ]
                
                batch_results = await classifier.predict_batch(batch_texts)
                print(f"Batch prediction completed: {len(batch_results)} results")
                
            else:
                print(f"⚠️ Model loading failed (expected in test environment): {model_result.get('error')}")
                
                # Test error handling with invalid input
                print(f"\n🧪 Testing error handling...")
                error_test_input = {"text": ""}
                error_result = await classifier.process(error_test_input)
                print("Error handling result:")
                pprint(error_result)

            # Show comprehensive metrics
            print(f"\n📊 Comprehensive metrics:")
            metrics = classifier.get_comprehensive_metrics()
            print(f"Agent type: {metrics.get('agent_type')}")
            print(f"Config integrated: {metrics.get('config_integrated')}")
            print(f"Enhanced exceptions: {metrics.get('enhanced_exceptions_available')}")
            print(f"BERT metrics: {metrics.get('bert_specific_metrics')}")

            print(f"\n✅ Enhanced BERT classifier test completed successfully!")

        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

    # Run the comprehensive test
    asyncio.run(run_comprehensive_test())
