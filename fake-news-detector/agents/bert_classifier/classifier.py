# agents/bert_classifier/classifier.py
"""
Enhanced BERT Classification Agent - Main Implementation with Config Integration

This is the main BERT classifier that brings together all modular components
for fake news detection with configuration integration.

Features:
- Modular architecture with separated concerns  
- Configuration integration from config files
- Built-in text preprocessing
- Enhanced error handling and logging
- Performance metrics tracking
- LangGraph integration ready
- Batch processing capabilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import base agent functionality
from agents.base.base_agent import BaseAgent

# Import modular components
from .preprocessing import TextPreprocessor
from .model_utils import DeviceManager, ModelManager

# ✅ IMPORT CONFIGURATION FILES
from config import get_model_config, get_settings
from utils.helpers import sanitize_text

class FakeNewsDataset(Dataset):
    """
    📊 DATASET CLASS FOR BATCH PROCESSING
    
    Handles multiple articles for efficient batch processing with
    integrated preprocessing capabilities.
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize dataset with preprocessing option
        
        Args:
            texts: List of article texts
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            preprocessor: Optional text preprocessor
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        
        # Apply preprocessing if preprocessor is provided
        if self.preprocessor:
            self.texts = [self.preprocessor.preprocess_text(text) for text in texts]
        else:
            self.texts = texts
    
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

class BERTClassifier(BaseAgent):
    """
    🤖 ENHANCED BERT CLASSIFICATION AGENT WITH CONFIG INTEGRATION
    
    Modular BERT-based fake news classifier that inherits from BaseAgent
    for consistent interface and LangGraph compatibility.
    
    Features:
    - Inherits from BaseAgent for consistent interface
    - Configuration integration from config files
    - Modular component architecture
    - Built-in preprocessing capabilities
    - Device optimization and management
    - Performance tracking and metrics
    - Batch processing support
    - LangGraph integration ready
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced BERT classifier with config integration
        
        Args:
            config: Configuration dictionary for runtime overrides
        """
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        bert_config = get_model_config('bert_classifier')
        system_settings = get_settings()
        
        # Merge with runtime overrides
        if config:
            bert_config.update(config)

        self.agent_name = "bert_classifier"
        
        # Initialize base agent with merged config
        super().__init__(bert_config)
        
        # ✅ USE CONFIG VALUES INSTEAD OF HARDCODED
        self.model_name = self.config.get('model_name', 'bert-base-uncased')
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 16)
        self.device_setting = self.config.get('device', 'auto')
        
        # ✅ USE PREPROCESSING CONFIG FROM FILES
        self.enable_preprocessing = self.config.get('enable_preprocessing', True)
        preprocessing_config = self.config.get('preprocessing_config', {})
        
        # ✅ USE SYSTEM SETTINGS FOR PATHS
        self.models_dir = system_settings.models_dir
        self.enable_metrics = self.config.get('enable_metrics', True)
        
        # Initialize modular components with config
        self.device_manager = DeviceManager()
        self.model_manager = ModelManager(self.device_manager)
        
        # Initialize preprocessor with config
        if self.enable_preprocessing:
            # Ensure max_length consistency
            preprocessing_config['max_length'] = self.max_length
            self.preprocessor = TextPreprocessor(preprocessing_config)
        else:
            self.preprocessor = None
        
        # Model components (will be loaded later)
        self.model = None
        self.tokenizer = None
        
        # Enhanced performance tracking
        self.bert_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'error_count': 0,
            'average_inference_time': 0.0,
            'batch_predictions': 0,
            'preprocessing_enabled': self.enable_preprocessing,
            'config_loaded': True
        }
        
        self.logger.info(f"✅ Enhanced BERT Classifier initialized with config")
        self.logger.info(f"🖥️ Device: {self.device_manager.get_device()}")
        self.logger.info(f"🧹 Preprocessing: {'Enabled' if self.preprocessor else 'Disabled'}")
        self.logger.info(f"⚙️ Model: {self.model_name}, Max Length: {self.max_length}")
    
    def load_model(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        🔧 LOAD BERT MODEL WITH CONFIG INTEGRATION
        
        Load the trained BERT model using the modular model manager.
        Uses config for default model path if not specified.
        
        Args:
            model_path: Path to the saved model directory (uses config default if None)
            
        Returns:
            Dictionary with loading results and model information
        """
        try:
            # ✅ USE MODEL PATH FROM CONFIG IF NOT PROVIDED
            if model_path is None:
                model_path = self.models_dir / self.config.get('model_path', 'bert_fake_news_classifier')
            
            self.logger.info(f"📂 Loading BERT model from: {model_path}")
            
            # Load model using model manager
            self.model, self.tokenizer = self.model_manager.load_model(
                model_path, self.model_name
            )
            
            # Get model information
            model_info = self.model_manager.get_model_info()
            device_info = self.device_manager.get_device_info()
            
            self.logger.info("🎉 BERT model ready for fake news detection!")
            
            return {
                "success": True,
                "model_info": model_info,
                "device_info": device_info,
                "preprocessing_enabled": self.preprocessor is not None,
                "config_integrated": True,
                "model_path": str(model_path)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_loaded": False,
                "config_integrated": True
            }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 MAIN PROCESSING METHOD - LANGGRAPH COMPATIBLE
        
        Process input according to BaseAgent interface for LangGraph compatibility.
        Now uses configuration for all processing parameters.
        
        Args:
            input_data: Dictionary containing:
                - text: Article text to analyze
                - skip_preprocessing: Optional flag to skip preprocessing
                - custom_config: Optional runtime configuration overrides
                
        Returns:
            Standardized output dictionary for LangGraph
        """
        # Validate input
        is_valid, error_msg = self.validate_input(input_data)
        if not is_valid:
            return self.format_error_output(ValueError(error_msg), input_data)
        
        # Start processing timer
        self._start_processing_timer()
        
        try:
            # Extract parameters
            article_text = input_data.get('text', '')
            skip_preprocessing = input_data.get('skip_preprocessing', False)
            
            # ✅ USE CONFIG FOR PROCESSING DECISIONS
            confidence_threshold = self.config.get('high_confidence_threshold', 0.8)
            
            # Perform prediction
            prediction_result = self.predict(
                article_text=article_text,
                skip_preprocessing=skip_preprocessing
            )
            
            # Check if prediction was successful
            if prediction_result.get('prediction') == 'ERROR':
                raise Exception(prediction_result.get('error', 'Unknown prediction error'))
            
            # Extract results with config context
            result = {
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'probabilities': {
                    'real': prediction_result['real_probability'],
                    'fake': prediction_result['fake_probability']
                },
                'text_analysis': {
                    'original_length': prediction_result['original_text_length'],
                    'processed_length': prediction_result['processed_text_length'],
                    'tokens_used': prediction_result['tokens_used'],
                    'preprocessing_applied': prediction_result['preprocessing_applied']
                },
                'quality_flags': {
                    'high_confidence': prediction_result['confidence'] >= confidence_threshold,
                    'config_applied': True
                }
            }
            
            # End processing timer and update metrics
            self._end_processing_timer()
            self._update_success_metrics(prediction_result['confidence'])
            self.bert_metrics['successful_predictions'] += 1
            
            # Format output for LangGraph
            return self.format_output(
                result=result,
                confidence=prediction_result['confidence'],
                metadata={
                    'processing_time': prediction_result['total_time_seconds'],
                    'device_used': prediction_result['device_used'],
                    'model_name': self.model_name,
                    'config_version': '2.0_integrated',
                    'agent_version': '2.0_modular'
                }
            )
            
        except Exception as e:
            self._end_processing_timer()
            self._update_error_metrics(e)
            self.bert_metrics['error_count'] += 1
            return self.format_error_output(e, input_data)
    
    def predict(self, article_text: str, skip_preprocessing: bool = False) -> Dict[str, Any]:
        """
        🎯 PREDICT SINGLE ARTICLE WITH CONFIG INTEGRATION
        
        Analyze a single news article with enhanced error handling and config usage.
        
        Args:
            article_text: The news article text to analyze
            skip_preprocessing: Skip preprocessing if text is already clean
            
        Returns:
            Detailed prediction results
        """
        start_time = time.time()
        
        if not self.model_manager.is_model_loaded():
            raise ValueError("❌ Model not loaded. Please call load_model() first.")
        
        try:
            # Step 1: Preprocess text using config settings
            preprocessing_applied = False
            if self.enable_preprocessing and not skip_preprocessing:
                cleaned_text = self.preprocessor.preprocess_text(article_text)
                preprocessing_applied = True
                self.logger.debug(f"📝 Text preprocessed: {len(article_text)} → {len(cleaned_text)} chars")
            else:
                cleaned_text = sanitize_text(article_text)  # Basic cleanup
            
            # Validate cleaned text
            min_length = self.config.get('min_text_length', 5)
            if not cleaned_text or len(cleaned_text.strip()) < min_length:
                raise ValueError(f"Article text is too short (< {min_length} chars) after preprocessing")
            
            # Step 2: Tokenization with config max_length
            encoding = self.tokenizer(
                cleaned_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device_manager.get_device())
            attention_mask = encoding['attention_mask'].to(self.device_manager.get_device())
            
            # Step 3: Model inference
            inference_start = time.time()
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probabilities = torch.softmax(outputs.logits, dim=1)
                real_prob = probabilities[0][0].item()
                fake_prob = probabilities[0][1].item()
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = max(real_prob, fake_prob)
            
            inference_time = time.time() - inference_start
            total_time = time.time() - start_time
            
            # Step 4: Format results with config metadata
            result = {
                # Main prediction results
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'prediction_numeric': prediction,
                'confidence': confidence,
                'real_probability': real_prob,
                'fake_probability': fake_prob,
                
                # Text analysis
                'original_text_length': len(article_text),
                'processed_text_length': len(cleaned_text),
                'tokens_used': int(attention_mask.sum().item()),
                'preprocessing_applied': preprocessing_applied,
                
                # System information with config context
                'model_name': self.model_name,
                'device_used': str(self.device_manager.get_device()),
                'predicted_at': datetime.now().isoformat(),
                'max_length_used': self.max_length,
                
                # Performance metrics
                'total_time_seconds': round(total_time, 3),
                'inference_time_seconds': round(inference_time, 3),
                
                # Agent integration
                'agent_name': 'BERT_Classifier',
                'agent_version': '2.0_modular',
                'config_integrated': True
            }
            
            # Update metrics
            self.bert_metrics['total_predictions'] += 1
            self._update_inference_metrics(inference_time)
            
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error(f"❌ Prediction failed: {str(e)}")
            return {
                'prediction': 'ERROR',
                'error': str(e),
                'confidence': 0.0,
                'predicted_at': datetime.now().isoformat(),
                'agent_name': 'BERT_Classifier',
                'total_time_seconds': error_time,
                'config_integrated': True
            }
    
    def predict_batch(self, texts: List[str], batch_size: Optional[int] = None,
                     skip_preprocessing: bool = False) -> List[Dict[str, Any]]:
        """
        🔄 BATCH PREDICTION WITH CONFIG INTEGRATION
        
        Process multiple articles efficiently using batch processing with config settings.
        
        Args:
            texts: List of article texts to analyze
            batch_size: Batch size for processing (uses config default if None)
            skip_preprocessing: Skip preprocessing for all texts
            
        Returns:
            List of prediction results
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("❌ Model not loaded. Please call load_model() first.")
        
        if not texts:
            return []
        
        # ✅ USE BATCH SIZE FROM CONFIG
        batch_size = batch_size or self.batch_size
        
        self.logger.info(f"🔄 Running batch prediction on {len(texts)} articles (batch_size={batch_size})...")
        
        try:
            # Create dataset with optional preprocessing
            preprocessor = None if skip_preprocessing else self.preprocessor
            dataset = FakeNewsDataset(texts, self.tokenizer, self.max_length, preprocessor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            self.model.eval()
            all_results = []
            device = self.device_manager.get_device()
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    batch_start = time.time()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    
                    batch_time = time.time() - batch_start
                    batch_idx_start = i * batch_size
                    
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
                            'agent_version': '2.0_modular',
                            'config_integrated': True,
                            'batch_size_used': batch_size
                        }
                        
                        all_results.append(result)
            
            # Update batch metrics
            self.bert_metrics['batch_predictions'] += 1
            self.logger.info(f"✅ Batch prediction completed: {len(all_results)} results")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"❌ Batch prediction failed: {str(e)}")
            raise
    
    def _update_inference_metrics(self, inference_time: float):
        """Update inference-specific metrics with config awareness"""
        total_predictions = self.bert_metrics['total_predictions']
        
        if total_predictions == 1:
            self.bert_metrics['average_inference_time'] = inference_time
        else:
            current_avg = self.bert_metrics['average_inference_time']
            self.bert_metrics['average_inference_time'] = (
                (current_avg * (total_predictions - 1) + inference_time) / total_predictions
            )
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        📊 GET COMPREHENSIVE PERFORMANCE METRICS WITH CONFIG INFO
        
        Combines base agent metrics with BERT-specific metrics, component metrics,
        and configuration information.
        
        Returns:
            Complete metrics dictionary including config details
        """
        # Get base metrics
        base_metrics = self.get_performance_metrics()
        
        # Get component metrics
        component_metrics = {}
        
        if self.preprocessor:
            component_metrics['preprocessing'] = self.preprocessor.get_preprocessing_stats()
        
        if self.model_manager.is_model_loaded():
            component_metrics['model'] = self.model_manager.get_model_info()
        
        component_metrics['device'] = self.device_manager.get_device_info()
        
        # ✅ ADD CONFIG INFORMATION TO METRICS
        config_metrics = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'preprocessing_enabled': self.enable_preprocessing,
            'device_setting': self.device_setting,
            'config_version': '2.0_integrated'
        }
        
        # Combine all metrics
        return {
            **base_metrics,
            'bert_specific_metrics': self.bert_metrics,
            'component_metrics': component_metrics,
            'config_metrics': config_metrics,
            'agent_type': 'bert_classifier',
            'modular_architecture': True,
            'config_integrated': True
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information with config context"""
        base_info = self.model_manager.get_model_info()
        base_info['config_integrated'] = True
        base_info['config_model_name'] = self.model_name
        return base_info
    
    def is_ready(self) -> bool:
        """Check if the classifier is ready for predictions"""
        return self.model_manager.is_model_loaded()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'preprocessing_enabled': self.enable_preprocessing,
            'device_setting': self.device_setting,
            'metrics_enabled': self.enable_metrics,
            'config_source': 'config_files',
            'models_dir': str(self.models_dir)
        }

# Testing functionality with config integration
if __name__ == "__main__":
    """Test the modular BERT classifier with config integration"""
    print("🧪 Testing Modular BERT Classifier with Config Integration")
    print("=" * 60)
    
    try:
        # Initialize classifier (will load from config files)
        classifier = BERTClassifier()
        print(f"✅ Classifier initialized with config: {classifier}")
        
        # Show config summary
        config_summary = classifier.get_config_summary()
        print(f"\n⚙️ Configuration Summary:")
        for key, value in config_summary.items():
            print(f"   {key}: {value}")
        
        # Test without model loading (to show error handling)
        test_input = {
            "text": "This is a test article for the modular BERT classifier with config integration."
        }
        
        print(f"\n📝 Testing without loaded model (should fail gracefully)...")
        result = classifier.process(test_input)
        print(f"Result success: {result['success']}")
        if not result['success']:
            print(f"Expected error: {result['error']['message']}")
        
        # Show comprehensive metrics with config info
        print(f"\n📊 Comprehensive metrics with config info:")
        metrics = classifier.get_comprehensive_metrics()
        print(f"Agent type: {metrics['agent_type']}")
        print(f"Config integrated: {metrics['config_integrated']}")
        print(f"Device: {metrics['component_metrics']['device']['device_type']}")
        print(f"Config model name: {metrics['config_metrics']['model_name']}")
        
        print(f"\n✅ Modular BERT classifier with config integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
