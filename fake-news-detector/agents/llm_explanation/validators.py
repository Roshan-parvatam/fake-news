# agents/llm_explanation/validators.py

"""
Input and Output Validators for LLM Explanation Agent

Comprehensive validation system for ensuring data quality and consistency
in explanation generation workflows with detailed error reporting.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime


class ValidationResult:
    """Container for validation results with detailed feedback."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
            errors: List of validation errors
            warnings: List of validation warnings
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str) -> None:
        """Add validation error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }


class InputValidator:
    """Validates input data for explanation generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize input validator.
        
        Args:
            config: Optional configuration for validation parameters
        """
        self.config = config or {}
        
        # Validation thresholds from config
        self.min_text_length = self.config.get('min_text_length', 50)
        self.max_text_length = self.config.get('max_text_length', 10000)
        self.min_confidence = self.config.get('min_confidence', 0.0)
        self.max_confidence = self.config.get('max_confidence', 1.0)
    
    def validate_explanation_input(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete explanation input data.
        
        Args:
            input_data: Input dictionary to validate
            
        Returns:
            ValidationResult with detailed feedback
        """
        result = ValidationResult(True)
        
        # Check required fields
        required_fields = ['text', 'prediction', 'confidence']
        for field in required_fields:
            if field not in input_data:
                result.add_error(f"Missing required field: {field}")
        
        # Validate individual fields if present
        if 'text' in input_data:
            text_validation = self.validate_article_text(input_data['text'])
            if not text_validation.is_valid:
                result.errors.extend(text_validation.errors)
                result.is_valid = False
            result.warnings.extend(text_validation.warnings)
        
        if 'prediction' in input_data:
            pred_validation = self.validate_prediction(input_data['prediction'])
            if not pred_validation.is_valid:
                result.errors.extend(pred_validation.errors)
                result.is_valid = False
            result.warnings.extend(pred_validation.warnings)
        
        if 'confidence' in input_data:
            conf_validation = self.validate_confidence(input_data['confidence'])
            if not conf_validation.is_valid:
                result.errors.extend(conf_validation.errors)
                result.is_valid = False
            result.warnings.extend(conf_validation.warnings)
        
        if 'metadata' in input_data:
            meta_validation = self.validate_metadata(input_data['metadata'])
            if not meta_validation.is_valid:
                result.errors.extend(meta_validation.errors)
                result.is_valid = False
            result.warnings.extend(meta_validation.warnings)
        
        return result
    
    def validate_article_text(self, text: Any) -> ValidationResult:
        """
        Validate article text content.
        
        Args:
            text: Article text to validate
            
        Returns:
            ValidationResult for text validation
        """
        result = ValidationResult(True)
        
        # Type validation
        if not isinstance(text, str):
            result.add_error(f"Article text must be string, got {type(text).__name__}")
            return result
        
        # Length validation
        text_length = len(text.strip())
        if text_length == 0:
            result.add_error("Article text cannot be empty")
        elif text_length < self.min_text_length:
            result.add_error(f"Article text too short: {text_length} chars (minimum: {self.min_text_length})")
        elif text_length > self.max_text_length:
            result.add_warning(f"Article text very long: {text_length} chars (may be truncated)")
        
        # Content quality checks
        if text.strip():
            # Check for suspicious patterns
            if text.count('http') > 10:
                result.add_warning("Text contains many URLs - may be spam or low quality")
            
            # Check for repeated content
            words = text.split()
            if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
                result.add_warning("Text has high word repetition - may be low quality")
            
            # Check encoding issues
            if any(ord(char) > 127 for char in text[:100]):  # Check first 100 chars
                result.add_warning("Text contains non-ASCII characters - encoding may need attention")
        
        return result
    
    def validate_prediction(self, prediction: Any) -> ValidationResult:
        """
        Validate prediction value.
        
        Args:
            prediction: Prediction to validate
            
        Returns:
            ValidationResult for prediction validation
        """
        result = ValidationResult(True)
        
        # Type validation
        if not isinstance(prediction, str):
            result.add_error(f"Prediction must be string, got {type(prediction).__name__}")
            return result
        
        # Value validation
        valid_predictions = ['FAKE', 'REAL', 'UNKNOWN']
        prediction_upper = prediction.strip().upper()
        
        if prediction_upper not in valid_predictions:
            result.add_error(f"Invalid prediction '{prediction}'. Must be one of: {valid_predictions}")
        elif prediction_upper == 'UNKNOWN':
            result.add_warning("Prediction is UNKNOWN - explanation quality may be limited")
        
        return result
    
    def validate_confidence(self, confidence: Any) -> ValidationResult:
        """
        Validate confidence score.
        
        Args:
            confidence: Confidence score to validate
            
        Returns:
            ValidationResult for confidence validation
        """
        result = ValidationResult(True)
        
        # Type validation
        if not isinstance(confidence, (int, float)):
            result.add_error(f"Confidence must be numeric, got {type(confidence).__name__}")
            return result
        
        # Range validation
        if confidence < self.min_confidence:
            result.add_error(f"Confidence {confidence} below minimum {self.min_confidence}")
        elif confidence > self.max_confidence:
            result.add_error(f"Confidence {confidence} above maximum {self.max_confidence}")
        
        # Quality warnings
        if confidence < 0.3:
            result.add_warning("Very low confidence - explanation may emphasize uncertainty")
        elif confidence > 0.95:
            result.add_warning("Very high confidence - ensure appropriate caveats")
        
        return result
    
    def validate_metadata(self, metadata: Any) -> ValidationResult:
        """
        Validate metadata dictionary.
        
        Args:
            metadata: Metadata to validate
            
        Returns:
            ValidationResult for metadata validation
        """
        result = ValidationResult(True)
        
        # Type validation
        if not isinstance(metadata, dict):
            result.add_error(f"Metadata must be dictionary, got {type(metadata).__name__}")
            return result
        
        # Validate common metadata fields
        if 'source' in metadata:
            source = metadata['source']
            if not isinstance(source, str):
                result.add_error("Metadata 'source' must be string")
            elif not source.strip():
                result.add_warning("Empty source in metadata")
        
        if 'date' in metadata:
            date_val = metadata['date']
            if isinstance(date_val, str):
                # Try to parse common date formats
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}'   # MM-DD-YYYY
                ]
                if not any(re.match(pattern, date_val) for pattern in date_patterns):
                    result.add_warning(f"Date format '{date_val}' may not be standard")
            else:
                result.add_warning("Date should be string for consistent processing")
        
        if 'subject' in metadata:
            subject = metadata['subject']
            if not isinstance(subject, str):
                result.add_error("Metadata 'subject' must be string")
        
        return result


class OutputValidator:
    """Validates output from explanation generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize output validator.
        
        Args:
            config: Optional configuration for validation parameters
        """
        self.config = config or {}
        self.min_explanation_length = self.config.get('min_explanation_length', 100)
        self.max_explanation_length = self.config.get('max_explanation_length', 5000)
    
    def validate_explanation_output(self, output: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete explanation output.
        
        Args:
            output: Output dictionary to validate
            
        Returns:
            ValidationResult with detailed feedback
        """
        result = ValidationResult(True)
        
        # Check required fields
        required_fields = ['explanation', 'metadata']
        for field in required_fields:
            if field not in output:
                result.add_error(f"Missing required output field: {field}")
        
        # Validate explanation content
        if 'explanation' in output:
            exp_validation = self.validate_explanation_content(output['explanation'])
            if not exp_validation.is_valid:
                result.errors.extend(exp_validation.errors)
                result.is_valid = False
            result.warnings.extend(exp_validation.warnings)
        
        # Validate metadata
        if 'metadata' in output:
            meta_validation = self.validate_output_metadata(output['metadata'])
            if not meta_validation.is_valid:
                result.errors.extend(meta_validation.errors)
                result.is_valid = False
            result.warnings.extend(meta_validation.warnings)
        
        # Validate optional fields
        optional_fields = ['detailed_analysis', 'confidence_analysis', 'source_assessment']
        for field in optional_fields:
            if field in output and output[field]:
                field_validation = self.validate_explanation_content(output[field])
                if not field_validation.is_valid:
                    result.add_warning(f"Issues with {field}: {', '.join(field_validation.errors)}")
        
        return result
    
    def validate_explanation_content(self, content: Any) -> ValidationResult:
        """
        Validate explanation content quality.
        
        Args:
            content: Explanation content to validate
            
        Returns:
            ValidationResult for content validation
        """
        result = ValidationResult(True)
        
        # Type validation
        if not isinstance(content, str):
            result.add_error(f"Explanation content must be string, got {type(content).__name__}")
            return result
        
        # Length validation
        content_length = len(content.strip())
        if content_length == 0:
            result.add_error("Explanation content cannot be empty")
            return result
        elif content_length < self.min_explanation_length:
            result.add_error(f"Explanation too short: {content_length} chars (minimum: {self.min_explanation_length})")
        elif content_length > self.max_explanation_length:
            result.add_warning(f"Explanation very long: {content_length} chars")
        
        # Content quality checks
        if content.strip():
            # Check for error messages in content
            error_indicators = [
                "error", "failed", "blocked", "unavailable", "not available",
                "something went wrong", "try again"
            ]
            content_lower = content.lower()
            if any(indicator in content_lower for indicator in error_indicators):
                result.add_warning("Explanation content may contain error messages")
            
            # Check for minimal content
            if len(content.split()) < 20:  # Less than 20 words
                result.add_warning("Explanation content is very brief")
            
            # Check for repeated phrases
            sentences = content.split('.')
            if len(sentences) > 3:
                sentence_starts = [s.strip()[:50] for s in sentences if s.strip()]
                unique_starts = set(sentence_starts)
                if len(unique_starts) < len(sentence_starts) * 0.8:
                    result.add_warning("Explanation has repetitive content")
        
        return result
    
    def validate_output_metadata(self, metadata: Any) -> ValidationResult:
        """
        Validate output metadata structure and content.
        
        Args:
            metadata: Output metadata to validate
            
        Returns:
            ValidationResult for metadata validation
        """
        result = ValidationResult(True)
        
        # Type validation
        if not isinstance(metadata, dict):
            result.add_error(f"Output metadata must be dictionary, got {type(metadata).__name__}")
            return result
        
        # Check expected metadata fields
        expected_fields = ['response_time_seconds', 'model_used', 'analysis_timestamp']
        for field in expected_fields:
            if field not in metadata:
                result.add_warning(f"Missing expected metadata field: {field}")
        
        # Validate specific fields
        if 'response_time_seconds' in metadata:
            response_time = metadata['response_time_seconds']
            if not isinstance(response_time, (int, float)):
                result.add_error("Response time must be numeric")
            elif response_time < 0:
                result.add_error("Response time cannot be negative")
            elif response_time > 300:  # 5 minutes
                result.add_warning(f"Very long response time: {response_time}s")
        
        if 'confidence_level' in metadata:
            confidence = metadata['confidence_level']
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                result.add_error("Confidence level must be numeric between 0 and 1")
        
        if 'analysis_timestamp' in metadata:
            timestamp = metadata['analysis_timestamp']
            if not isinstance(timestamp, str):
                result.add_error("Analysis timestamp must be string")
            else:
                # Basic ISO format check
                if not re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', timestamp):
                    result.add_warning("Timestamp format may not be ISO standard")
        
        return result


class BatchValidator:
    """Validates batch processing operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize batch validator.
        
        Args:
            config: Optional configuration for validation parameters
        """
        self.config = config or {}
        self.max_batch_size = self.config.get('max_batch_size', 100)
        self.input_validator = InputValidator(config)
        self.output_validator = OutputValidator(config)
    
    def validate_batch_input(self, batch_data: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate batch input data.
        
        Args:
            batch_data: List of input dictionaries to validate
            
        Returns:
            ValidationResult for batch validation
        """
        result = ValidationResult(True)
        
        # Type validation
        if not isinstance(batch_data, list):
            result.add_error(f"Batch data must be list, got {type(batch_data).__name__}")
            return result
        
        # Size validation
        if len(batch_data) == 0:
            result.add_error("Batch data cannot be empty")
            return result
        elif len(batch_data) > self.max_batch_size:
            result.add_error(f"Batch size {len(batch_data)} exceeds maximum {self.max_batch_size}")
        
        # Validate individual items
        error_count = 0
        for i, item in enumerate(batch_data):
            item_validation = self.input_validator.validate_explanation_input(item)
            if not item_validation.is_valid:
                error_count += 1
                for error in item_validation.errors:
                    result.add_error(f"Item {i}: {error}")
        
        # Summary warnings
        if error_count > 0:
            error_rate = error_count / len(batch_data)
            if error_rate > 0.1:  # More than 10% errors
                result.add_warning(f"High error rate in batch: {error_rate:.1%}")
        
        return result
    
    def validate_batch_output(self, batch_results: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate batch output results.
        
        Args:
            batch_results: List of output dictionaries to validate
            
        Returns:
            ValidationResult for batch output validation
        """
        result = ValidationResult(True)
        
        # Type validation
        if not isinstance(batch_results, list):
            result.add_error(f"Batch results must be list, got {type(batch_results).__name__}")
            return result
        
        # Validate individual results
        success_count = 0
        for i, item in enumerate(batch_results):
            if not isinstance(item, dict):
                result.add_error(f"Result {i} must be dictionary")
                continue
            
            # Check for success indicators
            if item.get('batch_success', True):
                success_count += 1
                # Validate successful results
                if 'explanation' in item:
                    item_validation = self.output_validator.validate_explanation_output(item)
                    if not item_validation.is_valid:
                        for error in item_validation.errors:
                            result.add_warning(f"Result {i}: {error}")
        
        # Success rate analysis
        if batch_results:
            success_rate = success_count / len(batch_results)
            if success_rate < 0.8:  # Less than 80% success
                result.add_warning(f"Low batch success rate: {success_rate:.1%}")
        
        return result


# Utility functions
def validate_prompt_parameters(prompt_type: str, **kwargs) -> ValidationResult:
    """
    Validate parameters for prompt generation.
    
    Args:
        prompt_type: Type of prompt to validate parameters for
        **kwargs: Parameters to validate
        
    Returns:
        ValidationResult for parameter validation
    """
    result = ValidationResult(True)
    
    # Define required parameters for each prompt type
    required_params = {
        'main': ['article_text', 'prediction', 'confidence', 'source', 'date', 'subject'],
        'detailed': ['article_text', 'prediction', 'confidence', 'metadata'],
        'confidence': ['article_text', 'prediction', 'confidence'],
        'source': ['source', 'article_context']
    }
    
    if prompt_type not in required_params:
        result.add_error(f"Unknown prompt type: {prompt_type}")
        return result
    
    # Check for missing parameters
    missing_params = [param for param in required_params[prompt_type] if param not in kwargs]
    if missing_params:
        for param in missing_params:
            result.add_error(f"Missing required parameter: {param}")
    
    # Validate parameter types and values
    for param, value in kwargs.items():
        if param == 'confidence' and isinstance(value, (int, float)):
            if not (0 <= value <= 1):
                result.add_error(f"Confidence must be between 0 and 1, got {value}")
        elif param in ['article_text', 'prediction', 'source', 'date', 'subject'] and not isinstance(value, str):
            result.add_warning(f"Parameter {param} should be string, got {type(value).__name__}")
    
    return result


# Testing functionality
if __name__ == "__main__":
    """Test validation functionality."""
    
    # Test input validation
    input_validator = InputValidator()
    
    test_input = {
        'text': 'This is a test article about fake news detection.',
        'prediction': 'FAKE',
        'confidence': 0.85,
        'metadata': {
            'source': 'test-source.com',
            'date': '2025-01-01',
            'subject': 'technology'
        }
    }
    
    input_result = input_validator.validate_explanation_input(test_input)
    print("=== INPUT VALIDATION TEST ===")
    print(f"Valid: {input_result.is_valid}")
    print(f"Errors: {input_result.errors}")
    print(f"Warnings: {input_result.warnings}")
    
    # Test output validation
    output_validator = OutputValidator()
    
    test_output = {
        'explanation': 'This article shows signs of misinformation based on several factors...',
        'metadata': {
            'response_time_seconds': 2.5,
            'model_used': 'gemini-1.5-pro',
            'analysis_timestamp': '2025-01-01T12:00:00',
            'confidence_level': 0.85
        }
    }
    
    output_result = output_validator.validate_explanation_output(test_output)
    print(f"\n=== OUTPUT VALIDATION TEST ===")
    print(f"Valid: {output_result.is_valid}")
    print(f"Errors: {output_result.errors}")
    print(f"Warnings: {output_result.warnings}")
    
    # Test prompt parameter validation
    prompt_result = validate_prompt_parameters(
        'main',
        article_text='test',
        prediction='REAL',
        confidence=0.9,
        source='test.com',
        date='2025-01-01',
        subject='test'
    )
    
    print(f"\n=== PROMPT VALIDATION TEST ===")
    print(f"Valid: {prompt_result.is_valid}")
    print(f"Errors: {prompt_result.errors}")
    
    print("\n=== VALIDATION TESTING COMPLETED ===")
