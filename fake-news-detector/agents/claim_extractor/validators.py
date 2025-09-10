# agents/claim_extractor/validators.py

"""
Claim Extractor Input and Output Validators

Production-ready validation system for claim extraction agent providing
comprehensive input validation, output verification, and data quality checks
with detailed error reporting and configurable validation rules.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """
    Structured validation result container.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of validation error messages
        warnings: List of validation warnings
        score: Validation quality score (0-100)
        details: Additional validation details
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    score: float = 0.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.details is None:
            self.details = {}

class InputValidator:
    """
    Comprehensive input validator for claim extraction requests.
    
    Validates article text, BERT results, configuration parameters,
    and other input data with configurable validation rules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize input validator with configuration.
        
        Args:
            config: Optional validation configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validation thresholds
        self.min_text_length = self.config.get('min_text_length', 50)
        self.max_text_length = self.config.get('max_text_length', 100000)
        self.min_word_count = self.config.get('min_word_count', 10)
        self.max_word_count = self.config.get('max_word_count', 15000)
        
        # Content quality thresholds
        self.min_sentence_count = self.config.get('min_sentence_count', 3)
        self.max_repeated_chars = self.config.get('max_repeated_chars', 10)
        self.min_unique_words_ratio = self.config.get('min_unique_words_ratio', 0.3)
        
        # Security validation
        self.enable_security_checks = self.config.get('enable_security_checks', True)
        self.blocked_patterns = self.config.get('blocked_patterns', [])
        
    def validate_article_text(self, text: str) -> ValidationResult:
        """
        Validate article text content comprehensively.
        
        Args:
            text: Article text to validate
            
        Returns:
            ValidationResult with detailed validation results
        """
        errors = []
        warnings = []
        score = 100.0
        details = {}
        
        # Basic type and existence checks
        if not isinstance(text, str):
            errors.append("Article text must be a string")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        if not text or not text.strip():
            errors.append("Article text cannot be empty")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        text_clean = text.strip()
        details['original_length'] = len(text)
        details['clean_length'] = len(text_clean)
        
        # Length validation
        if len(text_clean) < self.min_text_length:
            errors.append(f"Article text too short: {len(text_clean)} < {self.min_text_length} characters")
            score -= 30
        
        if len(text_clean) > self.max_text_length:
            errors.append(f"Article text too long: {len(text_clean)} > {self.max_text_length} characters")
            score -= 20
        
        # Word count validation
        words = text_clean.split()
        word_count = len(words)
        details['word_count'] = word_count
        
        if word_count < self.min_word_count:
            errors.append(f"Article has too few words: {word_count} < {self.min_word_count}")
            score -= 25
        
        if word_count > self.max_word_count:
            warnings.append(f"Article is very long: {word_count} words")
            score -= 5
        
        # Sentence structure validation
        sentences = re.split(r'[.!?]+', text_clean)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentence_count = len(valid_sentences)
        details['sentence_count'] = sentence_count
        
        if sentence_count < self.min_sentence_count:
            errors.append(f"Article has too few sentences: {sentence_count} < {self.min_sentence_count}")
            score -= 20
        
        # Content quality checks
        if words:
            unique_words = set(word.lower() for word in words if word.isalnum())
            unique_ratio = len(unique_words) / len(words)
            details['unique_words_ratio'] = unique_ratio
            
            if unique_ratio < self.min_unique_words_ratio:
                warnings.append(f"Low vocabulary diversity: {unique_ratio:.2f}")
                score -= 10
        
        # Repeated character check
        repeated_char_pattern = r'(.)\1{' + str(self.max_repeated_chars) + ',}'
        if re.search(repeated_char_pattern, text_clean):
            warnings.append("Contains excessive repeated characters")
            score -= 5
        
        # Security validation
        if self.enable_security_checks:
            security_issues = self._check_security_patterns(text_clean)
            if security_issues:
                errors.extend(security_issues)
                score -= 40
        
        # Language detection (basic)
        if not self._has_readable_content(text_clean):
            warnings.append("Content may not be in a readable format")
            score -= 15
        
        is_valid = len(errors) == 0 and score > 0
        final_score = max(0.0, min(100.0, score))
        
        return ValidationResult(is_valid, errors, warnings, final_score, details)
    
    def validate_bert_results(self, bert_results: Dict[str, Any]) -> ValidationResult:
        """
        Validate BERT classification results.
        
        Args:
            bert_results: BERT results dictionary
            
        Returns:
            ValidationResult with validation results
        """
        errors = []
        warnings = []
        score = 100.0
        details = {}
        
        if not isinstance(bert_results, dict):
            errors.append("BERT results must be a dictionary")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        # Check required fields
        required_fields = ['prediction', 'confidence']
        for field in required_fields:
            if field not in bert_results:
                errors.append(f"Missing required BERT field: {field}")
                score -= 30
        
        # Validate prediction
        if 'prediction' in bert_results:
            prediction = bert_results['prediction']
            if prediction not in ['REAL', 'FAKE', 'UNCERTAIN']:
                errors.append(f"Invalid BERT prediction: {prediction}")
                score -= 25
            details['prediction'] = prediction
        
        # Validate confidence
        if 'confidence' in bert_results:
            confidence = bert_results['confidence']
            if not isinstance(confidence, (int, float)):
                errors.append("BERT confidence must be a number")
                score -= 20
            elif not 0 <= confidence <= 1:
                errors.append(f"BERT confidence out of range: {confidence}")
                score -= 15
            elif confidence < 0.5:
                warnings.append(f"Low BERT confidence: {confidence:.2f}")
                score -= 5
            details['confidence'] = confidence
        
        is_valid = len(errors) == 0
        final_score = max(0.0, min(100.0, score))
        
        return ValidationResult(is_valid, errors, warnings, final_score, details)
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive input data validation for claim extraction.
        
        Args:
            input_data: Complete input data dictionary
            
        Returns:
            ValidationResult with comprehensive validation results
        """
        errors = []
        warnings = []
        score = 100.0
        details = {}
        
        # Basic structure validation
        if not isinstance(input_data, dict):
            errors.append("Input data must be a dictionary")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        if not input_data:
            errors.append("Input data cannot be empty")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        # Validate required text field
        if 'text' not in input_data:
            errors.append("Missing required field: text")
            score -= 40
        else:
            text_validation = self.validate_article_text(input_data['text'])
            if not text_validation.is_valid:
                errors.extend([f"Text validation: {error}" for error in text_validation.errors])
                score -= 30
            if text_validation.warnings:
                warnings.extend([f"Text warning: {warning}" for warning in text_validation.warnings])
                score -= 5
            details['text_validation'] = text_validation.details
        
        # Validate optional BERT results
        if 'bert_results' in input_data:
            bert_validation = self.validate_bert_results(input_data['bert_results'])
            if not bert_validation.is_valid:
                warnings.extend([f"BERT validation: {error}" for error in bert_validation.errors])
                score -= 10  # Non-critical for claim extraction
            details['bert_validation'] = bert_validation.details
        
        # Validate optional parameters
        optional_fields = {
            'topic_domain': str,
            'include_verification_analysis': bool,
            'max_claims': int
        }
        
        for field, expected_type in optional_fields.items():
            if field in input_data:
                if not isinstance(input_data[field], expected_type):
                    warnings.append(f"Field {field} should be {expected_type.__name__}")
                    score -= 3
        
        # Validate max_claims range
        if 'max_claims' in input_data:
            max_claims = input_data['max_claims']
            if not 1 <= max_claims <= 20:
                warnings.append(f"max_claims should be between 1-20, got {max_claims}")
                score -= 5
        
        is_valid = len(errors) == 0
        final_score = max(0.0, min(100.0, score))
        
        return ValidationResult(is_valid, errors, warnings, final_score, details)
    
    def _check_security_patterns(self, text: str) -> List[str]:
        """Check for security-related patterns in text."""
        security_issues = []
        
        # Check for script injection patterns
        script_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'document\s*\.\s*write'
        ]
        
        for pattern in script_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                security_issues.append("Potential script injection detected")
                break
        
        # Check custom blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                security_issues.append(f"Blocked pattern detected: {pattern}")
        
        return security_issues
    
    def _has_readable_content(self, text: str) -> bool:
        """Check if text contains readable content."""
        # Count alphabetic characters
        alpha_chars = sum(1 for char in text if char.isalpha())
        total_chars = len(text)
        
        if total_chars == 0:
            return False
        
        # At least 30% alphabetic characters for readability
        alpha_ratio = alpha_chars / total_chars
        return alpha_ratio >= 0.3

class OutputValidator:
    """
    Comprehensive output validator for claim extraction results.
    
    Validates extracted claims, verification analysis, metadata,
    and ensures output format consistency.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize output validator with configuration.
        
        Args:
            config: Optional validation configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validation parameters
        self.min_claims = self.config.get('min_claims', 0)
        self.max_claims = self.config.get('max_claims', 20)
        self.min_claim_length = self.config.get('min_claim_length', 10)
        self.required_claim_fields = self.config.get('required_claim_fields', [
            'claim_id', 'text', 'claim_type', 'priority', 'verifiability_score'
        ])
        
    def validate_extracted_claims(self, claims: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate extracted claims structure and content.
        
        Args:
            claims: List of extracted claim dictionaries
            
        Returns:
            ValidationResult with validation results
        """
        errors = []
        warnings = []
        score = 100.0
        details = {}
        
        if not isinstance(claims, list):
            errors.append("Extracted claims must be a list")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        claim_count = len(claims)
        details['claim_count'] = claim_count
        
        # Check claim count bounds
        if claim_count < self.min_claims:
            warnings.append(f"Few claims extracted: {claim_count}")
            score -= 10
        
        if claim_count > self.max_claims:
            warnings.append(f"Many claims extracted: {claim_count}")
            score -= 5
        
        # Validate individual claims
        valid_claims = 0
        claim_types = set()
        priorities = []
        
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                errors.append(f"Claim {i+1} must be a dictionary")
                score -= 15
                continue
            
            # Check required fields
            missing_fields = []
            for field in self.required_claim_fields:
                if field not in claim:
                    missing_fields.append(field)
            
            if missing_fields:
                errors.append(f"Claim {i+1} missing fields: {missing_fields}")
                score -= 10
                continue
            
            # Validate claim text
            claim_text = claim.get('text', '')
            if not isinstance(claim_text, str) or len(claim_text.strip()) < self.min_claim_length:
                errors.append(f"Claim {i+1} has invalid or short text")
                score -= 8
                continue
            
            # Validate claim type
            claim_type = claim.get('claim_type', '')
            valid_types = ['Statistical', 'Event', 'Attribution', 'Research', 'Policy', 'Causal', 'Other']
            if claim_type not in valid_types:
                warnings.append(f"Claim {i+1} has unknown type: {claim_type}")
                score -= 3
            else:
                claim_types.add(claim_type)
            
            # Validate priority
            priority = claim.get('priority', 0)
            if not isinstance(priority, int) or not 1 <= priority <= 3:
                warnings.append(f"Claim {i+1} has invalid priority: {priority}")
                score -= 3
            else:
                priorities.append(priority)
            
            # Validate verifiability score
            verif_score = claim.get('verifiability_score', 0)
            if not isinstance(verif_score, (int, float)) or not 1 <= verif_score <= 10:
                warnings.append(f"Claim {i+1} has invalid verifiability score: {verif_score}")
                score -= 3
            
            valid_claims += 1
        
        details['valid_claims'] = valid_claims
        details['claim_types'] = list(claim_types)
        details['priority_distribution'] = {
            'high': priorities.count(1),
            'medium': priorities.count(2), 
            'low': priorities.count(3)
        }
        
        # Quality checks
        if valid_claims == 0 and claim_count > 0:
            errors.append("No valid claims found in output")
            score = 0
        elif valid_claims < claim_count * 0.7:
            warnings.append("Many claims failed validation")
            score -= 20
        
        is_valid = len(errors) == 0
        final_score = max(0.0, min(100.0, score))
        
        return ValidationResult(is_valid, errors, warnings, final_score, details)
    
    def validate_output_data(self, output_data: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive output data validation.
        
        Args:
            output_data: Complete output dictionary from claim extraction
            
        Returns:
            ValidationResult with validation results
        """
        errors = []
        warnings = []
        score = 100.0
        details = {}
        
        # Basic structure validation
        if not isinstance(output_data, dict):
            errors.append("Output data must be a dictionary")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        # Check required top-level fields
        required_fields = ['extracted_claims', 'metadata']
        for field in required_fields:
            if field not in output_data:
                errors.append(f"Missing required output field: {field}")
                score -= 30
        
        # Validate extracted claims
        if 'extracted_claims' in output_data:
            claims_validation = self.validate_extracted_claims(output_data['extracted_claims'])
            if not claims_validation.is_valid:
                errors.extend([f"Claims validation: {error}" for error in claims_validation.errors])
                score -= 25
            if claims_validation.warnings:
                warnings.extend([f"Claims warning: {warning}" for warning in claims_validation.warnings])
                score -= 5
            details['claims_validation'] = claims_validation.details
        
        # Validate metadata
        if 'metadata' in output_data:
            metadata = output_data['metadata']
            if not isinstance(metadata, dict):
                warnings.append("Metadata should be a dictionary")
                score -= 5
            else:
                # Check for expected metadata fields
                expected_metadata = [
                    'total_claims_found', 'processing_time_seconds', 'model_used'
                ]
                for field in expected_metadata:
                    if field not in metadata:
                        warnings.append(f"Missing metadata field: {field}")
                        score -= 3
        
        # Validate optional fields
        optional_fields = ['verification_analysis', 'prioritization_analysis', 'pattern_analysis']
        for field in optional_fields:
            if field in output_data:
                if not isinstance(output_data[field], (str, dict, type(None))):
                    warnings.append(f"Field {field} has unexpected type")
                    score -= 5
        
        is_valid = len(errors) == 0
        final_score = max(0.0, min(100.0, score))
        
        return ValidationResult(is_valid, errors, warnings, final_score, details)

# Utility functions
def create_validation_config(
    min_text_length: int = 50,
    max_text_length: int = 100000,
    enable_security_checks: bool = True,
    max_claims: int = 20
) -> Dict[str, Any]:
    """
    Create a validation configuration dictionary.
    
    Args:
        min_text_length: Minimum required text length
        max_text_length: Maximum allowed text length  
        enable_security_checks: Whether to enable security validation
        max_claims: Maximum number of claims to validate
        
    Returns:
        Configuration dictionary
    """
    return {
        'min_text_length': min_text_length,
        'max_text_length': max_text_length,
        'enable_security_checks': enable_security_checks,
        'max_claims': max_claims,
        'min_claim_length': 10,
        'required_claim_fields': [
            'claim_id', 'text', 'claim_type', 'priority', 'verifiability_score'
        ]
    }

# Testing functionality
if __name__ == "__main__":
    """Test validators functionality."""
    
    # Test input validation
    input_validator = InputValidator()
    
    test_input = {
        "text": "This is a test article with sufficient content for validation testing.",
        "bert_results": {
            "prediction": "REAL", 
            "confidence": 0.85
        },
        "topic_domain": "general"
    }
    
    result = input_validator.validate_input_data(test_input)
    print(f"Input validation: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"Score: {result.score:.1f}/100")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    # Test output validation
    output_validator = OutputValidator()
    
    test_output = {
        "extracted_claims": [
            {
                "claim_id": 1,
                "text": "Test claim text here",
                "claim_type": "Statistical",
                "priority": 1,
                "verifiability_score": 8
            }
        ],
        "metadata": {
            "total_claims_found": 1,
            "processing_time_seconds": 2.5,
            "model_used": "gemini-1.5-pro"
        }
    }
    
    output_result = output_validator.validate_output_data(test_output)
    print(f"\nOutput validation: {'PASSED' if output_result.is_valid else 'FAILED'}")
    print(f"Score: {output_result.score:.1f}/100")
    
    print("\n=== CLAIM EXTRACTOR VALIDATORS TESTING COMPLETED ===")
