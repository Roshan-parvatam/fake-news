# agents/credible_source/validators.py

"""
Validators for Credible Source Agent

Comprehensive input and output validation for credible source recommendations
with detailed error reporting and configurable validation rules.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
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

    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)

class InputValidator:
    """
    Comprehensive input validator for credible source agent.
    
    Validates article text, extracted claims, evidence evaluation,
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
        
        # Content quality thresholds
        self.min_sentence_count = self.config.get('min_sentence_count', 3)
        
        # Security validation
        self.enable_security_checks = self.config.get('enable_security_checks', True)

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
        
        # Sentence structure validation
        sentences = re.split(r'[.!?]+', text_clean)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentence_count = len(valid_sentences)
        details['sentence_count'] = sentence_count
        
        if sentence_count < self.min_sentence_count:
            errors.append(f"Article has too few sentences: {sentence_count} < {self.min_sentence_count}")
            score -= 20
        
        # Security validation
        if self.enable_security_checks:
            security_issues = self._check_security_patterns(text_clean)
            if security_issues:
                errors.extend(security_issues)
                score -= 40
        
        is_valid = len(errors) == 0 and score > 0
        final_score = max(0.0, min(100.0, score))
        
        return ValidationResult(is_valid, errors, warnings, final_score, details)

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
        
        if claim_count == 0:
            warnings.append("No claims provided for validation")
            return ValidationResult(True, errors, warnings, 80.0, details)
        
        # Validate individual claims
        valid_claims = 0
        
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                errors.append(f"Claim {i+1} must be a dictionary")
                score -= 15
                continue
            
            # Check for required fields
            required_fields = ['text', 'claim_type']
            missing_fields = []
            for field in required_fields:
                if field not in claim:
                    missing_fields.append(field)
            
            if missing_fields:
                errors.append(f"Claim {i+1} missing fields: {missing_fields}")
                score -= 10
                continue
            
            # Validate claim text
            claim_text = claim.get('text', '')
            if not isinstance(claim_text, str) or len(claim_text.strip()) < 5:
                errors.append(f"Claim {i+1} has invalid or short text")
                score -= 8
                continue
            
            # Validate claim type
            claim_type = claim.get('claim_type', '')
            valid_types = ['Statistical', 'Event', 'Attribution', 'Research', 'Policy', 'Causal', 'Other']
            if claim_type not in valid_types:
                warnings.append(f"Claim {i+1} has unknown type: {claim_type}")
                score -= 3
            
            valid_claims += 1
        
        details['valid_claims'] = valid_claims
        
        # Quality checks
        if valid_claims == 0 and claim_count > 0:
            errors.append("No valid claims found in input")
            score = 0
        elif valid_claims < claim_count * 0.7:
            warnings.append("Many claims failed validation")
            score -= 20
        
        is_valid = len(errors) == 0
        final_score = max(0.0, min(100.0, score))
        
        return ValidationResult(is_valid, errors, warnings, final_score, details)

    def validate_evidence_evaluation(self, evidence: Dict[str, Any]) -> ValidationResult:
        """
        Validate evidence evaluation structure.
        
        Args:
            evidence: Evidence evaluation dictionary
            
        Returns:
            ValidationResult with validation results
        """
        errors = []
        warnings = []
        score = 100.0
        details = {}
        
        if not isinstance(evidence, dict):
            errors.append("Evidence evaluation must be a dictionary")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        # Check for overall evidence score
        if 'overall_evidence_score' in evidence:
            overall_score = evidence['overall_evidence_score']
            if not isinstance(overall_score, (int, float)):
                errors.append("Overall evidence score must be numeric")
                score -= 20
            elif not 0 <= overall_score <= 10:
                errors.append("Overall evidence score must be between 0-10")
                score -= 15
            else:
                details['evidence_score'] = overall_score
        else:
            warnings.append("No overall evidence score provided")
            score -= 10
        
        # Check for individual evidence components
        evidence_components = ['source_quality', 'factual_accuracy', 'supporting_evidence']
        for component in evidence_components:
            if component in evidence:
                component_score = evidence[component]
                if not isinstance(component_score, (int, float)) or not 0 <= component_score <= 10:
                    warnings.append(f"Invalid {component} score: {component_score}")
                    score -= 5
        
        is_valid = len(errors) == 0
        final_score = max(0.0, min(100.0, score))
        
        return ValidationResult(is_valid, errors, warnings, final_score, details)

    def validate_input_data(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive input data validation for credible source recommendations.
        
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
        
        # Validate optional extracted claims
        if 'extracted_claims' in input_data:
            claims_validation = self.validate_extracted_claims(input_data['extracted_claims'])
            if not claims_validation.is_valid:
                warnings.extend([f"Claims validation: {error}" for error in claims_validation.errors])
                score -= 10  # Non-critical for source recommendation
            details['claims_validation'] = claims_validation.details
        
        # Validate optional evidence evaluation
        if 'evidence_evaluation' in input_data:
            evidence_validation = self.validate_evidence_evaluation(input_data['evidence_evaluation'])
            if not evidence_validation.is_valid:
                warnings.extend([f"Evidence validation: {error}" for error in evidence_validation.errors])
                score -= 5  # Non-critical for source recommendation
            details['evidence_validation'] = evidence_validation.details
        
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
        ]
        
        for pattern in script_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                security_issues.append("Potential script injection detected")
                break
        
        return security_issues

    # Legacy method for backward compatibility
    def validate(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Legacy validation method for backward compatibility.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        result = self.validate_input_data(input_data)
        if result.is_valid:
            return True, ""
        else:
            return False, result.errors[0] if result.errors else "Validation failed"

class OutputValidator:
    """
    Comprehensive output validator for credible source recommendations.
    
    Validates source recommendations, analysis results, metadata,
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
        self.min_sources = self.config.get('min_sources', 0)
        self.max_sources = self.config.get('max_sources', 20)
        self.required_source_fields = self.config.get('required_source_fields', [
            'name', 'type', 'reliability_score'
        ])

    def validate_source_recommendation_output(self, output: Any) -> ValidationResult:
        """
        Validate complete source recommendation output.
        
        Args:
            output: Source recommendation output to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(output, dict):
            errors.append(f"Source recommendation output must be dictionary, got {type(output).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Required fields validation
        required_fields = ['result', 'confidence', 'metadata']
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required output field: {field}")
        
        # Validate result structure
        if 'result' in output:
            result_validation = self.validate_recommendation_result(output['result'])
            if not result_validation.is_valid:
                errors.extend(result_validation.errors)
                warnings.extend(result_validation.warnings)
        
        # Validate confidence score
        if 'confidence' in output:
            conf_validation = self.validate_confidence_score(output['confidence'])
            if not conf_validation.is_valid:
                errors.extend(conf_validation.errors)
                warnings.extend(conf_validation.warnings)
        
        # Validate metadata
        if 'metadata' in output:
            meta_validation = self.validate_output_metadata(output['metadata'])
            if not meta_validation.is_valid:
                errors.extend(meta_validation.errors)
                warnings.extend(meta_validation.warnings)
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_recommendation_result(self, result: Any) -> ValidationResult:
        """
        Validate recommendation result structure.
        
        Args:
            result: Recommendation result to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        if not isinstance(result, dict):
            errors.append(f"Recommendation result must be dictionary, got {type(result).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Check for contextual sources
        if 'contextual_sources' in result:
            sources_validation = self.validate_contextual_sources(result['contextual_sources'])
            if not sources_validation.is_valid:
                errors.extend(sources_validation.errors)
                warnings.extend(sources_validation.warnings)
        
        # Check for recommendation scores
        if 'recommendation_scores' in result:
            scores_validation = self.validate_recommendation_scores(result['recommendation_scores'])
            if not scores_validation.is_valid:
                errors.extend(scores_validation.errors)
                warnings.extend(scores_validation.warnings)
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_contextual_sources(self, sources: Any) -> ValidationResult:
        """
        Validate contextual sources list.
        
        Args:
            sources: Sources list to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        if not isinstance(sources, list):
            errors.append(f"Contextual sources must be list, got {type(sources).__name__}")
            return ValidationResult(False, errors, warnings)
        
        if len(sources) == 0:
            warnings.append("No contextual sources provided")
            return ValidationResult(True, errors, warnings)
        
        if len(sources) > self.max_sources:
            warnings.append(f"Many sources provided ({len(sources)})")
        
        # Validate each source
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                errors.append(f"Source {i} must be dictionary, got {type(source).__name__}")
                continue
            
            # Check required source fields
            for field in self.required_source_fields:
                if field not in source:
                    errors.append(f"Source {i} missing required field: {field}")
                elif not isinstance(source[field], str) and field != 'reliability_score':
                    errors.append(f"Source {i} field '{field}' must be string")
                elif isinstance(source[field], str) and len(source[field].strip()) == 0:
                    errors.append(f"Source {i} field '{field}' cannot be empty")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_recommendation_scores(self, scores: Any) -> ValidationResult:
        """
        Validate recommendation scores.
        
        Args:
            scores: Scores dictionary to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        if not isinstance(scores, dict):
            errors.append(f"Recommendation scores must be dictionary, got {type(scores).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Check for overall recommendation score
        if 'overall_recommendation_score' not in scores:
            errors.append("Missing required 'overall_recommendation_score' field")
        else:
            overall_score = scores['overall_recommendation_score']
            if not isinstance(overall_score, (int, float)):
                errors.append("Overall recommendation score must be numeric")
            elif not (0 <= overall_score <= 10):
                errors.append("Overall recommendation score must be between 0 and 10")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_confidence_score(self, confidence: Any) -> ValidationResult:
        """
        Validate confidence score.
        
        Args:
            confidence: Confidence score to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        if not isinstance(confidence, (int, float)):
            errors.append(f"Confidence score must be numeric, got {type(confidence).__name__}")
        elif not (0 <= confidence <= 1):
            errors.append("Confidence score must be between 0 and 1")
        elif confidence < 0.3:
            warnings.append("Very low confidence score")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_output_metadata(self, metadata: Any) -> ValidationResult:
        """
        Validate output metadata.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        if not isinstance(metadata, dict):
            errors.append(f"Metadata must be dictionary, got {type(metadata).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Check for processing time
        if 'processing_time' in metadata:
            processing_time = metadata['processing_time']
            if not isinstance(processing_time, (int, float)):
                errors.append("Processing time must be numeric")
            elif processing_time < 0:
                errors.append("Processing time cannot be negative")
            elif processing_time > 300:  # 5 minutes
                warnings.append("Very long processing time")
        
        # Check for model used
        if 'model_used' in metadata:
            if not isinstance(metadata['model_used'], str):
                errors.append("Model used must be string")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    # Legacy method for backward compatibility
    def validate(self, output_data: Dict[str, Any]) -> bool:
        """
        Legacy validation method for backward compatibility.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            True if valid, False otherwise
        """
        result = self.validate_source_recommendation_output(output_data)
        return result.is_valid


# Testing functionality
if __name__ == "__main__":
    """Test validators functionality."""
    
    # Test input validation
    input_validator = InputValidator()
    
    test_input = {
        "text": "This is a test article with sufficient content for validation testing.",
        "extracted_claims": [
            {
                "text": "Test claim text here",
                "claim_type": "Research",
                "priority": 1
            }
        ],
        "evidence_evaluation": {
            "overall_evidence_score": 7.5
        }
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
        "result": {
            "contextual_sources": [
                {
                    "name": "Test Source",
                    "type": "academic",
                    "reliability_score": 8
                }
            ],
            "recommendation_scores": {
                "overall_recommendation_score": 7.5
            }
        },
        "confidence": 0.75,
        "metadata": {
            "processing_time": 2.5,
            "model_used": "gemini-1.5-pro"
        }
    }
    
    output_result = output_validator.validate_source_recommendation_output(test_output)
    print(f"\nOutput validation: {'PASSED' if output_result.is_valid else 'FAILED'}")
    
    print("\n=== CREDIBLE SOURCE VALIDATORS TESTING COMPLETED ===")
