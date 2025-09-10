# agents/context_analyzer/validators.py

"""
Context Analyzer Input/Output Validation

Validation utilities for the Context Analyzer Agent providing comprehensive
input validation, output validation, and bias analysis result checking.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass

from .exceptions import (
    InputValidationError, 
    BiasDetectionError, 
    ManipulationDetectionError,
    ScoringConsistencyError,
    raise_input_validation_error
)


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class InputValidator:
    """
    Input validation for context analyzer processing.
    
    Validates article text, previous analysis data, and configuration
    parameters before processing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize input validator.
        
        Args:
            config: Optional configuration for validation rules
        """
        self.config = config or {}
        self.min_article_length = self.config.get('min_article_length', 50)
        self.max_article_length = self.config.get('max_article_length', 50000)
        self.min_analysis_confidence = self.config.get('min_analysis_confidence', 0.0)

    def validate_article_text(self, text: Any) -> ValidationResult:
        """
        Validate article text input.
        
        Args:
            text: Article text to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(text, str):
            errors.append(f"Article text must be string, got {type(text).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Content validation
        if not text.strip():
            errors.append("Article text cannot be empty or whitespace only")
        
        # Length validation
        text_length = len(text.strip())
        if text_length < self.min_article_length:
            errors.append(f"Article text too short: {text_length} chars (minimum: {self.min_article_length})")
        
        if text_length > self.max_article_length:
            warnings.append(f"Article text very long: {text_length} chars (maximum recommended: {self.max_article_length})")
        
        # Content quality checks
        if text_length < 200:
            warnings.append("Article text may be too short for meaningful context analysis")
        
        # Check for suspicious patterns that might affect analysis
        if text.count('\n') > text_length * 0.1:
            warnings.append("Article contains excessive line breaks - may affect analysis quality")
        
        # Check for adequate sentence structure
        if len(re.findall(r'[.!?]', text)) < text_length / 200:
            warnings.append("Article may lack proper sentence structure")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_previous_analysis(self, analysis: Any) -> ValidationResult:
        """
        Validate previous analysis input.
        
        Args:
            analysis: Previous analysis results to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(analysis, dict):
            errors.append(f"Previous analysis must be a dictionary, got {type(analysis).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Required fields validation
        required_fields = ['prediction', 'confidence']
        for field in required_fields:
            if field not in analysis:
                errors.append(f"Missing required field in previous analysis: '{field}'")
        
        # Field value validation
        if 'confidence' in analysis:
            confidence = analysis['confidence']
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                errors.append(f"Invalid confidence value: {confidence} (must be between 0 and 1)")
            elif confidence < self.min_analysis_confidence:
                warnings.append(f"Low confidence score: {confidence:.2f}")
        
        if 'prediction' in analysis:
            prediction = analysis['prediction']
            if not isinstance(prediction, str) or not prediction.strip():
                errors.append("Prediction must be a non-empty string")
        
        # Optional field validation
        if 'source' in analysis:
            source = analysis['source']
            if not isinstance(source, str):
                warnings.append("Source field should be a string")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_processing_input(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete input data for context analysis processing.
        
        Args:
            input_data: Complete input data dictionary
            
        Returns:
            ValidationResult with validation status and any errors
        """
        all_errors = []
        all_warnings = []
        
        # Required fields
        if 'text' not in input_data:
            all_errors.append("Missing required 'text' field")
        else:
            text_result = self.validate_article_text(input_data['text'])
            all_errors.extend(text_result.errors)
            all_warnings.extend(text_result.warnings)
        
        # Optional fields
        if 'previous_analysis' in input_data:
            analysis_result = self.validate_previous_analysis(input_data['previous_analysis'])
            all_errors.extend(analysis_result.errors)
            all_warnings.extend(analysis_result.warnings)
        
        # Additional validation
        if 'include_detailed_analysis' in input_data:
            detailed = input_data['include_detailed_analysis']
            if not isinstance(detailed, bool):
                all_errors.append(f"'include_detailed_analysis' must be boolean, got {type(detailed).__name__}")
        
        return ValidationResult(len(all_errors) == 0, all_errors, all_warnings)


class BiasAnalysisValidator:
    """
    Validation for bias analysis results and scoring consistency.
    
    Ensures bias detection results are properly formatted and consistent
    with textual analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize bias analysis validator.
        
        Args:
            config: Optional configuration for validation rules
        """
        self.config = config or {}

    def validate_bias_scores(self, scores: Any) -> ValidationResult:
        """
        Validate bias scoring results.
        
        Args:
            scores: Bias scores to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(scores, dict):
            errors.append(f"Bias scores must be a dictionary, got {type(scores).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Required score fields
        required_scores = ['bias', 'manipulation', 'credibility', 'risk']
        
        for score_field in required_scores:
            if score_field not in scores:
                errors.append(f"Missing required score field: {score_field}")
            else:
                score = scores[score_field]
                if not isinstance(score, (int, float)) or score < 0 or score > 100:
                    errors.append(f"{score_field} score must be number between 0-100, got {score}")
        
        # Score consistency checks
        bias_score = scores.get('bias', 0)
        manipulation_score = scores.get('manipulation', 0)
        credibility_score = scores.get('credibility', 50)
        
        # High bias should generally correlate with lower credibility
        if bias_score > 75 and credibility_score > 75:
            warnings.append("High bias score with high credibility may be inconsistent")
        
        # High manipulation should correlate with higher bias
        if manipulation_score > 75 and bias_score < 25:
            warnings.append("High manipulation score with low bias may be inconsistent")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_political_bias_analysis(self, analysis: Any) -> ValidationResult:
        """
        Validate political bias analysis results.
        
        Args:
            analysis: Political bias analysis to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        if not isinstance(analysis, dict):
            errors.append(f"Political bias analysis must be dictionary, got {type(analysis).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Check required fields
        required_fields = ['political_leaning', 'confidence']
        for field in required_fields:
            if field not in analysis:
                errors.append(f"Missing required field: {field}")
        
        # Validate political leaning
        if 'political_leaning' in analysis:
            leaning = analysis['political_leaning']
            valid_leanings = ['left', 'right', 'center', 'mixed', 'neutral']
            if leaning not in valid_leanings:
                errors.append(f"Invalid political leaning: {leaning} (must be one of {valid_leanings})")
        
        # Validate confidence
        if 'confidence' in analysis:
            confidence = analysis['confidence']
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                errors.append(f"Invalid confidence: {confidence} (must be between 0 and 1)")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class ManipulationAnalysisValidator:
    """
    Validation for manipulation analysis results and technique detection.
    
    Ensures manipulation detection results are properly formatted and
    technique classifications are accurate.
    """
    
    def validate_manipulation_report(self, report: Any) -> ValidationResult:
        """
        Validate manipulation detection report.
        
        Args:
            report: Manipulation report to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        if not isinstance(report, dict):
            errors.append(f"Manipulation report must be dictionary, got {type(report).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Check required sections
        required_sections = ['propaganda_techniques', 'manipulation_patterns', 'logical_fallacies']
        for section in required_sections:
            if section not in report:
                errors.append(f"Missing required section: {section}")
        
        # Validate overall manipulation score
        if 'overall_manipulation_score' in report:
            score = report['overall_manipulation_score']
            if not isinstance(score, (int, float)) or score < 0 or score > 10:
                errors.append(f"Invalid manipulation score: {score} (must be between 0 and 10)")
        
        # Validate risk level
        if 'risk_level' in report:
            risk_level = report['risk_level']
            valid_levels = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            if risk_level not in valid_levels:
                errors.append(f"Invalid risk level: {risk_level} (must be one of {valid_levels})")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_propaganda_techniques(self, techniques: Any) -> ValidationResult:
        """
        Validate detected propaganda techniques.
        
        Args:
            techniques: Propaganda techniques detection results
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        if not isinstance(techniques, dict):
            errors.append(f"Propaganda techniques must be dictionary, got {type(techniques).__name__}")
            return ValidationResult(False, errors, warnings)
        
        if 'detected' not in techniques:
            errors.append("Missing 'detected' field in propaganda techniques")
            return ValidationResult(False, errors, warnings)
        
        detected = techniques['detected']
        if not isinstance(detected, dict):
            errors.append("'detected' field must be dictionary")
            return ValidationResult(False, errors, warnings)
        
        # Validate each detected technique
        for technique_name, technique_data in detected.items():
            if not isinstance(technique_data, dict):
                errors.append(f"Technique '{technique_name}' data must be dictionary")
                continue
            
            # Check required fields for each technique
            required_fields = ['confidence', 'severity']
            for field in required_fields:
                if field not in technique_data:
                    errors.append(f"Technique '{technique_name}' missing required field: {field}")
            
            # Validate confidence
            if 'confidence' in technique_data:
                confidence = technique_data['confidence']
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    errors.append(f"Invalid confidence for '{technique_name}': {confidence}")
            
            # Validate severity
            if 'severity' in technique_data:
                severity = technique_data['severity']
                valid_severities = ['low', 'medium', 'high']
                if severity not in valid_severities:
                    errors.append(f"Invalid severity for '{technique_name}': {severity}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class ScoringConsistencyValidator:
    """
    Validator for checking consistency between textual analysis and numerical scores.
    
    Addresses the main issue where LLM text analysis doesn't match the provided scores.
    """
    
    def validate_score_text_consistency(self, analysis_text: str, scores: Dict[str, int]) -> ValidationResult:
        """
        Validate consistency between textual analysis and numerical scores.
        
        Args:
            analysis_text: Generated analysis text
            scores: Dictionary of numerical scores
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        text_lower = analysis_text.lower()
        
        # Check bias score consistency
        bias_score = scores.get('bias', 50)
        if bias_score <= 25:  # Low bias
            if any(word in text_lower for word in ['high bias', 'significant bias', 'extreme bias', 'heavily biased']):
                errors.append(f"Text indicates high bias but score is {bias_score}")
        elif bias_score >= 75:  # High bias
            if any(word in text_lower for word in ['minimal bias', 'low bias', 'neutral', 'unbiased', 'balanced']):
                errors.append(f"Text indicates low bias but score is {bias_score}")
        
        # Check manipulation score consistency
        manipulation_score = scores.get('manipulation', 50)
        if manipulation_score <= 25:  # Low manipulation
            if any(word in text_lower for word in ['high manipulation', 'extreme manipulation', 'heavily manipulative']):
                errors.append(f"Text indicates high manipulation but score is {manipulation_score}")
        elif manipulation_score >= 75:  # High manipulation
            if any(word in text_lower for word in ['minimal manipulation', 'no manipulation', 'straightforward']):
                errors.append(f"Text indicates low manipulation but score is {manipulation_score}")
        
        # Check credibility score consistency
        credibility_score = scores.get('credibility', 50)
        if credibility_score <= 25:  # Low credibility
            if any(word in text_lower for word in ['highly credible', 'very reliable', 'trustworthy', 'authoritative']):
                errors.append(f"Text indicates high credibility but score is {credibility_score}")
        elif credibility_score >= 75:  # High credibility
            if any(word in text_lower for word in ['not credible', 'unreliable', 'questionable', 'dubious']):
                errors.append(f"Text indicates low credibility but score is {credibility_score}")
        
        # Check for general consistency indicators
        inconsistent_patterns = [
            (r'minimal\s+\w+', lambda s: s > 60, "minimal"),
            (r'extreme\s+\w+', lambda s: s < 40, "extreme"),
            (r'no\s+\w+', lambda s: s > 30, "no"),
            (r'significant\s+\w+', lambda s: s < 50, "significant")
        ]
        
        for pattern, score_check, descriptor in inconsistent_patterns:
            if re.search(pattern, text_lower):
                for score_type, score_value in scores.items():
                    if score_check(score_value):
                        warnings.append(f"Text uses '{descriptor}' language but {score_type} score is {score_value}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_risk_assessment_consistency(self, analysis_text: str, risk_level: str, risk_score: int) -> ValidationResult:
        """
        Validate consistency between risk assessment text and risk classification.
        
        Args:
            analysis_text: Risk assessment text
            risk_level: Assigned risk level (LOW, MEDIUM, HIGH, etc.)
            risk_score: Numerical risk score
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        text_lower = analysis_text.lower()
        
        # Risk level to score mapping
        risk_mappings = {
            'MINIMAL': (0, 20),
            'LOW': (20, 40),
            'MEDIUM': (40, 60),
            'HIGH': (60, 80),
            'CRITICAL': (80, 100)
        }
        
        # Check if score matches level
        if risk_level in risk_mappings:
            min_score, max_score = risk_mappings[risk_level]
            if not (min_score <= risk_score <= max_score):
                errors.append(f"Risk level '{risk_level}' inconsistent with score {risk_score}")
        
        # Check text consistency with level
        risk_text_indicators = {
            'MINIMAL': ['no risk', 'minimal risk', 'very low risk'],
            'LOW': ['low risk', 'minor concerns', 'slight risk'],
            'MEDIUM': ['moderate risk', 'some concerns', 'medium risk'],
            'HIGH': ['high risk', 'significant concerns', 'major risk'],
            'CRITICAL': ['critical risk', 'severe risk', 'dangerous', 'urgent']
        }
        
        expected_indicators = risk_text_indicators.get(risk_level, [])
        if expected_indicators and not any(indicator in text_lower for indicator in expected_indicators):
            warnings.append(f"Text doesn't contain expected language for risk level '{risk_level}'")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class OutputValidator:
    """
    Comprehensive output validation for context analyzer results.
    
    Validates all aspects of context analysis output including scores,
    analysis text, and result consistency.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize output validator.
        
        Args:
            config: Optional configuration for validation rules
        """
        self.config = config or {}
        self.bias_validator = BiasAnalysisValidator(config)
        self.manipulation_validator = ManipulationAnalysisValidator()
        self.consistency_validator = ScoringConsistencyValidator()

    def validate_context_analysis_output(self, output: Any) -> ValidationResult:
        """
        Validate complete context analysis output.
        
        Args:
            output: Context analysis output to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(output, dict):
            errors.append(f"Context analysis output must be dictionary, got {type(output).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Required sections validation
        required_sections = ['llm_analysis', 'llm_scores', 'context_scores']
        for section in required_sections:
            if section not in output:
                errors.append(f"Missing required section: {section}")
        
        # Validate LLM scores
        if 'llm_scores' in output:
            scores_result = self.bias_validator.validate_bias_scores(output['llm_scores'])
            errors.extend(scores_result.errors)
            warnings.extend(scores_result.warnings)
        
        # Validate manipulation report if present
        if 'manipulation_report' in output:
            manipulation_result = self.manipulation_validator.validate_manipulation_report(output['manipulation_report'])
            errors.extend(manipulation_result.errors)
            warnings.extend(manipulation_result.warnings)
        
        # Validate scoring consistency
        if 'llm_analysis' in output and 'llm_scores' in output:
            consistency_result = self.consistency_validator.validate_score_text_consistency(
                output['llm_analysis'], output['llm_scores']
            )
            errors.extend(consistency_result.errors)
            warnings.extend(consistency_result.warnings)
        
        return ValidationResult(len(errors) == 0, errors, warnings)


# Convenience validation functions
def validate_context_input(input_data: Dict[str, Any], config: Dict[str, Any] = None) -> ValidationResult:
    """
    Validate complete context analyzer input.
    
    Args:
        input_data: Input data to validate
        config: Optional validation configuration
        
    Returns:
        ValidationResult with validation status and any errors
    """
    validator = InputValidator(config)
    return validator.validate_processing_input(input_data)


def validate_bias_analysis(analysis_text: str, scores: Dict[str, int], config: Dict[str, Any] = None) -> bool:
    """
    Quick validation for bias analysis consistency.
    
    Args:
        analysis_text: Analysis text to validate
        scores: Numerical scores to validate
        config: Optional validation configuration
        
    Returns:
        True if consistent, False if inconsistent
    """
    validator = ScoringConsistencyValidator()
    result = validator.validate_score_text_consistency(analysis_text, scores)
    return result.is_valid


def validate_context_output(output: Dict[str, Any], config: Dict[str, Any] = None) -> ValidationResult:
    """
    Validate context analysis output.
    
    Args:
        output: Context analysis output to validate
        config: Optional validation configuration
        
    Returns:
        ValidationResult with validation status and any errors
    """
    validator = OutputValidator(config)
    return validator.validate_context_analysis_output(output)


# Testing functionality
if __name__ == "__main__":
    """Test context analyzer validation functionality."""
    
    # Test input validation
    test_input = {
        'text': 'This is a test article with sufficient length for context analysis purposes and proper structure.',
        'previous_analysis': {
            'prediction': 'REAL',
            'confidence': 0.85,
            'source': 'Test Source'
        },
        'include_detailed_analysis': True
    }
    
    input_result = validate_context_input(test_input)
    print(f"Input validation: {'✓' if input_result.is_valid else '✗'}")
    if input_result.errors:
        print(f"Errors: {input_result.errors}")
    if input_result.warnings:
        print(f"Warnings: {input_result.warnings}")
    
    # Test bias analysis consistency
    test_analysis = "This article shows minimal bias and straightforward presentation with high credibility."
    test_scores = {'bias': 15, 'manipulation': 10, 'credibility': 85, 'risk': 20}
    
    is_consistent = validate_bias_analysis(test_analysis, test_scores)
    print(f"\nBias analysis consistency: {'✓' if is_consistent else '✗'}")
    
    # Test inconsistent scoring (should fail)
    inconsistent_analysis = "This article shows extreme bias and heavy manipulation with questionable credibility."
    inconsistent_scores = {'bias': 10, 'manipulation': 15, 'credibility': 90, 'risk': 5}
    
    is_inconsistent = validate_bias_analysis(inconsistent_analysis, inconsistent_scores)
    print(f"Inconsistent analysis detection: {'✓' if not is_inconsistent else '✗'}")
    
    # Test output validation
    test_output = {
        'llm_analysis': test_analysis,
        'llm_scores': test_scores,
        'context_scores': {
            'bias_score': test_scores['bias'],
            'manipulation_score': test_scores['manipulation'],
            'credibility': test_scores['credibility'],
            'risk_level': 'LOW'
        }
    }
    
    output_result = validate_context_output(test_output)
    print(f"\nOutput validation: {'✓' if output_result.is_valid else '✗'}")
    if output_result.errors:
        print(f"Errors: {output_result.errors}")
    if output_result.warnings:
        print(f"Warnings: {output_result.warnings}")
    
    print("\n=== VALIDATION TESTS COMPLETED ===")
