# agents/claim_extractor/parsers.py

"""
Claim Parsing Utilities

Production-ready claim parsing utilities for claim extractor agent.
Enhanced parsing with multiple fallback methods, configuration support,
and comprehensive validation for structured claim extraction.
"""

from typing import Dict, List, Any, Optional
import re
import logging
import time


class ClaimParser:
    """
    Parses AI-generated claim extraction results into structured data.
    
    Provides multiple parsing strategies including structured format parsing,
    alternative format parsing, and basic sentence extraction with fallbacks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize claim parser with configuration.
        
        Args:
            config: Optional configuration dictionary for parsing parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default claim structure template
        self.default_claim_structure = {
            'claim_id': 0,
            'text': '',
            'claim_type': 'Other',
            'priority': 2,
            'verifiability_score': 5,
            'source': 'Not specified',
            'verification_strategy': 'Standard fact-checking',
            'importance': 'Supporting claim'
        }
        
        # Performance tracking
        self.parsing_count = 0
        self.successful_parses = 0
        self.fallback_parses = 0
        self.error_parses = 0
        self.total_processing_time = 0.0
        
        self.logger.info(f"Claim parser initialized with configuration: {bool(config)}")

    def parse_extracted_claims(self, raw_extraction: str) -> List[Dict[str, Any]]:
        """
        Parse raw LLM extraction output into structured claim data.
        
        Args:
            raw_extraction: Raw text output from claim extraction LLM
            
        Returns:
            List of structured claim dictionaries
        """
        start_time = time.time()
        self.parsing_count += 1
        
        try:
            # Primary parsing method - structured format
            claims = self._parse_structured_format(raw_extraction)
            
            # Fallback to alternative parsing if insufficient results
            min_expected = self.config.get('min_expected_claims', 1)
            if len(claims) < min_expected:
                self.logger.info("Primary parsing yielded insufficient results, trying alternative methods")
                claims = self._parse_alternative_format(raw_extraction)
                self.fallback_parses += 1
            
            # Final fallback to basic sentence extraction
            if len(claims) < min_expected:
                self.logger.info("Alternative parsing failed, using basic sentence extraction")
                claims = self._parse_basic_sentences(raw_extraction)
            
            # Validate and clean parsed claims
            claims = self._validate_and_clean_claims(claims)
            
            # Apply configuration limits
            max_claims = self.config.get('max_parsed_claims', 20)
            if len(claims) > max_claims:
                claims = claims[:max_claims]
                self.logger.info(f"Limited parsed claims to configured maximum: {max_claims}")
            
            # Assign sequential IDs
            for i, claim in enumerate(claims):
                claim['claim_id'] = i + 1
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_parsing_statistics(processing_time, len(claims), success=True)
            self.successful_parses += 1
            
            self.logger.info(f"Successfully parsed {len(claims)} structured claims")
            return claims
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_parsing_statistics(processing_time, 0, success=False)
            self.error_parses += 1
            
            self.logger.error(f"Claim parsing failed: {str(e)}")
            return self._create_error_claim(str(e))

    def _parse_structured_format(self, raw_extraction: str) -> List[Dict[str, Any]]:
        """Parse structured LLM output format with field labels."""
        claims = []
        lines = raw_extraction.split('\n')
        current_claim = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for claim start markers
            if self._is_claim_start(line):
                # Save previous claim if valid
                if current_claim and self._is_valid_claim(current_claim):
                    claims.append(current_claim.copy())
                
                # Initialize new claim
                current_claim = self.default_claim_structure.copy()
                
                # Extract priority from claim marker
                priority = self._extract_priority_from_line(line)
                if priority:
                    current_claim['priority'] = priority
            
            # Parse field lines within claims
            elif current_claim:
                self._parse_field_line(line, current_claim)
        
        # Add final claim if valid
        if current_claim and self._is_valid_claim(current_claim):
            claims.append(current_claim)
        
        return claims

    def _parse_alternative_format(self, raw_extraction: str) -> List[Dict[str, Any]]:
        """Parse alternative formats using regex patterns."""
        claims = []
        
        # Alternative parsing patterns
        patterns = [
            r'(?:Claim\s*\d+|^\d+\.)\s*[:\-]?\s*(.+?)(?=(?:Claim\s*\d+|^\d+\.)|$)',
            r'\*\*.*?\*\*\s*[:\-]?\s*(.+?)(?=\*\*.*?\*\*|$)',
            r'-\s*(.+?)(?=\n-|\n\n|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, raw_extraction, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                claim_text = match.strip()
                min_length = self.config.get('min_claim_text_length', 10)
                
                if len(claim_text) > min_length:
                    claim = self.default_claim_structure.copy()
                    claim['text'] = self._extract_claim_text(claim_text)
                    claim['claim_type'] = self._infer_claim_type(claim['text'])
                    claim['verifiability_score'] = self._extract_numeric_value(claim_text, 'verifiability', 5)
                    claim['priority'] = self._extract_numeric_value(claim_text, 'priority', 2)
                    claims.append(claim)
            
            # Use first successful pattern
            if claims:
                break
        
        return claims

    def _parse_basic_sentences(self, raw_extraction: str) -> List[Dict[str, Any]]:
        """Basic fallback parsing using sentence-level claim indicators."""
        claims = []
        sentences = raw_extraction.split('.')
        
        # Claim indicator keywords
        claim_indicators = [
            'study', 'research', 'found', 'showed', 'announced', 'said',
            'according to', 'reported', 'confirmed', 'revealed', 'data shows',
            'statistics indicate', 'poll found', 'survey revealed', 'experts claim'
        ]
        
        max_basic_claims = self.config.get('max_basic_claims', 5)
        min_sentence_length = self.config.get('min_sentence_length', 20)
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            if (len(sentence) > min_sentence_length and 
                any(indicator in sentence.lower() for indicator in claim_indicators)):
                
                claim = self.default_claim_structure.copy()
                claim['text'] = sentence + '.'
                claim['claim_type'] = self._infer_claim_type(sentence)
                claims.append(claim)
                
                if len(claims) >= max_basic_claims:
                    break
        
        return claims

    def _is_claim_start(self, line: str) -> bool:
        """Check if line indicates the start of a new claim."""
        claim_start_patterns = [
            r'\*\*Claim\s*\d+',
            r'Claim\s*\d+:',
            r'^\d+\.\s*\*\*',
            r'Priority\s*\d+.*Claim',
            r'##\s*Claim\s*\d+',
            r'\*\*\d+\.\s*'
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in claim_start_patterns)

    def _extract_priority_from_line(self, line: str) -> Optional[int]:
        """Extract priority number from claim marker line."""
        priority_match = re.search(r'Priority\s*(\d+)', line, re.IGNORECASE)
        if priority_match:
            priority = int(priority_match.group(1))
            return max(1, min(3, priority))  # Clamp to valid range
        return None

    def _parse_field_line(self, line: str, current_claim: Dict[str, Any]) -> None:
        """Parse individual field lines within a claim block."""
        # Text field
        if self._line_contains_field(line, ['text', 'claim']):
            text = self._extract_field_value(line)
            if text:
                current_claim['text'] = text.strip('"\'')
        
        # Type field
        elif self._line_contains_field(line, ['type']):
            claim_type = self._extract_field_value(line)
            if claim_type:
                current_claim['claim_type'] = claim_type
        
        # Verifiability field
        elif self._line_contains_field(line, ['verifiability']):
            score = self._extract_numeric_value(line, 'verifiability')
            if score is not None:
                current_claim['verifiability_score'] = score
        
        # Source field
        elif self._line_contains_field(line, ['source']):
            source = self._extract_field_value(line)
            if source:
                current_claim['source'] = source
        
        # Verification strategy field
        elif self._line_contains_field(line, ['verification strategy', 'verification']):
            strategy = self._extract_field_value(line)
            if strategy:
                current_claim['verification_strategy'] = strategy
        
        # Importance field
        elif self._line_contains_field(line, ['why important', 'importance']):
            importance = self._extract_field_value(line)
            if importance:
                current_claim['importance'] = importance

    def _line_contains_field(self, line: str, field_names: List[str]) -> bool:
        """Check if line contains any of the specified field names."""
        line_lower = line.lower()
        return any(
            f'**{field}**' in line_lower or f'- **{field}**' in line_lower 
            for field in field_names
        )

    def _extract_field_value(self, line: str) -> Optional[str]:
        """Extract value from a field line."""
        field_value_patterns = [
            r'\*\*[^*]+\*\*:\s*(.+)',
            r'-\s*\*\*[^*]+\*\*:\s*(.+)',
            r':\s*(.+)'
        ]
        
        for pattern in field_value_patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_numeric_value(self, text: str, field_name: str, default: int = 5) -> int:
        """Extract numeric value from text with validation."""
        try:
            # Look for number after field name
            pattern = rf'{field_name}[:\s]*(\d+)'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                value = int(match.group(1))
                return max(1, min(10, value))  # Clamp to 1-10 range
            
            # Look for any number in the text
            number_match = re.search(r'(\d+)', text)
            if number_match:
                value = int(number_match.group(1))
                if 1 <= value <= 10:
                    return value
            
            return default
            
        except (ValueError, AttributeError):
            return default

    def _infer_claim_type(self, claim_text: str) -> str:
        """Infer claim type from claim text content."""
        text_lower = claim_text.lower()
        
        # Statistical claims
        if any(indicator in text_lower for indicator in ['%', 'percent', 'study', 'research', 'data']):
            return 'Statistical'
        
        # Event claims
        elif any(indicator in text_lower for indicator in ['announced', 'occurred', 'happened', 'will']):
            return 'Event'
        
        # Attribution claims
        elif any(indicator in text_lower for indicator in ['said', 'according to', 'spokesperson']):
            return 'Attribution'
        
        # Research claims
        elif any(indicator in text_lower for indicator in ['study', 'research', 'scientists', 'published']):
            return 'Research'
        
        # Policy claims
        elif any(indicator in text_lower for indicator in ['law', 'policy', 'regulation', 'court']):
            return 'Policy'
        
        # Causal claims
        elif any(indicator in text_lower for indicator in ['caused by', 'due to', 'because of']):
            return 'Causal'
        
        else:
            return 'Other'

    def _extract_claim_text(self, raw_text: str) -> str:
        """Extract and clean claim text from raw input."""
        text = raw_text.strip()
        
        # Remove field indicators
        text = re.sub(r'^\s*-?\s*\*\*[^*]+\*\*:\s*', '', text)
        text = re.sub(r'^\s*Text:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*Claim:\s*', '', text, flags=re.IGNORECASE)
        
        # Clean quotes
        text = text.strip('"\'')
        
        # Take first sentence if multiple sentences
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[0]) > 20:
            text = sentences[0] + '.'
        
        return text.strip()

    def _is_valid_claim(self, claim: Dict[str, Any]) -> bool:
        """Check if claim meets minimum validity requirements."""
        min_text_length = self.config.get('min_claim_text_length', 5)
        return (
            claim.get('text') and 
            isinstance(claim['text'], str) and
            len(claim['text'].strip()) >= min_text_length
        )

    def _validate_and_clean_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean parsed claims."""
        validated_claims = []
        min_text_length = self.config.get('min_claim_text_length', 5)
        valid_claim_types = ['Statistical', 'Event', 'Attribution', 'Research', 'Policy', 'Causal', 'Other']
        
        for claim in claims:
            # Ensure all required fields exist with defaults
            for key, default_value in self.default_claim_structure.items():
                if key not in claim:
                    claim[key] = default_value
            
            # Validate text field
            if not claim.get('text') or len(claim['text'].strip()) < min_text_length:
                continue  # Skip claims with insufficient text
            
            # Validate and clamp numeric fields
            claim['verifiability_score'] = max(1, min(10, claim.get('verifiability_score', 5)))
            claim['priority'] = max(1, min(3, claim.get('priority', 2)))
            
            # Clean text field
            claim['text'] = claim['text'].strip()
            if not claim['text'].endswith('.'):
                claim['text'] += '.'
            
            # Validate claim type
            if claim.get('claim_type') not in valid_claim_types:
                claim['claim_type'] = 'Other'
            
            validated_claims.append(claim)
        
        return validated_claims

    def _create_error_claim(self, error_msg: str) -> List[Dict[str, Any]]:
        """Create error claim when parsing fails completely."""
        error_claim = self.default_claim_structure.copy()
        error_claim.update({
            'text': f"Error parsing claims: {error_msg}",
            'claim_type': 'Error',
            'priority': 3,
            'verifiability_score': 1,
            'claim_id': 1
        })
        return [error_claim]

    def _update_parsing_statistics(self, processing_time: float, claims_found: int, success: bool) -> None:
        """Update parsing performance statistics."""
        self.total_processing_time += processing_time

    # Utility methods for working with parsed claims
    def format_claims_summary(self, claims: List[Dict[str, Any]]) -> str:
        """Format claims into a readable summary."""
        if not claims:
            return "No claims extracted from the article."
        
        summary_lines = [f"EXTRACTED CLAIMS ({len(claims)} total):", ""]
        
        for claim in claims:
            priority_indicator = "🔴" if claim['priority'] == 1 else "🟡" if claim['priority'] == 2 else "🟢"
            
            summary_lines.extend([
                f"Claim {claim['claim_id']}: {priority_indicator}",
                f" Text: {claim['text']}",
                f" Type: {claim['claim_type']}",
                f" Priority: {claim['priority']}",
                f" Verifiability: {claim['verifiability_score']}/10",
                ""
            ])
        
        return "\n".join(summary_lines)

    def get_claims_by_priority(self, claims: List[Dict[str, Any]], priority: int) -> List[Dict[str, Any]]:
        """Get claims filtered by priority level."""
        return [claim for claim in claims if claim.get('priority') == priority]

    def get_most_verifiable_claims(self, claims: List[Dict[str, Any]], min_score: int = 7) -> List[Dict[str, Any]]:
        """Get claims with high verifiability scores."""
        return [claim for claim in claims if claim.get('verifiability_score', 0) >= min_score]

    def get_claims_by_type(self, claims: List[Dict[str, Any]], claim_type: str) -> List[Dict[str, Any]]:
        """Get claims filtered by claim type."""
        return [claim for claim in claims if claim.get('claim_type') == claim_type]

    def calculate_parsing_quality(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics for parsed claims."""
        if not claims:
            return {'quality': 'poor', 'score': 0}
        
        # Calculate quality indicators
        has_text = sum(1 for claim in claims if claim.get('text') and len(claim['text']) > 5)
        has_type = sum(1 for claim in claims if claim.get('claim_type') != 'Other')
        has_verifiability = sum(1 for claim in claims if claim.get('verifiability_score', 0) > 5)
        has_priority = sum(1 for claim in claims if claim.get('priority', 0) in [1, 2, 3])
        
        total_claims = len(claims)
        
        # Calculate weighted quality score
        text_weight = self.config.get('text_quality_weight', 0.3)
        type_weight = self.config.get('type_quality_weight', 0.25)
        verif_weight = self.config.get('verifiability_weight', 0.25)
        priority_weight = self.config.get('priority_weight', 0.2)
        
        quality_score = (
            (has_text / total_claims) * text_weight +
            (has_type / total_claims) * type_weight +
            (has_verifiability / total_claims) * verif_weight +
            (has_priority / total_claims) * priority_weight
        ) * 100
        
        # Determine quality level
        excellent_threshold = self.config.get('excellent_threshold', 80)
        good_threshold = self.config.get('good_threshold', 60)
        fair_threshold = self.config.get('fair_threshold', 40)
        
        if quality_score >= excellent_threshold:
            quality_level = 'excellent'
        elif quality_score >= good_threshold:
            quality_level = 'good'
        elif quality_score >= fair_threshold:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        return {
            'quality': quality_level,
            'score': round(quality_score, 2),
            'claims_with_text': has_text,
            'claims_with_type': has_type,
            'claims_with_verifiability': has_verifiability,
            'claims_with_priority': has_priority,
            'total_claims': total_claims
        }

    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive parsing performance statistics."""
        success_rate = (self.successful_parses / self.parsing_count * 100) if self.parsing_count > 0 else 0
        fallback_rate = (self.fallback_parses / self.parsing_count * 100) if self.parsing_count > 0 else 0
        error_rate = (self.error_parses / self.parsing_count * 100) if self.parsing_count > 0 else 0
        avg_processing_time = (self.total_processing_time / self.parsing_count) if self.parsing_count > 0 else 0
        
        return {
            'total_parses': self.parsing_count,
            'successful_parses': self.successful_parses,
            'fallback_parses': self.fallback_parses,
            'error_parses': self.error_parses,
            'success_rate': round(success_rate, 2),
            'fallback_rate': round(fallback_rate, 2),
            'error_rate': round(error_rate, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'configuration_applied': bool(self.config)
        }


# Testing functionality
if __name__ == "__main__":
    """Test claim parser functionality."""
    
    # Initialize parser with test configuration
    test_config = {
        'min_claim_text_length': 10,
        'max_parsed_claims': 5,
        'excellent_threshold': 85
    }
    
    parser = ClaimParser(test_config)
    
    # Test structured format parsing
    test_extraction = """
    **Claim 1**: Priority 1
    - **Text**: "Study found 85% of patients showed improvement"
    - **Type**: Statistical
    - **Verifiability**: 8/10
    - **Source**: Research team

    **Claim 2**: Priority 2
    - **Text**: "CEO announced new product launch next month"
    - **Type**: Event
    - **Verifiability**: 6/10
    - **Source**: Company spokesperson
    """
    
    print("=== CLAIM PARSING TEST ===")
    claims = parser.parse_extracted_claims(test_extraction)
    
    print(f"Parsed {len(claims)} claims:")
    for claim in claims:
        print(f" - {claim['text']} (Type: {claim['claim_type']}, Priority: {claim['priority']})")
    
    # Test quality calculation
    quality = parser.calculate_parsing_quality(claims)
    print(f"\nParsing quality: {quality['quality']} ({quality['score']:.1f}%)")
    
    # Test claims summary formatting
    summary = parser.format_claims_summary(claims)
    print(f"\nFormatted summary:\n{summary}")
    
    # Test filtering methods
    high_priority_claims = parser.get_claims_by_priority(claims, 1)
    print(f"\nHigh priority claims: {len(high_priority_claims)}")
    
    verifiable_claims = parser.get_most_verifiable_claims(claims, min_score=7)
    print(f"Highly verifiable claims: {len(verifiable_claims)}")
    
    # Test performance statistics
    stats = parser.get_parsing_statistics()
    print(f"\nParsing statistics:")
    print(f" - Success rate: {stats['success_rate']:.1f}%")
    print(f" - Processing time: {stats['average_processing_time_ms']:.1f}ms")
    
    print("\n=== CLAIM PARSER TESTING COMPLETED ===")
