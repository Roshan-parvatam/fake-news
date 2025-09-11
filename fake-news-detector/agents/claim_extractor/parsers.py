# agents/claim_extractor/parsers.py

"""
Claim Parsing Utilities - Production Ready

Production-ready claim parsing utilities for claim extractor agent.
Enhanced parsing with multiple fallback methods, configuration support,
comprehensive validation, performance tracking, and robust error handling
for structured claim extraction in production environments.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

from .exceptions import (
    ClaimParsingError,
    DataFormatError,
    raise_claim_parsing_error
)


class ClaimParser:
    """
    Parses AI-generated claim extraction results into structured data.
    
    Provides multiple parsing strategies including structured format parsing,
    alternative format parsing, and basic sentence extraction with comprehensive
    fallbacks, validation, and quality assessment for production environments.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize claim parser with production configuration.

        Args:
            config: Optional configuration dictionary for parsing parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ClaimParser")
        
        # Claim validation parameters
        self.min_claim_text_length = self.config.get('min_claim_text_length', 10)
        self.max_claim_text_length = self.config.get('max_claim_text_length', 500)
        self.max_parsed_claims = self.config.get('max_parsed_claims', 20)
        self.min_expected_claims = self.config.get('min_expected_claims', 1)
        
        # Quality thresholds
        self.text_quality_weight = self.config.get('text_quality_weight', 0.3)
        self.type_quality_weight = self.config.get('type_quality_weight', 0.25)
        self.verifiability_weight = self.config.get('verifiability_weight', 0.25)
        self.priority_weight = self.config.get('priority_weight', 0.2)
        
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
        
        # Parsing method success rates
        self.method_stats = {
            'structured_format': {'attempts': 0, 'successes': 0},
            'alternative_format': {'attempts': 0, 'successes': 0},
            'basic_sentences': {'attempts': 0, 'successes': 0},
            'json_format': {'attempts': 0, 'successes': 0},
            'regex_extraction': {'attempts': 0, 'successes': 0}
        }
        
        self.logger.info(f"ClaimParser initialized with max claims: {self.max_parsed_claims}")

    def parse_extracted_claims(self, raw_extraction: str, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Parse raw LLM extraction output into structured claim data with enhanced fallbacks.

        Args:
            raw_extraction: Raw text output from claim extraction LLM
            session_id: Optional session ID for tracking

        Returns:
            List of structured claim dictionaries
        """
        start_time = time.time()
        self.parsing_count += 1
        
        try:
            self.logger.info(f"Starting claim parsing", extra={'session_id': session_id})
            
            # Primary parsing method - structured format
            claims = self._parse_structured_format(raw_extraction, session_id)
            
            # Fallback to alternative parsing if insufficient results
            if len(claims) < self.min_expected_claims:
                self.logger.info("Primary parsing insufficient, trying alternative methods", 
                               extra={'session_id': session_id})
                claims = self._parse_alternative_format(raw_extraction, session_id)
                self.fallback_parses += 1
            
            # JSON format fallback
            if len(claims) < self.min_expected_claims:
                self.logger.info("Alternative parsing failed, trying JSON parsing", 
                               extra={'session_id': session_id})
                claims = self._parse_json_format(raw_extraction, session_id)
                self.fallback_parses += 1
            
            # Final fallback to basic sentence extraction
            if len(claims) < self.min_expected_claims:
                self.logger.info("JSON parsing failed, using basic sentence extraction", 
                               extra={'session_id': session_id})
                claims = self._parse_basic_sentences(raw_extraction, session_id)
                self.fallback_parses += 1
            
            # Validate and clean parsed claims
            claims = self._validate_and_clean_claims(claims, session_id)
            
            # Apply configuration limits
            if len(claims) > self.max_parsed_claims:
                claims = claims[:self.max_parsed_claims]
                self.logger.info(f"Limited claims to maximum: {self.max_parsed_claims}", 
                               extra={'session_id': session_id})
            
            # Assign sequential IDs
            for i, claim in enumerate(claims):
                claim['claim_id'] = i + 1
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_parsing_statistics(processing_time, len(claims), success=True)
            self.successful_parses += 1
            
            self.logger.info(
                f"Successfully parsed {len(claims)} structured claims in {processing_time:.3f}s",
                extra={'session_id': session_id}
            )
            
            return claims

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_parsing_statistics(processing_time, 0, success=False)
            self.error_parses += 1
            
            self.logger.error(f"Claim parsing failed: {str(e)}", extra={'session_id': session_id})
            
            # Return error claim instead of empty list
            return self._create_error_claim(str(e), session_id)

    def _parse_structured_format(self, raw_extraction: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Parse structured LLM output format with enhanced field detection."""
        self.method_stats['structured_format']['attempts'] += 1
        claims = []
        lines = raw_extraction.split('\n')
        current_claim = {}
        
        try:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for claim start markers with enhanced patterns
                if self._is_claim_start(line):
                    # Save previous claim if valid
                    if current_claim and self._is_valid_claim_structure(current_claim):
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
            if current_claim and self._is_valid_claim_structure(current_claim):
                claims.append(current_claim)
            
            if claims:
                self.method_stats['structured_format']['successes'] += 1
            
            return claims

        except Exception as e:
            self.logger.warning(f"Structured format parsing failed: {str(e)}", 
                              extra={'session_id': session_id})
            return []

    def _parse_alternative_format(self, raw_extraction: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Parse alternative formats using comprehensive regex patterns."""
        self.method_stats['alternative_format']['attempts'] += 1
        claims = []
        
        try:
            # Multiple parsing patterns for flexibility
            patterns = [
                # Numbered claims with colon
                r'(?:Claim\s*\d+|^\d+\.)\s*[:\-]?\s*(.+?)(?=(?:Claim\s*\d+|^\d+\.)|$)',
                # Bold formatted claims
                r'\*\*.*?\*\*\s*[:\-]?\s*(.+?)(?=\*\*.*?\*\*|$)',
                # Bullet point claims
                r'-\s*(.+?)(?=\n-|\n\n|$)',
                # Simple line-by-line claims
                r'^([A-Z][^.!?]*[.!?])\s*$'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, raw_extraction, re.MULTILINE | re.DOTALL | re.IGNORECASE)
                
                for match in matches:
                    claim_text = match.strip() if isinstance(match, str) else str(match).strip()
                    
                    if len(claim_text) > self.min_claim_text_length:
                        claim = self.default_claim_structure.copy()
                        claim['text'] = self._clean_claim_text(claim_text)
                        claim['claim_type'] = self._infer_claim_type(claim['text'])
                        claim['verifiability_score'] = self._extract_numeric_value(claim_text, 'verifiability', 5)
                        claim['priority'] = self._extract_numeric_value(claim_text, 'priority', 2)
                        
                        claims.append(claim)
                
                # Use first successful pattern
                if claims:
                    self.method_stats['alternative_format']['successes'] += 1
                    break
            
            return claims

        except Exception as e:
            self.logger.warning(f"Alternative format parsing failed: {str(e)}", 
                              extra={'session_id': session_id})
            return []

    def _parse_json_format(self, raw_extraction: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Parse JSON format output with error recovery."""
        self.method_stats['json_format']['attempts'] += 1
        claims = []
        
        try:
            import json
            
            # Try to find JSON block in the text
            json_patterns = [
                r'\{[\s\S]*\}',  # Basic JSON object
                r'\[[\s\S]*\]',  # JSON array
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, raw_extraction)
                if json_match:
                    json_text = json_match.group(0)
                    try:
                        data = json.loads(json_text)
                        
                        # Handle different JSON structures
                        if isinstance(data, dict):
                            if 'claims' in data:
                                claims_data = data['claims']
                            elif 'assertions' in data:
                                claims_data = data['assertions']
                            elif 'institutional_analysis' in data:
                                claims_data = data['institutional_analysis'].get('assertions', [])
                            else:
                                claims_data = [data]  # Single claim object
                        elif isinstance(data, list):
                            claims_data = data
                        else:
                            continue
                        
                        # Convert JSON claims to standard format
                        for claim_data in claims_data:
                            if isinstance(claim_data, dict):
                                claim = self.default_claim_structure.copy()
                                
                                # Map JSON fields to standard fields
                                claim['text'] = claim_data.get('assertion_text') or claim_data.get('text') or claim_data.get('content', '')
                                claim['claim_type'] = claim_data.get('classification_type') or claim_data.get('type') or claim_data.get('claim_type', 'Other')
                                claim['priority'] = claim_data.get('research_priority') or claim_data.get('priority', 2)
                                claim['verifiability_score'] = claim_data.get('verification_index') or claim_data.get('verifiability') or claim_data.get('verifiability_score', 5)
                                claim['source'] = claim_data.get('source_attribution') or claim_data.get('source', 'Not specified')
                                claim['verification_strategy'] = claim_data.get('research_methodology') or claim_data.get('verification_strategy', 'Standard fact-checking')
                                claim['importance'] = claim_data.get('academic_significance') or claim_data.get('importance', 'Supporting claim')
                                
                                if self._is_valid_claim_structure(claim):
                                    claims.append(claim)
                        
                        if claims:
                            self.method_stats['json_format']['successes'] += 1
                            break
                        
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"JSON decode error: {str(e)}", extra={'session_id': session_id})
                        continue
            
            return claims

        except Exception as e:
            self.logger.warning(f"JSON format parsing failed: {str(e)}", 
                              extra={'session_id': session_id})
            return []

    def _parse_basic_sentences(self, raw_extraction: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Basic fallback parsing using sentence-level claim indicators."""
        self.method_stats['basic_sentences']['attempts'] += 1
        claims = []
        
        try:
            sentences = re.split(r'[.!?]+', raw_extraction)
            
            # Enhanced claim indicator keywords
            claim_indicators = [
                'study', 'research', 'found', 'showed', 'announced', 'said',
                'according to', 'reported', 'confirmed', 'revealed', 'data shows',
                'statistics indicate', 'poll found', 'survey revealed', 'experts claim',
                'percent', 'percentage', 'rate', 'increased', 'decreased', 'rose', 'fell'
            ]

            max_basic_claims = self.config.get('max_basic_claims', 10)
            min_sentence_length = self.config.get('min_sentence_length', 20)

            for sentence in sentences:
                sentence = sentence.strip()
                
                if (len(sentence) > min_sentence_length and
                    any(indicator in sentence.lower() for indicator in claim_indicators)):
                    
                    claim = self.default_claim_structure.copy()
                    claim['text'] = sentence + '.'
                    claim['claim_type'] = self._infer_claim_type(sentence)
                    claim['priority'] = 3  # Lower priority for basic extraction
                    claim['verifiability_score'] = 4  # Conservative score
                    
                    claims.append(claim)
                    
                    if len(claims) >= max_basic_claims:
                        break
            
            if claims:
                self.method_stats['basic_sentences']['successes'] += 1
            
            return claims

        except Exception as e:
            self.logger.warning(f"Basic sentence parsing failed: {str(e)}", 
                              extra={'session_id': session_id})
            return []

    def _is_claim_start(self, line: str) -> bool:
        """Check if line indicates the start of a new claim with enhanced patterns."""
        claim_start_patterns = [
            r'\*\*Claim\s*\d+',        # **Claim 1
            r'Claim\s*\d+:',           # Claim 1:
            r'^\d+\.\s*\*\*',          # 1. **
            r'Priority\s*\d+.*Claim',  # Priority 1 Claim
            r'##\s*Claim\s*\d+',       # ## Claim 1
            r'\*\*\d+\.\s*',           # **1.
            r'Assertion\s*\d+',        # Assertion 1
            r'Research\s*ID\s*\d+'     # Research ID 1
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in claim_start_patterns)

    def _extract_priority_from_line(self, line: str) -> Optional[int]:
        """Extract priority number from claim marker line."""
        priority_patterns = [
            r'Priority\s*(\d+)',
            r'\[Priority[:\s]*(\d+)',
            r'Research\s*Priority[:\s]*(\d+)'
        ]
        
        for pattern in priority_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                priority = int(match.group(1))
                return max(1, min(3, priority))  # Clamp to valid range
        
        return None

    def _parse_field_line(self, line: str, current_claim: Dict[str, Any]) -> None:
        """Parse individual field lines within a claim block with enhanced field detection."""
        
        # Enhanced field mapping
        field_mappings = {
            'text': ['text', 'claim', 'content', 'assertion_text', 'assertion content'],
            'claim_type': ['type', 'classification', 'claim_type', 'classification_type'],
            'priority': ['priority', 'research_priority'],
            'verifiability_score': ['verifiability', 'verification_index', 'verifiability score'],
            'source': ['source', 'source_attribution', 'attribution'],
            'verification_strategy': ['verification strategy', 'research_methodology', 'verification method', 'methodology'],
            'importance': ['importance', 'why important', 'academic_significance', 'significance']
        }
        
        for target_field, field_variants in field_mappings.items():
            if self._line_contains_field(line, field_variants):
                value = self._extract_field_value(line)
                if value:
                    if target_field == 'text':
                        current_claim[target_field] = value.strip('"\'')
                    elif target_field == 'verifiability_score':
                        current_claim[target_field] = self._extract_numeric_value(value, 'verifiability', 5)
                    elif target_field == 'priority':
                        current_claim[target_field] = self._extract_numeric_value(value, 'priority', 2)
                    else:
                        current_claim[target_field] = value
                break

    def _line_contains_field(self, line: str, field_names: List[str]) -> bool:
        """Check if line contains any of the specified field names with enhanced matching."""
        line_lower = line.lower()
        
        for field in field_names:
            patterns = [
                f'**{field}**',
                f'- **{field}**',
                f'{field}:',
                f'- {field}:',
                f'**{field}:**'
            ]
            
            if any(pattern in line_lower for pattern in patterns):
                return True
        
        return False

    def _extract_field_value(self, line: str) -> Optional[str]:
        """Extract value from a field line with enhanced pattern matching."""
        field_value_patterns = [
            r'\*\*[^*]+\*\*:\s*[""]?(.+?)[""]?$',       # **Field**: "Value"
            r'-\s*\*\*[^*]+\*\*:\s*[""]?(.+?)[""]?$',   # - **Field**: "Value"
            r'[^:]+:\s*[""]?(.+?)[""]?$',                # Field: "Value"
            r'\|\s*([^|]+)\s*\|',                        # | Value |
        ]

        for pattern in field_value_patterns:
            match = re.search(pattern, line)
            if match:
                value = match.group(1).strip()
                # Clean up common formatting artifacts
                value = re.sub(r'^["\'`]+|["\'`]+$', '', value)  # Remove quotes
                value = re.sub(r'^\[|\]$', '', value)            # Remove brackets
                return value
        
        return None

    def _extract_numeric_value(self, text: str, field_name: str, default: int = 5) -> int:
        """Extract numeric value from text with enhanced validation."""
        try:
            # Look for number after field name
            patterns = [
                rf'{field_name}[:\s]*(\d+)',
                r'(\d+)/10',       # X/10 format
                r'(\d+)\s*out\s*of\s*10',  # X out of 10
                r'(\d+)'           # Any number
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
                    # Apply appropriate ranges based on field
                    if field_name == 'verifiability':
                        return max(1, min(10, value))
                    elif field_name == 'priority':
                        return max(1, min(3, value))
                    else:
                        return value
            
            return default

        except (ValueError, AttributeError):
            return default

    def _infer_claim_type(self, claim_text: str) -> str:
        """Infer claim type from claim text content with enhanced detection."""
        text_lower = claim_text.lower()
        
        # Enhanced type detection patterns
        type_patterns = {
            'Statistical': ['%', 'percent', 'study', 'research', 'data', 'rate', 'increased', 'decreased', 'statistics', 'survey', 'poll'],
            'Event': ['announced', 'occurred', 'happened', 'will', 'meeting', 'conference', 'launched', 'signed', 'voted'],
            'Attribution': ['said', 'according to', 'spokesperson', 'stated', 'declared', 'claimed', 'confirmed', 'denied'],
            'Research': ['study', 'research', 'scientists', 'published', 'peer-reviewed', 'clinical trial', 'experiment', 'findings'],
            'Policy': ['law', 'policy', 'regulation', 'court', 'ruled', 'legislation', 'bill', 'executive order'],
            'Causal': ['caused by', 'due to', 'because of', 'result of', 'leads to', 'triggered', 'stems from']
        }
        
        # Score each type based on keyword matches
        type_scores = {}
        for claim_type, keywords in type_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[claim_type] = score
        
        # Return type with highest score, or 'Other' if no matches
        return max(type_scores, key=type_scores.get) if type_scores else 'Other'

    def _clean_claim_text(self, raw_text: str) -> str:
        """Extract and clean claim text from raw input with enhanced cleaning."""
        text = raw_text.strip()
        
        # Remove field indicators and formatting
        text = re.sub(r'^\s*-?\s*\*\*[^*]+\*\*:\s*', '', text)      # **Field**:
        text = re.sub(r'^\s*Text:\s*', '', text, flags=re.IGNORECASE)  # Text:
        text = re.sub(r'^\s*Claim:\s*', '', text, flags=re.IGNORECASE)  # Claim:
        text = re.sub(r'^\s*Content:\s*', '', text, flags=re.IGNORECASE)  # Content:
        
        # Clean quotes and brackets
        text = re.sub(r'^["\'`\[\(]+|["\'`\]\)]+$', '', text)
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text.strip()

    def _is_valid_claim_structure(self, claim: Dict[str, Any]) -> bool:
        """Check if claim meets minimum validity requirements with enhanced validation."""
        # Check required fields
        required_fields = ['text', 'claim_type']
        for field in required_fields:
            if field not in claim or not claim[field]:
                return False
        
        # Validate text length
        text = claim.get('text', '').strip()
        if len(text) < self.min_claim_text_length:
            return False
        
        # Validate claim type
        valid_types = ['Statistical', 'Event', 'Attribution', 'Research', 'Policy', 'Causal', 'Other']
        if claim.get('claim_type') not in valid_types:
            return False
        
        # Validate numeric fields
        priority = claim.get('priority', 2)
        if not isinstance(priority, int) or not 1 <= priority <= 3:
            return False
        
        verifiability = claim.get('verifiability_score', 5)
        if not isinstance(verifiability, (int, float)) or not 1 <= verifiability <= 10:
            return False
        
        return True

    def _validate_and_clean_claims(self, claims: List[Dict[str, Any]], session_id: str = None) -> List[Dict[str, Any]]:
        """Validate and clean parsed claims with comprehensive quality checks."""
        validated_claims = []
        
        for claim in claims:
            try:
                # Ensure all required fields exist with defaults
                for key, default_value in self.default_claim_structure.items():
                    if key not in claim or claim[key] is None:
                        claim[key] = default_value

                # Validate and clean text field
                text = claim.get('text', '').strip()
                if len(text) < self.min_claim_text_length:
                    continue  # Skip claims with insufficient text
                
                if len(text) > self.max_claim_text_length:
                    text = text[:self.max_claim_text_length] + '...'
                
                claim['text'] = text

                # Validate and clamp numeric fields
                claim['verifiability_score'] = max(1, min(10, int(claim.get('verifiability_score', 5))))
                claim['priority'] = max(1, min(3, int(claim.get('priority', 2))))

                # Validate claim type
                valid_claim_types = ['Statistical', 'Attribution', 'Event', 'Research', 'Policy', 'Causal', 'Other']
                if claim.get('claim_type') not in valid_claim_types:
                    claim['claim_type'] = 'Other'

                # Ensure string fields are strings
                string_fields = ['source', 'verification_strategy', 'importance']
                for field in string_fields:
                    if not isinstance(claim.get(field), str):
                        claim[field] = str(claim.get(field, ''))

                validated_claims.append(claim)

            except Exception as e:
                self.logger.warning(f"Failed to validate claim: {str(e)}", 
                                  extra={'session_id': session_id})
                continue

        self.logger.info(f"Validated {len(validated_claims)} out of {len(claims)} claims", 
                        extra={'session_id': session_id})
        
        return validated_claims

    def _create_error_claim(self, error_msg: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Create error claim when parsing fails completely."""
        error_claim = self.default_claim_structure.copy()
        error_claim.update({
            'text': f"Error parsing claims: {error_msg}",
            'claim_type': 'Error',
            'priority': 3,
            'verifiability_score': 1,
            'claim_id': 1,
            'source': 'Parser Error',
            'verification_strategy': 'Review parsing logs',
            'importance': 'Error indicator'
        })
        
        return [error_claim]

    def _update_parsing_statistics(self, processing_time: float, claims_found: int, success: bool) -> None:
        """Update parsing performance statistics."""
        self.total_processing_time += processing_time

    # Utility methods for working with parsed claims

    def format_claims_summary(self, claims: List[Dict[str, Any]]) -> str:
        """Format claims into a readable summary with enhanced formatting."""
        if not claims:
            return "No claims extracted from the article."

        summary_lines = [f"EXTRACTED CLAIMS ({len(claims)} total):", ""]

        for claim in claims:
            priority_indicator = "🔴" if claim.get('priority') == 1 else "🟡" if claim.get('priority') == 2 else "🟢"
            
            summary_lines.extend([
                f"Claim {claim.get('claim_id', '?')}: {priority_indicator}",
                f"  Text: {claim.get('text', 'No text')}",
                f"  Type: {claim.get('claim_type', 'Unknown')}",
                f"  Priority: {claim.get('priority', '?')}",
                f"  Verifiability: {claim.get('verifiability_score', '?')}/10",
                f"  Source: {claim.get('source', 'Not specified')}",
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
        """Calculate quality metrics for parsed claims with enhanced assessment."""
        if not claims:
            return {'quality': 'poor', 'score': 0, 'details': 'No claims parsed'}

        # Calculate quality indicators
        has_text = sum(1 for claim in claims if claim.get('text') and len(claim['text']) > 5)
        has_valid_type = sum(1 for claim in claims if claim.get('claim_type') != 'Other')
        has_good_verifiability = sum(1 for claim in claims if claim.get('verifiability_score', 0) > 5)
        has_priority = sum(1 for claim in claims if claim.get('priority', 0) in [1, 2, 3])
        has_source = sum(1 for claim in claims if claim.get('source', 'Not specified') != 'Not specified')

        total_claims = len(claims)

        # Calculate weighted quality score
        quality_score = (
            (has_text / total_claims) * self.text_quality_weight +
            (has_valid_type / total_claims) * self.type_quality_weight +
            (has_good_verifiability / total_claims) * self.verifiability_weight +
            (has_priority / total_claims) * self.priority_weight
        ) * 100

        # Determine quality level with enhanced thresholds
        excellent_threshold = self.config.get('excellent_threshold', 85)
        good_threshold = self.config.get('good_threshold', 70)
        fair_threshold = self.config.get('fair_threshold', 50)

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
            'claims_with_valid_type': has_valid_type,
            'claims_with_good_verifiability': has_good_verifiability,
            'claims_with_priority': has_priority,
            'claims_with_source': has_source,
            'total_claims': total_claims,
            'completeness_ratio': round((has_text + has_valid_type + has_good_verifiability) / (total_claims * 3), 2)
        }

    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive parsing performance statistics."""
        success_rate = (self.successful_parses / self.parsing_count * 100) if self.parsing_count > 0 else 0
        fallback_rate = (self.fallback_parses / self.parsing_count * 100) if self.parsing_count > 0 else 0
        error_rate = (self.error_parses / self.parsing_count * 100) if self.parsing_count > 0 else 0
        avg_processing_time = (self.total_processing_time / self.parsing_count) if self.parsing_count > 0 else 0

        # Calculate method success rates
        method_success_rates = {}
        for method, stats in self.method_stats.items():
            if stats['attempts'] > 0:
                method_success_rates[method] = round((stats['successes'] / stats['attempts']) * 100, 1)
            else:
                method_success_rates[method] = 0.0

        return {
            'total_parses': self.parsing_count,
            'successful_parses': self.successful_parses,
            'fallback_parses': self.fallback_parses,
            'error_parses': self.error_parses,
            'success_rate': round(success_rate, 2),
            'fallback_rate': round(fallback_rate, 2),
            'error_rate': round(error_rate, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'method_success_rates': method_success_rates,
            'method_stats': self.method_stats,
            'configuration_applied': bool(self.config)
        }

    def get_claim_statistics(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed statistics about parsed claims."""
        if not claims:
            return {'total_claims': 0, 'statistics': 'No claims to analyze'}

        # Type distribution
        type_counts = defaultdict(int)
        for claim in claims:
            type_counts[claim.get('claim_type', 'Unknown')] += 1

        # Priority distribution
        priority_counts = defaultdict(int)
        for claim in claims:
            priority_counts[claim.get('priority', 0)] += 1

        # Verifiability distribution
        verifiability_scores = [claim.get('verifiability_score', 0) for claim in claims]
        avg_verifiability = sum(verifiability_scores) / len(verifiability_scores) if verifiability_scores else 0

        # Text length statistics
        text_lengths = [len(claim.get('text', '')) for claim in claims]
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

        return {
            'total_claims': len(claims),
            'type_distribution': dict(type_counts),
            'priority_distribution': dict(priority_counts),
            'average_verifiability': round(avg_verifiability, 2),
            'average_text_length': round(avg_text_length, 1),
            'verifiability_range': {
                'min': min(verifiability_scores) if verifiability_scores else 0,
                'max': max(verifiability_scores) if verifiability_scores else 0
            },
            'text_length_range': {
                'min': min(text_lengths) if text_lengths else 0,
                'max': max(text_lengths) if text_lengths else 0
            }
        }


# Testing functionality
if __name__ == "__main__":
    """Test claim parser functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== CLAIM PARSER TESTING ===")
    
    # Initialize parser with test configuration
    test_config = {
        'min_claim_text_length': 15,
        'max_parsed_claims': 10,
        'excellent_threshold': 80
    }

    parser = ClaimParser(test_config)

    # Test structured format parsing
    structured_test = """
    **Claim 1**: Priority 1
    - **Text**: "Study published in Nature Medicine found 85% of patients showed improvement"
    - **Type**: Research
    - **Verifiability**: 9/10
    - **Source**: Nature Medicine publication

    **Claim 2**: Priority 2
    - **Text**: "Dr. Sarah Johnson announced results at yesterday's conference"
    - **Type**: Attribution
    - **Verifiability**: 7/10
    - **Source**: Conference announcement
    """

    print("--- Structured Format Test ---")
    structured_claims = parser.parse_extracted_claims(structured_test, session_id="test_parser_001")
    print(f"✅ Parsed {len(structured_claims)} claims from structured format")
    for claim in structured_claims:
        print(f"   • {claim['text'][:50]}... (Type: {claim['claim_type']}, Priority: {claim['priority']})")

    # Test JSON format parsing
    json_test = '''
    {
        "institutional_analysis": {
            "assertions": [
                {
                    "research_id": 1,
                    "assertion_text": "Clinical trial involved 1,200 participants across 15 hospitals",
                    "classification_type": "Statistical",
                    "research_priority": 1,
                    "verification_index": 8
                }
            ]
        }
    }
    '''

    print("\n--- JSON Format Test ---")
    json_claims = parser.parse_extracted_claims(json_test, session_id="test_parser_002")
    print(f"✅ Parsed {len(json_claims)} claims from JSON format")
    if json_claims:
        print(f"   • {json_claims[0]['text'][:50]}... (Score: {json_claims[0]['verifiability_score']})")

    # Test alternative format parsing
    alternative_test = """
    1. According to government data, healthcare spending increased by $50 billion this year
    2. The Supreme Court ruled that the new regulation violates constitutional rights
    3. Research indicates 70% improvement in patient outcomes
    """

    print("\n--- Alternative Format Test ---")
    alt_claims = parser.parse_extracted_claims(alternative_test, session_id="test_parser_003")
    print(f"✅ Parsed {len(alt_claims)} claims from alternative format")
    
    # Test basic sentence extraction (fallback)
    basic_test = """
    This article discusses healthcare improvements. A recent study found significant results.
    The research showed promising outcomes. Data indicates positive trends in treatment.
    """

    print("\n--- Basic Extraction Test ---")
    basic_claims = parser.parse_extracted_claims(basic_test, session_id="test_parser_004")
    print(f"✅ Parsed {len(basic_claims)} claims from basic extraction")

    # Test quality assessment
    print("\n--- Quality Assessment Test ---")
    all_claims = structured_claims + json_claims + alt_claims
    quality = parser.calculate_parsing_quality(all_claims)
    print(f"✅ Parsing quality: {quality['quality']} ({quality['score']:.1f}%)")
    print(f"   Claims with text: {quality['claims_with_text']}/{quality['total_claims']}")
    print(f"   Claims with valid type: {quality['claims_with_valid_type']}/{quality['total_claims']}")

    # Test utility methods
    print("\n--- Utility Methods Test ---")
    high_priority = parser.get_claims_by_priority(all_claims, 1)
    verifiable_claims = parser.get_most_verifiable_claims(all_claims, 7)
    statistical_claims = parser.get_claims_by_type(all_claims, 'Statistical')

    print(f"✅ High priority claims: {len(high_priority)}")
    print(f"✅ Highly verifiable claims: {len(verifiable_claims)}")
    print(f"✅ Statistical claims: {len(statistical_claims)}")

    # Test statistics
    print("\n--- Parser Statistics ---")
    parser_stats = parser.get_parsing_statistics()
    print(f"✅ Total parses: {parser_stats['total_parses']}")
    print(f"✅ Success rate: {parser_stats['success_rate']:.1f}%")
    print(f"✅ Average time: {parser_stats['average_processing_time_ms']:.1f}ms")
    print(f"✅ Method success rates: {parser_stats['method_success_rates']}")

    # Test claim statistics
    claim_stats = parser.get_claim_statistics(all_claims)
    print(f"✅ Claim type distribution: {claim_stats['type_distribution']}")
    print(f"✅ Average verifiability: {claim_stats['average_verifiability']}/10")

    # Test summary formatting
    print("\n--- Summary Formatting Test ---")
    summary = parser.format_claims_summary(structured_claims[:2])
    print("✅ Claims summary generated")
    print(f"   Preview: {summary.split(chr(10))[0]}")  # First line

    print("\n🎯 Claim parser tests completed successfully!")
