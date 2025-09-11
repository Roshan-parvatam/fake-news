# utils/helpers.py

"""
Production-Ready Helper Utilities for Fake News Detection

Comprehensive utility functions for secure text processing, metadata extraction,
input validation, and system monitoring with enterprise-grade security practices.

Features:
- Advanced content sanitization with XSS/injection prevention
- Robust input validation with detailed error responses
- Secure metadata extraction with normalization
- High-performance implementations for large content processing
- Integrated logging with structured formatting
- System monitoring and health utilities
- Memory-efficient implementations for production use

Version: 3.2.0 - Enhanced Production Edition
"""

import re
import html
import hashlib
import json
import os
import sys
import time
import logging
import unicodedata
import psutil
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass
import platform

# Configure structured logging
logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for performance
SCRIPT_TAG_RE = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
STYLE_TAG_RE = re.compile(r'<style[^>]*>.*?</style>', re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)

# Event handlers and dangerous attributes
EVENT_HANDLER_RE = re.compile(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', re.IGNORECASE)
DANGEROUS_ATTRS_RE = re.compile(r'\s*(src|href|action)\s*=\s*["\']javascript:[^"\']*["\']', re.IGNORECASE)

# Dangerous URI schemes
DANGEROUS_SCHEMES_RE = re.compile(r'(javascript:|data:|vbscript:|file:|ftp:)[^\s]*', re.IGNORECASE)

# Dangerous HTML tags
DANGEROUS_TAGS = [
    'script', 'style', 'iframe', 'object', 'embed', 'applet', 'form', 
    'input', 'textarea', 'button', 'select', 'option', 'link', 'meta',
    'base', 'frame', 'frameset', 'noframes', 'noscript', 'canvas'
]

# Pre-compile dangerous tag patterns
DANGEROUS_TAGS_RE = {
    tag: re.compile(rf'<{tag}[^>]*>.*?</{tag}>', re.IGNORECASE | re.DOTALL)
    for tag in DANGEROUS_TAGS
}

# Self-closing dangerous tags
DANGEROUS_SELF_CLOSING_RE = re.compile(
    r'<(img|area|base|br|col|embed|hr|input|link|meta|param|source|track|wbr)[^>]*/?>', 
    re.IGNORECASE
)

# Email validation pattern
EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# URL validation pattern  
URL_RE = re.compile(
    r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
    re.IGNORECASE
)


def sanitize_text(text: str, max_length: int = 50000, preserve_structure: bool = False, 
                 strict_mode: bool = True) -> str:
    """
    Enhanced text sanitization with comprehensive security measures.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        preserve_structure: Whether to preserve basic formatting
        strict_mode: Enable strict security filtering
        
    Returns:
        Sanitized and safe text string
        
    Raises:
        ValueError: If input is invalid type
    """
    start_time = time.time()
    
    if not isinstance(text, str):
        if text is None:
            return ""
        raise ValueError(f"Expected string, got {type(text).__name__}")
    
    if not text.strip():
        return ""
    
    original_length = len(text)
    
    # Step 1: Length validation and truncation
    if len(text) > max_length:
        logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]
    
    # Step 2: Basic cleanup
    text = text.strip()
    
    # Step 3: HTML entity decoding (but be careful of double-decoding attacks)
    previous_text = ""
    decode_count = 0
    while text != previous_text and decode_count < 3:  # Prevent infinite loops
        previous_text = text
        text = html.unescape(text)
        decode_count += 1
    
    # Step 4: Remove comments first (can hide malicious content)
    text = COMMENT_RE.sub('', text)
    
    # Step 5: Remove script and style tags with content
    text = SCRIPT_TAG_RE.sub('', text)
    text = STYLE_TAG_RE.sub('', text)
    
    # Step 6: Remove dangerous tags with content
    for tag_pattern in DANGEROUS_TAGS_RE.values():
        text = tag_pattern.sub('', text)
    
    # Step 7: Remove self-closing dangerous tags
    text = DANGEROUS_SELF_CLOSING_RE.sub('', text)
    
    # Step 8: Remove event handlers and dangerous attributes
    text = EVENT_HANDLER_RE.sub('', text)
    text = DANGEROUS_ATTRS_RE.sub('', text)
    
    # Step 9: Remove dangerous URI schemes
    text = DANGEROUS_SCHEMES_RE.sub('', text)
    
    # Step 10: Handle remaining HTML tags
    if preserve_structure:
        # Convert basic formatting to text equivalents
        text = re.sub(r'<\s*br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*p\s*[^>]*>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</\s*p\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*div\s*[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</\s*div\s*>', '', text, flags=re.IGNORECASE)
        
        # Remove all remaining tags
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
    
    # Step 11: Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)   # Multiple spaces to single
    text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading whitespace on lines
    text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing whitespace on lines
    
    # Step 12: Remove control characters and normalize Unicode
    if strict_mode:
        # Remove all control characters except newlines and tabs
        text = ''.join(char for char in text 
                      if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Normalize Unicode to prevent homograph attacks
        text = unicodedata.normalize('NFKC', text)
    
    # Step 13: Final cleanup
    text = text.strip()
    
    # Log sanitization metrics
    processing_time = time.time() - start_time
    size_reduction = ((original_length - len(text)) / original_length * 100) if original_length > 0 else 0
    
    logger.debug(
        f"Text sanitized - Original: {original_length} chars, Final: {len(text)} chars, "
        f"Reduction: {size_reduction:.1f}%, Time: {processing_time:.3f}s"
    )
    
    return text


def validate_input_data(data: Dict[str, Any], required_fields: List[str], 
                       max_lengths: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Comprehensive input data validation with detailed error reporting.
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
        max_lengths: Optional maximum lengths for string fields
        
    Returns:
        Validation result dictionary
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'sanitized_data': {},
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    if not isinstance(data, dict):
        validation_result['valid'] = False
        validation_result['errors'].append('Input data must be a dictionary')
        return validation_result
    
    max_lengths = max_lengths or {}
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Required field missing: {field}')
        elif not data[field] or (isinstance(data[field], str) and not data[field].strip()):
            validation_result['valid'] = False
            validation_result['errors'].append(f'Required field empty: {field}')
    
    # Validate and sanitize each field
    for field, value in data.items():
        try:
            if isinstance(value, str):
                # Sanitize string fields
                max_len = max_lengths.get(field, 10000)
                sanitized_value = sanitize_text(value, max_length=max_len)
                validation_result['sanitized_data'][field] = sanitized_value
                
                # Check length after sanitization
                if len(sanitized_value) != len(value):
                    validation_result['warnings'].append(
                        f'Field "{field}" was sanitized (content modified)'
                    )
                
            elif isinstance(value, (int, float, bool)):
                validation_result['sanitized_data'][field] = value
                
            elif isinstance(value, (list, dict)):
                validation_result['sanitized_data'][field] = value
                validation_result['warnings'].append(
                    f'Field "{field}" contains complex data - manual validation recommended'
                )
                
            else:
                validation_result['sanitized_data'][field] = str(value)
                validation_result['warnings'].append(
                    f'Field "{field}" converted to string from {type(value).__name__}'
                )
                
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Validation error for field "{field}": {str(e)}')
    
    return validation_result


def extract_metadata(text: str, url: str = "") -> Dict[str, Any]:
    """
    Extract comprehensive metadata from text content.
    
    Args:
        text: Input text
        url: Optional source URL
        
    Returns:
        Dictionary containing extracted metadata
    """
    if not text or not isinstance(text, str):
        return {'error': 'Invalid or empty text input'}
    
    start_time = time.time()
    
    metadata = {
        'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
        'source_url': url,
        'content_stats': {},
        'linguistic_features': {},
        'quality_indicators': {},
        'processing_time': 0.0
    }
    
    # Basic content statistics
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    metadata['content_stats'] = {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'paragraph_count': len(paragraphs),
        'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'average_sentence_length': len(words) / len(sentences) if sentences else 0
    }
    
    # Linguistic features
    if words:
        uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        capitalized_words = sum(1 for word in words if word.istitle())
        
        metadata['linguistic_features'] = {
            'uppercase_ratio': uppercase_words / len(words),
            'capitalized_ratio': capitalized_words / len(words),
            'punctuation_density': sum(1 for char in text if char in '.,;:!?') / len(text),
            'numeric_content_ratio': sum(1 for char in text if char.isdigit()) / len(text),
            'contains_urls': bool(re.search(r'https?://\S+', text)),
            'contains_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        }
    
    # Quality indicators
    quality_score = 0.0
    quality_factors = []
    
    # Length quality
    if 500 <= len(text) <= 10000:
        quality_score += 0.3
        quality_factors.append('appropriate_length')
    elif 100 <= len(text) < 500:
        quality_score += 0.1
        quality_factors.append('short_length')
    
    # Structure quality
    if len(paragraphs) >= 2:
        quality_score += 0.2
        quality_factors.append('good_structure')
    
    # Sentence variety
    if metadata['content_stats']['sentence_count'] >= 3:
        quality_score += 0.2
        quality_factors.append('sentence_variety')
    
    # Language quality
    if metadata['linguistic_features'].get('uppercase_ratio', 0) < 0.1:
        quality_score += 0.1
        quality_factors.append('appropriate_capitalization')
    
    # Coherence indicators
    if re.search(r'\b(however|therefore|moreover|furthermore|consequently|meanwhile)\b', text.lower()):
        quality_score += 0.1
        quality_factors.append('logical_connectors')
    
    # Professional language indicators
    if re.search(r'\b(according to|research|study|analysis|report|data|evidence)\b', text.lower()):
        quality_score += 0.1
        quality_factors.append('professional_language')
    
    metadata['quality_indicators'] = {
        'overall_quality_score': min(quality_score, 1.0),
        'quality_factors': quality_factors,
        'readability_estimate': _estimate_readability(text),
        'content_type_prediction': _predict_content_type(text)
    }
    
    # Domain extraction if URL provided
    if url:
        domain = extract_domain(url)
        if domain:
            metadata['source_domain'] = domain
    
    # Processing time
    metadata['processing_time'] = time.time() - start_time
    
    return metadata


def _estimate_readability(text: str) -> str:
    """Simple readability estimation based on sentence and word length."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    if not words or not sentences:
        return 'unknown'
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simple readability classification
    if avg_sentence_length <= 15 and avg_word_length <= 5:
        return 'easy'
    elif avg_sentence_length <= 25 and avg_word_length <= 6:
        return 'moderate'
    else:
        return 'difficult'


def _predict_content_type(text: str) -> str:
    """Predict content type based on linguistic patterns."""
    text_lower = text.lower()
    
    # News article indicators
    news_indicators = ['reported', 'according to', 'breaking', 'update', 'announced', 'confirmed']
    news_score = sum(1 for indicator in news_indicators if indicator in text_lower)
    
    # Opinion piece indicators
    opinion_indicators = ['i believe', 'in my opinion', 'i think', 'personally', 'should', 'must']
    opinion_score = sum(1 for indicator in opinion_indicators if indicator in text_lower)
    
    # Academic indicators
    academic_indicators = ['research', 'study', 'analysis', 'methodology', 'findings', 'conclusion']
    academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)
    
    # Blog/informal indicators
    informal_indicators = ['hey', 'guys', 'awesome', 'totally', 'check out', 'click here']
    informal_score = sum(1 for indicator in informal_indicators if indicator in text_lower)
    
    scores = {
        'news': news_score,
        'opinion': opinion_score,
        'academic': academic_score,
        'blog': informal_score
    }
    
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'


def generate_content_hash(content: str, algorithm: str = 'sha256') -> str:
    """
    Generate cryptographic hash of content for deduplication.
    
    Args:
        content: Input content
        algorithm: Hash algorithm (sha256, sha1, md5)
        
    Returns:
        Hexadecimal hash string
    """
    if not content:
        return ''
    
    try:
        # Normalize content for consistent hashing
        normalized_content = sanitize_text(content, preserve_structure=False)
        normalized_content = ' '.join(normalized_content.split())  # Normalize whitespace
        
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(normalized_content.encode('utf-8'))
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.error(f"Hash generation failed: {str(e)}")
        return ''


def clean_filename(filename: str, max_length: int = 255) -> str:
    """
    Clean filename to be filesystem-safe.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Clean, filesystem-safe filename
    """
    if not filename:
        return 'unnamed_file'
    
    # Remove dangerous characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32)
    
    # Normalize spaces and remove excessive underscores
    cleaned = re.sub(r'[\s_]+', '_', cleaned)
    
    # Remove leading/trailing dots and underscores
    cleaned = cleaned.strip('._')
    
    # Ensure length limit
    if len(cleaned) > max_length:
        name_part = cleaned[:max_length-10]  # Leave room for extension
        extension = cleaned[max_length-10:]
        cleaned = name_part + extension
    
    return cleaned or 'unnamed_file'


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for monitoring.
    
    Returns:
        Dictionary containing system metrics
    """
    try:
        # Get memory info
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get Python info
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system': {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'hostname': platform.node(),
                'python_version': python_version
            },
            'resources': {
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent_used': memory.percent,
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'disk_percent_used': round((disk.used / disk.total) * 100, 1),
                'cpu_count': cpu_count,
                'cpu_percent': cpu_percent
            },
            'process': {
                'pid': os.getpid(),
                'memory_mb': round(psutil.Process().memory_info().rss / (1024**2), 1),
                'cpu_percent': psutil.Process().cpu_percent()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(e),
            'basic_info': {
                'platform': platform.system(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        }


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable format.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.2 MB")
    """
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024**2:
        return f"{bytes_value/1024:.1f} KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value/1024**2:.1f} MB"
    else:
        return f"{bytes_value/1024**3:.1f} GB"


def safe_json_parse(json_str: str) -> Optional[Any]:
    """
    Safely parse JSON string with error handling.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed object or None if parsing fails
    """
    if not json_str or not isinstance(json_str, str):
        return None
    
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parsing failed: {str(e)}")
        return None


def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Input text
        max_length: Maximum length including suffix
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated text
    """
    if not text or not isinstance(text, str):
        return ''
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].rstrip() + suffix


def is_valid_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email format is valid
    """
    if not email or not isinstance(email, str):
        return False
    
    return bool(EMAIL_RE.match(email.strip()))


def is_valid_url(url: str) -> bool:
    """
    Validate URL format with security checks.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid and safe
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    
    # Basic format check
    if not URL_RE.match(url):
        return False
    
    try:
        parsed = urlparse(url)
        
        # Security checks
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            return False
        
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Check for suspicious patterns
        if any(dangerous in url.lower() for dangerous in ['javascript:', 'data:', 'file:']):
            return False
        
        return True
        
    except Exception:
        return False


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL safely.
    
    Args:
        url: URL string
        
    Returns:
        Domain name or None if extraction fails
    """
    if not is_valid_url(url):
        return None
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain
        
    except Exception:
        return None


def create_error_response(message: str, code: str = "ERROR", 
                         details: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        message: Error message
        code: Error code
        details: Optional additional details
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        'success': False,
        'error': {
            'message': message,
            'code': code,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    }
    
    if details is not None:
        response['error']['details'] = details
    
    return response


# Export all public functions
__all__ = [
    'sanitize_text',
    'validate_input_data', 
    'extract_metadata',
    'generate_content_hash',
    'clean_filename',
    'get_system_info',
    'format_bytes',
    'safe_json_parse',
    'truncate_text',
    'is_valid_email',
    'is_valid_url',
    'extract_domain',
    'create_error_response'
]

# Log successful module initialization
logger.debug(f"Helper utilities module loaded successfully - {len(__all__)} functions available")
