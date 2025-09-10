"""
Production-Ready Helper Utilities for Fake News Detection

Comprehensive utility functions for text processing, metadata extraction,
input sanitization, and system utilities with security best practices.

Features:
- Advanced text sanitization with security considerations
- Comprehensive metadata extraction and normalization
- Input validation and safety checks
- System monitoring utilities
- Error handling helpers
- Performance optimization tools
"""

import re
import time
import html
import hashlib
import json
import os
import sys
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
import logging
from urllib.parse import urlparse
import unicodedata

logger = logging.getLogger(__name__)


def sanitize_text(text: str, max_length: int = 50000, preserve_structure: bool = False) -> str:
    """
    Enhanced text sanitization for safe processing with security considerations.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        preserve_structure: Whether to preserve basic formatting
        
    Returns:
        Sanitized and safe text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Basic cleaning
    text = text.strip()
    
    # Step 2: HTML entity decoding
    text = html.unescape(text)
    
    # Step 3: Remove potentially dangerous elements
    # Remove script tags and content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove style tags and content
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove other potentially dangerous tags
    dangerous_tags = ['iframe', 'object', 'embed', 'applet', 'form']
    for tag in dangerous_tags:
        text = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove event handlers
    text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    
    # Remove suspicious URLs
    text = re.sub(r'(javascript:|data:|vbscript:|file:)[^\s]*', '', text, flags=re.IGNORECASE)
    
    # Step 4: Handle HTML tags based on preserve_structure flag
    if preserve_structure:
        # Convert basic HTML to plain text equivalents
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<p[^>]*>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<h[1-6][^>]*>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</h[1-6]>', '\n', text, flags=re.IGNORECASE)
        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', '', text)
    
    # Step 5: Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Step 6: Character filtering - keep only safe characters
    # Allow alphanumeric, common punctuation, and whitespace
    text = re.sub(r'[^\w\s\-.,!?;:()\'"/@#$%&*+=<>[\]{}|\\`~]', '', text, flags=re.UNICODE)
    
    # Step 7: Whitespace normalization
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    
    # Step 8: Length control
    text = text[:max_length]
    
    # Step 9: Final cleanup
    text = text.strip()
    
    return text


def extract_metadata(article_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from article data with validation and normalization.
    
    Args:
        article_data: Dictionary containing raw article data
        
    Returns:
        Dictionary with extracted, validated, and normalized metadata
    """
    metadata = {}
    
    # Extract basic information with fallbacks
    metadata['title'] = _extract_title(article_data)
    metadata['author'] = _extract_author(article_data)
    metadata['date'] = _extract_date(article_data)
    metadata['source'] = _extract_source(article_data)
    metadata['url'] = _extract_url(article_data)
    metadata['subject'] = _extract_subject(article_data)
    
    # Calculate content metrics
    content = article_data.get('text', '') or article_data.get('content', '')
    metadata['content_metrics'] = _calculate_content_metrics(content)
    
    # Add processing metadata
    metadata['processing_info'] = {
        'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
        'extractor_version': '3.2.0'
    }
    
    return metadata


def _extract_title(article_data: Dict[str, Any]) -> str:
    """Extract and validate article title"""
    title_fields = ['title', 'headline', 'subject', 'name']
    
    for field in title_fields:
        if field in article_data and article_data[field]:
            title = str(article_data[field]).strip()
            if 5 <= len(title) <= 300:  # Reasonable title length
                return title
    
    return 'Untitled'


def _extract_author(article_data: Dict[str, Any]) -> str:
    """Extract and normalize author information"""
    author_fields = ['author', 'authors', 'byline', 'writer', 'creator']
    
    for field in author_fields:
        if field in article_data and article_data[field]:
            author = article_data[field]
            
            if isinstance(author, list):
                # Handle list of authors
                authors = [str(a).strip() for a in author if a]
                return ', '.join(authors[:3])  # Max 3 authors
            elif isinstance(author, str):
                author = author.strip()
                # Remove "By" prefix if present
                author = re.sub(r'^by\s+', '', author, flags=re.IGNORECASE)
                if author and len(author) <= 100:
                    return author
    
    return 'Unknown'


def _extract_date(article_data: Dict[str, Any]) -> str:
    """Extract and validate publication date"""
    date_fields = ['publish_date', 'date', 'published', 'created_date', 'timestamp']
    
    for field in date_fields:
        if field in article_data and article_data[field]:
            date_value = article_data[field]
            
            # Try to parse and validate the date
            parsed_date = _parse_date(date_value)
            if parsed_date:
                return parsed_date.isoformat()
    
    return 'Unknown'


def _extract_source(article_data: Dict[str, Any]) -> str:
    """Extract source information"""
    source_fields = ['source', 'publication', 'site_name', 'domain']
    
    for field in source_fields:
        if field in article_data and article_data[field]:
            source = str(article_data[field]).strip()
            if source and len(source) <= 100:
                return source
    
    # Fallback: extract from URL if available
    url = article_data.get('url', '')
    if url:
        try:
            domain = urlparse(url).netloc
            # Remove www. prefix
            domain = domain[4:] if domain.startswith('www.') else domain
            return domain
        except:
            pass
    
    return 'Unknown'


def _extract_url(article_data: Dict[str, Any]) -> str:
    """Extract and validate URL"""
    url_fields = ['url', 'link', 'uri', 'permalink']
    
    for field in url_fields:
        if field in article_data and article_data[field]:
            url = str(article_data[field]).strip()
            if _validate_url(url):
                return url
    
    return ''


def _extract_subject(article_data: Dict[str, Any]) -> str:
    """Extract article subject/category"""
    subject_fields = ['subject', 'category', 'topic', 'section', 'genre']
    
    for field in subject_fields:
        if field in article_data and article_data[field]:
            subject = str(article_data[field]).strip()
            if subject and len(subject) <= 50:
                return subject
    
    return 'General'


def _calculate_content_metrics(content: str) -> Dict[str, Any]:
    """Calculate comprehensive content metrics"""
    if not content:
        return {
            'word_count': 0,
            'character_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'reading_time_minutes': 0
        }
    
    words = content.split()
    sentences = re.split(r'[.!?]+', content)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 20]
    
    return {
        'word_count': len(words),
        'character_count': len(content),
        'character_count_no_spaces': len(content.replace(' ', '')),
        'sentence_count': len(valid_sentences),
        'paragraph_count': len(paragraphs),
        'average_words_per_sentence': round(len(words) / max(len(valid_sentences), 1), 1),
        'reading_time_minutes': round(len(words) / 200, 1)  # ~200 WPM average
    }


def _parse_date(date_value: Any) -> Optional[datetime]:
    """Parse various date formats safely"""
    if not date_value:
        return None
    
    date_str = str(date_value).strip()
    
    # Common date patterns
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',   # SQL datetime
        r'\d{4}-\d{2}-\d{2}',                      # Date only
        r'\d{1,2}/\d{1,2}/\d{4}',                  # US format
    ]
    
    try:
        # Try dateutil parser if available
        import dateutil.parser
        return dateutil.parser.parse(date_str)
    except ImportError:
        # Fallback to basic parsing
        for pattern in date_patterns:
            if re.match(pattern, date_str):
                try:
                    if 'T' in date_str:
                        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        return datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')
                except:
                    continue
    except:
        pass
    
    return None


def _validate_url(url: str) -> bool:
    """Validate URL format and safety"""
    if not url or not isinstance(url, str):
        return False
    
    try:
        parsed = urlparse(url)
        return all([
            parsed.scheme in ['http', 'https'],
            parsed.netloc,
            '.' in parsed.netloc,
            len(parsed.netloc) > 3,
            not parsed.netloc.startswith('.'),
            # Security check - avoid localhost and internal IPs
            not any(blocked in parsed.netloc.lower() 
                   for blocked in ['localhost', '127.0.0.1', '0.0.0.0', '192.168.', '10.0.'])
        ])
    except:
        return False


def generate_content_hash(content: str, algorithm: str = 'sha256') -> str:
    """
    Generate hash of content for deduplication and caching.
    
    Args:
        content: Content to hash
        algorithm: Hashing algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string
    """
    if not content:
        return ''
    
    content_bytes = content.encode('utf-8')
    
    if algorithm == 'md5':
        return hashlib.md5(content_bytes).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(content_bytes).hexdigest()
    else:  # Default to SHA256
        return hashlib.sha256(content_bytes).hexdigest()


def validate_input_data(data: Dict[str, Any], required_fields: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate input data structure and required fields.
    
    Args:
        data: Input data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if required_fields is None:
        required_fields = ['text']
    
    errors = []
    
    # Basic validation
    if not isinstance(data, dict):
        errors.append("Input must be a dictionary")
        return False, errors
    
    if not data:
        errors.append("Input data cannot be empty")
        return False, errors
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not data[field]:
            errors.append(f"Field '{field}' cannot be empty")
    
    # Validate text content if present
    if 'text' in data:
        text = data['text']
        if not isinstance(text, str):
            errors.append("Text field must be a string")
        elif len(text.strip()) < 10:
            errors.append("Text content too short (minimum 10 characters)")
        elif len(text) > 100000:
            errors.append("Text content too long (maximum 100,000 characters)")
    
    return len(errors) == 0, errors


def clean_filename(filename: str, max_length: int = 255) -> str:
    """
    Clean filename for safe filesystem usage.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Cleaned, safe filename
    """
    if not filename:
        return 'untitled'
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure reasonable length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename or 'untitled'


def get_system_info() -> Dict[str, Any]:
    """
    Get basic system information for monitoring and debugging.
    
    Returns:
        Dictionary with system information
    """
    info = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'python_version': sys.version,
        'platform': sys.platform,
    }
    
    try:
        import psutil
        info.update({
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        })
    except ImportError:
        info['note'] = 'psutil not available - limited system info'
    
    return info


def format_bytes(byte_count: int) -> str:
    """
    Format byte count into human-readable string.
    
    Args:
        byte_count: Number of bytes
        
    Returns:
        Human-readable byte string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if byte_count < 1024.0:
            return f"{byte_count:.1f} {unit}"
        byte_count /= 1024.0
    return f"{byte_count:.1f} PB"


def safe_json_parse(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback.
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON data or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Safely truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text with suffix if needed
    """
    if not text or len(text) <= max_length:
        return text
    
    truncated_length = max_length - len(suffix)
    return text[:truncated_length] + suffix


def is_valid_email(email: str) -> bool:
    """
    Basic email validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email format appears valid
    """
    if not email or not isinstance(email, str):
        return False
    
    # Basic regex pattern for email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def extract_domain_from_email(email: str) -> str:
    """
    Extract domain from email address.
    
    Args:
        email: Email address
        
    Returns:
        Domain part of email or empty string if invalid
    """
    if not is_valid_email(email):
        return ''
    
    try:
        return email.split('@')[1].lower()
    except:
        return ''


# Performance and monitoring utilities
def measure_execution_time(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def create_error_response(error_message: str, error_code: str = "PROCESSING_ERROR", 
                         additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error_message: Human-readable error message
        error_code: Machine-readable error code
        additional_info: Additional error context
        
    Returns:
        Standardized error response dictionary
    """
    error_response = {
        'success': False,
        'error': {
            'message': error_message,
            'code': error_code,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    }
    
    if additional_info:
        error_response['error']['details'] = additional_info
    
    return error_response


# Example usage and testing
if __name__ == "__main__":
    # Test text sanitization
    test_text = '<script>alert("xss")</script><p>This is a <b>test</b> article.</p>'
    cleaned = sanitize_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test metadata extraction
    test_article = {
        'title': 'Test Article',
        'author': 'John Doe',
        'text': 'This is test content with multiple sentences. It has enough content for testing.',
        'url': 'https://example.com/article'
    }
    
    metadata = extract_metadata(test_article)
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    
    # Test system info
    sys_info = get_system_info()
    print(f"System Info: {json.dumps(sys_info, indent=2)}")
