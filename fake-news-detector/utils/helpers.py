"""
helpers.py - Part of Fake News Detection System
"""

# utils/helpers.py (add these functions)

import re
from typing import Dict, Any
import html

def sanitize_text(text: str) -> str:
    """Sanitize text input for LLM processing"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove potentially problematic characters
    text = re.sub(r'[^\w\s\-.,!?;:()\'"/@#$%&*+=<>[\]{}|\\`~]', '', text)
    
    return text

def extract_metadata(article_data: Dict) -> Dict[str, Any]:
    """Extract and normalize metadata from article data"""
    metadata = {
        'source': article_data.get('source', 'Unknown'),
        'date': article_data.get('publish_date', article_data.get('date', 'Unknown')),
        'author': article_data.get('author', 'Unknown'),
        'subject': article_data.get('subject', article_data.get('category', 'General')),
        'url': article_data.get('url', ''),
        'word_count': len(article_data.get('text', '').split())
    }
    
    return metadata


