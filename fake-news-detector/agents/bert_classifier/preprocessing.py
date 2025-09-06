# agents/bert_classifier/preprocessing.py
"""
Text Preprocessing Utilities for BERT Classifier - Config Enhanced

Enhanced text preprocessing with configuration support and better metrics.
"""

import re
import time
from typing import Dict, Any, Optional
import logging

class TextPreprocessor:
    """
    📝 ENHANCED TEXT PREPROCESSING WITH CONFIG SUPPORT
    
    Configurable text preprocessing pipeline optimized for BERT input
    with comprehensive metrics and logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        # ✅ USE CONFIG PARAMETERS WITH DEFAULTS
        self.config = config or {}
        
        self.max_length = self.config.get('max_length', 2000)
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', True)
        self.normalize_quotes = self.config.get('normalize_quotes', True)
        self.remove_excessive_punctuation = self.config.get('remove_excessive_punctuation', True)
        self.handle_special_characters = self.config.get('handle_special_characters', True)
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.stats = {
            'total_texts_processed': 0,
            'total_characters_processed': 0,
            'total_urls_removed': 0,
            'total_emails_removed': 0,
            'average_processing_time': 0.0,
            'config_applied': True
        }
        
        self.logger.info(f"✅ TextPreprocessor initialized with config: max_length={self.max_length}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Main preprocessing pipeline with config-driven processing
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and processed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        start_time = time.time()
        original_length = len(text)
        
        try:
            # Apply preprocessing steps based on config
            processed_text = text
            
            # Step 1: Normalize whitespace
            processed_text = self._normalize_whitespace(processed_text)
            
            # Step 2: Remove URLs (if enabled)
            if self.remove_urls:
                processed_text, urls_removed = self._remove_urls(processed_text)
                self.stats['total_urls_removed'] += urls_removed
            
            # Step 3: Remove emails (if enabled)
            if self.remove_emails:
                processed_text, emails_removed = self._remove_emails(processed_text)
                self.stats['total_emails_removed'] += emails_removed
            
            # Step 4: Handle special characters (if enabled)
            if self.handle_special_characters:
                processed_text = self._handle_special_characters(processed_text)
            
            # Step 5: Normalize quotes (if enabled)
            if self.normalize_quotes:
                processed_text = self._normalize_quotes(processed_text)
            
            # Step 6: Remove excessive punctuation (if enabled)
            if self.remove_excessive_punctuation:
                processed_text = self._remove_excessive_punctuation(processed_text)
            
            # Step 7: Limit length
            processed_text = self._limit_length(processed_text, self.max_length)
            
            # Step 8: Final cleanup
            processed_text = self._final_cleanup(processed_text)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(original_length, processing_time)
            
            return processed_text
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {str(e)}")
            return text  # Return original text on error
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        return re.sub(r'\s+', ' ', text.strip())
    
    def _remove_urls(self, text: str) -> tuple:
        """Remove URLs from text"""
        url_pattern = r'https?://[^\s]+|www\.[^\s]+|[^\s]+\.(com|org|net|edu|gov)[^\s]*'
        urls_found = len(re.findall(url_pattern, text, re.IGNORECASE))
        cleaned_text = re.sub(url_pattern, ' ', text, flags=re.IGNORECASE)
        return cleaned_text, urls_found
    
    def _remove_emails(self, text: str) -> tuple:
        """Remove email addresses from text"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails_found = len(re.findall(email_pattern, text))
        cleaned_text = re.sub(email_pattern, ' ', text)
        return cleaned_text, emails_found
    
    def _handle_special_characters(self, text: str) -> str:
        """Keep only alphanumeric and basic punctuation"""
        return re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize different types of quotes"""
        text = re.sub(r'[""„‚]', '"', text)  # Smart quotes to regular quotes
        text = re.sub(r'[''`]', "'", text)   # Smart apostrophes to regular
        return text
    
    def _remove_excessive_punctuation(self, text: str) -> str:
        """Remove excessive punctuation"""
        text = re.sub(r'[!]{3,}', '!!', text)    # Multiple exclamations
        text = re.sub(r'[?]{3,}', '??', text)    # Multiple questions
        text = re.sub(r'[.]{3,}', '...', text)   # Multiple periods
        text = re.sub(r'[,]{2,}', ',', text)     # Multiple commas
        return text
    
    def _limit_length(self, text: str, max_length: int) -> str:
        """Limit text length with smart truncation"""
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length-3]  # Leave room for "..."
        
        # Try to cut at last sentence
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # If period is reasonably close to end
            return truncated[:last_period+1] + "..."
        else:
            return truncated + "..."
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass"""
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces again
        text = text.strip()
        return text
    
    def _update_stats(self, original_length: int, processing_time: float):
        """Update processing statistics"""
        self.stats['total_texts_processed'] += 1
        self.stats['total_characters_processed'] += original_length
        
        # Update average processing time
        total_processed = self.stats['total_texts_processed']
        current_avg = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg * (total_processed - 1) + processing_time) / total_processed
        )
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get preprocessing statistics with config info
        
        Returns:
            Dictionary with comprehensive preprocessing statistics
        """
        return {
            **self.stats,
            'config': {
                'max_length': self.max_length,
                'remove_urls': self.remove_urls,
                'remove_emails': self.remove_emails,
                'normalize_quotes': self.normalize_quotes,
                'remove_excessive_punctuation': self.remove_excessive_punctuation,
                'handle_special_characters': self.handle_special_characters
            },
            'preprocessor_version': '2.0_config_enhanced'
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_texts_processed': 0,
            'total_characters_processed': 0,
            'total_urls_removed': 0,
            'total_emails_removed': 0,
            'average_processing_time': 0.0,
            'config_applied': True
        }
        self.logger.info("📊 Preprocessing statistics reset")

# Testing
if __name__ == "__main__":
    """Test preprocessor with config"""
    test_config = {
        'max_length': 1000,
        'remove_urls': True,
        'remove_emails': True
    }
    
    preprocessor = TextPreprocessor(test_config)
    
    test_text = """
    This is a test article with URLs like https://example.com and 
    emails like test@example.com. It has lots    of    spaces!!!
    """
    
    result = preprocessor.preprocess_text(test_text)
    stats = preprocessor.get_preprocessing_stats()
    
    print(f"Original: {test_text}")
    print(f"Processed: {result}")
    print(f"Stats: {stats}")
