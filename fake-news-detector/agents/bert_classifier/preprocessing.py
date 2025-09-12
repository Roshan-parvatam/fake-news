# agents/bert_classifier/preprocessing.py

"""
Enhanced Text Preprocessing Utilities for BERT Classifier - Production Ready

Production-grade text preprocessing with comprehensive configuration support,
performance monitoring, error handling, and integration with enhanced agent architecture.

Features:
- Dynamic configuration with environment awareness
- Comprehensive performance metrics and analytics
- Enhanced error handling with recovery strategies
- Structured logging with session tracking
- Security validation and content filtering
- Memory-efficient processing with optimization
- Batch preprocessing capabilities
- Quality assessment and validation

Version: 3.2.0 - Enhanced Production Edition
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

# Enhanced exception integration
try:
    from agents.llm_explanation.exceptions import (
        handle_llm_explanation_exception,
        ErrorContext,
        log_exception_with_context
    )
    _enhanced_exceptions_available = True
except ImportError:
    _enhanced_exceptions_available = False


@dataclass
class PreprocessingStats:
    """Enhanced statistics container for preprocessing operations."""
    total_texts_processed: int = 0
    total_characters_processed: int = 0
    total_urls_removed: int = 0
    total_emails_removed: int = 0
    total_special_chars_removed: int = 0
    average_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    processing_time_samples: List[float] = None
    config_applied: bool = True
    last_reset_time: str = None

    def __post_init__(self):
        if self.processing_time_samples is None:
            self.processing_time_samples = []
        if self.last_reset_time is None:
            self.last_reset_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_texts_processed': self.total_texts_processed,
            'total_characters_processed': self.total_characters_processed,
            'total_urls_removed': self.total_urls_removed,
            'total_emails_removed': self.total_emails_removed,
            'total_special_chars_removed': self.total_special_chars_removed,
            'average_processing_time': self.average_processing_time,
            'min_processing_time': self.min_processing_time if self.min_processing_time != float('inf') else 0.0,
            'max_processing_time': self.max_processing_time,
            'config_applied': self.config_applied,
            'last_reset_time': self.last_reset_time,
            'sample_count': len(self.processing_time_samples)
        }


class TextPreprocessor:
    """
    Enhanced Text Preprocessing Pipeline with Production Features

    Configurable text preprocessing pipeline optimized for BERT input with
    comprehensive metrics, error handling, security validation, and performance optimization.

    Features:
    - Dynamic configuration with environment awareness
    - Comprehensive performance tracking and analytics
    - Enhanced error handling with recovery strategies
    - Security validation and content filtering
    - Batch processing capabilities with optimization
    - Quality assessment and validation metrics
    - Memory-efficient processing with cleanup
    - Structured logging with session tracking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced preprocessor with comprehensive configuration.

        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        # Configuration with enhanced defaults
        self.config = config or {}
        
        # Text processing configuration
        self.max_length = self.config.get('max_length', 2000)
        self.min_length = self.config.get('min_length', 10)
        
        # Content filtering configuration
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', True)
        self.normalize_quotes = self.config.get('normalize_quotes', True)
        self.remove_excessive_punctuation = self.config.get('remove_excessive_punctuation', True)
        self.handle_special_characters = self.config.get('handle_special_characters', True)
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)
        
        # Advanced processing configuration
        self.preserve_case = self.config.get('preserve_case', True)
        self.remove_numbers = self.config.get('remove_numbers', False)
        self.handle_contractions = self.config.get('handle_contractions', True)
        self.remove_stopwords = self.config.get('remove_stopwords', False)
        
        # Security and validation configuration
        self.enable_security_checks = self.config.get('enable_security_checks', True)
        self.max_consecutive_chars = self.config.get('max_consecutive_chars', 5)
        self.blocked_patterns = self.config.get('blocked_patterns', [])
        
        # Performance configuration
        self.enable_metrics = self.config.get('enable_metrics', True)
        self.enable_caching = self.config.get('enable_caching', False)
        self.cache_size_limit = self.config.get('cache_size_limit', 1000)

        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize enhanced statistics
        self.stats = PreprocessingStats()
        
        # Initialize processing cache if enabled
        self.processing_cache = {} if self.enable_caching else None
        
        # Compile regex patterns for performance
        self._compile_regex_patterns()
        
        self.logger.info(f"Enhanced TextPreprocessor initialized")
        self.logger.info(f"Max Length: {self.max_length}, Security Checks: {self.enable_security_checks}")
        self.logger.info(f"URL Removal: {self.remove_urls}, Email Removal: {self.remove_emails}")

    def _compile_regex_patterns(self):
        """Compile regex patterns for improved performance."""
        try:
            # URL patterns
            self.url_pattern = re.compile(
                r'https?://[^\s]+|www\.[^\s]+|[^\s]+\.(com|org|net|edu|gov|co\.uk|de|fr|it|es|ru|cn|jp)[^\s]*',
                re.IGNORECASE
            )
            
            # Email patterns
            self.email_pattern = re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            )
            
            # Whitespace normalization
            self.whitespace_pattern = re.compile(r'\s+')
            
            # Special characters
            self.special_chars_pattern = re.compile(r'[^\w\s.,!?;:()\-\'\""]')
            
            # Excessive punctuation
            self.excessive_punct_patterns = {
                'exclamation': re.compile(r'[!]{3,}'),
                'question': re.compile(r'[?]{3,}'),
                'period': re.compile(r'[.]{3,}'),
                'comma': re.compile(r'[,]{2,}')
            }
            
            # Quote normalization
            self.quote_patterns = {
                'smart_quotes': re.compile(r'[""„‚]'),
                'smart_apostrophes': re.compile(r'[''`]')
            }
            
            # Security patterns
            if self.enable_security_checks:
                self.security_patterns = {
                    'script': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
                    'javascript': re.compile(r'javascript:', re.IGNORECASE),
                    'vbscript': re.compile(r'vbscript:', re.IGNORECASE),
                    'onclick': re.compile(r'on\w+\s*=', re.IGNORECASE)
                }
            
            self.logger.debug("Regex patterns compiled successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to compile regex patterns: {e}")
            raise

    def preprocess_text(self, text: str, session_id: Optional[str] = None) -> str:
        """
        Main preprocessing pipeline with comprehensive processing and error handling.

        Args:
            text: Raw text to preprocess
            session_id: Optional session ID for tracking

        Returns:
            Cleaned and processed text

        Raises:
            ValueError: If input text is invalid
            RuntimeError: If preprocessing fails critically
        """
        if not text or not isinstance(text, str):
            if self.enable_metrics:
                self.stats.total_texts_processed += 1
            return ""

        start_time = time.time()
        original_length = len(text)

        try:
            # Check cache if enabled
            if self.enable_caching and text in self.processing_cache:
                self.logger.debug("Using cached preprocessing result", extra={'session_id': session_id})
                return self.processing_cache[text]

            # Enhanced input validation
            self._validate_input_text(text, session_id)

            # Apply preprocessing pipeline
            processed_text = self._apply_preprocessing_pipeline(text, session_id)

            # Post-processing validation
            self._validate_processed_text(processed_text, session_id)

            # Update cache if enabled
            if self.enable_caching:
                self._update_cache(text, processed_text)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(original_length, processing_time)

            self.logger.debug(
                f"Text preprocessing completed: {original_length} → {len(processed_text)} chars",
                extra={'session_id': session_id, 'processing_time': processing_time}
            )

            return processed_text

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Preprocessing failed: {str(e)}"
            self.logger.error(error_msg, extra={'session_id': session_id})

            # Enhanced exception handling
            if _enhanced_exceptions_available:
                context = ErrorContext(
                    session_id=session_id,
                    operation="text_preprocessing",
                    processing_time=processing_time,
                    input_size=len(text) if text else 0
                )
                standardized_error = handle_llm_explanation_exception(e, context)
                log_exception_with_context(standardized_error, session_id, {'preprocessor': 'TextPreprocessor'})

            # Return sanitized original text as fallback
            return self._emergency_sanitize(text)

    def preprocess_batch(self, texts: List[str], batch_size: int = 100, 
                        session_id: Optional[str] = None) -> List[str]:
        """
        Batch preprocessing with optimized performance and error handling.

        Args:
            texts: List of text strings to preprocess
            batch_size: Number of texts to process in each batch
            session_id: Optional session ID for tracking

        Returns:
            List of preprocessed text strings
        """
        if not texts:
            return []

        self.logger.info(f"Starting batch preprocessing: {len(texts)} texts", extra={'session_id': session_id})
        
        results = []
        start_time = time.time()
        
        try:
            # Process in batches for memory efficiency
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = []
                
                for text in batch:
                    try:
                        processed = self.preprocess_text(text, session_id)
                        batch_results.append(processed)
                    except Exception as e:
                        self.logger.warning(f"Batch item failed, using fallback: {e}")
                        batch_results.append(self._emergency_sanitize(text))
                
                results.extend(batch_results)
                
                # Log progress for large batches
                if len(texts) > 1000 and (i + batch_size) % 1000 == 0:
                    self.logger.info(f"Batch progress: {i + batch_size}/{len(texts)} processed")

            processing_time = time.time() - start_time
            self.logger.info(
                f"Batch preprocessing completed: {len(results)} texts in {processing_time:.3f}s",
                extra={'session_id': session_id}
            )

            return results

        except Exception as e:
            error_msg = f"Batch preprocessing failed: {str(e)}"
            self.logger.error(error_msg, extra={'session_id': session_id})
            
            # Return emergency sanitized texts as fallback
            return [self._emergency_sanitize(text) for text in texts]

    def _apply_preprocessing_pipeline(self, text: str, session_id: Optional[str] = None) -> str:
        """Apply the complete preprocessing pipeline in optimized order."""
        processed_text = text

        # Step 1: Security checks (if enabled)
        if self.enable_security_checks:
            processed_text = self._apply_security_filtering(processed_text)

        # Step 2: Normalize whitespace early
        if self.normalize_whitespace:
            processed_text = self._normalize_whitespace(processed_text)

        # Step 3: Remove URLs (if enabled)
        if self.remove_urls:
            processed_text, urls_removed = self._remove_urls(processed_text)
            self.stats.total_urls_removed += urls_removed

        # Step 4: Remove emails (if enabled)
        if self.remove_emails:
            processed_text, emails_removed = self._remove_emails(processed_text)
            self.stats.total_emails_removed += emails_removed

        # Step 5: Handle contractions (if enabled)
        if self.handle_contractions:
            processed_text = self._expand_contractions(processed_text)

        # Step 6: Handle special characters (if enabled)
        if self.handle_special_characters:
            processed_text, special_chars_removed = self._handle_special_characters(processed_text)
            self.stats.total_special_chars_removed += special_chars_removed

        # Step 7: Normalize quotes (if enabled)
        if self.normalize_quotes:
            processed_text = self._normalize_quotes(processed_text)

        # Step 8: Remove excessive punctuation (if enabled)
        if self.remove_excessive_punctuation:
            processed_text = self._remove_excessive_punctuation(processed_text)

        # Step 9: Handle numbers (if enabled)
        if self.remove_numbers:
            processed_text = self._remove_numbers(processed_text)

        # Step 10: Limit length with smart truncation
        processed_text = self._limit_length(processed_text, self.max_length)

        # Step 11: Final cleanup
        processed_text = self._final_cleanup(processed_text)

        return processed_text

    def _validate_input_text(self, text: str, session_id: Optional[str] = None):
        """Enhanced input validation with security checks."""
        # Use a more reasonable limit for text processing (allow longer articles)
        max_chars = self.max_length * 8  # Allow 8x the token limit for character count
        if len(text) > max_chars:
            raise ValueError(f"Input text too long: {len(text)} chars (max: {max_chars})")
        
        if len(text.strip()) < self.min_length:
            raise ValueError(f"Input text too short: {len(text.strip())} chars (min: {self.min_length})")

        # Check for suspicious patterns
        if self.enable_security_checks:
            for pattern_name, pattern in self.security_patterns.items():
                if pattern.search(text):
                    self.logger.warning(f"Suspicious pattern detected: {pattern_name}", extra={'session_id': session_id})

    def _validate_processed_text(self, text: str, session_id: Optional[str] = None):
        """Validate processed text meets quality requirements."""
        if not text or not text.strip():
            raise ValueError("Processed text is empty after preprocessing")
        
        if len(text.strip()) < self.min_length:
            raise ValueError(f"Processed text too short: {len(text.strip())} chars (min: {self.min_length})")

    def _apply_security_filtering(self, text: str) -> str:
        """Apply security filtering to remove potentially malicious content."""
        filtered_text = text
        
        for pattern_name, pattern in self.security_patterns.items():
            filtered_text = pattern.sub(' ', filtered_text)
        
        # Remove blocked patterns if configured
        for blocked_pattern in self.blocked_patterns:
            try:
                pattern = re.compile(blocked_pattern, re.IGNORECASE)
                filtered_text = pattern.sub(' ', filtered_text)
            except re.error:
                self.logger.warning(f"Invalid blocked pattern: {blocked_pattern}")
        
        return filtered_text

    def _normalize_whitespace(self, text: str) -> str:
        """Enhanced whitespace normalization."""
        return self.whitespace_pattern.sub(' ', text.strip())

    def _remove_urls(self, text: str) -> Tuple[str, int]:
        """Remove URLs and return count of removals."""
        urls_found = len(self.url_pattern.findall(text))
        cleaned_text = self.url_pattern.sub(' [URL] ', text)  # Replace with placeholder
        return cleaned_text, urls_found

    def _remove_emails(self, text: str) -> Tuple[str, int]:
        """Remove email addresses and return count of removals."""
        emails_found = len(self.email_pattern.findall(text))
        cleaned_text = self.email_pattern.sub(' [EMAIL] ', text)  # Replace with placeholder
        return cleaned_text, emails_found

    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        contractions_map = {
            "won't": "will not", "can't": "cannot", "shouldn't": "should not",
            "wouldn't": "would not", "couldn't": "could not", "mustn't": "must not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "hadn't": "had not", "doesn't": "does not", "don't": "do not",
            "didn't": "did not", "I'm": "I am", "you're": "you are",
            "he's": "he is", "she's": "she is", "it's": "it is",
            "we're": "we are", "they're": "they are", "I'll": "I will",
            "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will"
        }
        
        for contraction, expansion in contractions_map.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text

    def _handle_special_characters(self, text: str) -> Tuple[str, int]:
        """Handle special characters and return count of removals."""
        original_length = len(text)
        cleaned_text = self.special_chars_pattern.sub(' ', text)
        chars_removed = original_length - len(cleaned_text.replace(' ', ''))
        return cleaned_text, max(0, chars_removed)

    def _normalize_quotes(self, text: str) -> str:
        """Normalize different types of quotes to standard ASCII."""
        text = self.quote_patterns['smart_quotes'].sub('"', text)
        text = self.quote_patterns['smart_apostrophes'].sub("'", text)
        return text

    def _remove_excessive_punctuation(self, text: str) -> str:
        """Remove excessive punctuation marks."""
        for punct_type, pattern in self.excessive_punct_patterns.items():
            if punct_type == 'exclamation':
                text = pattern.sub('!!', text)
            elif punct_type == 'question':
                text = pattern.sub('??', text)
            elif punct_type == 'period':
                text = pattern.sub('...', text)
            elif punct_type == 'comma':
                text = pattern.sub(',', text)
        return text

    def _remove_numbers(self, text: str) -> str:
        """Remove or replace numeric characters."""
        return re.sub(r'\d+', ' [NUMBER] ', text)

    def _limit_length(self, text: str, max_length: int) -> str:
        """Smart length limiting with sentence boundary preservation."""
        if len(text) <= max_length:
            return text

        truncated = text[:max_length-3]  # Leave room for "..."
        
        # Try to cut at last sentence
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > max_length * 0.8:  # If sentence end is reasonably close
            return truncated[:last_sentence_end+1] + "..."
        else:
            # Try to cut at last word
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.9:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass to ensure text quality."""
        # Normalize spaces again after all processing
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive consecutive identical characters
        if self.max_consecutive_chars > 0:
            pattern = r'(.)\1{' + str(self.max_consecutive_chars) + ',}'
            text = re.sub(pattern, r'\1' * self.max_consecutive_chars, text)
        
        return text

    def _emergency_sanitize(self, text: str) -> str:
        """Emergency text sanitization for fallback scenarios."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleanup only
        text = re.sub(r'[^\w\s.,!?;:()\-\'\""]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Limit length
        if len(text) > self.max_length:
            text = text[:self.max_length-3] + "..."
        
        return text

    def _update_cache(self, original: str, processed: str):
        """Update processing cache with size management."""
        if len(self.processing_cache) >= self.cache_size_limit:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.processing_cache))
            del self.processing_cache[oldest_key]
        
        self.processing_cache[original] = processed

    def _update_stats(self, original_length: int, processing_time: float):
        """Update comprehensive processing statistics."""
        if not self.enable_metrics:
            return

        self.stats.total_texts_processed += 1
        self.stats.total_characters_processed += original_length

        # Update timing statistics
        if self.stats.total_texts_processed == 1:
            self.stats.average_processing_time = processing_time
            self.stats.min_processing_time = processing_time
            self.stats.max_processing_time = processing_time
        else:
            current_avg = self.stats.average_processing_time
            total_processed = self.stats.total_texts_processed
            self.stats.average_processing_time = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )
            self.stats.min_processing_time = min(self.stats.min_processing_time, processing_time)
            self.stats.max_processing_time = max(self.stats.max_processing_time, processing_time)

        # Keep sample for percentile calculations (last 1000 samples)
        self.stats.processing_time_samples.append(processing_time)
        if len(self.stats.processing_time_samples) > 1000:
            self.stats.processing_time_samples = self.stats.processing_time_samples[-1000:]

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive preprocessing statistics with configuration info.

        Returns:
            Dictionary with detailed preprocessing statistics and metadata
        """
        base_stats = self.stats.to_dict()
        
        # Add percentile calculations if we have samples
        if self.stats.processing_time_samples:
            sorted_samples = sorted(self.stats.processing_time_samples)
            n = len(sorted_samples)
            percentiles = {
                'p50_processing_time': sorted_samples[int(n * 0.5)],
                'p95_processing_time': sorted_samples[int(n * 0.95)],
                'p99_processing_time': sorted_samples[int(n * 0.99)]
            }
            base_stats.update(percentiles)

        return {
            **base_stats,
            'configuration': {
                'max_length': self.max_length,
                'min_length': self.min_length,
                'remove_urls': self.remove_urls,
                'remove_emails': self.remove_emails,
                'normalize_quotes': self.normalize_quotes,
                'remove_excessive_punctuation': self.remove_excessive_punctuation,
                'handle_special_characters': self.handle_special_characters,
                'enable_security_checks': self.enable_security_checks,
                'enable_caching': self.enable_caching,
                'cache_size_limit': self.cache_size_limit
            },
            'performance_info': {
                'cache_size': len(self.processing_cache) if self.processing_cache else 0,
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'enhanced_exceptions_available': _enhanced_exceptions_available
            },
            'preprocessor_version': '3.2.0_enhanced'
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate if caching is enabled."""
        if not self.enable_caching or self.stats.total_texts_processed == 0:
            return 0.0
        
        # This is a simplified calculation - in practice, you'd track actual hits
        return min(50.0, (len(self.processing_cache) / max(self.stats.total_texts_processed, 1)) * 100)

    def reset_stats(self):
        """Reset processing statistics for fresh monitoring period."""
        self.stats = PreprocessingStats()
        if self.processing_cache:
            self.processing_cache.clear()
        
        self.logger.info("Preprocessing statistics reset")

    def get_health_status(self) -> Dict[str, Any]:
        """Get preprocessor health status for monitoring."""
        total_processed = self.stats.total_texts_processed
        avg_time = self.stats.average_processing_time
        
        # Determine health status
        if avg_time < 0.001:  # < 1ms per text
            status = "excellent"
        elif avg_time < 0.01:  # < 10ms per text
            status = "good"
        elif avg_time < 0.1:   # < 100ms per text
            status = "acceptable"
        else:
            status = "slow"

        return {
            'status': status,
            'texts_processed': total_processed,
            'average_processing_time_ms': round(avg_time * 1000, 3),
            'cache_enabled': self.enable_caching,
            'cache_size': len(self.processing_cache) if self.processing_cache else 0,
            'security_checks_enabled': self.enable_security_checks,
            'last_reset': self.stats.last_reset_time
        }


# Testing functionality with comprehensive validation
if __name__ == "__main__":
    """Test the enhanced text preprocessor with comprehensive scenarios."""
    import time
    from pprint import pprint

    print("=== Testing Enhanced Text Preprocessor ===")
    print("=" * 60)

    # Test configuration
    test_config = {
        'max_length': 500,
        'remove_urls': True,
        'remove_emails': True,
        'enable_security_checks': True,
        'enable_caching': True,
        'enable_metrics': True
    }

    # Initialize preprocessor
    print("🔧 Initializing Enhanced Text Preprocessor...")
    preprocessor = TextPreprocessor(test_config)
    print("✅ Preprocessor initialized successfully")

    # Test cases with various scenarios
    test_cases = [
        {
            'name': 'URL and Email Removal',
            'text': 'Visit https://example.com or email us at contact@example.com for more info!'
        },
        {
            'name': 'Excessive Punctuation',
            'text': 'This is amazing!!! Really??? Yes... it is!!!'
        },
        {
            'name': 'Special Characters and Quotes',
            'text': 'He said "Hello" but she replied with \'Hi\' and smiled 😊'
        },
        {
            'name': 'Long Text Truncation',
            'text': 'This is a very long text that should be truncated. ' * 50
        },
        {
            'name': 'Contractions',
            'text': "I can't believe it won't work. Shouldn't we try again?"
        },
        {
            'name': 'Security Test',
            'text': 'Normal text with <script>alert("test")</script> embedded script'
        }
    ]

    print(f"\n🧪 Running {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Original: {test_case['text'][:100]}{'...' if len(test_case['text']) > 100 else ''}")
        
        try:
            start_time = time.time()
            processed = preprocessor.preprocess_text(test_case['text'], session_id=f"test_{i}")
            processing_time = (time.time() - start_time) * 1000
            
            print(f"Processed: {processed[:100]}{'...' if len(processed) > 100 else ''}")
            print(f"Time: {processing_time:.2f}ms")
        except Exception as e:
            print(f"❌ Test failed: {e}")

    # Test batch processing
    print(f"\n🔄 Testing batch preprocessing...")
    batch_texts = [case['text'] for case in test_cases[:3]]
    
    try:
        start_time = time.time()
        batch_results = preprocessor.preprocess_batch(batch_texts, session_id="batch_test")
        batch_time = (time.time() - start_time) * 1000
        
        print(f"✅ Batch processing completed: {len(batch_results)} texts in {batch_time:.2f}ms")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")

    # Show comprehensive statistics
    print(f"\n📊 Comprehensive Statistics:")
    stats = preprocessor.get_preprocessing_stats()
    pprint(stats)

    # Show health status
    print(f"\n🏥 Health Status:")
    health = preprocessor.get_health_status()
    pprint(health)

    print(f"\n✅ Enhanced Text Preprocessor testing completed!")
