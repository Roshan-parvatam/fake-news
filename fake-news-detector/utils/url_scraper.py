# utils/url_scraper.py

"""
Production-Ready Advanced News Article Scraper

Comprehensive multi-strategy scraper with async support, intelligent rate limiting,
content validation, and robust error handling for maximum reliability.

Features:
- Multiple extraction strategies with intelligent fallbacks
- Adaptive rate limiting with domain-specific controls and respect for robots.txt
- Async processing with proper error isolation and timeout handling
- Advanced content validation with quality scoring and reliability metrics
- User-agent rotation and anti-detection measures
- Memory-efficient processing for high-volume scraping
- Comprehensive logging and performance monitoring

Version: 3.2.0 - Enhanced Production Edition
"""

import asyncio
import re
import time
import logging
import json
import hashlib
from urllib.parse import urlparse, urljoin, urlencode
from urllib.robotparser import RobotFileParser
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor
import socket

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import html2text

# Optional dependencies with graceful fallbacks
try:
    from newspaper import Article, Config as NewsConfig
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logging.warning("newspaper3k not available - using fallback extraction only")

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("playwright not available - async scraping disabled")

try:
    import cchardet as chardet
except ImportError:
    try:
        import chardet
    except ImportError:
        chardet = None
        logging.warning("No character encoding detection library available - using fallback encoding detection")

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Enhanced scraping result with comprehensive metadata."""
    success: bool
    text: str = ""
    title: str = ""
    author: str = ""
    publish_date: Optional[str] = None
    url: str = ""
    method: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Add default metadata fields."""
        if not self.metadata:
            self.metadata = {}
        
        # Add processing timestamp
        self.metadata.setdefault('scraped_at', datetime.now().isoformat())
        self.metadata.setdefault('content_length', len(self.text))
        self.metadata.setdefault('word_count', len(self.text.split()) if self.text else 0)


@dataclass 
class RateLimitConfig:
    """Configuration for rate limiting behavior."""
    requests_per_minute: int = 12
    base_delay: float = 2.0
    burst_limit: int = 3
    adaptive_delay: bool = True
    respect_robots: bool = True
    max_concurrent: int = 5


class EnhancedRateLimiter:
    """
    Advanced rate limiter with domain-specific controls, robots.txt respect, and adaptive delays.
    
    Features:
    - Per-domain request tracking and rate limiting
    - Robots.txt compliance with caching
    - Adaptive delays based on response times and server behavior
    - User-agent rotation with realistic browser profiles
    - Concurrent request management
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.domain_requests = {}
        self.domain_response_times = {}
        self.domain_errors = {}
        self.robots_cache = {}
        self.lock = threading.RLock()
        
        # Realistic user agents from different browsers and platforms
        self.user_agents = [
            # Chrome on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            # Chrome on macOS  
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            # Firefox on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            # Safari on macOS
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            # Chrome on Linux
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        self.ua_index = 0
        
        # Concurrent request tracking
        self.active_requests = {}
        
    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.config.respect_robots:
            return True
            
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            with self.lock:
                if domain not in self.robots_cache:
                    robots_url = f"{parsed.scheme}://{domain}/robots.txt"
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    
                    try:
                        rp.read()
                        self.robots_cache[domain] = rp
                    except Exception as e:
                        logger.debug(f"Could not read robots.txt for {domain}: {e}")
                        # If we can't read robots.txt, assume we can fetch
                        self.robots_cache[domain] = None
                        return True
                
                rp = self.robots_cache[domain]
                if rp is None:
                    return True
                    
                return rp.can_fetch(user_agent, url)
                
        except Exception as e:
            logger.debug(f"Error checking robots.txt for {url}: {e}")
            return True
    
    def get_headers(self, url: str) -> Dict[str, str]:
        """Get headers with rate limiting enforcement and user-agent rotation."""
        domain = urlparse(url).netloc
        
        with self.lock:
            self._clean_old_requests(domain)
            self._enforce_rate_limit(domain)
            self._manage_concurrent_requests(domain)
            
            # Rotate user agent
            user_agent = self.user_agents[self.ua_index]
            self.ua_index = (self.ua_index + 1) % len(self.user_agents)
            
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            # Add referer for more realistic requests
            if domain != urlparse(url).netloc:
                headers['Referer'] = f"https://www.google.com/"
            
            self._record_request(domain)
            return headers
    
    def _clean_old_requests(self, domain: str):
        """Remove requests older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        if domain in self.domain_requests:
            self.domain_requests[domain] = [
                timestamp for timestamp in self.domain_requests[domain] 
                if timestamp > cutoff
            ]
    
    def _enforce_rate_limit(self, domain: str):
        """Enforce rate limiting with adaptive delays."""
        if domain not in self.domain_requests:
            self.domain_requests[domain] = []
        
        recent_requests = len(self.domain_requests[domain])
        
        if recent_requests >= self.config.requests_per_minute:
            wait_time = self._calculate_adaptive_delay(domain) * 2
            logger.warning(f"Rate limit exceeded for {domain}, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        elif recent_requests >= self.config.burst_limit:
            wait_time = self._calculate_adaptive_delay(domain)
            logger.debug(f"Burst limit approached for {domain}, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
    
    def _manage_concurrent_requests(self, domain: str):
        """Manage concurrent requests to avoid overwhelming servers."""
        if domain not in self.active_requests:
            self.active_requests[domain] = 0
        
        while self.active_requests[domain] >= self.config.max_concurrent:
            logger.debug(f"Too many concurrent requests to {domain}, waiting...")
            time.sleep(0.1)
    
    def _calculate_adaptive_delay(self, domain: str) -> float:
        """Calculate adaptive delay based on server response patterns."""
        if not self.config.adaptive_delay or domain not in self.domain_response_times:
            return self.config.base_delay
        
        recent_times = self.domain_response_times[domain][-10:]
        if not recent_times:
            return self.config.base_delay
        
        avg_response_time = sum(recent_times) / len(recent_times)
        error_count = self.domain_errors.get(domain, 0)
        
        # Adjust delay based on response time and error rate
        if avg_response_time > 5.0 or error_count > 2:
            return self.config.base_delay * 2.0
        elif avg_response_time < 1.0 and error_count == 0:
            return max(self.config.base_delay * 0.7, 1.0)
        
        return self.config.base_delay
    
    def _record_request(self, domain: str):
        """Record request timestamp for rate limiting."""
        self.domain_requests.setdefault(domain, []).append(datetime.now())
        self.active_requests.setdefault(domain, 0)
        self.active_requests[domain] += 1
    
    def record_response(self, domain: str, response_time: float, success: bool = True):
        """Record response metrics for adaptive behavior."""
        with self.lock:
            if domain in self.active_requests:
                self.active_requests[domain] = max(0, self.active_requests[domain] - 1)
            
            # Record response time
            self.domain_response_times.setdefault(domain, []).append(response_time)
            # Keep only last 20 response times
            self.domain_response_times[domain] = self.domain_response_times[domain][-20:]
            
            # Record errors
            if not success:
                self.domain_errors.setdefault(domain, 0)
                self.domain_errors[domain] += 1
            else:
                # Decay error count on success
                if domain in self.domain_errors:
                    self.domain_errors[domain] = max(0, self.domain_errors[domain] - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiter statistics."""
        with self.lock:
            return {
                'domain_requests': {k: len(v) for k, v in self.domain_requests.items()},
                'active_requests': self.active_requests.copy(),
                'domain_response_times': {
                    k: {
                        'avg': sum(v) / len(v) if v else 0,
                        'count': len(v)
                    } for k, v in self.domain_response_times.items()
                },
                'domain_errors': self.domain_errors.copy(),
                'robots_cache_size': len(self.robots_cache)
            }


class AdvancedContentExtractor:
    """
    Advanced content extraction with quality assessment and cleaning.
    
    Features:
    - Intelligent element selection with priority scoring
    - Content quality assessment with multiple metrics
    - Noise removal and content cleaning
    - Metadata extraction (title, author, date)
    - Readability analysis and content classification
    """
    
    def __init__(self):
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = True  
        self.html2text.ignore_images = True
        self.html2text.body_width = 0
        self.html2text.unicode_snob = True
        
        # Quality thresholds
        self.min_content_length = 200
        self.min_word_count = 50
        self.min_sentence_count = 3
        
        # Pre-compiled regex patterns for performance
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r'\b\w+\b')
        self.url_pattern = re.compile(r'https?://\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    def extract_from_soup(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content using intelligent BeautifulSoup analysis."""
        try:
            # Remove noise elements first
            self._remove_noise_elements(soup)
            
            # Extract metadata
            title = self._extract_title(soup)
            author = self._extract_author(soup)
            publish_date = self._extract_publish_date(soup)
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            if not content:
                return {
                    'success': False,
                    'error': 'No content could be extracted',
                    'url': url
                }
            
            # Clean and validate content
            cleaned_content = self._clean_content(content)
            if not self._validate_content_quality(cleaned_content):
                return {
                    'success': False,
                    'error': 'Content failed quality validation',
                    'url': url,
                    'content_preview': cleaned_content[:200] + '...' if len(cleaned_content) > 200 else cleaned_content
                }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(cleaned_content)
            
            return {
                'success': True,
                'title': title,
                'content': cleaned_content,
                'author': author,
                'publish_date': publish_date,
                'url': url,
                'domain': urlparse(url).netloc,
                'quality_metrics': quality_metrics,
                'extraction_metadata': {
                    'word_count': len(cleaned_content.split()),
                    'char_count': len(cleaned_content),
                    'extraction_method': 'advanced_soup'
                }
            }
            
        except Exception as e:
            logger.error(f"Content extraction error for {url}: {str(e)}")
            return {
                'success': False,
                'error': f'Extraction failed: {str(e)}',
                'url': url
            }
    
    def _remove_noise_elements(self, soup: BeautifulSoup):
        """Remove noise elements that don't contain article content."""
        noise_selectors = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            '.advertisement', '.ad', '.ads', '.social-share', '.social-media',
            '.comments', '.comment', '.sidebar', '.navigation', '.menu',
            '.popup', '.modal', '.overlay', '.cookie-notice', '.newsletter',
            '[class*="share"]', '[class*="social"]', '[class*="ad-"]',
            '[class*="advertisement"]', '[id*="ad-"]', '[class*="popup"]',
            '.related-articles', '.recommended', '.trending'
        ]
        
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title using priority selectors."""
        title_selectors = [
            'h1.headline', 'h1.title', 'h1[class*="headline"]', 'h1[class*="title"]',
            '.article-title', '.story-headline', '.entry-title', '.post-title',
            '.content-title', '.page-title', 'h1.article-header', 'h1.post-header',
            '[property="og:title"]', '[name="twitter:title"]',
            '[property="article:title"]', 'title', 'h1'
        ]
        
        for selector in title_selectors:
            elements = soup.select(selector)
            for element in elements:
                if selector.startswith('['):
                    # Meta tag
                    title = element.get('content', '')
                else:
                    title = element.get_text(strip=True)
                
                if title and len(title) >= 10:
                    # Clean title (remove site name suffixes)
                    for separator in [' | ', ' - ', ' :: ', ' • ', ' — ', ' – ']:
                        if separator in title:
                            title = title.split(separator)[0].strip()
                            break
                    
                    # Validate title quality
                    if 10 <= len(title) <= 200 and not self._is_navigation_text(title):
                        return title
        
        return ""
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract author information with multiple strategies."""
        # Try meta tags first
        meta_selectors = [
            'meta[name="author"]', 'meta[property="article:author"]',
            'meta[name="dc.creator"]', 'meta[name="sailthru.author"]',
            'meta[property="author"]', 'meta[name="byl"]'
        ]
        
        for selector in meta_selectors:
            element = soup.select_one(selector)
            if element and element.get('content'):
                return element['content'].strip()
        
        # Try structured data
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    author = data.get('author')
                    if author:
                        if isinstance(author, dict):
                            return author.get('name', '')
                        elif isinstance(author, str):
                            return author
            except (json.JSONDecodeError, AttributeError):
                continue
        
        # Try semantic elements
        author_selectors = [
            '.author', '.byline', '.by-author', '.article-author',
            '[class*="author"]', '[class*="byline"]', '.writer',
            '.journalist', '.reporter', '[rel="author"]'
        ]
        
        for selector in author_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) < 100:
                    # Clean author text
                    text = re.sub(r'^by\s+', '', text, flags=re.IGNORECASE).strip()
                    text = re.sub(r'\s+', ' ', text)
                    if text and not self._is_navigation_text(text):
                        return text
        
        return ""
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date using multiple strategies."""
        # Try structured data first
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    for date_field in ['datePublished', 'dateCreated', 'dateModified']:
                        date_value = data.get(date_field)
                        if date_value:
                            return date_value
            except (json.JSONDecodeError, AttributeError):
                continue
        
        # Try time elements
        for time_elem in soup.find_all('time'):
            datetime_attr = time_elem.get('datetime')
            if datetime_attr:
                return datetime_attr
        
        # Try meta tags
        meta_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publish-date"]', 'meta[name="date"]',
            'meta[name="dcterms.created"]', 'meta[name="sailthru.date"]',
            'meta[property="og:publish_date"]', 'meta[name="pubdate"]'
        ]
        
        for selector in meta_selectors:
            element = soup.select_one(selector)
            if element and element.get('content'):
                return element['content']
        
        return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content with intelligent selection."""
        # Priority content selectors
        content_selectors = [
            'article', '[role="main"]', 'main', '.main-content',
            '.article-body', '.story-body', '.post-content', '.entry-content',
            '.article-content', '.content-body', '.post-body', '.story-content',
            '#content', '.content', '.article-text', '.story-text'
        ]
        
        # Try each selector and score the content
        best_content = ""
        best_score = 0
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content = self._extract_text_from_element(element)
                score = self._score_content(content)
                
                if score > best_score:
                    best_content = content
                    best_score = score
        
        # Fallback: aggregate paragraphs
        if not best_content or best_score < 3:
            paragraph_content = self._extract_from_paragraphs(soup)
            paragraph_score = self._score_content(paragraph_content)
            
            if paragraph_score > best_score:
                best_content = paragraph_content
        
        return best_content
    
    def _extract_text_from_element(self, element) -> str:
        """Extract clean text from HTML element."""
        # Remove remaining noise within the element
        for unwanted in element.select('.ad, .social, .share, .related, .comments, .popup'):
            unwanted.decompose()
        
        # Convert to text using html2text for better formatting
        html_content = str(element)
        text = self.html2text.handle(html_content)
        
        return self._clean_content(text)
    
    def _extract_from_paragraphs(self, soup: BeautifulSoup) -> str:
        """Fallback content extraction by intelligently aggregating paragraphs."""
        paragraphs = soup.find_all('p')
        if not paragraphs:
            return ""
        
        # Score and filter paragraphs
        scored_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) < 30:  # Skip very short paragraphs
                continue
                
            score = self._score_paragraph(p, text)
            if score > 0:
                scored_paragraphs.append((score, text))
        
        # Sort by score and take the best paragraphs
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
        
        # Take top paragraphs that collectively form good content
        selected_paragraphs = []
        total_length = 0
        
        for score, text in scored_paragraphs:
            if len(selected_paragraphs) >= 20:  # Limit number of paragraphs
                break
            if total_length > 5000:  # Reasonable content length limit
                break
                
            selected_paragraphs.append(text)
            total_length += len(text)
        
        content = '\n\n'.join(selected_paragraphs)
        return self._clean_content(content)
    
    def _score_paragraph(self, element, text: str) -> float:
        """Score paragraph likelihood of being main content."""
        score = 0.0
        
        # Length scoring (optimal range)
        length = len(text)
        if 50 <= length <= 1000:
            score += min(length / 100, 5)
        
        # Word count scoring
        words = text.split()
        word_count = len(words)
        if 10 <= word_count <= 200:
            score += word_count / 20
        
        # Content quality indicators
        if any(indicator in text.lower() for indicator in [
            'said', 'according to', 'reported', 'announced', 'revealed',
            'study', 'research', 'data', 'analysis', 'expert'
        ]):
            score += 2
        
        # Penalize navigation/UI text
        if self._is_navigation_text(text):
            score -= 10
        
        # Parent element context
        parent_classes = ' '.join(element.parent.get('class', []))
        if any(indicator in parent_classes.lower() for indicator in [
            'article', 'content', 'story', 'post', 'main'
        ]):
            score += 3
        
        # Penalize if in likely sidebar/footer areas
        if any(indicator in parent_classes.lower() for indicator in [
            'sidebar', 'footer', 'nav', 'menu', 'ad', 'widget'
        ]):
            score -= 5
        
        return max(0, score)
    
    def _score_content(self, content: str) -> float:
        """Score content quality for selection purposes."""
        if not content:
            return 0
        
        words = content.split()
        word_count = len(words)
        
        score = 0
        
        # Word count scoring
        if word_count >= 100:
            score += 5
        elif word_count >= 50:
            score += 3
        elif word_count >= 25:
            score += 1
        
        # Sentence structure
        sentences = self.sentence_pattern.split(content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(valid_sentences) >= 3:
            score += 2
        
        # Content indicators
        content_lower = content.lower()
        quality_indicators = [
            'according to', 'reported', 'study', 'research', 'analysis',
            'data', 'expert', 'official', 'confirmed', 'announced'
        ]
        indicator_count = sum(1 for indicator in quality_indicators if indicator in content_lower)
        score += min(indicator_count * 0.5, 3)
        
        return score
    
    def _clean_content(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove markdown artifacts from html2text conversion
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
        text = re.sub(r'#{1,6}\s*', '', text)      # Remove headers
        text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)  # Remove bold/italic
        text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)    # Remove underline
        
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)    # Max 2 consecutive newlines
        text = re.sub(r'[ \t]{2,}', ' ', text)    # Multiple spaces to single
        text = re.sub(r'\n[ \t]+', '\n', text)    # Remove leading whitespace
        text = re.sub(r'[ \t]+\n', '\n', text)    # Remove trailing whitespace
        
        # Remove common artifacts
        text = re.sub(r'^\s*\n', '', text)        # Leading newlines
        text = re.sub(r'\n\s*$', '', text)        # Trailing newlines
        
        return text.strip()
    
    def _validate_content_quality(self, text: str) -> bool:
        """Validate if content meets minimum quality standards."""
        if not text or not text.strip():
            return False
        
        words = text.split()
        if len(words) < self.min_word_count:
            return False
        
        if len(text) < self.min_content_length:
            return False
        
        # Check sentence count
        sentences = self.sentence_pattern.split(text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(valid_sentences) < self.min_sentence_count:
            return False
        
        # Check alphabetic content ratio
        alpha_chars = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha_chars / len(text) < 0.6:
            return False
        
        # Check for error/placeholder content
        error_patterns = [
            r'404|not found|page not found|error occurred',
            r'access denied|forbidden|unauthorized',
            r'please enable javascript|javascript required',
            r'subscription required|premium content|paywall',
            r'loading\.\.\.|please wait|content unavailable'
        ]
        
        text_lower = text.lower()
        for pattern in error_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    def _calculate_quality_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for content."""
        if not text:
            return {'overall_score': 0.0}
        
        words = text.split()
        sentences = self.sentence_pattern.split(text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        metrics = {
            'word_count': len(words),
            'sentence_count': len(valid_sentences),
            'character_count': len(text),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(valid_sentences) if valid_sentences else 0
        }
        
        # Quality scoring (0-1 scale)
        quality_score = 0.0
        
        # Length quality (0-0.3)
        if len(words) >= 100:
            quality_score += 0.3
        elif len(words) >= 50:
            quality_score += 0.2
        elif len(words) >= 25:
            quality_score += 0.1
        
        # Structure quality (0-0.2) 
        if len(valid_sentences) >= 5:
            quality_score += 0.2
        elif len(valid_sentences) >= 3:
            quality_score += 0.1
        
        # Content indicators (0-0.3)
        content_indicators = [
            'according to', 'reported', 'announced', 'study', 'research',
            'data', 'analysis', 'expert', 'official', 'confirmed'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in content_indicators if indicator in text_lower)
        quality_score += min(indicator_count * 0.05, 0.3)
        
        # Readability (0-0.2)
        avg_word_length = metrics['avg_word_length']
        if 4 <= avg_word_length <= 6:  # Good readability range
            quality_score += 0.2
        elif 3 <= avg_word_length <= 7:
            quality_score += 0.1
        
        metrics['overall_score'] = min(quality_score, 1.0)
        metrics['quality_factors'] = {
            'sufficient_length': len(words) >= 50,
            'good_structure': len(valid_sentences) >= 3,
            'has_indicators': indicator_count > 0,
            'readable': 3 <= avg_word_length <= 7
        }
        
        return metrics
    
    def _is_navigation_text(self, text: str) -> bool:
        """Check if text is likely navigation/UI element."""
        text_lower = text.lower().strip()
        
        # Short navigation phrases
        if len(text_lower) < 50:
            nav_phrases = [
                'click here', 'read more', 'subscribe', 'sign up', 'log in',
                'home', 'about', 'contact', 'menu', 'search', 'share',
                'follow us', 'newsletter', 'privacy policy', 'terms of service',
                'cookie policy', 'skip to content', 'back to top'
            ]
            
            if any(phrase in text_lower for phrase in nav_phrases):
                return True
        
        # Check for mostly uppercase (likely headings/navigation)
        if len(text) > 5 and sum(1 for c in text if c.isupper()) / len(text) > 0.7:
            return True
        
        return False


class ProductionNewsScraper:
    """
    Production-ready news article scraper with comprehensive error handling,
    multiple extraction strategies, and advanced quality control.
    
    Features:
    - Multi-strategy extraction (newspaper3k, Playwright, requests+BeautifulSoup)
    - Intelligent fallback system with quality-based selection
    - Advanced rate limiting with robots.txt respect
    - Content validation and quality scoring
    - Comprehensive error handling and recovery
    - Performance monitoring and statistics
    - Memory-efficient processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize rate limiter
        rate_config = RateLimitConfig(
            requests_per_minute=self.config.get('requests_per_minute', 12),
            base_delay=self.config.get('base_delay', 2.0),
            burst_limit=self.config.get('burst_limit', 3),
            adaptive_delay=self.config.get('adaptive_delay', True),
            respect_robots=self.config.get('respect_robots', True),
            max_concurrent=self.config.get('max_concurrent', 5)
        )
        self.rate_limiter = EnhancedRateLimiter(rate_config)
        
        # Initialize content extractor
        self.content_extractor = AdvancedContentExtractor()
        
        # Initialize requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'method_success': {
                'newspaper3k': 0,
                'playwright': 0, 
                'requests': 0
            },
            'quality_scores': [],
            'processing_times': []
        }
    
    def scrape_article(self, url: str, timeout: int = 30, 
                      require_quality_threshold: float = 0.3) -> ScrapingResult:
        """
        Main scraping method with intelligent multi-strategy approach.
        
        Args:
            url: Article URL to scrape
            timeout: Request timeout in seconds
            require_quality_threshold: Minimum quality score (0.0-1.0)
            
        Returns:
            ScrapingResult with comprehensive extraction data
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # Validate URL
        if not self._validate_url(url):
            return ScrapingResult(
                success=False,
                url=url,
                error="Invalid URL format or blocked domain",
                method="validation"
            )
        
        # Check robots.txt
        if not self.rate_limiter.can_fetch(url):
            return ScrapingResult(
                success=False,
                url=url,
                error="Blocked by robots.txt",
                method="robots_check"
            )
        
        logger.info(f"Scraping article: {url}")
        
        # Strategy 1: newspaper3k (fastest for standard news sites)
        if NEWSPAPER_AVAILABLE:
            result = self._scrape_with_newspaper(url, timeout)
            if self._is_result_acceptable(result, require_quality_threshold):
                processing_time = time.time() - start_time
                self._record_success('newspaper3k', result, processing_time)
                return result
            
            logger.debug(f"Newspaper3k result not acceptable: {result.error}")
        
        # Strategy 2: Playwright (for JavaScript-heavy sites)
        if PLAYWRIGHT_AVAILABLE:
            try:
                result = asyncio.run(self._scrape_with_playwright(url, timeout))
                if self._is_result_acceptable(result, require_quality_threshold):
                    processing_time = time.time() - start_time
                    self._record_success('playwright', result, processing_time)
                    return result
                
                logger.debug(f"Playwright result not acceptable: {result.error}")
            except Exception as e:
                logger.debug(f"Playwright strategy failed: {str(e)}")
        
        # Strategy 3: Requests + BeautifulSoup (reliable fallback)
        result = self._scrape_with_requests(url, timeout)
        processing_time = time.time() - start_time
        
        if self._is_result_acceptable(result, require_quality_threshold):
            self._record_success('requests', result, processing_time)
        else:
            self._record_failure(processing_time)
        
        return result
    
    def _scrape_with_newspaper(self, url: str, timeout: int) -> ScrapingResult:
        """Scrape using newspaper3k library with enhanced configuration."""
        try:
            start_time = time.time()
            
            # Get headers from rate limiter
            headers = self.rate_limiter.get_headers(url)
            
            # Configure newspaper
            news_config = NewsConfig()
            news_config.browser_user_agent = headers['User-Agent']
            news_config.request_timeout = timeout
            news_config.number_threads = 1
            
            # Create and process article
            article = Article(url, config=news_config)
            article.download()
            article.parse()
            
            # Record response time
            domain = urlparse(url).netloc
            response_time = time.time() - start_time
            self.rate_limiter.record_response(domain, response_time, True)
            
            # Validate content
            if not article.text or len(article.text.strip()) < 100:
                return ScrapingResult(
                    success=False,
                    url=url,
                    method="newspaper3k",
                    error="Content too short or empty"
                )
            
            # Calculate quality score
            quality_metrics = self.content_extractor._calculate_quality_metrics(article.text)
            
            return ScrapingResult(
                success=True,
                text=article.text.strip(),
                title=article.title.strip() if article.title else "",
                author=', '.join(article.authors) if article.authors else "",
                publish_date=article.publish_date.isoformat() if article.publish_date else None,
                url=url,
                method="newspaper3k",
                metadata={
                    'quality_metrics': quality_metrics,
                    'response_time': response_time,
                    'top_image': article.top_image,
                    'meta_keywords': article.meta_keywords,
                    'summary': article.summary[:200] + '...' if len(article.summary) > 200 else article.summary
                }
            )
            
        except Exception as e:
            domain = urlparse(url).netloc
            self.rate_limiter.record_response(domain, time.time() - start_time, False)
            
            return ScrapingResult(
                success=False,
                url=url,
                method="newspaper3k",
                error=f"Newspaper3k extraction failed: {str(e)}"
            )
    
    async def _scrape_with_playwright(self, url: str, timeout: int) -> ScrapingResult:
        """Scrape using Playwright for JavaScript-heavy sites."""
        try:
            start_time = time.time()
            headers = self.rate_limiter.get_headers(url)
            
            async with async_playwright() as p:
                # Launch browser with minimal resources
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-images',
                        '--disable-plugins',
                        '--disable-extensions',
                        '--no-first-run'
                    ]
                )
                
                try:
                    context = await browser.new_context(
                        user_agent=headers['User-Agent'],
                        viewport={'width': 1920, 'height': 1080},
                        ignore_https_errors=True
                    )
                    
                    page = await context.new_page()
                    
                    # Navigate with comprehensive error handling
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                        
                        # Wait for content to load
                        await page.wait_for_timeout(2000)
                        
                        # Try to wait for article content
                        try:
                            await page.wait_for_selector(
                                'article, .article-body, .story-body, main, .content', 
                                timeout=5000
                            )
                        except PlaywrightTimeout:
                            pass  # Continue even if specific selectors not found
                        
                    except PlaywrightTimeout:
                        return ScrapingResult(
                            success=False,
                            url=url,
                            method="playwright",
                            error="Page load timeout"
                        )
                    
                    # Extract content
                    title = await page.title()
                    
                    # Get page content
                    content_html = await page.content()
                    soup = BeautifulSoup(content_html, 'html.parser')
                    
                    # Use content extractor
                    extraction_result = self.content_extractor.extract_from_soup(soup, url)
                    
                    if not extraction_result['success']:
                        return ScrapingResult(
                            success=False,
                            url=url,
                            method="playwright",
                            error=extraction_result['error']
                        )
                    
                    # Record response time
                    response_time = time.time() - start_time
                    domain = urlparse(url).netloc
                    self.rate_limiter.record_response(domain, response_time, True)
                    
                    return ScrapingResult(
                        success=True,
                        text=extraction_result['content'],
                        title=title.strip(),
                        author=extraction_result.get('author', ''),
                        publish_date=extraction_result.get('publish_date'),
                        url=url,
                        method="playwright",
                        metadata={
                            **extraction_result.get('quality_metrics', {}),
                            'response_time': response_time,
                            'extraction_metadata': extraction_result.get('extraction_metadata', {})
                        }
                    )
                    
                finally:
                    await browser.close()
                    
        except Exception as e:
            domain = urlparse(url).netloc
            response_time = time.time() - start_time
            self.rate_limiter.record_response(domain, response_time, False)
            
            return ScrapingResult(
                success=False,
                url=url,
                method="playwright",
                error=f"Playwright extraction failed: {str(e)}"
            )
    
    def _scrape_with_requests(self, url: str, timeout: int) -> ScrapingResult:
        """Fallback scraper using requests + BeautifulSoup with advanced content extraction."""
        try:
            start_time = time.time()
            headers = self.rate_limiter.get_headers(url)
            
            # Make request
            response = self.session.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=False
            )
            
            response.raise_for_status()
            
            # Record response time
            response_time = time.time() - start_time
            domain = urlparse(url).netloc
            self.rate_limiter.record_response(domain, response_time, True)
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return ScrapingResult(
                    success=False,
                    url=url,
                    method="requests",
                    error=f"Invalid content type: {content_type}"
                )
            
            # Detect encoding
            encoding = response.encoding
            if not encoding or encoding.lower() == 'iso-8859-1':
                # Try to detect encoding
                if chardet:
                    detected = chardet.detect(response.content)
                    if detected and detected['confidence'] > 0.7:
                        encoding = detected['encoding']
                
                if not encoding:
                    encoding = 'utf-8'
            
            # Parse content
            try:
                content = response.content.decode(encoding, errors='replace')
            except (UnicodeDecodeError, LookupError):
                content = response.content.decode('utf-8', errors='replace')
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract using advanced content extractor
            extraction_result = self.content_extractor.extract_from_soup(soup, url)
            
            if not extraction_result['success']:
                return ScrapingResult(
                    success=False,
                    url=url, 
                    method="requests",
                    error=extraction_result['error']
                )
            
            return ScrapingResult(
                success=True,
                text=extraction_result['content'],
                title=extraction_result.get('title', ''),
                author=extraction_result.get('author', ''),
                publish_date=extraction_result.get('publish_date'),
                url=url,
                method="requests",
                metadata={
                    **extraction_result.get('quality_metrics', {}),
                    'response_time': response_time,
                    'encoding': encoding,
                    'content_type': content_type,
                    'status_code': response.status_code,
                    'extraction_metadata': extraction_result.get('extraction_metadata', {})
                }
            )
            
        except requests.exceptions.RequestException as e:
            domain = urlparse(url).netloc
            response_time = time.time() - start_time
            self.rate_limiter.record_response(domain, response_time, False)
            
            return ScrapingResult(
                success=False,
                url=url,
                method="requests",
                error=f"Request failed: {str(e)}"
            )
        except Exception as e:
            domain = urlparse(url).netloc
            response_time = time.time() - start_time
            self.rate_limiter.record_response(domain, response_time, False)
            
            return ScrapingResult(
                success=False,
                url=url,
                method="requests",
                error=f"Processing failed: {str(e)}"
            )
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format and check against blocklist."""
        if not url or not isinstance(url, str):
            return False
        
        url = url.strip()
        
        try:
            parsed = urlparse(url)
            
            # Basic format validation
            if not parsed.scheme or not parsed.netloc:
                return False
            
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Security checks
            hostname = parsed.hostname
            if not hostname:
                return False
            
            # Block local/private networks
            blocked_hosts = [
                'localhost', '127.0.0.1', '0.0.0.0', '::1',
                '10.', '172.16.', '192.168.'
            ]
            
            if any(hostname.startswith(blocked) for blocked in blocked_hosts):
                return False
            
            # Check for suspicious patterns
            suspicious_patterns = [
                'javascript:', 'data:', 'file:', 'ftp:'
            ]
            
            if any(pattern in url.lower() for pattern in suspicious_patterns):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_result_acceptable(self, result: ScrapingResult, threshold: float) -> bool:
        """Check if scraping result meets quality threshold."""
        if not result.success or not result.text:
            return False
        
        # Check basic content requirements
        if len(result.text.strip()) < 100:
            return False
        
        # Check quality score if available
        if result.metadata and 'quality_metrics' in result.metadata:
            quality_score = result.metadata['quality_metrics'].get('overall_score', 0)
            return quality_score >= threshold
        
        # Fallback: basic word count check
        word_count = len(result.text.split())
        return word_count >= 50
    
    def _record_success(self, method: str, result: ScrapingResult, processing_time: float):
        """Record successful extraction metrics."""
        self.stats['successful_extractions'] += 1
        self.stats['method_success'][method] += 1
        self.stats['processing_times'].append(processing_time)
        
        if result.metadata and 'quality_metrics' in result.metadata:
            quality_score = result.metadata['quality_metrics'].get('overall_score', 0)
            self.stats['quality_scores'].append(quality_score)
    
    def _record_failure(self, processing_time: float):
        """Record failed extraction metrics."""
        self.stats['failed_extractions'] += 1
        self.stats['processing_times'].append(processing_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive scraper performance statistics."""
        total_requests = self.stats['total_requests']
        if total_requests == 0:
            return self.stats
        
        success_rate = (self.stats['successful_extractions'] / total_requests) * 100
        
        # Calculate averages
        avg_processing_time = (
            sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            if self.stats['processing_times'] else 0
        )
        
        avg_quality_score = (
            sum(self.stats['quality_scores']) / len(self.stats['quality_scores'])
            if self.stats['quality_scores'] else 0
        )
        
        return {
            **self.stats,
            'success_rate': round(success_rate, 2),
            'failure_rate': round(100 - success_rate, 2),
            'avg_processing_time': round(avg_processing_time, 2),
            'avg_quality_score': round(avg_quality_score, 3),
            'method_preference_order': sorted(
                self.stats['method_success'].items(),
                key=lambda x: x[1],
                reverse=True
            ),
            'rate_limiter_stats': self.rate_limiter.get_stats()
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'method_success': {
                'newspaper3k': 0,
                'playwright': 0,
                'requests': 0
            },
            'quality_scores': [],
            'processing_times': []
        }


# Utility functions for backward compatibility and convenience

def validate_url(url: str) -> bool:
    """Validate URL format - utility function."""
    scraper = ProductionNewsScraper()
    return scraper._validate_url(url)


def is_news_url(url: str) -> bool:
    """Detect if URL appears to be from a news website."""
    if not validate_url(url):
        return False
    
    news_patterns = [
        r'/news/', r'/article/', r'/story/', r'/post/', r'/blog/',
        r'/\d{4}/\d{2}/\d{2}/', r'/\d{4}/\d{2}/',
        r'/(politics|world|business|technology|health|sports|breaking|latest|opinion)/',
        r'/(local|national|international|breaking-news)/'
    ]
    
    url_lower = url.lower()
    return any(re.search(pattern, url_lower) for pattern in news_patterns)


def extract_domain(url: str) -> Optional[str]:
    """Extract clean domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain
    except Exception:
        return None


def create_scraper(config: Optional[Dict[str, Any]] = None) -> ProductionNewsScraper:
    """Factory function to create configured scraper instance."""
    default_config = {
        'requests_per_minute': 12,
        'base_delay': 2.0,
        'adaptive_delay': True,
        'respect_robots': True,
        'max_concurrent': 5
    }
    
    if config:
        default_config.update(config)
    
    return ProductionNewsScraper(default_config)


# Export all public interfaces
__all__ = [
    'ProductionNewsScraper',
    'ScrapingResult', 
    'RateLimitConfig',
    'create_scraper',
    'validate_url',
    'is_news_url',
    'extract_domain'
]


# Example usage and testing
if __name__ == "__main__":
    # Example usage with enhanced configuration
    scraper = create_scraper({
        'requests_per_minute': 10,
        'base_delay': 1.5,
        'respect_robots': True,
        'adaptive_delay': True
    })
    
    test_urls = [
        'https://www.bbc.com/news/world-asia-india-52002734',
        'https://edition.cnn.com/2024/01/15/world/example/index.html',
        'https://www.reuters.com/world/sample-article-2024-01-15/',
    ]
    
    for url in test_urls:
        print(f"\n--- Scraping: {url} ---")
        result = scraper.scrape_article(url, require_quality_threshold=0.3)
        
        if result.success:
            print(f"✅ Success - Method: {result.method}")
            print(f"Title: {result.title[:100]}..." if len(result.title) > 100 else f"Title: {result.title}")
            print(f"Content: {len(result.text)} chars, {len(result.text.split())} words")
            
            if result.metadata and 'quality_metrics' in result.metadata:
                quality = result.metadata['quality_metrics'].get('overall_score', 0)
                print(f"Quality Score: {quality:.3f}")
        else:
            print(f"❌ Failed: {result.error}")
    
    print(f"\n--- Performance Stats ---")
    stats = scraper.get_stats()
    print(f"Success Rate: {stats['success_rate']}%")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Average Processing Time: {stats['avg_processing_time']}s")
    print(f"Average Quality Score: {stats['avg_quality_score']}")
    print(f"Method Success: {dict(stats['method_success'])}")
