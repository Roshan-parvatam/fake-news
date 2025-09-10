"""
Production-Ready Advanced News Article Scraper

Multi-strategy scraper with comprehensive error handling, rate limiting,
and content validation for maximum news source coverage.

Features:
- Multiple extraction strategies (newspaper3k, Playwright, custom BeautifulSoup)
- Intelligent rate limiting with domain-specific controls
- Content validation and quality assessment
- Async support for high-performance scraping
- Comprehensive error handling and recovery
- User-agent rotation and anti-detection measures
"""

import asyncio
import re
import time
import logging
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import html2text
from dataclasses import dataclass
import hashlib
import json

# Optional dependencies with graceful fallbacks
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Standardized scraping result structure"""
    success: bool
    text: str = ""
    title: str = ""
    author: str = ""
    publish_date: Optional[str] = None
    url: str = ""
    method: str = ""
    error: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EnhancedRateLimiter:
    """
    Advanced rate limiter with domain-specific controls and adaptive delays.
    
    Features:
    - Per-domain request tracking
    - Adaptive delay based on response times
    - Burst request handling
    - Respectful crawling patterns
    """
    
    def __init__(self, requests_per_minute: int = 12, base_delay: float = 2.0, 
                 burst_limit: int = 3, adaptive_delay: bool = True):
        self.rpm = requests_per_minute
        self.base_delay = base_delay
        self.burst_limit = burst_limit
        self.adaptive_delay = adaptive_delay
        
        # Domain-specific tracking
        self.domain_requests = {}
        self.domain_response_times = {}
        self.domain_errors = {}
        
        # User agent rotation
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        self.current_ua_index = 0

    def get_user_agent(self) -> str:
        """Rotate user agents to avoid detection"""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua

    def _clean_old_requests(self, domain: str):
        """Remove requests older than tracking window"""
        cutoff = datetime.now() - timedelta(minutes=1)
        if domain in self.domain_requests:
            self.domain_requests[domain] = [
                t for t in self.domain_requests[domain] if t > cutoff
            ]

    def _calculate_adaptive_delay(self, domain: str) -> float:
        """Calculate delay based on domain response patterns"""
        if not self.adaptive_delay or domain not in self.domain_response_times:
            return self.base_delay
        
        # Get recent response times
        recent_times = self.domain_response_times[domain][-10:]
        if not recent_times:
            return self.base_delay
        
        avg_response_time = sum(recent_times) / len(recent_times)
        
        # Adjust delay based on response time
        if avg_response_time > 5.0:  # Slow responses
            return self.base_delay * 2
        elif avg_response_time < 1.0:  # Fast responses
            return max(self.base_delay * 0.7, 1.0)
        
        return self.base_delay

    def wait_if_needed(self, url: str) -> Dict[str, Any]:
        """
        Wait if necessary to respect rate limits and return request headers
        
        Returns:
            Dictionary with headers including rotated user agent
        """
        domain = urlparse(url).netloc
        self._clean_old_requests(domain)
        
        if domain not in self.domain_requests:
            self.domain_requests[domain] = []
        
        # Check if we need to wait
        recent_requests = len(self.domain_requests[domain])
        if recent_requests >= self.rpm:
            delay = self._calculate_adaptive_delay(domain) * 2
            logger.warning(f"Rate limit reached for {domain}, waiting {delay:.1f}s")
            time.sleep(delay)
        elif recent_requests >= self.burst_limit:
            delay = self._calculate_adaptive_delay(domain)
            time.sleep(delay)
        
        # Record this request
        self.domain_requests[domain].append(datetime.now())
        
        # Return headers with rotated user agent
        return {
            'User-Agent': self.get_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Cache-Control': 'max-age=0'
        }

    def record_response_time(self, domain: str, response_time: float):
        """Record response time for adaptive delay calculation"""
        if domain not in self.domain_response_times:
            self.domain_response_times[domain] = []
        
        self.domain_response_times[domain].append(response_time)
        
        # Keep only recent response times
        if len(self.domain_response_times[domain]) > 20:
            self.domain_response_times[domain] = self.domain_response_times[domain][-20:]

class AdvancedContentExtractor:
    """
    Advanced content extraction with multiple strategies and quality validation
    
    Features:
    - Intelligent element selection with priority scoring
    - Content quality assessment
    - Metadata extraction (title, date, author)
    - Noise removal and content cleaning
    - Readability analysis
    """
    
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.h2t.body_width = 0
        
        # Content quality thresholds
        self.min_content_length = 200
        self.min_word_count = 50
        self.min_sentences = 3

    def extract_from_soup(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content using intelligent BeautifulSoup selectors"""
        
        # Remove noise elements
        self._remove_noise_elements(soup)
        
        # Extract components
        title = self._extract_title(soup)
        content = self._extract_content(soup)
        date = self._extract_date(soup)
        author = self._extract_author(soup)
        
        # Validate and clean content
        if content:
            content = self._clean_content(content)
            quality_score = self._assess_content_quality(content)
        else:
            quality_score = 0.0
        
        return {
            'title': title,
            'content': content,
            'date': date,
            'author': author,
            'url': url,
            'domain': urlparse(url).netloc,
            'quality_score': quality_score,
            'word_count': len(content.split()) if content else 0,
            'char_count': len(content) if content else 0
        }

    def _remove_noise_elements(self, soup: BeautifulSoup):
        """Remove common noise elements that don't contain article content"""
        noise_selectors = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            '.advertisement', '.ad', '.ads', '.social-share', '.social-media',
            '.comments', '.comment', '.sidebar', '.navigation',
            '.menu', '.popup', '.modal', '.overlay', '.cookie-notice',
            '[class*="share"]', '[class*="social"]', '[class*="ad-"]',
            '[id*="ad-"]', '[class*="advertisement"]'
        ]
        
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article title using priority selectors"""
        title_selectors = [
            'h1.headline', 'h1.title', 'h1[class*="headline"]', 'h1[class*="title"]',
            '.article-title', '.story-headline', '.entry-title', '.post-title',
            '[property="og:title"]', '[name="twitter:title"]',
            'h1', 'title'
        ]
        
        for selector in title_selectors:
            if selector.startswith('['):
                # Meta tag selector
                element = soup.select_one(selector)
                if element:
                    title = element.get('content', '')
            else:
                element = soup.select_one(selector)
                if element:
                    title = element.get_text().strip()
            
            if 'title' in locals() and title:
                # Clean title (remove site name suffixes)
                for separator in [' | ', ' - ', ' :: ', ' • ']:
                    if separator in title:
                        title = title.split(separator)[0].strip()
                        break
                
                # Validate title length
                if 10 <= len(title) <= 200 and not self._is_navigation_text(title):
                    return title
        
        return None

    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main article content with fallback strategies"""
        
        # Priority content selectors
        content_selectors = [
            'article', '[role="main"]', 'main',
            '.article-body', '.story-body', '.post-content', '.entry-content',
            '.article-content', '.content-body', '#content', '.content'
        ]
        
        # Try priority selectors first
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                text = self._extract_text_from_element(element)
                if self._validate_content_quality(text):
                    return text
        
        # Fallback: Smart paragraph aggregation
        return self._extract_from_paragraphs(soup)

    def _extract_text_from_element(self, element) -> str:
        """Extract clean text from HTML element"""
        # Remove remaining noise within the element
        for unwanted in element.select('.ad, .social, .share, .related, .comments'):
            unwanted.decompose()
        
        text = self.h2t.handle(str(element))
        return self._clean_content(text)

    def _extract_from_paragraphs(self, soup: BeautifulSoup) -> Optional[str]:
        """Fallback content extraction by aggregating paragraphs"""
        paragraphs = soup.find_all('p')
        if not paragraphs:
            return None
        
        # Score paragraphs by likely content quality
        scored_paragraphs = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) < 30:
                continue
            
            score = self._score_paragraph(p, text)
            scored_paragraphs.append((score, text))
        
        # Sort by score and take top paragraphs
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
        top_paragraphs = [text for score, text in scored_paragraphs[:20] if score > 0]
        
        if top_paragraphs:
            content = '\n\n'.join(top_paragraphs)
            return self._clean_content(content) if self._validate_content_quality(content) else None
        
        return None

    def _score_paragraph(self, element, text: str) -> float:
        """Score paragraph likelihood of being main content"""
        score = 0.0
        
        # Length score (sweet spot around 100-500 chars)
        length = len(text)
        if 50 <= length <= 1000:
            score += min(length / 100, 5)
        
        # Word count score
        words = text.split()
        if 10 <= len(words) <= 200:
            score += len(words) / 20
        
        # Penalize if likely navigation/UI text
        if self._is_navigation_text(text):
            score -= 10
        
        # Boost if contains typical article indicators
        article_indicators = ['said', 'according to', 'reported', 'announced', 'revealed']
        score += sum(2 for indicator in article_indicators if indicator in text.lower())
        
        # Check parent element context
        parent_classes = ' '.join(element.parent.get('class', []))
        if any(indicator in parent_classes.lower() 
               for indicator in ['article', 'content', 'story', 'post']):
            score += 3
        
        return score

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date using multiple strategies"""
        
        # Try structured data first
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    date = data.get('datePublished') or data.get('dateCreated')
                    if date:
                        return date
            except:
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
            'meta[property="og:publish_date"]'
        ]
        
        for selector in meta_selectors:
            meta = soup.select_one(selector)
            if meta and meta.get('content'):
                return meta['content']
        
        return None

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information"""
        
        # Try meta tags first
        author_selectors = [
            'meta[name="author"]', 'meta[property="article:author"]',
            'meta[name="dc.creator"]', 'meta[name="sailthru.author"]'
        ]
        
        for selector in author_selectors:
            meta = soup.select_one(selector)
            if meta and meta.get('content'):
                return meta['content'].strip()
        
        # Try structured elements
        author_elements = soup.select('.author, .byline, [class*="author"], [class*="byline"]')
        for element in author_elements:
            text = element.get_text().strip()
            if text and len(text) < 100 and 'by' in text.lower():
                # Clean author text
                author = re.sub(r'^by\s+', '', text, flags=re.IGNORECASE).strip()
                if author:
                    return author
        
        return None

    def _clean_content(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove markdown artifacts
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
        text = re.sub(r'#{1,6}\s*', '', text)      # Remove headers
        text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)  # Remove bold/italic
        
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)     # Max 2 consecutive newlines
        text = re.sub(r'[ \t]{2,}', ' ', text)      # Multiple spaces to single
        text = re.sub(r'\n[ \t]+', '\n', text)      # Remove leading whitespace on lines
        
        # Remove common artifacts
        text = re.sub(r'^\s*\n', '', text)          # Leading newlines
        text = re.sub(r'\n\s*$', '', text)          # Trailing newlines
        
        return text.strip()

    def _validate_content_quality(self, text: str) -> bool:
        """Validate if extracted content meets quality standards"""
        if not text or not text.strip():
            return False
        
        words = text.split()
        if len(words) < self.min_word_count:
            return False
        
        if len(text) < self.min_content_length:
            return False
        
        # Check sentence count
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(valid_sentences) < self.min_sentences:
            return False
        
        # Check alphabetic content ratio
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.6:
            return False
        
        # Check for error indicators
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

    def _assess_content_quality(self, text: str) -> float:
        """Assess content quality on a 0-1 scale"""
        if not text:
            return 0.0
        
        score = 0.0
        words = text.split()
        
        # Length score (0-0.3)
        if len(words) >= 100:
            score += 0.3
        elif len(words) >= 50:
            score += 0.2
        elif len(words) >= 25:
            score += 0.1
        
        # Sentence structure score (0-0.2)
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(valid_sentences) >= 5:
            score += 0.2
        elif len(valid_sentences) >= 3:
            score += 0.1
        
        # Content indicators score (0-0.3)
        quality_indicators = [
            'according to', 'reported', 'announced', 'revealed', 'confirmed',
            'study', 'research', 'analysis', 'data', 'expert', 'official'
        ]
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in quality_indicators if indicator in text_lower)
        score += min(indicator_count * 0.05, 0.3)
        
        # Readability score (0-0.2)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        if 4 <= avg_word_length <= 6:  # Good readability range
            score += 0.2
        elif 3 <= avg_word_length <= 7:
            score += 0.1
        
        return min(score, 1.0)

    def _is_navigation_text(self, text: str) -> bool:
        """Check if text is likely navigation/UI element rather than content"""
        text_lower = text.lower()
        nav_indicators = [
            'click here', 'read more', 'subscribe', 'sign up', 'log in',
            'home', 'about', 'contact', 'menu', 'search', 'share',
            'follow us', 'newsletter', 'privacy policy', 'terms of service'
        ]
        return any(indicator in text_lower for indicator in nav_indicators)

class ProductionNewsScraper:
    """
    Production-ready news article scraper with multiple strategies and comprehensive error handling.
    
    Features:
    - Multi-strategy extraction (newspaper3k, Playwright, custom)
    - Intelligent fallback system
    - Advanced rate limiting and respectful crawling
    - Content quality assessment and validation
    - Comprehensive error handling and recovery
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.rate_limiter = EnhancedRateLimiter(
            requests_per_minute=self.config.get('requests_per_minute', 12),
            base_delay=self.config.get('base_delay', 2.0),
            adaptive_delay=self.config.get('adaptive_delay', True)
        )
        self.extractor = AdvancedContentExtractor()
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'method_success': {'newspaper3k': 0, 'playwright': 0, 'requests': 0}
        }

    def scrape_article(self, url: str, timeout: int = 30, 
                      require_quality_threshold: float = 0.5) -> ScrapingResult:
        """
        Main scraping method with intelligent fallback strategies.
        
        Args:
            url: Article URL to scrape
            timeout: Request timeout in seconds
            require_quality_threshold: Minimum quality score required (0.0-1.0)
            
        Returns:
            ScrapingResult with extraction data and metadata
        """
        
        self.stats['total_requests'] += 1
        
        if not self._validate_url(url):
            return ScrapingResult(
                success=False, 
                url=url, 
                error='Invalid URL format'
            )
        
        logger.info(f"Scraping article: {url}")
        
        # Strategy 1: newspaper3k (fastest for supported sites)
        if NEWSPAPER_AVAILABLE:
            result = self._scrape_with_newspaper(url, timeout)
            if result.success and self._meets_quality_threshold(result, require_quality_threshold):
                self.stats['successful_extractions'] += 1
                self.stats['method_success']['newspaper3k'] += 1
                return result
            logger.debug(f"Newspaper3k failed or low quality: {result.error}")
        
        # Strategy 2: Playwright (for JavaScript-heavy sites)
        if PLAYWRIGHT_AVAILABLE:
            try:
                result = asyncio.run(self._scrape_with_playwright(url, timeout))
                if result.success and self._meets_quality_threshold(result, require_quality_threshold):
                    self.stats['successful_extractions'] += 1
                    self.stats['method_success']['playwright'] += 1
                    return result
                logger.debug(f"Playwright failed or low quality: {result.error}")
            except Exception as e:
                logger.debug(f"Playwright strategy failed: {str(e)}")
        
        # Strategy 3: Custom requests + BeautifulSoup (fallback)
        result = self._scrape_with_requests(url, timeout)
        if result.success:
            if self._meets_quality_threshold(result, require_quality_threshold):
                self.stats['successful_extractions'] += 1
                self.stats['method_success']['requests'] += 1
            else:
                result.error = f"Content quality below threshold ({require_quality_threshold})"
                result.success = False
        
        if not result.success:
            self.stats['failed_extractions'] += 1
        
        return result

    def _scrape_with_newspaper(self, url: str, timeout: int) -> ScrapingResult:
        """Scrape using newspaper3k library"""
        try:
            headers = self.rate_limiter.wait_if_needed(url)
            
            article = Article(url)
            article.set_config(timeout=timeout)
            
            start_time = time.time()
            article.download()
            article.parse()
            
            response_time = time.time() - start_time
            domain = urlparse(url).netloc
            self.rate_limiter.record_response_time(domain, response_time)
            
            if not article.text or len(article.text.strip()) < 100:
                return ScrapingResult(
                    success=False,
                    url=url,
                    method='newspaper3k',
                    error='Content too short or empty'
                )
            
            # Calculate quality score
            quality_score = self.extractor._assess_content_quality(article.text)
            
            return ScrapingResult(
                success=True,
                text=article.text.strip(),
                title=article.title.strip() if article.title else '',
                author=', '.join(article.authors) if article.authors else '',
                publish_date=article.publish_date.isoformat() if article.publish_date else None,
                url=url,
                method='newspaper3k',
                metadata={
                    'quality_score': quality_score,
                    'word_count': len(article.text.split()),
                    'response_time': response_time,
                    'top_image': article.top_image,
                    'meta_keywords': article.meta_keywords
                }
            )
            
        except Exception as e:
            return ScrapingResult(
                success=False,
                url=url,
                method='newspaper3k',
                error=f'Newspaper3k extraction failed: {str(e)}'
            )

    async def _scrape_with_playwright(self, url: str, timeout: int) -> ScrapingResult:
        """Scrape using Playwright for JavaScript-heavy sites"""
        try:
            headers = self.rate_limiter.wait_if_needed(url)
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--no-first-run',
                        '--disable-background-timer-throttling'
                    ]
                )
                
                context = await browser.new_context(
                    user_agent=headers['User-Agent'],
                    viewport={'width': 1920, 'height': 1080},
                    ignore_https_errors=True
                )
                
                page = await context.new_page()
                
                start_time = time.time()
                
                # Navigate with comprehensive error handling
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                    
                    # Wait for content to load
                    await page.wait_for_timeout(2000)
                    
                    # Try to wait for article content
                    try:
                        await page.wait_for_selector('article, .article-body, .story-body, main', timeout=5000)
                    except:
                        pass  # Continue if specific selectors not found
                    
                except Exception as nav_error:
                    await browser.close()
                    return ScrapingResult(
                        success=False,
                        url=url,
                        method='playwright',
                        error=f'Navigation failed: {str(nav_error)}'
                    )
                
                response_time = time.time() - start_time
                domain = urlparse(url).netloc
                self.rate_limiter.record_response_time(domain, response_time)
                
                # Extract title
                title = await page.title()
                
                # Extract content using multiple selectors
                content = ""
                content_selectors = [
                    'article', '[role="main"]', 'main',
                    '.article-body', '.story-body', '.post-content', '.entry-content'
                ]
                
                for selector in content_selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            text = await element.inner_text()
                            if len(text.split()) > 50:
                                content = text
                                break
                    except:
                        continue
                
                # Fallback to paragraphs
                if not content:
                    try:
                        paragraphs = await page.query_selector_all('p')
                        texts = []
                        for p in paragraphs:
                            try:
                                text = await p.inner_text()
                                if len(text.strip()) > 30:
                                    texts.append(text.strip())
                            except:
                                continue
                        content = '\n\n'.join(texts)
                    except:
                        pass
                
                await browser.close()
                
                if not content or len(content.strip()) < 100:
                    return ScrapingResult(
                        success=False,
                        url=url,
                        method='playwright',
                        error='Insufficient content extracted'
                    )
                
                # Clean and assess content
                cleaned_content = self.extractor._clean_content(content)
                quality_score = self.extractor._assess_content_quality(cleaned_content)
                
                return ScrapingResult(
                    success=True,
                    text=cleaned_content,
                    title=title.strip(),
                    url=url,
                    method='playwright',
                    metadata={
                        'quality_score': quality_score,
                        'word_count': len(cleaned_content.split()),
                        'response_time': response_time
                    }
                )
                
        except Exception as e:
            return ScrapingResult(
                success=False,
                url=url,
                method='playwright',
                error=f'Playwright extraction failed: {str(e)}'
            )

    def _scrape_with_requests(self, url: str, timeout: int) -> ScrapingResult:
        """Fallback scraper using requests + BeautifulSoup"""
        try:
            headers = self.rate_limiter.wait_if_needed(url)
            
            start_time = time.time()
            
            response = requests.get(
                url, 
                headers=headers, 
                timeout=timeout,
                allow_redirects=True,
                stream=False
            )
            response.raise_for_status()
            
            response_time = time.time() - start_time
            domain = urlparse(url).netloc
            self.rate_limiter.record_response_time(domain, response_time)
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract using advanced content extractor
            extracted_data = self.extractor.extract_from_soup(soup, url)
            
            if not extracted_data.get('content') or not self.extractor._validate_content_quality(extracted_data['content']):
                return ScrapingResult(
                    success=False,
                    url=url,
                    method='requests',
                    error='No valid content extracted or content failed quality validation'
                )
            
            return ScrapingResult(
                success=True,
                text=extracted_data['content'],
                title=extracted_data.get('title', ''),
                author=extracted_data.get('author', ''),
                publish_date=extracted_data.get('date'),
                url=url,
                method='requests',
                metadata={
                    'quality_score': extracted_data.get('quality_score', 0.0),
                    'word_count': extracted_data.get('word_count', 0),
                    'response_time': response_time,
                    'domain': extracted_data.get('domain', ''),
                    'char_count': extracted_data.get('char_count', 0)
                }
            )
            
        except requests.exceptions.RequestException as e:
            return ScrapingResult(
                success=False,
                url=url,
                method='requests',
                error=f'Request failed: {str(e)}'
            )
        except Exception as e:
            return ScrapingResult(
                success=False,
                url=url,
                method='requests',
                error=f'Extraction failed: {str(e)}'
            )

    def _validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility"""
        try:
            parsed = urlparse(url)
            return all([
                parsed.scheme in ['http', 'https'],
                parsed.netloc,
                '.' in parsed.netloc,
                not parsed.netloc.startswith('.'),
                len(parsed.netloc) > 3,
                not any(blocked in parsed.netloc.lower() 
                       for blocked in ['localhost', '127.0.0.1', '0.0.0.0'])
            ])
        except:
            return False

    def _meets_quality_threshold(self, result: ScrapingResult, threshold: float) -> bool:
        """Check if scraping result meets quality threshold"""
        if not result.success or not result.text:
            return False
        
        quality_score = result.metadata.get('quality_score', 0.0) if result.metadata else 0.0
        return quality_score >= threshold

    def get_stats(self) -> Dict[str, Any]:
        """Get scraper performance statistics"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        success_rate = (self.stats['successful_extractions'] / total) * 100
        
        return {
            **self.stats,
            'success_rate': round(success_rate, 2),
            'failure_rate': round(100 - success_rate, 2),
            'method_preference_order': sorted(
                self.stats['method_success'].items(),
                key=lambda x: x[1],
                reverse=True
            )
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'method_success': {'newspaper3k': 0, 'playwright': 0, 'requests': 0}
        }

# Utility functions for backward compatibility and convenience

def validate_url(url: str) -> bool:
    """Validate URL format - utility function"""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
    except:
        return False

def is_news_url(url: str) -> bool:
    """Detect if URL appears to be a news article"""
    news_patterns = [
        r'/news/', r'/article/', r'/story/', r'/post/', r'/blog/',
        r'/\d{4}/\d{2}/\d{2}/', r'/\d{4}/\d{2}/',
        r'/(politics|world|business|technology|health|sports|breaking|latest|opinion)/',
        r'/(local|national|international|breaking-news)/'
    ]
    
    url_lower = url.lower()
    return any(re.search(pattern, url_lower) for pattern in news_patterns)

def extract_domain(url: str) -> str:
    """Extract clean domain from URL"""
    try:
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix
        return domain[4:] if domain.startswith('www.') else domain
    except:
        return 'unknown'

def create_scraper(config: Optional[Dict[str, Any]] = None) -> ProductionNewsScraper:
    """Factory function to create configured scraper instance"""
    default_config = {
        'requests_per_minute': 12,
        'base_delay': 2.0,
        'adaptive_delay': True,
        'quality_threshold': 0.5
    }
    
    if config:
        default_config.update(config)
    
    return ProductionNewsScraper(default_config)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    scraper = create_scraper({
        'requests_per_minute': 10,
        'base_delay': 1.5
    })
    
    test_urls = [
        'https://www.bbc.com/news/world-asia-india-52002734',
        'https://edition.cnn.com/2024/01/15/world/example/index.html'
    ]
    
    for url in test_urls:
        print(f"\n--- Scraping: {url} ---")
        result = scraper.scrape_article(url)
        
        if result.success:
            print(f"✅ Success - Method: {result.method}")
            print(f"Title: {result.title[:80]}...")
            print(f"Content: {len(result.text)} chars")
            print(f"Quality: {result.metadata.get('quality_score', 0):.2f}")
        else:
            print(f"❌ Failed: {result.error}")
    
    print(f"\n--- Performance Stats ---")
    stats = scraper.get_stats()
    print(f"Success Rate: {stats['success_rate']}%")
    print(f"Total Requests: {stats['total_requests']}")
