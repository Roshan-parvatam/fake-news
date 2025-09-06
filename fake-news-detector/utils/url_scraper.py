"""
Advanced News Article Scraper - Production Ready
Combines newspaper3k, Playwright, and custom extraction for maximum coverage.
"""

import asyncio
import re
import time
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import html2text

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


class RateLimiter:
    """Smart rate limiter with domain-specific controls."""
    
    def __init__(self, requests_per_minute: int = 15, base_delay: float = 1.0):
        self.rpm = requests_per_minute
        self.delay = base_delay
        self.domain_requests = {}
    
    def _clean_old_requests(self, domain: str):
        """Remove requests older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        if domain in self.domain_requests:
            self.domain_requests[domain] = [
                t for t in self.domain_requests[domain] if t > cutoff
            ]
    
    def wait_if_needed(self, url: str):
        """Wait if necessary to respect rate limits."""
        domain = urlparse(url).netloc
        self._clean_old_requests(domain)
        
        if domain not in self.domain_requests:
            self.domain_requests[domain] = []
        
        # Check if we need to wait
        if len(self.domain_requests[domain]) >= self.rpm:
            time.sleep(self.delay * 2)  # Wait longer if hitting limits
        
        # Record this request
        self.domain_requests[domain].append(datetime.now())
        time.sleep(self.delay)


class ContentExtractor:
    """Advanced content extraction with multiple strategies."""
    
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.h2t.body_width = 0
    
    def extract_from_soup(self, soup: BeautifulSoup, url: str) -> Dict[str, Optional[str]]:
        """Extract content using BeautifulSoup with intelligent selectors."""
        
        # Remove noise elements
        for unwanted in soup.select('script, style, nav, header, footer, aside, .ad, .advertisement, .social-share, .comments'):
            unwanted.decompose()
        
        # Extract title with priority selectors
        title = self._extract_title(soup)
        
        # Extract main content
        content = self._extract_content(soup)
        
        # Extract publish date
        date = self._extract_date(soup)
        
        return {
            'title': title,
            'content': content,
            'date': date,
            'url': url,
            'domain': urlparse(url).netloc
        }
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article title using multiple strategies."""
        selectors = [
            'h1.headline', 'h1.title', 'h1[class*="headline"]',
            '.article-title', '.story-headline', '.entry-title',
            'h1', 'title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                # Clean up title (remove site name suffixes)
                for sep in [' | ', ' - ', ' :: ']:
                    if sep in title:
                        title = title.split(sep)[0].strip()
                        break
                
                if 10 <= len(title) <= 200:
                    return title
        return None
    
    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main article content with fallback strategies."""
        
        # Priority selectors for content
        content_selectors = [
            'article', '[role="main"]', '.article-body', '.story-body',
            '.post-content', '.entry-content', '.article-content',
            'main', '.content-body', '#content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                text = self.h2t.handle(str(element))
                cleaned = self._clean_text(text)
                if self._validate_content(cleaned):
                    return cleaned
        
        # Fallback: combine all paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            texts = []
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 30:  # Minimum paragraph length
                    texts.append(text)
            
            if texts:
                combined = '\n\n'.join(texts)
                cleaned = self._clean_text(combined)
                if self._validate_content(cleaned):
                    return cleaned
        
        return None
    
    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date using multiple methods."""
        
        # Try time elements first
        for time_elem in soup.find_all('time'):
            if time_elem.get('datetime'):
                return time_elem['datetime']
        
        # Try meta tags
        meta_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publish-date"]',
            'meta[name="date"]',
            'meta[name="dcterms.created"]'
        ]
        
        for selector in meta_selectors:
            meta = soup.select_one(selector)
            if meta and meta.get('content'):
                return meta['content']
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text of artifacts."""
        if not text:
            return ""
        
        # Remove markdown artifacts
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n[ \t]+', '\n', text)
        
        return text.strip()
    
    def _validate_content(self, text: str) -> bool:
        """Validate content quality."""
        if not text:
            return False
        
        words = text.split()
        if len(words) < 50:
            return False
        
        # Check alphabetic content ratio
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.6:
            return False
        
        # Check for error messages
        error_patterns = [
            r'404|not found|page not found',
            r'access denied|forbidden',
            r'please enable javascript',
            r'subscription required'
        ]
        
        text_lower = text.lower()
        for pattern in error_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True


class NewsArticleScraper:
    """Production-ready news article scraper with multiple extraction methods."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.extractor = ContentExtractor()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
    
    def scrape_article(self, url: str) -> Dict[str, Any]:
        """
        Main scraping method with intelligent fallback strategies.
        
        Returns:
            Dict with keys: success, text, title, author, publish_date, url, method
        """
        if not self._validate_url(url):
            return {'success': False, 'error': 'Invalid URL format'}
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed(url)
        
        # Strategy 1: newspaper3k (fastest, handles most sites)
        if NEWSPAPER_AVAILABLE:
            result = self._scrape_with_newspaper(url)
            if result['success'] and len(result.get('text', '').split()) >= 50:
                return result
        
        # Strategy 2: Playwright (for JavaScript sites)
        if PLAYWRIGHT_AVAILABLE:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self._scrape_with_playwright(url))
                loop.close()
                
                if result['success'] and len(result.get('text', '').split()) >= 50:
                    return result
            except Exception:
                pass
        
        # Strategy 3: Custom HTTP + BeautifulSoup
        return self._scrape_with_requests(url)
    
    def _scrape_with_newspaper(self, url: str) -> Dict[str, Any]:
        """Scrape using newspaper3k library."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if len(article.text.strip()) < 100:
                return {'success': False, 'error': 'Content too short'}
            
            return {
                'success': True,
                'text': article.text.strip(),
                'title': article.title.strip() if article.title else '',
                'author': ', '.join(article.authors) if article.authors else '',
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'url': url,
                'method': 'newspaper3k'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Newspaper3k failed: {str(e)}'}
    
    async def _scrape_with_playwright(self, url: str) -> Dict[str, Any]:
        """Scrape using Playwright for JavaScript-heavy sites."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
                
                context = await browser.new_context(
                    user_agent=self.headers['User-Agent'],
                    viewport={'width': 1920, 'height': 1080}
                )
                
                page = await context.new_page()
                
                # Navigate with timeout
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(3000)  # Wait for dynamic content
                
                # Extract title
                title = await page.title()
                
                # Extract content using multiple selectors
                content = ""
                selectors = [
                    'article', '[role="main"]', '.article-body', '.story-body',
                    '.post-content', 'main'
                ]
                
                for selector in selectors:
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
                
                await browser.close()
                
                if len(content.strip()) < 100:
                    return {'success': False, 'error': 'Insufficient content'}
                
                return {
                    'success': True,
                    'text': content.strip(),
                    'title': title.strip(),
                    'author': '',
                    'publish_date': None,
                    'url': url,
                    'method': 'playwright'
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Playwright failed: {str(e)}'}
    
    def _scrape_with_requests(self, url: str) -> Dict[str, Any]:
        """Fallback scraper using requests + BeautifulSoup."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            data = self.extractor.extract_from_soup(soup, url)
            
            if not data.get('content'):
                return {'success': False, 'error': 'No content extracted'}
            
            return {
                'success': True,
                'text': data['content'],
                'title': data.get('title', ''),
                'author': '',
                'publish_date': data.get('date'),
                'url': url,
                'method': 'requests'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Requests method failed: {str(e)}'}
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format and basic accessibility."""
        try:
            parsed = urlparse(url)
            return all([
                parsed.scheme in ['http', 'https'],
                parsed.netloc,
                '.' in parsed.netloc,
                not parsed.netloc.startswith('.'),
                len(parsed.netloc) > 3
            ])
        except:
            return False


# Utility functions for backwards compatibility
def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
    except:
        return False


def is_news_url(url: str) -> bool:
    """Detect if URL appears to be a news article."""
    patterns = [
        r'/news/', r'/article/', r'/story/', r'/post/',
        r'/\d{4}/\d{2}/\d{2}/', r'/\d{4}/\d{2}/',
        r'/(politics|world|business|technology|health|sports|breaking|latest)/',
    ]
    
    url_lower = url.lower()
    return any(re.search(pattern, url_lower) for pattern in patterns)


def extract_domain(url: str) -> str:
    """Extract clean domain from URL."""
    try:
        domain = urlparse(url).netloc.lower()
        return domain[4:] if domain.startswith('www.') else domain
    except:
        return 'unknown'


# Example usage and testing
def test_scraper():
    """Test the scraper with various URLs."""
    scraper = NewsArticleScraper()
    
    test_urls = [
        'https://timesofindia.indiatimes.com/india/pm-modi-announces-extension-of-lockdown-till-may-3/articleshow/75005449.cms',
        'https://edition.cnn.com/2024/01/15/world/example-news-article/index.html',
        'https://www.bbc.com/news/world-asia-india-52002734'
    ]
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n--- Test {i}: {url} ---")
        result = scraper.scrape_article(url)
        
        if result['success']:
            print(f"✅ Success - Method: {result['method']}")
            print(f"Title: {result.get('title', 'No title')[:80]}...")
            print(f"Content: {len(result.get('text', ''))} characters")
            print(f"Preview: {result.get('text', '')[:150]}...")
        else:
            print(f"❌ Failed: {result['error']}")


if __name__ == "__main__":
    test_scraper()
