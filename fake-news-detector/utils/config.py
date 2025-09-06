"""
config.py - Part of Fake News Detection System
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration class for the entire fake news detection system
    """
    
    # =================================================================
    # PROJECT STRUCTURE PATHS
    # =================================================================
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model directories
    MODELS_DIR = PROJECT_ROOT / "models"
    SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
    
    # Logging directory
    LOGS_DIR = DATA_DIR / "training_logs"
    
    # Create all directories if they don't exist
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                     MODELS_DIR, SAVED_MODELS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # =================================================================
    # LOGGING CONFIGURATION
    # =================================================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # =================================================================
    # TEXT PREPROCESSING CONFIGURATION
    # =================================================================
    PREPROCESSING_CONFIG = {
        'min_text_length': 50,
        'max_text_length': 5000,
        'target_language': 'en',
        'language_confidence': 0.7,
        'remove_html': True,
        'remove_urls': True,
        'remove_mentions': True,
        'remove_hashtags': False,
        'normalize_whitespace': True,
        'lowercase': True,
        'min_word_count': 10,
        'max_duplicate_ratio': 0.1,
    }
    
    # =================================================================
    # WEB SCRAPER CONFIGURATION
    # =================================================================
    SCRAPER_CONFIG = {
        'request_delay': 2.0,
        'timeout': 30,
        'max_retries': 3,
        'backoff_factor': 2,
        'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'reliable_sources': [
            'reuters.com',
            'bbc.com',
            'apnews.com',
            'npr.org',
            'theguardian.com',
            'cnn.com',
            'abcnews.go.com'
        ],
        'min_article_length': 200,
        'max_article_length': 10000,
        'content_selectors': {
            'default': {
                'title': ['h1', '.headline', '.title', '.article-title'],
                'content': ['article', '.content', '.story-body', 'main', '.article-body'],
                'date': ['.date', '.published', 'time', '.article-date']
            }
        },
        'requests_per_minute': 30,
        'concurrent_requests': 3,
        'save_raw_html': False,
        'batch_size': 10,
    }
    
    # =================================================================
    # API CONFIGURATION
    # =================================================================
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # =================================================================
    # UTILITY METHODS
    # =================================================================
    
    @classmethod
    def get_scraper_paths(cls) -> Dict[str, Path]:
        """Returns file paths for web scraper data storage"""
        return {
            'scraped_data': cls.RAW_DATA_DIR / "scraped_articles.csv",
            'scraper_logs': cls.LOGS_DIR / "scraper.log",
            'failed_urls': cls.LOGS_DIR / "failed_urls.json"
        }
    
    @classmethod
    def get_dual_dataset_paths(cls) -> Dict[str, Path]:
        """Returns paths for both True.csv and Fake.csv files from Kaggle dataset"""
        return {
            'true_file': cls.RAW_DATA_DIR / "True.csv",
            'fake_file': cls.RAW_DATA_DIR / "Fake.csv"
        }
    
    @classmethod
    def validate_dual_setup(cls) -> Dict[str, bool]:
        """Enhanced setup validation for dual dataset structure"""
        dual_paths = cls.get_dual_dataset_paths()
        
        status = {
            'directories_exist': all(d.exists() for d in [
                cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
                cls.MODELS_DIR, cls.SAVED_MODELS_DIR, cls.LOGS_DIR
            ]),
            'true_csv_exists': dual_paths['true_file'].exists(),
            'fake_csv_exists': dual_paths['fake_file'].exists(),
            'both_datasets_exist': dual_paths['true_file'].exists() and dual_paths['fake_file'].exists(),
            'api_keys_configured': bool(cls.GEMINI_API_KEY or cls.OPENAI_API_KEY),
            'gemini_key_exists': bool(cls.GEMINI_API_KEY),
            'openai_key_exists': bool(cls.OPENAI_API_KEY),
        }
        return status
    
    @classmethod
    def print_setup_status(cls):
        """Print a detailed setup status report for debugging"""
        print("🔍 Project Setup Status Report")
        print("=" * 50)
        
        status = cls.validate_dual_setup()
        
        # Directory structure
        print("📁 Directory Structure:")
        if status['directories_exist']:
            print("  ✅ All required directories exist")
        else:
            print("  ❌ Some directories are missing")
        
        # Dataset availability
        print("\n📊 Dataset Availability:")
        if status['both_datasets_exist']:
            print("  ✅ Dual dataset files found (True.csv + Fake.csv)")
        else:
            print("  ❌ No dataset files found")
        
        # API configuration
        print("\n🔑 API Configuration:")
        if status['api_keys_configured']:
            api_details = []
            if status['gemini_key_exists']:
                api_details.append("Google Gemini")
            if status['openai_key_exists']:
                api_details.append("OpenAI")
            print(f"  ✅ API keys configured: {', '.join(api_details)}")
        else:
            print("  ⚠️  No API keys configured")
        
        print("=" * 50)
