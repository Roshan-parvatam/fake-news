"""
Enhanced Data Preprocessing Agent for Fake News Detection System
Updated to handle the dual-file Kaggle dataset structure (True.csv + Fake.csv)

THEORY: Multi-File Dataset Handling
The Kaggle fake news dataset consists of two separate files:
- True.csv: Legitimate news articles from Reuters (12,600+ articles)
- Fake.csv: Fake news articles from various unreliable sources (12,600+ articles)

This structure requires:
1. Loading both files separately to handle potential differences
2. Adding appropriate labels (0=real, 1=fake) to distinguish the classes
3. Combining them into a single dataset for unified processing
4. Handling potential column differences between files gracefully

WHY PREPROCESSING IS CRITICAL:
- Raw text contains noise that hurts model performance
- Inconsistent formatting confuses machine learning algorithms
- Language mixing reduces model accuracy
- Poor quality data leads to poor model predictions
- Balanced datasets prevent model bias toward majority class
"""

# =================================================================
# IMPORTS: External Libraries and Internal Modules
# =================================================================

import re  # Regular expressions for pattern matching and text cleaning
import pandas as pd  # Data manipulation and analysis (DataFrames)
import numpy as np  # Numerical operations and mathematical functions
from typing import Dict, List, Tuple, Optional, Any  # Type hints for better code documentation
from pathlib import Path  # Modern, cross-platform file path handling
import html  # HTML entity decoding (e.g., &amp; -> &)
from urllib.parse import urlparse  # URL parsing and validation
import langdetect  # Statistical language detection library
from langdetect.lang_detect_exception import LangDetectException  # Handle language detection errors
import nltk  # Natural Language Toolkit - comprehensive NLP library
from nltk.corpus import stopwords  # Common words that usually don't carry semantic meaning
from nltk.tokenize import word_tokenize  # Split text into individual words/tokens
import spacy  # Industrial-strength NLP library with advanced features
from sklearn.model_selection import train_test_split  # Split datasets for ML workflows
from sklearn.utils import resample  # Resampling utilities for dataset balancing

# Import our custom configuration and logging systems
from utils.config import Config  # Centralized configuration management
from utils.logger import setup_logger  # Professional logging system

class DataPreprocessorAgent:
    """
    Enhanced preprocessing agent that handles the dual-file Kaggle dataset structure
    
    THEORY: Agent-Based Architecture
    In software engineering, an "agent" is a component that:
    - Has a specific, well-defined responsibility (Single Responsibility Principle)
    - Can operate independently of other components
    - Provides a clear interface for interaction
    - Can be tested, debugged, and modified in isolation
    
    This preprocessing agent is responsible for:
    1. Loading raw data from various sources (single file, dual files)
    2. Cleaning and normalizing text data
    3. Validating data quality
    4. Balancing class distributions
    5. Splitting data for machine learning workflows
    6. Saving processed data for downstream use
    
    Key Enhancements for Dual-File Support:
    1. Separate loading methods for True.csv and Fake.csv
    2. Automatic label assignment (0=real, 1=fake)
    3. Dataset combination with validation
    4. Enhanced column standardization for both file formats
    5. Comprehensive statistics tracking
    """
    
    def __init__(self):
        """
        Initialize the preprocessing agent with necessary resources
        
        THEORY: Initialization Strategy
        During initialization, we:
        1. Set up logging for debugging and monitoring
        2. Load configuration parameters
        3. Initialize heavy NLP resources (download if needed)
        4. Set up statistics tracking
        5. Fail fast if critical resources are unavailable
        
        This front-loads the expensive operations so they don't slow down
        processing later, and ensures all dependencies are available.
        """
        
        # Set up logging system with file output
        # __name__ gives us the module name for clear log identification
        self.logger = setup_logger(__name__, "preprocessing.log")
        
        # Load preprocessing configuration from centralized config
        # This allows easy parameter tuning without code changes
        self.config = Config.PREPROCESSING_CONFIG
        
        # Initialize NLP resources (NLTK, spaCy)
        # This may take time on first run as it downloads required data
        self._initialize_nlp_resources()
        
        # Enhanced statistics tracking for dual datasets
        # These counters help us understand data quality and processing effectiveness
        self.stats = {
            'original_count': 0,          # Total records at start
            'true_news_count': 0,         # Records from True.csv
            'fake_news_count': 0,         # Records from Fake.csv
            'after_cleaning': 0,          # Records after text cleaning
            'after_language_filter': 0,   # Records after language filtering
            'after_quality_filter': 0,    # Records after quality validation
            'final_count': 0,             # Final processed records
            'removed_html': 0,            # Texts with HTML tags removed
            'removed_urls': 0,            # Texts with URLs removed
            'removed_short_texts': 0,     # Texts removed for being too short
            'removed_non_english': 0,     # Non-English texts removed
        }
        
        self.logger.info("Enhanced DataPreprocessorAgent initialized successfully")
    
    def _initialize_nlp_resources(self):
        """
        Initialize NLTK and spaCy resources with automatic downloads
        
        THEORY: Resource Management in NLP
        NLP libraries require large data files (models, corpora, dictionaries):
        - NLTK: Tokenization models, part-of-speech taggers, word lists
        - spaCy: Language models with word vectors and parsing capabilities
        
        Best practices:
        1. Check if resources exist before downloading
        2. Provide clear error messages for missing resources
        3. Handle download failures gracefully
        4. Initialize resources once and reuse them
        
        NLTK Resources we use:
        - punkt: Pre-trained sentence and word tokenization models
        - stopwords: Lists of common words in various languages
        - wordnet: English lexical database for semantic relationships
        - averaged_perceptron_tagger: Part-of-speech tagging model
        
        spaCy Resources:
        - en_core_web_sm: Small English language model (50MB)
        - Includes tokenization, POS tagging, parsing, NER
        """
        try:
            # Download required NLTK data if not already present
            nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            
            for resource in nltk_downloads:
                try:
                    # Check if resource already exists to avoid unnecessary downloads
                    nltk.data.find(f'tokenizers/{resource}')
                    self.logger.debug(f"NLTK resource already available: {resource}")
                except LookupError:
                    # Resource not found, download it
                    self.logger.info(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource, quiet=True)  # quiet=True suppresses verbose output
            
            # Initialize spaCy English language model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("spaCy model loaded successfully")
            except OSError:
                # spaCy model not installed - provide helpful error message
                self.logger.warning(
                    "spaCy 'en_core_web_sm' model not found. "
                    "Please install it with: python -m spacy download en_core_web_sm"
                )
                self.nlp = None  # Set to None so we can check later
            
            # Load English stopwords for text analysis
            # Stopwords are common words (the, and, or, etc.) that usually don't carry meaning
            # We use them for analysis but not for filtering (BERT can handle them)
            self.stop_words = set(stopwords.words('english'))
            self.logger.debug(f"Loaded {len(self.stop_words)} English stopwords")
            
        except Exception as e:
            # Log the error and re-raise to fail fast
            self.logger.error(f"Error initializing NLP resources: {str(e)}")
            raise  # Re-raise the exception to stop initialization
    
    def load_dual_kaggle_dataset(self, data_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Load both True.csv and Fake.csv from the Kaggle dataset
        
        THEORY: Dual-File Dataset Loading Strategy
        The Kaggle fake news dataset is structured as two separate files:
        - True.csv: Contains 12,600+ legitimate news articles from Reuters
        - Fake.csv: Contains 12,600+ fake news articles from various sources
        
        Why separate files?
        1. Clear source separation (real vs fake)
        2. Easier to verify data quality
        3. Allows for different processing if needed
        4. Maintains data provenance
        
        Both files have the same structure:
        - title: Article headline/title
        - text: Main article content
        - subject: News category (politics, world news, etc.)
        - date: Publication date
        
        Our process:
        1. Load each file separately
        2. Add labels (0=real, 1=fake)
        3. Combine into single DataFrame
        4. Shuffle to mix real and fake articles
        5. Standardize column names
        
        Args:
            data_dir: Directory containing True.csv and Fake.csv files
                     If None, uses Config.RAW_DATA_DIR
        
        Returns:
            Combined pandas DataFrame with both real and fake news
            
        Raises:
            FileNotFoundError: If either True.csv or Fake.csv is missing
            ValueError: If files don't have expected structure
        """
        
        # Use default data directory if none provided
        if data_dir is None:
            data_dir = Config.RAW_DATA_DIR
        
        # Define expected file paths
        true_file = data_dir / "True.csv"
        fake_file = data_dir / "Fake.csv"
        
        self.logger.info(f"Loading dual dataset from: {data_dir}")
        
        # Check if both files exist before attempting to load
        # This provides clear error messages for missing files
        missing_files = []
        if not true_file.exists():
            missing_files.append("True.csv")
        if not fake_file.exists():
            missing_files.append("Fake.csv")
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files: {missing_files}. "
                f"Please download the Kaggle dataset and extract both CSV files to {data_dir}"
            )
        
        try:
            # =================================================================
            # LOAD TRUE.CSV (REAL NEWS)
            # =================================================================
            self.logger.info("Loading True.csv (real news)...")
            true_df = pd.read_csv(true_file)
            
            # Add label column: 0 = Real news
            # This is the standard binary classification convention
            true_df['label'] = 0
            
            # Update statistics
            self.stats['true_news_count'] = len(true_df)
            self.logger.info(f"Loaded {len(true_df):,} real news articles")
            
            # =================================================================
            # LOAD FAKE.CSV (FAKE NEWS)
            # =================================================================
            self.logger.info("Loading Fake.csv (fake news)...")
            fake_df = pd.read_csv(fake_file)
            
            # Add label column: 1 = Fake news
            fake_df['label'] = 1
            
            # Update statistics
            self.stats['fake_news_count'] = len(fake_df)
            self.logger.info(f"Loaded {len(fake_df):,} fake news articles")
            
            # =================================================================
            # VALIDATE FILE STRUCTURES
            # =================================================================
            # Display column information for debugging and verification
            self.logger.info(f"True.csv columns: {list(true_df.columns)}")
            self.logger.info(f"Fake.csv columns: {list(fake_df.columns)}")
            
            # Check if both datasets have the same column structure
            # Different structures could indicate data issues
            if list(true_df.columns) != list(fake_df.columns):
                self.logger.warning("Column structures differ between True.csv and Fake.csv")
                self.logger.info(f"True.csv: {true_df.columns.tolist()}")
                self.logger.info(f"Fake.csv: {fake_df.columns.tolist()}")
                # We'll continue processing but log the difference
            
            # =================================================================
            # COMBINE DATASETS
            # =================================================================
            self.logger.info("Combining real and fake news datasets...")
            
            # Concatenate DataFrames vertically (stack rows)
            # ignore_index=True creates a new sequential index
            combined_df = pd.concat([true_df, fake_df], ignore_index=True)
            
            # Shuffle the combined dataset to mix real and fake articles
            # This prevents the model from learning order-based patterns
            # frac=1 means sample 100% of the data (all rows)
            # random_state=42 ensures reproducible shuffling
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # =================================================================
            # STANDARDIZE COLUMN NAMES
            # =================================================================
            # Apply our column standardization logic
            combined_df = self._standardize_dual_dataset_columns(combined_df)
            
            # Update statistics with total count
            self.stats['original_count'] = len(combined_df)
            
            # =================================================================
            # LOG DATASET SUMMARY
            # =================================================================
            self.logger.info(f"Combined dataset created: {len(combined_df):,} total articles")
            
            # Show class distribution to verify balance
            class_distribution = combined_df['label'].value_counts().sort_index()
            self.logger.info(f"Class distribution: {class_distribution.to_dict()}")
            
            # Final validation: ensure we have required columns
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in combined_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns after standardization: {missing_columns}")
            
            self.logger.info("Dual dataset loading completed successfully")
            return combined_df
            
        except Exception as e:
            # Log the error with context and re-raise
            self.logger.error(f"Error loading dual dataset: {str(e)}")
            raise
    
    def _standardize_dual_dataset_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for the dual Kaggle dataset format
        
        THEORY: Column Standardization Strategy
        The Kaggle fake news dataset typically has these columns:
        - title: Article headline (short, attention-grabbing)
        - text: Main article content (detailed information)
        - subject: News category/topic (politics, world news, etc.)
        - date: Publication date (when article was published)
        - label: Our added label (0=real, 1=fake)
        
        Why combine title and text?
        1. Headlines often contain crucial information for fake news detection
        2. Fake news headlines are often more sensational
        3. BERT can handle longer sequences (up to 512 tokens)
        4. More context improves classification accuracy
        
        Processing steps:
        1. Handle missing values (fill with empty strings)
        2. Combine title and text with proper punctuation
        3. Keep useful metadata columns
        4. Remove redundant columns
        5. Ensure consistent naming
        
        Args:
            df: Combined DataFrame from both CSV files
        
        Returns:
            DataFrame with standardized columns and combined text
        """
        
        self.logger.info("Standardizing dataset columns...")
        
        # Log original column structure for debugging
        original_columns = df.columns.tolist()
        self.logger.info(f"Original columns: {original_columns}")
        
        # =================================================================
        # HANDLE TITLE + TEXT COMBINATION
        # =================================================================
        if 'title' in df.columns and 'text' in df.columns:
            self.logger.info("Found both title and text columns - combining them")
            
            # Fill NaN values with empty strings to avoid concatenation issues
            # pandas concat with NaN results in NaN, which we don't want
            df['title'] = df['title'].fillna('')
            df['text'] = df['text'].fillna('')
            
            # Create combined text with proper punctuation
            # Format: "Title. Main article text..."
            # The period ensures proper sentence separation for BERT
            df['combined_text'] = df['title'] + '. ' + df['text']
            
            # Replace the original text column with combined content
            df['text'] = df['combined_text']
            
            # Clean up: remove temporary and original columns
            df = df.drop(columns=['combined_text', 'title'])
            
            self.logger.info("Successfully combined title and text columns")
        
        # =================================================================
        # HANDLE EDGE CASES
        # =================================================================
        elif 'title' in df.columns and 'text' not in df.columns:
            # Only title column exists - use it as main text
            df['text'] = df['title']
            df = df.drop(columns=['title'])
            self.logger.info("Used title as main text (no separate text column found)")
        
        elif 'text' not in df.columns:
            # No text column found - try alternative names
            text_alternatives = ['content', 'article', 'body', 'news']
            for alt in text_alternatives:
                if alt in df.columns:
                    df['text'] = df[alt]
                    self.logger.info(f"Used '{alt}' column as main text")
                    break
            else:
                # No suitable text column found - this is a critical error
                raise ValueError("No suitable text column found in dataset")
        
        # =================================================================
        # PRESERVE USEFUL METADATA
        # =================================================================
        # Keep metadata columns that might be useful for analysis
        metadata_columns = ['subject', 'date']
        available_metadata = [col for col in metadata_columns if col in df.columns]
        
        if available_metadata:
            self.logger.info(f"Preserving metadata columns: {available_metadata}")
        else:
            self.logger.info("No metadata columns found to preserve")
        
        # =================================================================
        # FINALIZE COLUMN STRUCTURE
        # =================================================================
        # Define required columns and optional metadata
        required_columns = ['text', 'label']
        final_columns = required_columns + available_metadata
        
        # Select only the columns we need (drops any other columns)
        df = df[final_columns]
        
        # =================================================================
        # LOG FINAL STRUCTURE AND SAMPLES
        # =================================================================
        self.logger.info(f"Final columns: {df.columns.tolist()}")
        self.logger.info(f"Dataset shape: {df.shape}")
        
        # Display sample data for verification and debugging
        self.logger.info("Sample of standardized data:")
        for i, row in df.head(2).iterrows():  # Show first 2 rows
            self.logger.info(f"  Sample {i+1}:")
            # Truncate text for readable logging
            text_preview = row['text'][:100] + '...' if len(row['text']) > 100 else row['text']
            self.logger.info(f"    Text: '{text_preview}'")
            
            # Show label with human-readable interpretation
            label_text = 'Fake' if row['label'] == 1 else 'Real'
            self.logger.info(f"    Label: {row['label']} ({label_text})")
            
            # Show metadata if available
            if 'subject' in row and pd.notna(row['subject']):
                self.logger.info(f"    Subject: {row['subject']}")
            if 'date' in row and pd.notna(row['date']):
                self.logger.info(f"    Date: {row['date']}")
        
        return df
    
    def load_kaggle_dataset(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Backward compatibility method - intelligently chooses loading strategy
        
        THEORY: Backward Compatibility Design
        This method maintains compatibility with the original interface while
        automatically handling the dual-file structure. This allows existing
        code to work without modification while supporting new features.
        
        Decision logic:
        - If file_path is provided: Use single-file loading (backward compatibility)
        - If file_path is None: Use dual-file loading (new default behavior)
        
        This design pattern is called "graceful evolution" - we enhance
        functionality without breaking existing usage patterns.
        
        Args:
            file_path: If provided, assumes single-file format
                      If None, uses dual-file format
        
        Returns:
            Combined DataFrame ready for processing
        """
        
        if file_path is not None:
            # Single file path provided - use original loading logic
            self.logger.info("Single file path provided, using single-file loading...")
            return self._load_single_file(file_path)
        else:
            # No specific file - use new dual-file loading
            self.logger.info("No specific file provided, using dual-file loading...")
            return self.load_dual_kaggle_dataset()
    
    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single CSV file (backward compatibility method)
        
        THEORY: Supporting Multiple Data Sources
        This method handles cases where:
        1. Someone has already combined the True.csv and Fake.csv files
        2. Using a different dataset with single-file format
        3. Working with custom or preprocessed datasets
        4. Migrating from older dataset versions
        
        The method applies the same validation and standardization as
        the dual-file loader to ensure consistent output format.
        
        Args:
            file_path: Path to single CSV file
        
        Returns:
            DataFrame from single file with standardized columns
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If file doesn't have required columns
        """
        
        self.logger.info(f"Loading single dataset file: {file_path}")
        
        # Validate file existence before attempting to load
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            # Load the CSV file using pandas
            df = pd.read_csv(file_path)
            
            # Log basic information about the loaded file
            self.logger.info(f"Loaded single file: {len(df)} rows, {len(df.columns)} columns")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Apply standardization for backward compatibility
            # This handles various column naming conventions
            df = self._standardize_column_names(df)
            
            # Validate that we have the minimum required columns
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Update statistics
            self.stats['original_count'] = len(df)
            
            self.logger.info("Single file loading completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading single file: {str(e)}")
            raise
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Original column standardization method (for backward compatibility)
        
        THEORY: Handling Dataset Variations
        Different fake news datasets use different column naming conventions:
        - Some use 'news' or 'content' instead of 'text'
        - Labels might be 'fake'/'real', 'FAKE'/'REAL', True/False, 0/1
        - Title columns might be 'headline', 'head', or 'title'
        
        This method normalizes these variations into our standard format:
        - 'text': Main content for classification
        - 'label': Binary label (0=real, 1=fake)
        - 'title': Optional title/headline
        
        Args:
            df: DataFrame with potentially non-standard column names
            
        Returns:
            DataFrame with standardized column names
        """
        
        # Define mappings from various column names to our standard names
        column_mappings = {
            # Text content columns - all map to 'text'
            'news': 'text',       # Some datasets use 'news'
            'content': 'text',    # Common alternative name
            'article': 'text',    # Another common name
            'body': 'text',       # Article body content
            
            # Label columns - all map to 'label'
            'fake': 'label',      # Simple fake/real labeling
            'is_fake': 'label',   # Boolean-style naming
            'class': 'label',     # Generic classification label
            'target': 'label',    # ML-style target variable
            
            # Title columns - keep as 'title'
            'headline': 'title',  # News headline
            'head': 'title',      # Short form of headline
        }
        
        # Apply the column name mappings
        df = df.rename(columns=column_mappings)
        self.logger.debug(f"Applied column mappings: {column_mappings}")
        
        # Handle title + text combination (same as dual dataset method)
        if 'title' in df.columns and 'text' in df.columns:
            # Combine title and text with space separator
            df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            self.logger.info("Combined title and text columns")
        
        # Ensure we have a text column by checking alternatives
        if 'text' not in df.columns:
            text_candidates = ['content', 'article', 'news', 'body']
            for candidate in text_candidates:
                if candidate in df.columns:
                    df['text'] = df[candidate]
                    self.logger.info(f"Using '{candidate}' as text column")
                    break
        
        # Handle different label formats and standardize to 0=real, 1=fake
        if 'label' in df.columns:
            unique_labels = df['label'].unique()
            self.logger.info(f"Original labels found: {unique_labels}")
            
            # Handle various label format conventions
            if set(unique_labels) == {'FAKE', 'REAL'}:
                df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})
                self.logger.info("Converted REAL/FAKE labels to 0/1")
            elif set(unique_labels) == {'fake', 'real'}:
                df['label'] = df['label'].map({'real': 0, 'fake': 1})
                self.logger.info("Converted real/fake labels to 0/1")
            elif set(unique_labels) == {'True', 'False'}:
                df['label'] = df['label'].map({'False': 0, 'True': 1})
                self.logger.info("Converted True/False labels to 0/1")
            # If labels are already 0/1, leave them as is
            
            # Log final label distribution
            final_labels = df['label'].unique()
            self.logger.info(f"Standardized labels: {sorted(final_labels)}")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning function with detailed explanations
        
        THEORY: Text Cleaning in NLP
        Raw text from web sources contains many elements that don't help
        with fake news classification and can actually hurt performance:
        
        1. HTML ENTITIES: &amp;, &lt;, &gt; - remnants from web scraping
        2. HTML TAGS: <p>, <div>, <strong> - markup that adds no semantic value
        3. URLS: http://example.com - usually not relevant for content classification
        4. SOCIAL MEDIA ARTIFACTS: @mentions, #hashtags - depend on context
        5. SPECIAL CHARACTERS: Various Unicode symbols and formatting characters
        6. INCONSISTENT WHITESPACE: Multiple spaces, tabs, newlines
        7. CASE INCONSISTENCY: MiXeD cAsE that confuses tokenization
        
        Our cleaning strategy balances thoroughness with information preservation:
        - Remove clear noise (HTML, URLs)
        - Normalize formatting (whitespace, case)
        - Preserve meaningful punctuation
        - Keep hashtag content (might indicate fake news patterns)
        
        The order of operations matters! For example, we decode HTML entities
        before removing HTML tags to handle cases like &lt;script&gt;.
        
        Args:
            text: Raw text string to clean
        
        Returns:
            Cleaned text string ready for tokenization
        """
        
        # Handle edge cases: non-string input or missing values
        if not isinstance(text, str) or pd.isna(text):
            return ""  # Return empty string for invalid input
        
        # Keep reference to original text for statistics tracking
        original_text = text
        
        # =================================================================
        # STEP 1: DECODE HTML ENTITIES
        # =================================================================
        # Convert HTML entities back to their original characters
        # Examples: &amp; → &, &lt; → <, &gt; → >, &quot; → "
        # This must happen BEFORE removing HTML tags
        text = html.unescape(text)
        
        # =================================================================
        # STEP 2: REMOVE HTML TAGS
        # =================================================================
        if self.config['remove_html']:
            # Regular expression explanation:
            # <       : Match literal '<' character
            # [^>]+   : Match one or more characters that are NOT '>'
            # >       : Match literal '>' character
            # Result: Matches complete HTML tags like <p>, <div>, <strong>
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Track statistics: count texts where HTML was actually removed
            if '<' in original_text and '<' not in text:
                self.stats['removed_html'] += 1
        
        # =================================================================
        # STEP 3: REMOVE URLS
        # =================================================================
        if self.config['remove_urls']:
            # Regular expression for HTTP/HTTPS URLs
            # This is a simplified but effective URL pattern
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            
            # Track statistics before removal
            if re.search(url_pattern, text):
                self.stats['removed_urls'] += 1
            
            # Replace URLs with single space to avoid word concatenation
            text = re.sub(url_pattern, ' ', text)
        
        # =================================================================
        # STEP 4: HANDLE @MENTIONS (Social Media)
        # =================================================================
        if self.config['remove_mentions']:
            # Remove @username patterns (common in Twitter-style data)
            # @\w+ matches @ followed by one or more word characters
            text = re.sub(r'@\w+', ' ', text)
        
        # =================================================================
        # STEP 5: HANDLE HASHTAGS
        # =================================================================
        if self.config['remove_hashtags']:
            # Remove hashtags completely: #fakenews → (removed)
            text = re.sub(r'#\w+', ' ', text)
        else:
            # Keep hashtag content but remove the # symbol
            # #fakenews → fakenews
            # Hashtag content might be informative for fake news detection
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # =================================================================
        # STEP 6: CLEAN SPECIAL CHARACTERS
        # =================================================================
        # Remove characters that aren't letters, numbers, basic punctuation, or spaces
        # Keep: a-z, A-Z, 0-9, whitespace, .,!?;:()-'"
        # Remove: emoji, weird Unicode symbols, excessive punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\""]', ' ', text)
        
        # =================================================================
        # STEP 7: NORMALIZE WHITESPACE
        # =================================================================
        if self.config['normalize_whitespace']:
            # Replace multiple consecutive whitespace characters with single space
            # \s+ matches one or more whitespace characters (spaces, tabs, newlines)
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading and trailing whitespace
            text = text.strip()
        
        # =================================================================
        # STEP 8: CONVERT TO LOWERCASE
        # =================================================================
        if self.config['lowercase']:
            # Convert to lowercase for consistency
            # This helps BERT's tokenizer work more effectively
            # "The" and "the" should be treated the same
            text = text.lower()
        
        return text
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of text using statistical analysis
        
        THEORY: Language Detection
        Language detection works by analyzing character and word patterns:
        1. CHARACTER N-GRAMS: Sequences of characters common in each language
        2. WORD PATTERNS: Common words and their frequencies
        3. STATISTICAL MODELS: Trained on large corpora of known languages
        
        The langdetect library uses Google's language detection algorithm:
        - Based on character n-gram frequency analysis
        - Trained on Wikipedia text in 55+ languages
        - Returns ISO 639-1 language codes (en, es, fr, de, etc.)
        
        Why filter by language?
        1. Our BERT model is primarily trained on English text
        2. Non-English text would confuse the classification
        3. Mixed languages reduce model accuracy
        4. Consistent language improves feature extraction
        
        Confidence calculation:
        - Run detection multiple times (langdetect has some randomness)
        - Calculate consistency across runs
        - Higher consistency = higher confidence
        
        Args:
            text: Text to analyze for language
        
        Returns:
            Tuple of (language_code, confidence_score)
            Examples: ('en', 0.95), ('es', 0.78), ('unknown', 0.0)
        """
        
        try:
            # Handle edge cases: empty or very short text
            if not text or len(text.strip()) < 10:
                # Too short for reliable language detection
                return 'unknown', 0.0
            
            # =================================================================
            # INITIAL LANGUAGE DETECTION
            # =================================================================
            # Use langdetect library for statistical language detection
            detected_lang = langdetect.detect(text)
            
            # =================================================================
            # CONFIDENCE CALCULATION
            # =================================================================
            # langdetect can vary slightly between runs due to internal randomness
            # We run it multiple times and measure consistency
            confidence_scores = []
            for _ in range(3):  # Run detection 3 times
                try:
                    lang = langdetect.detect(text)
                    # Score 1.0 if it matches our first detection, 0.0 if different
                    confidence_scores.append(1.0 if lang == detected_lang else 0.0)
                except:
                    # If detection fails, score as 0.0
                    confidence_scores.append(0.0)
            
            # Calculate average confidence
            # If all 3 runs agree: confidence = 1.0
            # If 2/3 agree: confidence = 0.67
            # If none agree: confidence = 0.33 or lower
            confidence = np.mean(confidence_scores)
            
            return detected_lang, confidence
            
        except LangDetectException:
            # langdetect couldn't determine the language
            # This happens with very short text or mixed languages
            return 'unknown', 0.0
        except Exception as e:
            # Log unexpected errors for debugging but don't crash
            self.logger.debug(f"Language detection error: {str(e)}")
            return 'unknown', 0.0
    
    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Validate text quality with comprehensive metrics and explanations
        
        THEORY: Quality Control in Machine Learning
        Poor quality training data leads to poor model performance:
        
        1. TOO SHORT: Insufficient context for meaningful classification
           - Single words or phrases lack semantic depth
           - Not representative of real news articles
           
        2. TOO LONG: Might be multiple articles or contain excessive noise
           - Could overwhelm BERT's attention mechanism
           - Might indicate data collection errors
           
        3. INSUFFICIENT WORDS: Text fragments don't represent complete thoughts
           - Headlines without content
           - Truncated or corrupted articles
           
        4. LOW LETTER RATIO: Text that's mostly numbers or symbols
           - Data tables, code snippets, or corrupted text
           - Not natural language suitable for classification
        
        Quality metrics we calculate:
        - Character count: Total text length
        - Word count: Number of words (whitespace-separated tokens)
        - Average word length: Indicator of language complexity
        - Letter ratio: Proportion of alphabetic characters
        
        This function implements a multi-stage validation process where
        each check can independently reject the text with a specific reason.
        
        Args:
            text: Text to validate for quality
        
        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,           # Overall validity
                'reason': str,              # Reason for rejection (if invalid)
                'char_count': int,          # Character count
                'word_count': int,          # Word count
                'avg_word_length': float    # Average word length
            }
        """
        
        # =================================================================
        # HANDLE NON-STRING INPUT
        # =================================================================
        if not isinstance(text, str):
            return {
                'is_valid': False,
                'reason': 'not_string',
                'char_count': 0,
                'word_count': 0,
                'avg_word_length': 0
            }
        
        # =================================================================
        # CALCULATE BASIC TEXT METRICS
        # =================================================================
        char_count = len(text)
        
        # Simple whitespace tokenization for word counting
        # More sophisticated than BERT's tokenization but adequate for quality checks
        words = text.split()
        word_count = len(words)
        
        # Calculate average word length (indicator of text complexity)
        # Real news tends to have longer, more complex words than spam
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Initialize validation results structure
        validation_results = {
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'is_valid': True,           # Assume valid until proven otherwise
            'reason': 'valid'           # Default reason
        }
        
        # =================================================================
        # QUALITY CHECK 1: MINIMUM CHARACTER LENGTH
        # =================================================================
        # Very short texts lack sufficient context for classification
        # Our threshold is based on typical news article lengths
        if char_count < self.config['min_text_length']:
            validation_results['is_valid'] = False
            validation_results['reason'] = 'too_short'
            return validation_results  # Early exit - no need for further checks
        
        # =================================================================
        # QUALITY CHECK 2: MAXIMUM CHARACTER LENGTH
        # =================================================================
        # Very long texts might be multiple articles or contain excessive noise
        # They could also cause memory issues during training
        if char_count > self.config['max_text_length']:
            validation_results['is_valid'] = False
            validation_results['reason'] = 'too_long'
            return validation_results
        
        # =================================================================
        # QUALITY CHECK 3: MINIMUM WORD COUNT
        # =================================================================
        # Text fragments with very few words don't represent complete thoughts
        # Headlines without content, truncated articles, etc.
        if word_count < self.config['min_word_count']:
            validation_results['is_valid'] = False
            validation_results['reason'] = 'insufficient_words'
            return validation_results
        
        # =================================================================
        # QUALITY CHECK 4: LETTER-TO-CHARACTER RATIO
        # =================================================================
        # Text that's mostly numbers, symbols, or punctuation isn't natural language
        # This catches data tables, code snippets, corrupted text, etc.
        letter_count = sum(1 for char in text if char.isalpha())
        letter_ratio = letter_count / char_count if char_count > 0 else 0
        
        if letter_ratio < 0.5:  # Less than 50% letters
            validation_results['is_valid'] = False
            validation_results['reason'] = 'low_letter_ratio'
            return validation_results
        
        # =================================================================
        # ALL QUALITY CHECKS PASSED
        # =================================================================
        # If we reach here, the text passed all quality checks
        return validation_results
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing pipeline that applies all preprocessing steps sequentially
        
        THEORY: Pipeline Processing Architecture
        A processing pipeline applies transformations in a specific order:
        1. Each step builds on the results of the previous step
        2. Order matters (e.g., clean before validate)
        3. Statistics tracking helps us understand data loss at each stage
        4. Error handling prevents pipeline crashes from bad data
        5. Logging provides visibility into the process
        
        Our pipeline stages:
        1. MISSING DATA REMOVAL: Remove rows without text content
        2. TEXT CLEANING: Apply comprehensive text cleaning
        3. EMPTY TEXT REMOVAL: Remove texts that became empty after cleaning
        4. LANGUAGE FILTERING: Keep only target language (English)
        5. QUALITY VALIDATION: Remove low-quality texts
        6. COLUMN CLEANUP: Remove temporary processing columns
        
        Each stage logs its progress and updates statistics for monitoring.
        
        Args:
            df: Input DataFrame with raw data
        
        Returns:
            Processed DataFrame ready for model training
        """
        
        self.logger.info("Starting enhanced dataset processing pipeline...")
        
        # =================================================================
        # LOG INITIAL DATASET COMPOSITION
        # =================================================================
        if 'label' in df.columns:
            initial_distribution = df['label'].value_counts().sort_index()
            self.logger.info(f"Initial class distribution: {initial_distribution.to_dict()}")
        
        # Initialize statistics with starting count
        self.stats['original_count'] = len(df)
        
        # =================================================================
        # PIPELINE STAGE 1: REMOVE MISSING TEXT
        # =================================================================
        # Remove rows where the text column is NaN or empty
        # We can't process what we don't have
        df = df.dropna(subset=['text'])
        self.logger.info(f"After removing missing text: {len(df):,} rows")
        
        # =================================================================
        # PIPELINE STAGE 2: CLEAN ALL TEXT
        # =================================================================
        self.logger.info("Cleaning text data...")
        
        # Apply our comprehensive cleaning function to every text
        # This is computationally expensive but crucial for quality
        df['text'] = df['text'].apply(self.clean_text)
        
        # Update statistics after cleaning
        self.stats['after_cleaning'] = len(df)
        
        # =================================================================
        # PIPELINE STAGE 3: REMOVE EMPTY TEXTS
        # =================================================================
        # Some texts might become empty after aggressive cleaning
        # (e.g., texts that were only HTML tags or URLs)
        df = df[df['text'].str.len() > 0]
        self.logger.info(f"After removing empty texts: {len(df):,} rows")
        
        # =================================================================
        # PIPELINE STAGE 4: LANGUAGE DETECTION AND FILTERING
        # =================================================================
        if self.config['target_language'] == 'en':
            self.logger.info("Filtering for English language...")
            
            # Apply language detection to all texts
            # This returns a Series of tuples: (language, confidence)
            self.logger.info("Running language detection (this may take a while)...")
            language_results = df['text'].apply(
                lambda x: self.detect_language(x)
            )
            
            # Extract language codes and confidence scores into separate columns
            df['detected_language'] = language_results.apply(lambda x: x[0])
            df['language_confidence'] = language_results.apply(lambda x: x[1])
            
            # Create boolean mask for English texts with high confidence
            english_mask = (
                (df['detected_language'] == 'en') & 
                (df['language_confidence'] >= self.config['language_confidence'])
            )
            
            # Count how many non-English texts we're removing
            non_english_count = len(df) - english_mask.sum()
            self.stats['removed_non_english'] = non_english_count
            
            # Apply the language filter
            df = df[english_mask]
            self.logger.info(f"After language filtering: {len(df):,} rows (removed {non_english_count:,} non-English)")
            self.stats['after_language_filter'] = len(df)
        else:
            # If not filtering for English, just update the statistic
            self.stats['after_language_filter'] = len(df)
        
        # =================================================================
        # PIPELINE STAGE 5: QUALITY VALIDATION
        # =================================================================
        self.logger.info("Performing quality validation...")
        
        # Apply quality validation to all texts
        # This returns a Series of dictionaries with validation results
        quality_results = df['text'].apply(self.validate_text_quality)
        
        # Extract validation results into separate columns for analysis
        df['is_valid'] = quality_results.apply(lambda x: x['is_valid'])
        df['validation_reason'] = quality_results.apply(lambda x: x['reason'])
        df['word_count'] = quality_results.apply(lambda x: x['word_count'])
        df['char_count'] = quality_results.apply(lambda x: x['char_count'])
        
        # Create boolean mask for valid texts
        valid_mask = df['is_valid']
        invalid_count = len(df) - valid_mask.sum()
        
        # Log reasons why texts were marked invalid (for debugging)
        if invalid_count > 0:
            invalid_reasons = df[~valid_mask]['validation_reason'].value_counts()
            self.logger.info(f"Invalid text reasons: {invalid_reasons.to_dict()}")
        
        # Apply the quality filter
        df = df[valid_mask]
        self.logger.info(f"After quality filtering: {len(df):,} rows (removed {invalid_count:,} invalid)")
        self.stats['after_quality_filter'] = len(df)
        
        # =================================================================
        # PIPELINE STAGE 6: CLEANUP TEMPORARY COLUMNS
        # =================================================================
        # Remove helper columns we created during processing
        # Keep word_count and char_count as they might be useful for analysis
        columns_to_remove = ['detected_language', 'language_confidence', 'is_valid', 'validation_reason']
        existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        if existing_columns_to_remove:
            df = df.drop(columns=existing_columns_to_remove)
        
        # =================================================================
        # FINALIZE PROCESSING
        # =================================================================
        # Update final statistics
        self.stats['final_count'] = len(df)
        
        # Log final class distribution to verify balance is maintained
        if 'label' in df.columns:
            final_distribution = df['label'].value_counts().sort_index()
            self.logger.info(f"Final class distribution: {final_distribution.to_dict()}")
        
        # Log comprehensive processing statistics
        self._log_enhanced_processing_stats()
        
        self.logger.info("Dataset processing pipeline completed successfully")
        return df
    
    def _log_enhanced_processing_stats(self):
        """
        Log detailed processing statistics for analysis and debugging
        
        THEORY: Process Monitoring and Transparency
        Comprehensive statistics help us:
        1. Understand data quality and composition
        2. Identify potential issues with processing parameters
        3. Optimize preprocessing settings
        4. Provide transparency about data transformations
        5. Debug problems when they occur
        
        We track:
        - Original data composition (real vs fake)
        - Data loss at each processing stage
        - Specific removal reasons and counts
        - Overall retention rate
        """
        
        self.logger.info("=== Enhanced Processing Statistics ===")
        
        # Original data composition
        self.logger.info(f"Original real news articles: {self.stats['true_news_count']:,}")
        self.logger.info(f"Original fake news articles: {self.stats['fake_news_count']:,}")
        self.logger.info(f"Total original records: {self.stats['original_count']:,}")
        
        # Processing pipeline results
        self.logger.info(f"After cleaning: {self.stats['after_cleaning']:,}")
        self.logger.info(f"After language filter: {self.stats['after_language_filter']:,}")
        self.logger.info(f"After quality filter: {self.stats['after_quality_filter']:,}")
        self.logger.info(f"Final records: {self.stats['final_count']:,}")
        
        # Calculate and log retention rate
        if self.stats['original_count'] > 0:
            retention_rate = (self.stats['final_count'] / self.stats['original_count']) * 100
            self.logger.info(f"Overall retention rate: {retention_rate:.1f}%")
            
            # Log retention at each stage
            if self.stats['after_cleaning'] > 0:
                cleaning_retention = (self.stats['after_cleaning'] / self.stats['original_count']) * 100
                self.logger.info(f"Cleaning retention rate: {cleaning_retention:.1f}%")
            
            if self.stats['after_language_filter'] > 0:
                language_retention = (self.stats['after_language_filter'] / self.stats['original_count']) * 100
                self.logger.info(f"Language filter retention rate: {language_retention:.1f}%")
        
        # Specific removal statistics
        self.logger.info("=== Removal Statistics ===")
        self.logger.info(f"HTML tags removed from: {self.stats['removed_html']:,} texts")
        self.logger.info(f"URLs removed from: {self.stats['removed_urls']:,} texts")
        self.logger.info(f"Non-English texts removed: {self.stats['removed_non_english']:,}")
        
        self.logger.info("=== Processing Complete ===")
    
    def balance_dataset(self, df: pd.DataFrame, method: str = 'undersample') -> pd.DataFrame:
        """
        Balance the dataset to handle class imbalance with detailed explanations
        
        THEORY: Class Imbalance Problem in Machine Learning
        Class imbalance occurs when one class has significantly more examples:
        - MODEL BIAS: Model learns to predict the majority class most of the time
        - POOR PERFORMANCE: High accuracy but poor recall on minority class
        - UNFAIR EVALUATION: Metrics like accuracy can be misleading
        
        In fake news detection:
        - Both classes (real and fake) are equally important
        - We want the model to perform well on both types
        - Imbalanced training leads to biased predictions
        
        Balancing Methods:
        
        1. UNDERSAMPLING:
           - Randomly remove samples from the majority class
           - Pros: Faster training, less memory usage, no overfitting risk
           - Cons: Lose potentially useful information
           - Best when: Lots of data, computational constraints
        
        2. OVERSAMPLING:
           - Randomly duplicate samples from the minority class
           - Pros: Keep all original data, more training examples
           - Cons: Risk of overfitting, longer training time
           - Best when: Limited data, sufficient computational resources
        
        Args:
            df: Input DataFrame with potentially imbalanced classes
            method: Balancing method ('undersample', 'oversample', or 'none')
        
        Returns:
            Balanced DataFrame with equal class representation
        """
        
        # Allow skipping balancing entirely
        if method == 'none':
            self.logger.info("Skipping dataset balancing (method='none')")
            return df
        
        # =================================================================
        # ANALYZE CURRENT CLASS DISTRIBUTION
        # =================================================================
        class_counts = df['label'].value_counts()
        self.logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        # Ensure we have exactly 2 classes for binary classification
        if len(class_counts) != 2:
            self.logger.warning("Dataset doesn't have exactly 2 classes. Skipping balancing.")
            return df
        
        # Identify minority and majority classes
        minority_class = class_counts.idxmin()  # Class with fewer samples
        majority_class = class_counts.idxmax()  # Class with more samples
        min_count = class_counts.min()          # Number of minority samples
        max_count = class_counts.max()          # Number of majority samples
        
        # Log class analysis
        class_names = {0: 'Real', 1: 'Fake'}
        self.logger.info(f"Minority class: {minority_class} ({class_names.get(minority_class, 'Unknown')}) - {min_count:,} samples")
        self.logger.info(f"Majority class: {majority_class} ({class_names.get(majority_class, 'Unknown')}) - {max_count:,} samples")
        
        # Calculate imbalance ratio
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        self.logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # =================================================================
        # APPLY BALANCING METHOD
        # =================================================================
        if method == 'undersample':
            self.logger.info("Applying undersampling to majority class...")
            
            # Separate classes into different DataFrames
            majority_df = df[df['label'] == majority_class]
            minority_df = df[df['label'] == minority_class]
            
            # Randomly sample from majority class without replacement
            # This reduces the majority class size to match minority class
            majority_sampled = resample(
                majority_df,
                replace=False,        # Sample without replacement (no duplicates)
                n_samples=min_count,  # Sample exactly min_count samples
                random_state=42       # For reproducible results
            )
            
            # Combine the balanced classes
            balanced_df = pd.concat([majority_sampled, minority_df], ignore_index=True)
            
            self.logger.info(f"Undersampling completed: removed {max_count - min_count:,} majority samples")
            
        elif method == 'oversample':
            self.logger.info("Applying oversampling to minority class...")
            
            # Separate classes into different DataFrames
            majority_df = df[df['label'] == majority_class]
            minority_df = df[df['label'] == minority_class]
            
            # Randomly sample from minority class with replacement
            # This increases the minority class size to match majority class
            minority_sampled = resample(
                minority_df,
                replace=True,         # Sample with replacement (allows duplicates)
                n_samples=max_count,  # Sample exactly max_count samples
                random_state=42       # For reproducible results
            )
            
            # Combine the balanced classes
            balanced_df = pd.concat([majority_df, minority_sampled], ignore_index=True)
            
            self.logger.info(f"Oversampling completed: added {max_count - min_count:,} minority samples")
            
        else:
            # Invalid method specified
            raise ValueError(f"Unknown balancing method: {method}. Use 'undersample', 'oversample', or 'none'")
        
        # =================================================================
        # FINALIZE BALANCED DATASET
        # =================================================================
        # Shuffle the balanced dataset to mix the classes
        # This prevents the model from learning order-based patterns
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # =================================================================
        # LOG BALANCING RESULTS
        # =================================================================
        new_class_counts = balanced_df['label'].value_counts().sort_index()
        self.logger.info(f"Balanced class distribution: {new_class_counts.to_dict()}")
        
        # Verify perfect balance (should be equal counts)
        if len(new_class_counts.unique()) == 1:
            self.logger.info("✅ Perfect class balance achieved")
        else:
            self.logger.warning("⚠️ Classes not perfectly balanced")
        
        # Log size change
        size_change = len(balanced_df) - len(df)
        if size_change > 0:
            self.logger.info(f"Dataset size increased by {size_change:,} samples")
        elif size_change < 0:
            self.logger.info(f"Dataset size decreased by {abs(size_change):,} samples")
        else:
            self.logger.info("Dataset size unchanged")
        
        return balanced_df
    
    def split_dataset(self, df: pd.DataFrame, 
                     train_size: float = 0.7,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets with stratification
        
        THEORY: Dataset Splitting Strategy in Machine Learning
        
        WHY THREE SPLITS?
        1. TRAINING SET (70%): Used to train model parameters
           - Model learns patterns from this data
           - Largest portion for maximum learning
           
        2. VALIDATION SET (15%): Used for hyperparameter tuning and model selection
           - Evaluate different model configurations
           - Select best performing model
           - Monitor for overfitting during training
           
        3. TEST SET (15%): Used for final, unbiased evaluation
           - Never used during development
           - Provides honest estimate of real-world performance
           - Must remain "unseen" until final evaluation
        
        WHY STRATIFIED SPLITTING?
        - Maintains the same class proportion across all splits
        - Ensures each split is representative of the overall dataset
        - Prevents bias where one split has mostly fake or mostly real news
        - Critical for balanced evaluation metrics
        
        RANDOM STATE IMPORTANCE:
        - Makes splits reproducible across different runs
        - Essential for comparing different experiments
        - Allows others to replicate your results
        - Standard practice in ML research
        
        Args:
            df: Input DataFrame to split
            train_size: Proportion for training set (0.0-1.0)
            val_size: Proportion for validation set (0.0-1.0)
            test_size: Proportion for test set (0.0-1.0)
            random_state: Seed for reproducible random splitting
        
        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        
        # =================================================================
        # VALIDATE SPLIT PROPORTIONS
        # =================================================================
        total_size = train_size + val_size + test_size
        if abs(total_size - 1.0) > 0.001:  # Allow for small floating point errors
            raise ValueError(f"Split sizes must sum to 1.0, got {total_size}")
        
        self.logger.info(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
        
        # =================================================================
        # FIRST SPLIT: SEPARATE TEST SET
        # =================================================================
        # We do this first to ensure the test set is completely separate
        # from any training or validation process
        self.logger.info("Performing first split: separating test set...")
        
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,        # Proportion for test set
            random_state=random_state,   # For reproducibility
            stratify=df['label']        # Maintain class ratio in both splits
        )
        
        self.logger.info(f"Test set separated: {len(test_df):,} samples")
        
        # =================================================================
        # SECOND SPLIT: SEPARATE TRAIN AND VALIDATION
        # =================================================================
        # Now we split the remaining data (train_val_df) into train and validation
        # We need to adjust the train_size because we're now splitting a smaller dataset
        self.logger.info("Performing second split: separating train and validation...")
        
        adjusted_train_size = train_size / (train_size + val_size)
        self.logger.debug(f"Adjusted train size for second split: {adjusted_train_size:.3f}")
        
        train_df, val_df = train_test_split(
            train_val_df,
            train_size=adjusted_train_size,  # Adjusted proportion
            random_state=random_state,       # For reproducibility
            stratify=train_val_df['label']   # Maintain class ratio
        )
        
        # =================================================================
        # LOG SPLIT RESULTS
        # =================================================================
        self.logger.info(f"Dataset split completed:")
        self.logger.info(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        self.logger.info(f"  Validation: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        self.logger.info(f"  Test: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        # =================================================================
        # VERIFY STRATIFICATION
        # =================================================================
        # Log class distribution for each split to verify stratification worked
        self.logger.info("Verifying stratification across splits:")
        
        for name, data in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            class_dist = data['label'].value_counts().sort_index()
            class_ratio = (class_dist / len(data) * 100).round(1)
            
            self.logger.info(f"  {name} class distribution:")
            self.logger.info(f"    Counts: {class_dist.to_dict()}")
            self.logger.info(f"    Ratios: Real {class_ratio[0]}%, Fake {class_ratio[1]}%")
        
        # =================================================================
        # FINAL VALIDATION
        # =================================================================
        # Verify that splits don't overlap and total correctly
        total_split_size = len(train_df) + len(val_df) + len(test_df)
        if total_split_size != len(df):
            self.logger.error(f"Split size mismatch: {total_split_size} != {len(df)}")
            raise ValueError("Dataset splitting produced incorrect total size")
        
        self.logger.info("✅ Dataset splitting completed successfully")
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, 
                          val_df: pd.DataFrame, 
                          test_df: pd.DataFrame) -> Dict[str, Path]:
        """
        Save processed datasets to CSV files with comprehensive metadata
        
        THEORY: Data Persistence and Reproducibility
        Saving processed data serves multiple purposes:
        1. AVOID REPROCESSING: Preprocessing is time-consuming, save results
        2. REPRODUCIBILITY: Others can use exact same processed data
        3. DEBUGGING: Inspect processed data to understand pipeline results
        4. EXPERIMENTATION: Try different models on same processed data
        5. AUDIT TRAIL: Keep record of data transformations
        
        File Organization:
        - train_processed.csv: Training data for model training
        - val_processed.csv: Validation data for hyperparameter tuning
        - test_processed.csv: Test data for final evaluation
        - processing_stats.json: Metadata about processing pipeline
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            test_df: Test DataFrame
        
        Returns:
            Dictionary mapping split names to their file paths
        """
        
        self.logger.info("Saving processed datasets...")
        
        file_paths = {}
        
        # =================================================================
        # SAVE EACH SPLIT TO SEPARATE CSV FILE
        # =================================================================
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Generate file path
            file_path = Config.PROCESSED_DATA_DIR / f"{name}_processed.csv"
            
            # Save to CSV without row indices (index=False)
            # This creates clean CSV files suitable for loading later
            df.to_csv(file_path, index=False)
            
            # Store path for return value
            file_paths[name] = file_path
            
            # Log save confirmation
            self.logger.info(f"Saved {name} data: {file_path} ({len(df):,} rows)")
            
            # Log first few columns for verification
            self.logger.debug(f"{name} columns: {list(df.columns)}")
        
        # =================================================================
        # NUMPY TYPE CONVERSION FUNCTION (FIX FOR JSON ERROR)
        # =================================================================
        def convert_numpy_types(obj):
            """
            Convert numpy types to native Python types for JSON serialization
            
            THEORY: JSON Serialization Issue Fix
            The error "Object of type int64 is not JSON serializable" occurs because:
            - Pandas operations often create numpy data types (int64, float64)
            - JSON encoder only handles native Python types (int, float, str)
            - We need to recursively convert all numpy types in our statistics
            
            This function handles:
            - numpy integers → Python int
            - numpy floats → Python float  
            - numpy arrays → Python lists
            - dictionaries and lists recursively
            """
            import numpy as np
            
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # =================================================================
        # SAVE PROCESSING STATISTICS (WITH NUMPY CONVERSION)
        # =================================================================
        # Save comprehensive metadata about the processing pipeline
        stats_path = Config.PROCESSED_DATA_DIR / "processing_stats.json"
        
        # Enhanced statistics with processing metadata
        enhanced_stats = {
            'processing_stats': self.stats,
            'dataset_info': {
                'original_total': self.stats['original_count'],
                'final_total': self.stats['final_count'],
                'retention_rate': (self.stats['final_count'] / self.stats['original_count'] * 100) if self.stats['original_count'] > 0 else 0,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df)
            },
            'processing_config': self.config,
            'split_info': {
                'train_file': str(file_paths['train']),
                'val_file': str(file_paths['val']),
                'test_file': str(file_paths['test'])
            },
            'timestamp': pd.Timestamp.now().isoformat(),
            'class_distributions': {
                'train': dict(train_df['label'].value_counts()),
                'val': dict(val_df['label'].value_counts()),
                'test': dict(test_df['label'].value_counts())
            }
        }
        
        # =================================================================
        # CONVERT NUMPY TYPES BEFORE JSON SERIALIZATION
        # =================================================================
        # This fixes the "Object of type int64 is not JSON serializable" error
        enhanced_stats = convert_numpy_types(enhanced_stats)
        
        # Save as JSON for easy reading and parsing
        import json
        try:
            with open(stats_path, 'w') as f:
                json.dump(enhanced_stats, f, indent=2)
            
            self.logger.info(f"Saved processing statistics: {stats_path}")
            
        except Exception as e:
            # If JSON saving still fails, log the error but don't crash
            self.logger.warning(f"Could not save processing statistics: {str(e)}")
            self.logger.info("CSV files saved successfully despite statistics error")
        
        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        self.logger.info(f"All processed data saved to: {Config.PROCESSED_DATA_DIR}")
        self.logger.info("✅ Data preprocessing and saving completed successfully")
        
        # Summary statistics
        total_samples = len(train_df) + len(val_df) + len(test_df)
        self.logger.info(f"📊 Final Summary:")
        self.logger.info(f"   Total processed samples: {total_samples:,}")
        self.logger.info(f"   Train/Val/Test split: {len(train_df)}/{len(val_df)}/{len(test_df)}")
        
        return file_paths



# =================================================================
# EXAMPLE USAGE AND TESTING
# =================================================================
if __name__ == "__main__":
    """
    Example usage with the dual Kaggle dataset (True.csv + Fake.csv)
    
    This demonstrates the complete preprocessing pipeline:
    1. Initialize agent with all NLP resources
    2. Load dual dataset (True.csv + Fake.csv)
    3. Process through complete cleaning pipeline
    4. Balance classes for fair training
    5. Split into train/validation/test sets
    6. Save processed data for model training
    
    Run this script to test the preprocessing pipeline on your data.
    """
    
    print("🚀 Starting Enhanced Dual-Dataset Preprocessing Pipeline")
    print("=" * 60)
    
    # =================================================================
    # INITIALIZE PREPROCESSING AGENT
    # =================================================================
    print("Initializing preprocessing agent...")
    try:
        preprocessor = DataPreprocessorAgent()
        print("✅ Preprocessing agent initialized successfully\n")
    except Exception as e:
        print(f"❌ Error initializing preprocessor: {e}")
        exit(1)
    
    try:
        # =================================================================
        # LOAD DUAL DATASET
        # =================================================================
        print("Loading dual Kaggle dataset (True.csv + Fake.csv)...")
        df = preprocessor.load_dual_kaggle_dataset()
        print(f"✅ Loaded combined dataset with {len(df):,} rows\n")
        
        # Show initial composition
        class_dist = df['label'].value_counts().sort_index()
        print(f"📊 Initial composition:")
        print(f"  Real news (0): {class_dist[0]:,} articles")
        print(f"  Fake news (1): {class_dist[1]:,} articles")
        print(f"  Total: {len(df):,} articles\n")
        
        # =================================================================
        # PROCESS DATASET
        # =================================================================
        print("Processing dataset through cleaning pipeline...")
        processed_df = preprocessor.process_dataset(df)
        print(f"✅ Processed dataset has {len(processed_df):,} rows\n")
        
        # Show processing impact
        retention_rate = (len(processed_df) / len(df)) * 100
        print(f"📈 Processing Results:")
        print(f"  Retention rate: {retention_rate:.1f}%")
        print(f"  Removed: {len(df) - len(processed_df):,} rows\n")
        
        # =================================================================
        # BALANCE DATASET
        # =================================================================
        print("Balancing dataset to handle class imbalance...")
        balanced_df = preprocessor.balance_dataset(processed_df, method='undersample')
        print(f"✅ Balanced dataset has {len(balanced_df):,} rows\n")
        
        # Show balancing results
        balanced_dist = balanced_df['label'].value_counts().sort_index()
        print(f"🏆 Balanced composition:")
        print(f"  Real news (0): {balanced_dist[0]:,} articles")
        print(f"  Fake news (1): {balanced_dist[1]:,} articles\n")
        
        # =================================================================
        # SPLIT DATASET
        # =================================================================
        print("Splitting dataset into train/validation/test sets...")
        train_df, val_df, test_df = preprocessor.split_dataset(balanced_df)
        print(f"✅ Split completed:")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Validation: {len(val_df):,} samples")
        print(f"  Test: {len(test_df):,} samples\n")
        
        # =================================================================
        # SAVE PROCESSED DATA
        # =================================================================
        print("Saving processed data...")
        file_paths = preprocessor.save_processed_data(train_df, val_df, test_df)
        
        print("🎉 Dual-dataset processing complete! Files saved:")
        for name, path in file_paths.items():
            print(f"  📁 {name}: {path}")
        
        # =================================================================
        # FINAL STATISTICS SUMMARY
        # =================================================================
        print(f"\n📊 Final Statistics:")
        print(f"  Original Real: {preprocessor.stats['true_news_count']:,} articles")
        print(f"  Original Fake: {preprocessor.stats['fake_news_count']:,} articles")
        print(f"  Total Original: {preprocessor.stats['original_count']:,} articles")
        print(f"  Final Processed: {preprocessor.stats['final_count']:,} articles")
        
        overall_retention = (preprocessor.stats['final_count'] / preprocessor.stats['original_count']) * 100
        print(f"  Overall Retention: {overall_retention:.1f}%")
        
        print(f"\n🎯 Ready for BERT model training!")
        print(f"   Use the files in: {Config.PROCESSED_DATA_DIR}")
            
    except FileNotFoundError as e:
        print("❌ Error: Kaggle dataset files not found!")
        print("Please ensure you have both files in your data/raw/ directory:")
        print("  📁 True.csv")
        print("  📁 Fake.csv")
        print(f"Expected location: {Config.RAW_DATA_DIR}")
        print("\nTo get the files:")
        print("1. Go to: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets")
        print("2. Download the dataset")
        print("3. Extract True.csv and Fake.csv to your data/raw/ folder")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        print("\nFor help debugging:")
        print("1. Check that all dependencies are installed")
        print("2. Verify the dataset files are properly formatted")
        print("3. Check the logs in data/training_logs/preprocessing.log")