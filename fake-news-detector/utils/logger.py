"""
logger.py - Part of Fake News Detection System
"""

"""
Logging System for Fake News Detection System

THEORY: Why Professional Logging Matters
1. Debugging: Trace what happens during processing and training
2. Monitoring: Track system performance and errors in production
3. Auditing: Keep records of data processing decisions
4. Development: Understand how your algorithms are working

This module provides:
- Colored console output for better readability during development
- File logging for permanent records
- Structured log messages with timestamps and severity levels
- Customizable log levels for different environments
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from utils.config import Config

class CustomFormatter(logging.Formatter):
    """
    Custom log formatter that adds color coding and structured formatting
    
    THEORY: Why custom formatting?
    - Colors help quickly identify different message types
    - Consistent format makes logs easier to parse
    - Timestamps help track when events occurred
    - Structured format works well with log analysis tools
    
    Color coding:
    - DEBUG (Cyan): Detailed information for diagnosing problems
    - INFO (Green): General information about system operation
    - WARNING (Yellow): Something unexpected happened but system continues
    - ERROR (Red): Error occurred but system might continue
    - CRITICAL (Magenta): Serious error, system might not continue
    """
    
    # ANSI color codes for terminal output
    # These work in most modern terminals (Mac Terminal, Linux, Windows 10+)
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan - for detailed debugging info
        'INFO': '\033[32m',      # Green - for normal operation info
        'WARNING': '\033[33m',   # Yellow - for warnings
        'ERROR': '\033[31m',     # Red - for errors
        'CRITICAL': '\033[35m',  # Magenta - for critical errors
    }
    RESET = '\033[0m'  # Reset to default color
    
    def format(self, record):
        """
        Format a log record with colors and structured information
        
        Args:
            record: LogRecord object containing the log message and metadata
            
        Returns:
            str: Formatted log message
        """
        # Add color to the level name for console output
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        
        # Format timestamp in a readable format
        # record.created is a Unix timestamp (seconds since epoch)
        record.asctime = datetime.fromtimestamp(record.created).strftime(
            '%Y-%m-%d %H:%M:%S'  # Format: 2024-01-15 14:30:25
        )
        
        # Create the final formatted message
        # Format: timestamp | level | logger_name | message
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
            # %(levelname)-8s means left-align levelname in 8-character field
        )
        return formatter.format(record)

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with both console and optional file output
    
    THEORY: Why both console and file logging?
    - Console: Immediate feedback during development and testing
    - File: Permanent record for analysis and debugging
    - Different formatters: Colors for console, plain text for files
    
    Args:
        name: Name for the logger (usually the module name)
        log_file: Optional filename for file logging
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get or create a logger with the specified name
    # If logger already exists, this returns the existing one
    logger = logging.getLogger(name)
    
    # Set the minimum log level from configuration
    # getattr gets the attribute value from the logging module
    # If LOG_LEVEL is invalid, defaults to INFO
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Prevent adding duplicate handlers if logger already exists
    # This is important when the same module is imported multiple times
    if logger.handlers:
        return logger
    
    # =================================================================
    # CONSOLE HANDLER - for immediate output during development
    # =================================================================
    console_handler = logging.StreamHandler(sys.stdout)  # Output to standard output
    console_handler.setFormatter(CustomFormatter())       # Use our colored formatter
    logger.addHandler(console_handler)
    
    # =================================================================
    # FILE HANDLER - for permanent logging (optional)
    # =================================================================
    if log_file:
        # Create the log file path in the logs directory
        log_path = Config.LOGS_DIR / log_file
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        
        # Use plain formatter for files (no colors)
        # Colors would show up as weird characters in text files
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a logger for this module
# This allows us to log messages from within the logger module itself
logger = setup_logger(__name__)
