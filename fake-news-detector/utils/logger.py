"""
Production-Ready Logging System for Fake News Detection

Comprehensive logging solution with colored console output, file rotation,
structured formatting, and performance monitoring capabilities.

Features:
- Colored console output for development visibility
- File logging with automatic rotation and compression
- Structured log formatting with contextual information
- Performance tracking and metrics logging
- Environment-aware log level configuration
- Memory-efficient logging with cleanup
- Integration with system monitoring
"""

import logging
import logging.handlers
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import threading
import time
from contextlib import contextmanager


@dataclass
class LogConfig:
    """Configuration for logging system"""
    log_level: str = "INFO"
    console_enabled: bool = True
    file_enabled: bool = True
    colored_output: bool = True
    structured_format: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    log_directory: Optional[Path] = None
    performance_logging: bool = True
    memory_threshold_mb: float = 100.0


class ColorFormatter(logging.Formatter):
    """
    Enhanced formatter with intelligent color coding and structured output
    
    Features:
    - ANSI color codes for different log levels
    - Contextual information display
    - Performance metrics integration
    - Clean formatting for both console and analysis
    """
    
    # Enhanced color scheme for better visibility
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan - detailed debugging information
        'INFO': '\033[32m',      # Green - normal operation information  
        'WARNING': '\033[33m',   # Yellow - warning conditions
        'ERROR': '\033[31m',     # Red - error conditions
        'CRITICAL': '\033[35m',  # Magenta - critical errors
        'PERFORMANCE': '\033[94m', # Light blue - performance metrics
        'SECURITY': '\033[91m'     # Light red - security events
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Component color coding for better readability
    COMPONENT_COLORS = {
        'agents': '\033[96m',      # Light cyan
        'workflow': '\033[95m',    # Light magenta  
        'api': '\033[93m',         # Light yellow
        'scraper': '\033[92m',     # Light green
        'database': '\033[94m'     # Light blue
    }

    def __init__(self, colored: bool = True, structured: bool = True):
        self.colored = colored
        self.structured = structured
        
        # Base format with comprehensive information
        if structured:
            format_string = (
                '%(asctime)s | %(levelname)-8s | %(name)s | '
                '%(funcName)s:%(lineno)d | %(process)d:%(thread)d | %(message)s'
            )
        else:
            format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        
        super().__init__(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and enhanced information"""
        
        # Add contextual information to record
        self._enhance_record(record)
        
        if self.colored and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            # Apply colors for terminal output
            original_levelname = record.levelname
            
            # Color the level name
            level_color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{level_color}{record.levelname}{self.RESET}"
            
            # Color the logger name based on component
            original_name = record.name
            component_color = self._get_component_color(record.name)
            if component_color:
                record.name = f"{component_color}{record.name}{self.RESET}"
            
            # Format the record
            formatted = super().format(record)
            
            # Restore original values
            record.levelname = original_levelname
            record.name = original_name
            
            return formatted
        else:
            # Plain formatting for files
            return super().format(record)

    def _enhance_record(self, record: logging.LogRecord):
        """Add contextual information to log record"""
        
        # Add timestamp in ISO format for structured logging
        record.iso_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = round(process.memory_info().rss / 1024 / 1024, 1)
        except:
            record.memory_mb = 0
        
        # Add execution context
        record.execution_id = getattr(record, 'execution_id', 'main')

    def _get_component_color(self, logger_name: str) -> Optional[str]:
        """Get color for logger component"""
        if not self.colored:
            return None
        
        name_lower = logger_name.lower()
        for component, color in self.COMPONENT_COLORS.items():
            if component in name_lower:
                return color
        return None


class PerformanceLogHandler(logging.Handler):
    """
    Specialized handler for performance metrics and monitoring
    
    Features:
    - Separate performance log files
    - Metrics aggregation and reporting
    - Memory and timing tracking
    - Alert generation for performance issues
    """
    
    def __init__(self, log_file: Path, threshold_ms: float = 1000.0):
        super().__init__()
        self.log_file = log_file
        self.threshold_ms = threshold_ms
        self.performance_data = []
        self.lock = threading.Lock()
        
        # Setup file handler
        self.file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=3
        )
        
        # JSON formatter for structured performance data
        self.setFormatter(JsonFormatter())

    def emit(self, record: logging.LogRecord):
        """Handle performance log records"""
        if hasattr(record, 'performance_data'):
            with self.lock:
                self.performance_data.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'logger': record.name,
                    'data': record.performance_data
                })
                
                # Check for performance alerts
                if record.performance_data.get('duration_ms', 0) > self.threshold_ms:
                    self._generate_performance_alert(record)
        
        # Also write to file
        self.file_handler.emit(record)

    def _generate_performance_alert(self, record: logging.LogRecord):
        """Generate alert for performance issues"""
        alert_record = logging.LogRecord(
            name=f"{record.name}.performance_alert",
            level=logging.WARNING,
            pathname=record.pathname,
            lineno=record.lineno,
            msg=f"Performance threshold exceeded: {record.performance_data.get('duration_ms')}ms > {self.threshold_ms}ms",
            args=(),
            exc_info=None
        )
        
        # Send to main logging system
        logging.getLogger('performance.alerts').handle(alert_record)

    def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes"""
        with self.lock:
            cutoff = datetime.now(timezone.utc).timestamp() - (minutes * 60)
            recent_data = [
                entry for entry in self.performance_data
                if datetime.fromisoformat(entry['timestamp']).timestamp() > cutoff
            ]
            
            if not recent_data:
                return {'message': 'No performance data available'}
            
            durations = [entry['data'].get('duration_ms', 0) for entry in recent_data]
            
            return {
                'period_minutes': minutes,
                'total_operations': len(recent_data),
                'avg_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations),
                'min_duration_ms': min(durations),
                'threshold_exceeded': len([d for d in durations if d > self.threshold_ms])
            }


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured log analysis"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON for structured analysis"""
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add any custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class EnhancedLogger:
    """
    Enhanced logger wrapper with additional functionality
    
    Features:
    - Performance timing context managers
    - Memory usage tracking
    - Structured logging helpers
    - Component-specific configuration
    """
    
    def __init__(self, name: str, config: LogConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with handlers and formatters"""
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Set log level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColorFormatter(
                colored=self.config.colored_output,
                structured=self.config.structured_format
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.file_enabled and self.config.log_directory:
            self._setup_file_handler()
        
        # Performance handler
        if self.config.performance_logging and self.config.log_directory:
            self._setup_performance_handler()

    def _setup_file_handler(self):
        """Setup rotating file handler"""
        log_file = self.config.log_directory / f"{self.name.replace('.', '_')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        
        # Use plain formatter for files (no colors)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | '
            '%(funcName)s:%(lineno)d | %(process)d:%(thread)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in files
        self.logger.addHandler(file_handler)

    def _setup_performance_handler(self):
        """Setup performance logging handler"""
        perf_log_file = self.config.log_directory / f"performance_{self.name.replace('.', '_')}.log"
        
        self.performance_handler = PerformanceLogHandler(
            perf_log_file,
            threshold_ms=1000.0
        )
        
        # Performance logs go to separate logger
        perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_logger.addHandler(self.performance_handler)
        perf_logger.setLevel(logging.INFO)

    @contextmanager
    def timer(self, operation: str, **kwargs):
        """Context manager for timing operations"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        operation_id = f"{operation}_{int(start_time * 1000)}"
        
        try:
            self.logger.debug(f"Starting operation: {operation} (ID: {operation_id})")
            yield operation_id
            
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            # Log performance data
            if self.config.performance_logging:
                perf_record = logging.LogRecord(
                    name=f"{self.name}.performance",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"Operation completed: {operation}",
                    args=(),
                    exc_info=None
                )
                
                perf_record.performance_data = {
                    'operation': operation,
                    'operation_id': operation_id,
                    'duration_ms': round(duration_ms, 2),
                    'memory_delta_mb': round(memory_delta, 2),
                    'start_memory_mb': round(start_memory, 2),
                    'end_memory_mb': round(end_memory, 2),
                    **kwargs
                }
                
                perf_logger = logging.getLogger(f"{self.name}.performance")
                perf_logger.handle(perf_record)
            
            # Log summary
            if duration_ms > 100:  # Log slow operations
                self.logger.info(
                    f"Operation '{operation}' completed in {duration_ms:.1f}ms "
                    f"(Memory: {memory_delta:+.1f}MB)"
                )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def log_structured(self, level: str, message: str, **kwargs):
        """Log with structured data"""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        
        # Create record with structured data
        record = logging.LogRecord(
            name=self.logger.name,
            level=getattr(logging, level.upper(), logging.INFO),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Add structured data
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        self.logger.handle(record)

    def debug(self, msg, *args, **kwargs):
        """Enhanced debug logging"""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Enhanced info logging"""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Enhanced warning logging"""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Enhanced error logging"""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Enhanced critical logging"""
        self.logger.critical(msg, *args, **kwargs)

    def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for this logger"""
        if hasattr(self, 'performance_handler'):
            return self.performance_handler.get_performance_summary(minutes)
        return {'message': 'Performance logging not enabled'}


class LoggingManager:
    """
    Central logging manager for the entire fake news detection system
    
    Features:
    - Centralized logger configuration
    - Environment-aware setup
    - Performance monitoring
    - Log cleanup and maintenance
    """
    
    def __init__(self, base_config: Optional[LogConfig] = None):
        self.base_config = base_config or LogConfig()
        self.loggers: Dict[str, EnhancedLogger] = {}
        self._setup_directories()

    def _setup_directories(self):
        """Setup logging directories"""
        if not self.base_config.log_directory:
            # Default to logs directory in project root
            project_root = Path(__file__).parent.parent
            self.base_config.log_directory = project_root / "logs"
        
        self.base_config.log_directory.mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str, config_override: Optional[Dict[str, Any]] = None) -> EnhancedLogger:
        """
        Get or create logger with optional configuration override
        
        Args:
            name: Logger name (typically module name)
            config_override: Optional configuration overrides
            
        Returns:
            EnhancedLogger instance
        """
        
        if name in self.loggers:
            return self.loggers[name]
        
        # Create config for this logger
        logger_config = LogConfig(
            log_level=self.base_config.log_level,
            console_enabled=self.base_config.console_enabled,
            file_enabled=self.base_config.file_enabled,
            colored_output=self.base_config.colored_output,
            structured_format=self.base_config.structured_format,
            max_file_size=self.base_config.max_file_size,
            backup_count=self.base_config.backup_count,
            log_directory=self.base_config.log_directory,
            performance_logging=self.base_config.performance_logging,
            memory_threshold_mb=self.base_config.memory_threshold_mb
        )
        
        # Apply overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(logger_config, key):
                    setattr(logger_config, key, value)
        
        # Create and cache logger
        enhanced_logger = EnhancedLogger(name, logger_config)
        self.loggers[name] = enhanced_logger
        
        return enhanced_logger

    def configure_for_environment(self, environment: str = "development"):
        """Configure logging based on environment"""
        
        env_configs = {
            "development": {
                "log_level": "DEBUG",
                "console_enabled": True,
                "file_enabled": True,
                "colored_output": True,
                "structured_format": True,
                "performance_logging": True
            },
            "testing": {
                "log_level": "WARNING",
                "console_enabled": False,
                "file_enabled": False,
                "colored_output": False,
                "structured_format": False,
                "performance_logging": False
            },
            "staging": {
                "log_level": "INFO", 
                "console_enabled": True,
                "file_enabled": True,
                "colored_output": True,
                "structured_format": True,
                "performance_logging": True
            },
            "production": {
                "log_level": "INFO",
                "console_enabled": False,
                "file_enabled": True,
                "colored_output": False,
                "structured_format": True,
                "performance_logging": True
            }
        }
        
        if environment in env_configs:
            config = env_configs[environment]
            for key, value in config.items():
                if hasattr(self.base_config, key):
                    setattr(self.base_config, key, value)

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive logging system summary"""
        summary = {
            "active_loggers": len(self.loggers),
            "logger_names": list(self.loggers.keys()),
            "configuration": {
                "log_level": self.base_config.log_level,
                "console_enabled": self.base_config.console_enabled,
                "file_enabled": self.base_config.file_enabled,
                "performance_logging": self.base_config.performance_logging,
                "log_directory": str(self.base_config.log_directory)
            },
            "performance_summaries": {}
        }
        
        # Get performance data from each logger
        for name, logger in self.loggers.items():
            try:
                perf_summary = logger.get_performance_summary(60)
                summary["performance_summaries"][name] = perf_summary
            except:
                continue
        
        return summary

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        if not self.base_config.log_directory.exists():
            return
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        cleaned_files = 0
        for log_file in self.base_config.log_directory.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_files += 1
                except:
                    continue
        
        if cleaned_files > 0:
            self.get_logger("logging_manager").info(f"Cleaned up {cleaned_files} old log files")


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None

def setup_logging(environment: str = "development", 
                 log_directory: Optional[Path] = None,
                 config_overrides: Optional[Dict[str, Any]] = None) -> LoggingManager:
    """
    Setup global logging system
    
    Args:
        environment: Environment name for configuration
        log_directory: Custom log directory path
        config_overrides: Additional configuration overrides
        
    Returns:
        LoggingManager instance
    """
    global _logging_manager
    
    # Create base configuration
    base_config = LogConfig()
    
    if log_directory:
        base_config.log_directory = log_directory
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
    
    # Create logging manager
    _logging_manager = LoggingManager(base_config)
    _logging_manager.configure_for_environment(environment)
    
    return _logging_manager

def get_logger(name: str, config_override: Optional[Dict[str, Any]] = None) -> EnhancedLogger:
    """
    Get logger instance (creates logging manager if not exists)
    
    Args:
        name: Logger name
        config_override: Optional configuration overrides
        
    Returns:
        EnhancedLogger instance
    """
    global _logging_manager
    
    if _logging_manager is None:
        # Auto-setup with default configuration
        environment = os.getenv('ENVIRONMENT', 'development')
        _logging_manager = setup_logging(environment)
    
    return _logging_manager.get_logger(name, config_override)

# Backward compatibility function
def setup_logger(name: str, log_file: Optional[str] = None) -> EnhancedLogger:
    """
    Backward compatibility function for existing code
    
    Args:
        name: Logger name
        log_file: Optional log file name
        
    Returns:
        EnhancedLogger instance
    """
    config_override = {}
    if log_file is not None:
        config_override['file_enabled'] = True
    
    return get_logger(name, config_override)

# Example usage
if __name__ == "__main__":
    # Setup logging for development
    logging_manager = setup_logging("development")
    
    # Get logger for testing
    logger = get_logger("test_logger")
    
    # Test different log levels
    logger.debug("Debug message with detailed information")
    logger.info("Information about normal operation")
    logger.warning("Warning about potential issue")
    logger.error("Error occurred but system continues")
    logger.critical("Critical error - immediate attention needed")
    
    # Test performance timing
    with logger.timer("example_operation", user_id=123, operation_type="test"):
        time.sleep(0.1)  # Simulate work
    
    # Test structured logging
    logger.log_structured("info", "User action completed", 
                         user_id=123, action="login", success=True)
    
    # Get performance summary
    summary = logger.get_performance_summary(5)
    print(f"Performance Summary: {summary}")
    
    # System summary
    sys_summary = logging_manager.get_system_summary()
    print(f"System Summary: {sys_summary}")
