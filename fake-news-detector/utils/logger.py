# utils/logger.py

"""
Production-Ready Logging System for Fake News Detection

Comprehensive logging solution with colored console output, file rotation,
structured formatting, performance monitoring, and environment awareness.

Features:
- Colored console output with intelligent formatting
- Rotating file handlers with compression and cleanup
- Structured JSON logging for analysis and monitoring
- Performance tracking with timing decorators
- Environment-aware configuration
- Memory-efficient implementation with resource management
- Thread-safe operations for concurrent usage
- Integration with system monitoring

Version: 3.2.0 - Enhanced Production Edition
"""

import logging
import logging.handlers
import sys
import os
import json
import time
import threading
import functools
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import uuid

# Optional dependencies
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import gzip
    _GZIP_AVAILABLE = True
except ImportError:
    _GZIP_AVAILABLE = False


@dataclass
class LogConfig:
    """Configuration for logging system with comprehensive options."""
    log_level: str = "INFO"
    console_enabled: bool = True
    file_enabled: bool = True
    colored_output: bool = True
    structured_logs: bool = True
    performance_tracking: bool = True
    
    # File handling
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    compress_backups: bool = True
    log_directory: Optional[Path] = None
    
    # Performance and monitoring
    memory_threshold_mb: float = 100.0
    enable_system_metrics: bool = True
    cleanup_days: int = 30
    
    # Environment-specific settings
    environment: str = "development"
    enable_debug_context: bool = False


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


class EnhancedColorFormatter(logging.Formatter):
    """
    Enhanced formatter with intelligent color coding and structured output.
    
    Features:
    - Level-based coloring with visual hierarchy
    - Component-specific colors for better readability
    - Performance indicators and timing information
    - Memory usage tracking
    - Session and context information
    """
    
    # Color mapping for log levels
    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD
    }
    
    # Component colors for different modules
    COMPONENT_COLORS = {
        'agents': Colors.BRIGHT_BLUE,
        'workflow': Colors.BRIGHT_MAGENTA,
        'scraper': Colors.BRIGHT_GREEN,
        'api': Colors.BRIGHT_YELLOW,
        'database': Colors.BRIGHT_CYAN,
        'performance': Colors.MAGENTA,
        'security': Colors.RED + Colors.BOLD
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, 
                 use_color: bool = True, include_performance: bool = True):
        if fmt is None:
            fmt = (
                '%(asctime)s | %(levelname)-8s | %(name)s | '
                '%(funcName)s:%(lineno)d | %(message)s'
            )
        
        super().__init__(fmt, datefmt or '%Y-%m-%d %H:%M:%S')
        self.use_color = use_color
        self.include_performance = include_performance
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and enhanced information."""
        # Add enhanced context to record
        self._enhance_record(record)
        
        if self.use_color and self._supports_color():
            # Store original values
            original_levelname = record.levelname
            original_name = record.name
            original_message = record.getMessage()
            
            # Apply level-based coloring
            level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            record.levelname = f"{level_color}{record.levelname}{Colors.RESET}"
            
            # Apply component-based coloring
            component_color = self._get_component_color(record.name)
            if component_color:
                record.name = f"{component_color}{record.name}{Colors.RESET}"
            
            # Format message with performance indicators
            if self.include_performance and hasattr(record, 'duration_ms'):
                duration_color = self._get_duration_color(record.duration_ms)
                performance_info = f" {duration_color}[{record.duration_ms:.1f}ms]{Colors.RESET}"
                record.msg = f"{record.msg}{performance_info}"
            
            # Format the record
            formatted = super().format(record)
            
            # Restore original values
            record.levelname = original_levelname
            record.name = original_name
            
            return formatted
        else:
            return super().format(record)
    
    def _enhance_record(self, record: logging.LogRecord):
        """Add contextual information to log record."""
        # Add ISO timestamp
        record.iso_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Add memory usage if available
        if _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                record.memory_mb = round(process.memory_info().rss / 1024 / 1024, 1)
            except Exception:
                record.memory_mb = 0
        else:
            record.memory_mb = 0
        
        # Add session context if available
        record.session_id = getattr(record, 'session_id', 'main')
        record.execution_id = getattr(record, 'execution_id', 'default')
    
    def _supports_color(self) -> bool:
        """Check if terminal supports color output."""
        return (
            hasattr(sys.stderr, 'isatty') and sys.stderr.isatty() and
            os.getenv('TERM') != 'dumb' and
            os.getenv('NO_COLOR') is None
        )
    
    def _get_component_color(self, logger_name: str) -> Optional[str]:
        """Get color for logger component."""
        if not self.use_color:
            return None
        
        name_lower = logger_name.lower()
        for component, color in self.COMPONENT_COLORS.items():
            if component in name_lower:
                return color
        return None
    
    def _get_duration_color(self, duration_ms: float) -> str:
        """Get color based on operation duration."""
        if duration_ms > 5000:  # > 5 seconds
            return Colors.RED
        elif duration_ms > 1000:  # > 1 second
            return Colors.YELLOW
        elif duration_ms > 100:  # > 100ms
            return Colors.CYAN
        else:
            return Colors.GREEN


class StructuredJsonFormatter(logging.Formatter):
    """
    JSON formatter for structured log analysis and monitoring.
    
    Features:
    - Consistent JSON structure for all log entries
    - Performance metrics integration
    - System resource tracking
    - Session and execution context
    - Error details and stack traces
    """
    
    def __init__(self, include_system_info: bool = True):
        super().__init__()
        self.include_system_info = include_system_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record as structured JSON."""
        # Base log structure
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'thread_name': getattr(record, 'threadName', 'MainThread')
        }
        
        # Add exception information
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add performance information
        if hasattr(record, 'duration_ms'):
            log_data['performance'] = {
                'duration_ms': record.duration_ms,
                'operation': getattr(record, 'operation', 'unknown')
            }
        
        # Add system information
        if self.include_system_info and _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                log_data['system'] = {
                    'memory_mb': round(process.memory_info().rss / 1024 / 1024, 1),
                    'cpu_percent': process.cpu_percent()
                }
            except Exception:
                pass
        
        # Add session context
        session_fields = ['session_id', 'execution_id', 'user_id', 'request_id']
        for field in session_fields:
            if hasattr(record, field):
                log_data.setdefault('context', {})[field] = getattr(record, field)
        
        # Add custom fields
        custom_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in self._get_standard_fields() and not k.startswith('_')
        }
        
        if custom_fields:
            log_data['custom'] = custom_fields
        
        return json.dumps(log_data, default=str, ensure_ascii=False)
    
    def _get_standard_fields(self) -> set:
        """Get standard logging record fields to exclude from custom fields."""
        return {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
            'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'message',
            'taskName', 'getMessage'
        }


class CompressingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler with compression support for space efficiency.
    """
    
    def __init__(self, filename: str, mode: str = 'a', maxBytes: int = 0,
                 backupCount: int = 0, encoding: Optional[str] = None,
                 delay: bool = False, compress: bool = True):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress and _GZIP_AVAILABLE
    
    def doRollover(self):
        """Override to add compression support."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                dfn = self.rotation_filename(f"{self.baseFilename}.{i+1}")
                
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            dfn = self.rotation_filename(f"{self.baseFilename}.1")
            if os.path.exists(dfn):
                os.remove(dfn)
            
            # Rename current file to .1
            self.rotate(self.baseFilename, dfn)
            
            # Compress the rotated file if enabled
            if self.compress:
                self._compress_file(dfn)
        
        if not self.delay:
            self.stream = self._open()
    
    def _compress_file(self, filename: str):
        """Compress a log file using gzip."""
        try:
            compressed_filename = f"{filename}.gz"
            with open(filename, 'rb') as f_in:
                with gzip.open(compressed_filename, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            os.remove(filename)
        except Exception as e:
            # If compression fails, keep the original file
            logging.getLogger(__name__).warning(f"Failed to compress log file {filename}: {e}")


class PerformanceTracker:
    """
    Performance tracking utility for timing operations and resource usage.
    
    Features:
    - Context manager for automatic timing
    - Decorator for function timing
    - Memory usage tracking
    - Operation categorization
    - Performance statistics
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.active_operations = {}
        self.performance_stats = {
            'total_operations': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'operations_by_category': {},
            'slowest_operations': []
        }
        self.lock = threading.Lock()
    
    @contextmanager
    def time_operation(self, operation_name: str, category: str = 'general', **context):
        """Context manager for timing operations."""
        operation_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Record operation start
        with self.lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'category': category,
                'start_time': start_time,
                'start_memory': start_memory,
                'context': context
            }
        
        try:
            self.logger.debug(f"Starting operation: {operation_name} (ID: {operation_id})")
            yield operation_id
            
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            # Remove from active operations
            with self.lock:
                operation_info = self.active_operations.pop(operation_id, {})
                self._update_stats(operation_name, category, duration_ms)
            
            # Log performance result
            extra = {
                'operation': operation_name,
                'operation_id': operation_id,
                'category': category,
                'duration_ms': duration_ms,
                'memory_delta_mb': memory_delta,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                **context
            }
            
            if duration_ms > 1000:  # Log slow operations as warnings
                self.logger.warning(
                    f"Slow operation completed: {operation_name} took {duration_ms:.1f}ms",
                    extra=extra
                )
            else:
                self.logger.info(
                    f"Operation completed: {operation_name} in {duration_ms:.1f}ms",
                    extra=extra
                )
    
    def time_function(self, category: str = 'function', include_args: bool = False):
        """Decorator for timing function calls."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                operation_name = f"{func.__module__}.{func.__name__}"
                context = {}
                
                if include_args:
                    context['args_count'] = len(args)
                    context['kwargs_keys'] = list(kwargs.keys())
                
                with self.time_operation(operation_name, category, **context):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
        return 0.0
    
    def _update_stats(self, operation_name: str, category: str, duration_ms: float):
        """Update performance statistics."""
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_time_ms'] += duration_ms
        self.performance_stats['average_time_ms'] = (
            self.performance_stats['total_time_ms'] / 
            self.performance_stats['total_operations']
        )
        
        # Update category stats
        if category not in self.performance_stats['operations_by_category']:
            self.performance_stats['operations_by_category'][category] = {
                'count': 0,
                'total_time_ms': 0.0,
                'average_time_ms': 0.0
            }
        
        cat_stats = self.performance_stats['operations_by_category'][category]
        cat_stats['count'] += 1
        cat_stats['total_time_ms'] += duration_ms
        cat_stats['average_time_ms'] = cat_stats['total_time_ms'] / cat_stats['count']
        
        # Track slowest operations
        slowest = self.performance_stats['slowest_operations']
        slowest.append({
            'operation': operation_name,
            'category': category,
            'duration_ms': duration_ms,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only top 10 slowest
        slowest.sort(key=lambda x: x['duration_ms'], reverse=True)
        self.performance_stats['slowest_operations'] = slowest[:10]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.lock:
            return {
                **self.performance_stats,
                'active_operations_count': len(self.active_operations),
                'active_operations': list(self.active_operations.values())
            }


class EnhancedLogger:
    """
    Enhanced logger wrapper with additional functionality.
    
    Features:
    - Performance timing integration
    - Structured logging helpers
    - Context management
    - Memory monitoring
    - Session tracking
    """
    
    def __init__(self, name: str, config: LogConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.performance_tracker = PerformanceTracker(self.logger)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            
            console_formatter = EnhancedColorFormatter(
                use_color=self.config.colored_output,
                include_performance=self.config.performance_tracking
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_enabled and self.config.log_directory:
            self._setup_file_handler(log_level)
        
        # Performance handler
        if self.config.performance_tracking and self.config.log_directory:
            self._setup_performance_handler()
    
    def _setup_file_handler(self, log_level):
        """Setup rotating file handler."""
        log_file = self.config.log_directory / f"{self.name.replace('.', '_')}.log"
        
        if self.config.compress_backups:
            file_handler = CompressingRotatingFileHandler(
                filename=str(log_file),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8',
                compress=True
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_file),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
        
        if self.config.structured_logs:
            file_formatter = StructuredJsonFormatter(
                include_system_info=self.config.enable_system_metrics
            )
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s | '
                    '%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in files
        self.logger.addHandler(file_handler)
    
    def _setup_performance_handler(self):
        """Setup dedicated performance logging."""
        perf_log_file = self.config.log_directory / f"performance_{self.name.replace('.', '_')}.log"
        
        perf_handler = CompressingRotatingFileHandler(
            filename=str(perf_log_file),
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding='utf-8',
            compress=self.config.compress_backups
        )
        
        perf_formatter = StructuredJsonFormatter(include_system_info=True)
        perf_handler.setFormatter(perf_formatter)
        perf_handler.setLevel(logging.INFO)
        
        # Create separate performance logger
        perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_logger.setLevel(logging.INFO)
        perf_logger.addHandler(perf_handler)
    
    # Enhanced logging methods
    def debug(self, msg, *args, **kwargs):
        """Enhanced debug logging with context."""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """Enhanced info logging with context."""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Enhanced warning logging with context."""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Enhanced error logging with context."""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """Enhanced critical logging with context."""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def _log_with_context(self, level, msg, *args, **kwargs):
        """Log with enhanced context information."""
        extra = kwargs.pop('extra', {})
        
        # Add memory usage if enabled
        if self.config.enable_system_metrics and _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                extra['memory_mb'] = round(process.memory_info().rss / 1024 / 1024, 1)
            except Exception:
                pass
        
        # Add debug context if enabled
        if self.config.enable_debug_context:
            import inspect
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back.f_back  # Skip _log_with_context frame
                extra['caller_locals_count'] = len(caller_frame.f_locals)
                extra['caller_filename'] = caller_frame.f_code.co_filename
            except (AttributeError, TypeError):
                pass
            finally:
                del frame
        
        kwargs['extra'] = extra
        self.logger._log(level, msg, args, **kwargs)
    
    def log_structured(self, level: str, message: str, **kwargs):
        """Log with structured data."""
        log_func = getattr(self, level.lower(), self.info)
        log_func(message, extra=kwargs)
    
    # Performance timing methods
    def time_operation(self, operation_name: str, category: str = 'general', **context):
        """Context manager for timing operations."""
        return self.performance_tracker.time_operation(operation_name, category, **context)
    
    def time_function(self, category: str = 'function', include_args: bool = False):
        """Decorator for timing functions."""
        return self.performance_tracker.time_function(category, include_args)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_tracker.get_stats()


class LoggingManager:
    """
    Central logging manager with comprehensive configuration and monitoring.
    
    Features:
    - Centralized logger configuration
    - Environment-aware setup
    - Performance monitoring
    - Log cleanup and maintenance
    - Health monitoring
    """
    
    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self.loggers: Dict[str, EnhancedLogger] = {}
        self._setup_directories()
        self._last_cleanup = datetime.now()
    
    def _setup_directories(self):
        """Setup logging directories."""
        if not self.config.log_directory:
            # Default to logs directory in project root
            project_root = Path(__file__).parent.parent
            self.config.log_directory = project_root / "logs"
        
        self.config.log_directory.mkdir(parents=True, exist_ok=True)
    
    def get_logger(self, name: str, config_override: Optional[Dict[str, Any]] = None) -> EnhancedLogger:
        """Get or create enhanced logger with optional configuration override."""
        if name in self.loggers:
            return self.loggers[name]
        
        # Create config for this logger
        logger_config = LogConfig(
            log_level=self.config.log_level,
            console_enabled=self.config.console_enabled,
            file_enabled=self.config.file_enabled,
            colored_output=self.config.colored_output,
            structured_logs=self.config.structured_logs,
            performance_tracking=self.config.performance_tracking,
            max_file_size=self.config.max_file_size,
            backup_count=self.config.backup_count,
            compress_backups=self.config.compress_backups,
            log_directory=self.config.log_directory,
            memory_threshold_mb=self.config.memory_threshold_mb,
            enable_system_metrics=self.config.enable_system_metrics,
            cleanup_days=self.config.cleanup_days,
            environment=self.config.environment,
            enable_debug_context=self.config.enable_debug_context
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
        """Configure logging based on environment."""
        env_configs = {
            "development": {
                "log_level": "DEBUG",
                "console_enabled": True,
                "file_enabled": True,
                "colored_output": True,
                "structured_logs": False,
                "performance_tracking": True,
                "enable_debug_context": True
            },
            "testing": {
                "log_level": "WARNING",
                "console_enabled": False,
                "file_enabled": False,
                "colored_output": False,
                "structured_logs": False,
                "performance_tracking": False,
                "enable_debug_context": False
            },
            "staging": {
                "log_level": "INFO",
                "console_enabled": True,
                "file_enabled": True,
                "colored_output": True,
                "structured_logs": True,
                "performance_tracking": True,
                "enable_debug_context": False
            },
            "production": {
                "log_level": "INFO",
                "console_enabled": False,
                "file_enabled": True,
                "colored_output": False,
                "structured_logs": True,
                "performance_tracking": True,
                "enable_debug_context": False,
                "compress_backups": True
            }
        }
        
        if environment in env_configs:
            config_updates = env_configs[environment]
            for key, value in config_updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self.config.environment = environment
    
    def cleanup_old_logs(self, days: Optional[int] = None):
        """Clean up old log files."""
        if not self.config.log_directory.exists():
            return
        
        days = days or self.config.cleanup_days
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        cleaned_files = 0
        cleaned_size = 0
        
        for log_file in self.config.log_directory.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    cleaned_files += 1
                    cleaned_size += file_size
            except Exception as e:
                logger = self.get_logger("logging_manager")
                logger.warning(f"Failed to clean up log file {log_file}: {e}")
        
        if cleaned_files > 0:
            logger = self.get_logger("logging_manager")
            logger.info(
                f"Cleaned up {cleaned_files} old log files "
                f"({cleaned_size / 1024 / 1024:.1f} MB freed)"
            )
        
        self._last_cleanup = datetime.now()
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive logging system summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.config.environment,
            "active_loggers": len(self.loggers),
            "logger_names": list(self.loggers.keys()),
            "configuration": {
                "log_level": self.config.log_level,
                "console_enabled": self.config.console_enabled,
                "file_enabled": self.config.file_enabled,
                "structured_logs": self.config.structured_logs,
                "performance_tracking": self.config.performance_tracking,
                "log_directory": str(self.config.log_directory)
            },
            "system_metrics": {},
            "performance_summaries": {},
            "log_files": []
        }
        
        # System metrics
        if _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                summary["system_metrics"] = {
                    "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
                    "cpu_percent": process.cpu_percent(),
                    "open_files": len(process.open_files())
                }
            except Exception:
                pass
        
        # Performance summaries
        for name, logger in self.loggers.items():
            try:
                summary["performance_summaries"][name] = logger.get_performance_stats()
            except Exception:
                continue
        
        # Log files information
        if self.config.log_directory.exists():
            for log_file in self.config.log_directory.glob("*.log*"):
                try:
                    stat = log_file.stat()
                    summary["log_files"].append({
                        "name": log_file.name,
                        "size_mb": round(stat.st_size / 1024 / 1024, 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception:
                    continue
        
        # Cleanup information
        summary["last_cleanup"] = self._last_cleanup.isoformat()
        summary["next_cleanup_due"] = (
            self._last_cleanup + timedelta(days=self.config.cleanup_days)
        ).isoformat()
        
        return summary
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Check log directory
            if not self.config.log_directory.exists():
                health["issues"].append("Log directory does not exist")
                health["status"] = "degraded"
            
            # Check disk space
            if self.config.log_directory.exists():
                try:
                    if _PSUTIL_AVAILABLE:
                        disk_usage = psutil.disk_usage(str(self.config.log_directory))
                        free_percent = (disk_usage.free / disk_usage.total) * 100
                        
                        if free_percent < 5:
                            health["issues"].append(f"Low disk space: {free_percent:.1f}% free")
                            health["status"] = "critical"
                        elif free_percent < 15:
                            health["warnings"].append(f"Disk space running low: {free_percent:.1f}% free")
                            if health["status"] == "healthy":
                                health["status"] = "warning"
                except Exception:
                    pass
            
            # Check memory usage
            if _PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    if memory_mb > self.config.memory_threshold_mb * 2:
                        health["issues"].append(f"High memory usage: {memory_mb:.1f}MB")
                        health["status"] = "degraded"
                    elif memory_mb > self.config.memory_threshold_mb:
                        health["warnings"].append(f"Elevated memory usage: {memory_mb:.1f}MB")
                        if health["status"] == "healthy":
                            health["status"] = "warning"
                except Exception:
                    pass
            
            # Check log file sizes
            total_log_size = 0
            large_files = []
            
            if self.config.log_directory.exists():
                for log_file in self.config.log_directory.glob("*.log*"):
                    try:
                        size = log_file.stat().st_size
                        total_log_size += size
                        
                        if size > self.config.max_file_size * 1.5:
                            large_files.append(log_file.name)
                    except Exception:
                        continue
            
            if large_files:
                health["warnings"].append(f"Large log files detected: {', '.join(large_files)}")
                if health["status"] == "healthy":
                    health["status"] = "warning"
            
            # Generate recommendations
            if health["status"] == "healthy":
                health["recommendations"].append("Logging system is operating normally")
            else:
                health["recommendations"].append("Review issues and warnings")
                
                if large_files:
                    health["recommendations"].append("Consider log rotation or cleanup")
                
                if total_log_size > 100 * 1024 * 1024:  # > 100MB
                    health["recommendations"].append("Consider reducing log retention period")
        
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            health["recommendations"] = ["Contact system administrator"]
        
        return health


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def setup_logging(environment: str = "development", 
                 log_directory: Optional[Path] = None,
                 config_overrides: Optional[Dict[str, Any]] = None) -> LoggingManager:
    """
    Setup global logging system with environment-aware configuration.
    
    Args:
        environment: Environment name ("development", "testing", "staging", "production")
        log_directory: Custom log directory path
        config_overrides: Additional configuration overrides
        
    Returns:
        LoggingManager instance
    """
    global _logging_manager
    
    # Create base configuration
    config = LogConfig()
    config.environment = environment
    
    if log_directory:
        config.log_directory = log_directory
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create logging manager
    _logging_manager = LoggingManager(config)
    _logging_manager.configure_for_environment(environment)
    
    return _logging_manager


def get_logger(name: str, config_override: Optional[Dict[str, Any]] = None) -> EnhancedLogger:
    """
    Get enhanced logger instance.
    
    Args:
        name: Logger name (typically module name)
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


# Backward compatibility functions
def setup_logger(name: str, log_file: Optional[str] = None) -> EnhancedLogger:
    """Backward compatibility function for existing code."""
    config_override = {}
    if log_file is not None:
        config_override['file_enabled'] = True
    
    return get_logger(name, config_override)


# Export all public interfaces
__all__ = [
    'LogConfig',
    'EnhancedLogger',
    'LoggingManager',
    'setup_logging',
    'get_logger',
    'setup_logger'  # Backward compatibility
]

# Log successful module initialization
logger = get_logger(__name__)
logger.debug(f"Enhanced logging system loaded successfully - Version 3.2.0")
