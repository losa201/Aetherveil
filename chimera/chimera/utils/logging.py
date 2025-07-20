"""
Enhanced logging configuration for Chimera
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for enhanced log analysis
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields from record
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
            
        if hasattr(record, 'persona'):
            log_entry["persona"] = record.persona
            
        if hasattr(record, 'target'):
            log_entry["target"] = record.target
            
        if hasattr(record, 'operation'):
            log_entry["operation"] = record.operation
            
        return json.dumps(log_entry)

class ChimeraLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds Chimera-specific context
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Add context from adapter
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        
        return msg, kwargs

def setup_logging(level: str = "INFO", file_path: str = "./data/logs/chimera.log", 
                 structured: bool = True, console_output: bool = True) -> logging.Logger:
    """
    Setup enhanced logging for Chimera
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_path: Path to log file
        structured: Whether to use structured JSON logging
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory
    log_file = Path(file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    if structured:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Use simpler format for console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    _configure_module_loggers(level)
    
    # Create main Chimera logger
    chimera_logger = logging.getLogger("chimera")
    
    # Add security-focused logging
    _setup_security_logging(file_path)
    
    chimera_logger.info("Logging system initialized", extra={
        "level": level,
        "file_path": file_path,
        "structured": structured
    })
    
    return chimera_logger

def _configure_module_loggers(level: str):
    """Configure logging for specific modules"""
    
    # Suppress noisy third-party loggers
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Set Chimera module levels
    chimera_modules = [
        "chimera.core",
        "chimera.reasoner", 
        "chimera.memory",
        "chimera.web",
        "chimera.llm",
        "chimera.planner",
        "chimera.executor",
        "chimera.validator",
        "chimera.reporter"
    ]
    
    for module in chimera_modules:
        logging.getLogger(module).setLevel(getattr(logging, level.upper()))

def _setup_security_logging(base_path: str):
    """Setup security-specific logging"""
    
    # Create security audit logger
    security_logger = logging.getLogger("chimera.security")
    
    # Security log file
    security_log_path = Path(base_path).parent / "security_audit.log"
    
    security_handler = logging.handlers.RotatingFileHandler(
        security_log_path,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    
    # Structured format for security logs
    security_formatter = StructuredFormatter()
    security_handler.setFormatter(security_formatter)
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger
    security_logger.propagate = False

def get_logger(name: str, **context) -> ChimeraLoggerAdapter:
    """
    Get a logger with Chimera-specific context
    
    Args:
        name: Logger name
        **context: Additional context to include in logs
        
    Returns:
        Logger adapter with context
    """
    
    logger = logging.getLogger(name)
    return ChimeraLoggerAdapter(logger, context)

def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "INFO"):
    """
    Log security-relevant events
    
    Args:
        event_type: Type of security event
        details: Event details
        severity: Log severity level
    """
    
    security_logger = logging.getLogger("chimera.security")
    
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details
    }
    
    level = getattr(logging, severity.upper())
    security_logger.log(level, f"Security Event: {event_type}", extra=log_data)

def log_operation(operation: str, target: str, persona: str, **kwargs):
    """
    Log operational activities
    
    Args:
        operation: Operation being performed
        target: Target of operation
        persona: Active persona
        **kwargs: Additional operation details
    """
    
    logger = logging.getLogger("chimera.operations")
    
    extra_data = {
        "operation": operation,
        "target": target,
        "persona": persona,
        **kwargs
    }
    
    logger.info(f"Operation: {operation} on {target}", extra=extra_data)

def log_performance(operation: str, duration: float, success: bool, **metrics):
    """
    Log performance metrics
    
    Args:
        operation: Operation name
        duration: Duration in seconds
        success: Whether operation succeeded
        **metrics: Additional performance metrics
    """
    
    logger = logging.getLogger("chimera.performance")
    
    perf_data = {
        "operation": operation,
        "duration_seconds": duration,
        "success": success,
        "metrics": metrics
    }
    
    logger.info(f"Performance: {operation}", extra=perf_data)

# Context managers for structured logging

class LoggedOperation:
    """Context manager for logging operations with timing"""
    
    def __init__(self, operation: str, logger: logging.Logger, **context):
        self.operation = operation
        self.logger = logger
        self.context = context
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting {self.operation}", extra=self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        success = exc_type is None
        
        self.context.update({
            "duration_seconds": duration,
            "success": success
        })
        
        if success:
            self.logger.info(f"Completed {self.operation}", extra=self.context)
        else:
            self.context["error"] = str(exc_val)
            self.logger.error(f"Failed {self.operation}", extra=self.context)

class CorrelatedLogs:
    """Context manager for correlated logs with ID"""
    
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.original_factory = logging.getLogRecordFactory()
        
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.original_factory(*args, **kwargs)
            record.correlation_id = self.correlation_id
            return record
            
        logging.setLogRecordFactory(record_factory)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.original_factory)