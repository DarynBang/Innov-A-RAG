"""
Centralized logging utility for InnovARAG system.
Provides logging to both console and file with consistent formatting.
Creates timestamped log files for each run.
"""
import logging
import os
from datetime import datetime

# Global flag to track if logging has been initialized
_logging_initialized = False

def generate_timestamped_logfile(base_name: str = "assistant") -> str:
    """
    Generate a timestamped log filename.
    
    Args:
        base_name: Base name for the log file (default: "assistant")
        
    Returns:
        str: Timestamped log filename (e.g., "assistant_2025-01-31_12-34-56.log")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base_name}_{timestamp}.log"

def setup_logging(logfile: str = None, level: int = logging.INFO, use_timestamp: bool = True):
    """
    Set up logging to both console and file.
    
    Args:
        logfile: Path to the log file (if None, uses timestamped filename)
        level: Logging level (default: INFO)
        use_timestamp: Whether to use timestamped log files (default: True)
    """
    global _logging_initialized
    
    # Only initialize once per process
    if _logging_initialized:
        return
    
    # Generate timestamped logfile if not provided or if use_timestamp is True
    if logfile is None or use_timestamp:
        if logfile is None:
            logfile = generate_timestamped_logfile("assistant")
        else:
            # Extract base name and extension
            base_name = os.path.splitext(logfile)[0]
            logfile = generate_timestamped_logfile(base_name)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(logfile) if os.path.dirname(logfile) else "logs"
    if log_dir and log_dir != "." and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # Use full path for log file if directory was created
    if log_dir and log_dir != ".":
        logfile = os.path.join(log_dir, os.path.basename(logfile))
    
    # File handler - create new log file each time
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    _logging_initialized = True
    logger.info(f"Logging initialized - Console: {level}, File: {logfile}")
    
    return logfile

def get_logger(name: str):
    """Get a logger with the specified name."""
    return logging.getLogger(name)


