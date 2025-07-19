"""
Centralized logging utility for InnovARAG system.
Provides logging to both console and file with consistent formatting.
"""
import logging
import os

# Global flag to track if logging has been initialized
_logging_initialized = False

def setup_logging(logfile: str = "assistant.log", level: int = logging.INFO):
    """
    Set up logging to both console and file.
    
    Args:
        logfile: Path to the log file
        level: Logging level (default: INFO)
    """
    global _logging_initialized
    
    # Only initialize once per process
    if _logging_initialized:
        return
    
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
    
    # File handler
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    _logging_initialized = True
    logger.info(f"Logging initialized - Console: {level}, File: {logfile}")

def get_logger(name: str):
    """Get a logger with the specified name."""
    return logging.getLogger(name)