"""
Logging utilities for bzanalysis package.

This module provides a centralized logging configuration for the entire package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Name of the logger (usually __name__)
        level: Logging level (default: logging.INFO)
        log_file: Optional path to log file
        console: Whether to log to console (default: True)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # For module loggers (child loggers), don't set level or add handlers
    # They will inherit from parent and propagate to root
    if '.' in name:
        # Don't set level - let it inherit from parent/root
        return logger
    
    # For root or top-level loggers, set the level
    logger.setLevel(level)
    
    # Check if already has handlers
    if logger.hasHandlers() and len(logger.handlers) > 0:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger