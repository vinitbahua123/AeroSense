"""
Structured logging configuration using loguru.
Every module imports: from src.utils.logger import logger
"""
import sys
from loguru import logger
from src.config import LOG_LEVEL, LOGS_DIR

# Remove default logger
logger.remove()

# Console output (colorful, human-readable)
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# File output (JSON format, machine-readable, rotated daily)
logger.add(
    LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="1 day",     # New file every day
    retention="30 days",  # Keep logs for 30 days
    compression="zip",    # Compress old logs
)
