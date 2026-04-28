"""Logging setup utilities."""

import logging


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logger for the CLI application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
