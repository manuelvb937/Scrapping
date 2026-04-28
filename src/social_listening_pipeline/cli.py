"""Command-line interface for the social listening pipeline."""

from __future__ import annotations

import argparse
import logging

from .config import load_settings
from .logging_setup import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="social-listening",
        description="Scaffold CLI for the social listening pipeline.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved runtime configuration and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.log_level)

    logger.info("Starting social-listening CLI in '%s' environment", settings.app_env)

    if args.show_config:
        print("Resolved configuration:")
        print(f"  APP_ENV: {settings.app_env}")
        print(f"  LOG_LEVEL: {settings.log_level}")
        print(f"  PROJECT_ROOT: {settings.project_root}")
        print(f"  DATA_RAW_DIR: {settings.data_raw_dir}")
        print(f"  DATA_PROCESSED_DIR: {settings.data_processed_dir}")
        print(f"  REPORTS_DIR: {settings.reports_dir}")
        return

    logger.info("Project scaffold is ready. No scraping is implemented yet.")


if __name__ == "__main__":
    main()
