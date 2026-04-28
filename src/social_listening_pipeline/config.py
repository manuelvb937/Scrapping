"""Configuration management for the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    project_root: Path = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
    data_raw_dir: Path = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    data_processed_dir: Path = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", "data/reports"))

    def resolved(self) -> "Settings":
        """Return settings with project-root-relative paths resolved to absolute paths."""
        return Settings(
            app_env=self.app_env,
            log_level=self.log_level,
            project_root=self.project_root,
            data_raw_dir=(self.project_root / self.data_raw_dir).resolve(),
            data_processed_dir=(self.project_root / self.data_processed_dir).resolve(),
            reports_dir=(self.project_root / self.reports_dir).resolve(),
        )


def load_settings() -> Settings:
    """Build and return application settings."""
    return Settings().resolved()
