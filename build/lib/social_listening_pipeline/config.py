"""Configuration management for the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_topic_labeling_model() -> str:
    explicit_model = os.getenv("TOPIC_LABELING_MODEL")
    if explicit_model:
        return explicit_model

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "gemini" or os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "dev"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    project_root: Path = field(default_factory=lambda: _env_path("PROJECT_ROOT", Path.cwd()).resolve())
    data_raw_dir: Path = field(default_factory=lambda: _env_path("DATA_RAW_DIR", "data/raw"))
    data_processed_dir: Path = field(default_factory=lambda: _env_path("DATA_PROCESSED_DIR", "data/processed"))
    reports_dir: Path = field(default_factory=lambda: _env_path("REPORTS_DIR", "data/reports"))
    sentiment_method: str = field(default_factory=lambda: os.getenv("SENTIMENT_METHOD", "hybrid").lower())
    sentiment_model_name: str = field(
        default_factory=lambda: os.getenv(
            "SENTIMENT_MODEL_NAME",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        )
    )
    sentiment_confidence_threshold: float = field(
        default_factory=lambda: _env_float("SENTIMENT_CONFIDENCE_THRESHOLD", 0.70)
    )
    enable_llm_sentiment_fallback: bool = field(
        default_factory=lambda: _env_bool("ENABLE_LLM_SENTIMENT_FALLBACK", True)
    )
    topic_labeling_model: str = field(default_factory=_env_topic_labeling_model)

    def resolved(self) -> "Settings":
        """Return settings with project-root-relative paths resolved to absolute paths."""
        return Settings(
            app_env=self.app_env,
            log_level=self.log_level,
            project_root=self.project_root,
            data_raw_dir=(self.project_root / self.data_raw_dir).resolve(),
            data_processed_dir=(self.project_root / self.data_processed_dir).resolve(),
            reports_dir=(self.project_root / self.reports_dir).resolve(),
            sentiment_method=self.sentiment_method,
            sentiment_model_name=self.sentiment_model_name,
            sentiment_confidence_threshold=self.sentiment_confidence_threshold,
            enable_llm_sentiment_fallback=self.enable_llm_sentiment_fallback,
            topic_labeling_model=self.topic_labeling_model,
        )


def load_settings() -> Settings:
    """Build and return application settings."""
    return Settings().resolved()
