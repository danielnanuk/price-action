"""Typed YAML configuration loader for the Price Action pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(Exception):
    """Raised when config file is missing, malformed, or fails validation."""


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    members_file: Path


class DateRangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: date
    end: date


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_dir: Path
    api_base_url: str
    api_key_env: str
    rate_limit_per_min: int = Field(gt=0)
    timespan: str = "day"
    timespan_multiplier: int = Field(default=1, gt=0)


class DetectorsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: list[str]
    param_tiers: list[str]


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_stop_bars: int = Field(gt=0)
    same_bar_priority: Literal["stop_first", "target_first"]


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: Path
    samples_per_setup: int = Field(ge=0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    universe: UniverseConfig
    date_range: DateRangeConfig
    data: DataConfig
    detectors: DetectorsConfig
    backtest: BacktestConfig
    report: ReportConfig


def load_config(path: Path) -> Config:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
        return Config.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"Invalid config {path}: {exc}") from exc
