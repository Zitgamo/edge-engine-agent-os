from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Config

log = logging.getLogger(__name__)


class PriceStorage:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def save_raw(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.config.raw_data_dir / filename
        df.to_parquet(path)
        log.info("Saved raw data to %s", path)
        return path

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.config.processed_data_dir / filename
        df.to_parquet(path)
        log.info("Saved processed data to %s", path)
        return path

    def load_raw(self, filename: str) -> pd.DataFrame:
        path = self.config.raw_data_dir / filename
        return pd.read_parquet(path)

    def load_processed(self, filename: str) -> pd.DataFrame:
        path = self.config.processed_data_dir / filename
        return pd.read_parquet(path)

    def load_all_raw(self) -> pd.DataFrame:
        files = list(self.config.raw_data_dir.glob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def load_all_processed(self) -> pd.DataFrame:
        files = list(self.config.processed_data_dir.glob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
