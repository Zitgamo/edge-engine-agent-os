from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

log = logging.getLogger(__name__)


class Strategy(ABC):
    name: str = ""
    description: str = ""
    requires_ml: bool = False

    @abstractmethod
    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def features_needed(self) -> list[str]:
        return []
