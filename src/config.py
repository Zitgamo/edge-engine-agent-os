import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    data_source: str = os.getenv("DATA_SOURCE", "yfinance")
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/xgboost_model.json"))
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8501"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    ssi_api_key: str | None = os.getenv("SSI_API_KEY")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.raw_data_dir.mkdir(parents=True, exist_ok=True)
        cls.processed_data_dir.mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
