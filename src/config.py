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

    telegram_bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = os.getenv("TELEGRAM_CHAT_ID")

    supabase_url: str | None = os.getenv("SUPABASE_URL")
    supabase_anon_key: str | None = os.getenv("SUPABASE_ANON_KEY")
    supabase_service_key: str | None = os.getenv("SUPABASE_SERVICE_KEY")

    stop_loss: float = float(os.getenv("STOP_LOSS", "-0.03"))
    take_profit: float = float(os.getenv("TAKE_PROFIT", "0.08"))

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.raw_data_dir.mkdir(parents=True, exist_ok=True)
        cls.processed_data_dir.mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
