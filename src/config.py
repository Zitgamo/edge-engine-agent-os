import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_horizon_weights(name: str, default: str) -> dict[int, float]:
    """Parse comma-separated horizon weights such as ``5:0.4,10:0.3``."""
    raw = os.getenv(name, default)
    weights: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            horizon_text, weight_text = item.split(":", 1)
            horizon = int(horizon_text.strip())
            weight = float(weight_text.strip())
        except (TypeError, ValueError):
            continue
        if horizon > 0 and weight >= 0:
            weights[horizon] = weight
    return weights


class Config:
    data_source: str = os.getenv("DATA_SOURCE", "yfinance")
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/xgboost_model_h20.json"))
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
    # Conservative default for a complete buy/sell round trip. Override with
    # the broker-specific all-in cost when deploying.
    round_trip_cost: float = float(os.getenv("ROUND_TRIP_COST", "0.003"))
    # Use execution-aligned labels for T+20 when a fresh pipeline run builds
    # the feature file. Older files fall back to close-to-close columns.
    execution_target_enabled: bool = _env_bool("EXECUTION_TARGET_ENABLED", True)
    # T+1 is noisy for a T+20 portfolio. Keep it out of the default blend,
    # while allowing deployment-specific weights without code changes.
    ensemble_horizon_weights: dict[int, float] = _env_horizon_weights(
        "ENSEMBLE_HORIZON_WEIGHTS", "1:0.0,5:0.0,10:0.0,20:1.0"
    )
    ensemble_blend_mode: str = os.getenv("ENSEMBLE_BLEND_MODE", "raw").strip().lower()
    primary_ranking_strategy: str = os.getenv(
        "PRIMARY_RANKING_STRATEGY", "outperform"
    ).strip().lower()
    # Fail closed in weak market regimes; set this false only for a controlled
    # paper-test comparison.
    enable_entry_filters: bool = _env_bool("ENABLE_ENTRY_FILTERS", True)
    # Trend/breakout/fundamental modules remain available for research, but
    # their latest blocked OOS run does not justify letting them dilute the
    # production ensemble by default.
    enable_research_strategies: bool = _env_bool("ENABLE_RESEARCH_STRATEGIES", False)
    min_market_breadth_20d: float = float(os.getenv("MIN_MARKET_BREADTH_20D", "0.50"))
    min_entry_atr_percentile: float = float(os.getenv("MIN_ENTRY_ATR_PERCENTILE", "0.20"))
    min_entry_trend_percentile: float = float(os.getenv("MIN_ENTRY_TREND_PERCENTILE", "0.00"))
    min_entry_picks: int = int(os.getenv("MIN_ENTRY_PICKS", "3"))
    min_entry_score: float = float(os.getenv("MIN_ENTRY_SCORE", "0.10"))
    min_entry_score_margin: float = float(
        os.getenv("MIN_ENTRY_SCORE_MARGIN", "0.02")
    )
    # Keep the supervised model aligned with the current market regime.  A
    # bounded window also prevents stale 2025 observations from dominating a
    # 2026 signal when the cross-sectional regime has changed.
    model_training_days: int = int(os.getenv("MODEL_TRAINING_DAYS", "180"))
    model_quality_test_days: int = int(os.getenv("MODEL_QUALITY_TEST_DAYS", "40"))
    min_model_quality_dates: int = int(os.getenv("MIN_MODEL_QUALITY_DATES", "30"))
    min_model_roc_auc: float = float(os.getenv("MIN_MODEL_ROC_AUC", "0.52"))
    min_model_top3_excess_return: float = float(
        os.getenv("MIN_MODEL_TOP3_EXCESS_RETURN", "0.0")
    )
    min_model_top3_spread: float = float(
        os.getenv("MIN_MODEL_TOP3_SPREAD", "0.0")
    )

    def ensure_dirs(self) -> None:
        """Create directories using this config instance's paths."""
        Path(self.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def model_path_for_horizon(self, horizon: int) -> Path:
        """Return the persisted path for an ensemble horizon.

        ``MODEL_PATH`` is the primary production model used by inference and
        therefore owns the execution-aligned T+20 model.  Auxiliary horizon
        models keep their conventional paths so an override cannot silently
        be ignored for the model that inference actually loads.
        """
        if int(horizon) == 20:
            return Path(self.model_path)
        return Path("models") / f"xgboost_model_h{int(horizon)}.json"
