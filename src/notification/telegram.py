from __future__ import annotations

import logging

import requests

from src.config import Config

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_signal(signal_df, strategy_name: str = "ensemble") -> bool:
    """Send today's signal to Telegram chat."""
    token = Config.telegram_bot_token
    chat_id = Config.telegram_chat_id

    if not token or not chat_id or token == "" or chat_id == "":
        log.info("Telegram not configured — skipping notification")
        return False

    if signal_df.empty:
        log.warning("Empty signal — nothing to send")
        return False

    date = signal_df["signal_date"].iloc[0] if "signal_date" in signal_df.columns else ""
    picks = []
    for _, row in signal_df.iterrows():
        ticker = row.get("ticker", "?")
        rank = row.get("rank", "?")
        score = row.get("score", 0)
        sl = row.get("stop_loss", Config.stop_loss)
        tp = row.get("take_profit", Config.take_profit)
        picks.append(f"  #{rank} {ticker}  score={score:.4f}  SL={sl:+.0%}  TP={tp:+.0%}")

    lines = [
        f"\U0001f4c8 *Edge Engine Signal*",
        f"Date: {date}",
        f"Strategy: {strategy_name}",
        "",
        "Top Picks:",
    ] + picks + [
        "",
        "_T+20 holding | N=3_",
    ]

    text = "\n".join(lines)

    try:
        resp = requests.post(
            TELEGRAM_API.format(token=token),
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            log.info("Telegram notification sent successfully")
            return True
        else:
            log.warning("Telegram API error: %s %s", resp.status_code, resp.text[:200])
            return False
    except requests.RequestException as e:
        log.warning("Telegram request failed: %s", e)
        return False


def send_message(message: str) -> bool:
    """Send a plain message to Telegram."""
    token = Config.telegram_bot_token
    chat_id = Config.telegram_chat_id

    if not token or not chat_id or token == "" or chat_id == "":
        log.info("Telegram not configured — skipping")
        return False

    try:
        resp = requests.post(
            TELEGRAM_API.format(token=token),
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException as e:
        log.warning("Telegram request failed: %s", e)
        return False
