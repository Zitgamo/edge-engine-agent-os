"""Time helpers for the Vietnam trading timezone."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def now_vn() -> datetime:
    """Return the current timezone-aware datetime in Vietnam."""
    return datetime.now(VN_TZ)


def today_vn() -> date:
    """Return today's calendar date in Vietnam."""
    return now_vn().date()

