from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

HOSE_TICKERS: list[str] = [
    # VN30 (30 tickers)
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "KDH", "MBB", "MSN", "MWG", "NVL", "PNJ", "POW", "SAB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    # Banking & Finance
    "EIB", "LPB", "MSB", "NAB", "OCB", "SHB", "VBB", "BAB", "KLB",
    # Securities
    "BSI", "VCI", "VND", "FTS", "MBS", "AGR", "SHS", "ORS", "TVB", "HCM",
    # Real Estate & Construction
    "DIG", "DXG", "HDG", "KBC", "NLG", "PDR", "SCR", "SJS", "CTD", "CTI",
    "FCN", "LCG", "VCG", "HBC", "NDN", "HDC", "SZC",
    # Steel & Materials
    "HSG", "NKG", "POM", "TLH", "BMP", "DGC", "DCM", "DPM",
    # Consumer & Retail
    "DBC", "DGW", "FRT", "PET", "SBT", "HAX", "PAN",
    # Pharmaceuticals
    "DHG", "DCL", "IMP", "TRA",
    # Logistics
    "GMD", "VSC", "TCL", "HAH",
    # Textile & Garment
    "TCM", "MSH", "GIL", "TNG",
    # Utilities & Energy
    "REE", "NT2", "VSH", "PC1", "QTP", "PPC",
    # Food & Beverage
    "LIX", "RAL", "TIP", "VHC", "SGC", "KDC",
    # Others (liquid)
    "GEX", "YEG", "VTP", "DPR", "ITA", "PHR", "TDM",
    "CAP", "COM", "D2D", "LHG", "PAC", "SAM", "SFC",
    "SJD", "SMC", "TNC", "TNT", "TVC", "VLF", "VTO",
]

VN30_TICKERS: list[str] = HOSE_TICKERS[:30]

MIN_HISTORY_DAYS = 200
MIN_AVG_VOLUME = 50_000


def get_ticker_universe() -> list[str]:
    return HOSE_TICKERS


def filter_quality(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if df.empty:
        log.warning("Skipping %s: no data", ticker)
        return None
    if len(df) < MIN_HISTORY_DAYS:
        log.warning("Skipping %s: only %d days (need %d)", ticker, len(df), MIN_HISTORY_DAYS)
        return None
    avg_vol = df["volume"].mean()
    if avg_vol < MIN_AVG_VOLUME:
        log.warning("Skipping %s: avg volume %.0f (need %d)", ticker, avg_vol, MIN_AVG_VOLUME)
        return None
    return df
