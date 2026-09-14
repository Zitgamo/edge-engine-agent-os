"""Exploratory strategy pivot audit; cached data only, no publication.

Run from repository root. No tuning or promotion: this cache has already
been inspected, and its static VN30 membership has survivorship bias.
"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.universe import VN30_TICKERS

OUT = Path(__file__).resolve().parent
CACHE = ROOT / 'data/research_kbs_5y'
FEATURES = CACHE / 'processed/features_exact.parquet'
OUTCOMES = CACHE / 'research_results/outcomes_atr2_tp10.parquet'
COLS = ['rs_5d', 'rs_20d', 'rs_60d', 'return_5d', 'return_20d', 'return_60d']


def main():
    f = pd.read_parquet(FEATURES)
    f['date'] = pd.to_datetime(f['date'])
    f = f[f.ticker.isin(VN30_TICKERS)].dropna(subset=COLS + ['atr_pct'])
    assert not f.duplicated(['date', 'ticker']).any()
    f = f[f.groupby('date').ticker.transform('size') == 30].copy()
    f['breadth'] = f.groupby('date').return_20d.transform(lambda s: (s > 0).mean())
    ranks = f.groupby('date')[COLS].rank(pct=True)
    f['momentum'] = ranks.mean(axis=1)
    f['reversion'] = ((1 - ranks).sum(axis=1) + f.groupby('date').atr_pct.rank(pct=True)) / 7
    o = pd.read_parquet(OUTCOMES)
    for col in ['date', 'exit_date', 'label_end_date']:
        o[col] = pd.to_datetime(o[col])
    assert not o.duplicated(['date', 'ticker']).any()
    assert ((o.gross_return - o.net_return - .003).abs() < 1e-9).all()
    periods = [('2021-2023', '2021-01-01', '2023-12-31'),
               ('2024-2025', '2024-01-01', '2025-12-31'),
               ('2026_seen', '2026-01-01', '2026-08-27')]
    policies = {
        'momentum_breadth60': (f.breadth >= .6, f.momentum),
        'reversion_weak': (f.breadth < .5, f.reversion),
        'regime_switch': (pd.Series(True, index=f.index), f.momentum.where(f.breadth >= .5, f.reversion)),
    }
    rows = []
    for name, (mask, score) in policies.items():
        picks = f.assign(score=score).loc[mask].sort_values(
            ['date', 'score', 'ticker'], ascending=[True, False, True]
        ).groupby('date').head(3)
        trades = picks.merge(o, on=['date', 'ticker'], validate='one_to_one')
        for period, start, end in periods:
            # All labels mature within the period; purge boundary-crossing trades.
            t = trades[(trades.date >= start) & (trades.label_end_date <= end)].copy()
            t = t[t.groupby('date').ticker.transform('size') == 3]
            calendar = sorted(f.loc[f.date.between(start, end), 'date'].unique())
            selected = set(picks.loc[picks.date.between(start, end), 'date'])
            longest = streak = 0
            for date in calendar:
                streak = 0 if pd.Timestamp(date) in selected else streak + 1
                longest = max(longest, streak)
            wins = t.loc[t.net_return > 0, 'net_return'].sum()
            losses = -t.loc[t.net_return < 0, 'net_return'].sum()
            rows.append(dict(strategy=name, period=period, eligible_sessions=len(calendar),
                signal_sessions=len(selected), max_no_signal_sessions=longest,
                completed_baskets=t.date.nunique(), trades=len(t),
                avg_net_return=t.net_return.mean(), avg_excess_return=t.excess_return.mean(),
                win_rate=(t.net_return > 0).mean(), profit_factor=wins / losses if losses else None,
                worst_trade=t.net_return.min(), status='exploratory_only'))
    report = pd.DataFrame(rows)
    report.to_csv(OUT / 'comparison.csv', index=False)
    (OUT / 'manifest.json').write_text(json.dumps({
        'features_sha256': hashlib.sha256(FEATURES.read_bytes()).hexdigest(),
        'outcomes_sha256': hashlib.sha256(OUTCOMES.read_bytes()).hexdigest(),
        'feature_end': str(f.date.max()), 'cost': .003, 'exit': 'ATR2 / TP10 / T20',
        'limitations': ['Previously seen data; not a fresh holdout',
            'Static VN30 membership, not point-in-time constituents',
            'Overlapping daily baskets are correlated, not independent samples',
            'Cached execution model does not establish liquidity or actual fills',
            'Signal frequency counts complete-feature sessions, not an exchange calendar',
            'No capital portfolio simulation or automatic promotion']
    }, indent=2), encoding='utf-8')
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
