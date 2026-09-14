"""Isolated US ETF research prototype. No orders, cloud writes or alerts.

Fixed rule: every 20 sessions select the strongest 126-session return among
SPY/QQQ/IWM/GLD/TLT above its 200-session mean, else cash. Signal at prior
close, execution at next open. Fractional shares, no leverage, cash earns 0.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']


def targets(close):
    momentum = close.pct_change(126, fill_method=None)
    allowed = (close > close.rolling(200).mean()) & (momentum > 0)
    score = momentum.where(allowed)
    return score.apply(lambda row: row.idxmax() if row.notna().any() else 'CASH', axis=1)


def simulate(data, cost, benchmark=False):
    close, opens = data['Close'][SYMBOLS], data['Open'][SYMBOLS]
    target = targets(close)
    returns, events = [], []
    equity, position, units, cash = 1.0, 'CASH', 0.0, 1.0
    start = int(opens.index.searchsorted(pd.Timestamp('2016-01-01')))
    for i in range(start, len(opens) - 1):
        today, tomorrow = opens.index[i], opens.index[i + 1]
        before = cash if position == 'CASH' else units * opens.iloc[i][position]
        if (i - start) % 20 == 0:
            desired = 'SPY' if benchmark else target.iloc[i - 1]
            if desired != position:
                capital = before * (1 - cost if position != 'CASH' else 1)
                if desired == 'CASH':
                    cash, units = capital, 0.0
                else:
                    units, cash = capital * (1 - cost) / opens.iloc[i][desired], 0.0
                events.append(dict(signal_date=str(opens.index[i - 1].date()),
                    execution_date=str(today.date()), old=position, target=desired))
                position = desired
        equity = cash if position == 'CASH' else units * opens.iloc[i + 1][position]
        returns.append(dict(date=tomorrow, daily_return=equity / before - 1,
                            equity=equity, position=position))
    return pd.DataFrame(returns).set_index('date'), events


def main():
    path = ROOT / 'us_etf_adjusted.parquet'
    data = pd.read_parquet(path).sort_index()
    assert not data.index.duplicated().any()
    for field in ['Open', 'Close']:
        assert data[field][SYMBOLS].notna().all().all()
        assert (data[field][SYMBOLS] > 0).all().all()
    close = data['Close'][SYMBOLS]
    # Future bars cannot alter historical targets.
    pd.testing.assert_series_equal(targets(close).iloc[:1000], targets(close.iloc[:1000]))
    assert (targets(close.iloc[:199]) == 'CASH').all()
    rows = []
    for name, cost, benchmark in [('rotation_10bps_side', .001, False),
                                   ('rotation_30bps_side', .003, False),
                                   ('SPY_buy_hold_10bps_side', .001, True)]:
        curve, events = simulate(data, cost, benchmark)
        curve.to_csv(ROOT / f'{name}_curve.csv')
        pd.DataFrame(events).to_csv(ROOT / f'{name}_events.csv', index=False)
        for period, start, end in [('2016-2023', '2016-01-01', '2023-12-31'),
                                   ('2024-2025', '2024-01-01', '2025-12-31'),
                                   ('2026', '2026-01-01', '2026-09-11')]:
            p = curve.loc[start:end]
            equity = np.r_[1.0, (1 + p.daily_return).cumprod().to_numpy()]
            total = equity[-1] - 1
            rows.append(dict(strategy=name, period=period, sessions=len(p),
                total_return=total, annualized_return=(1 + total)**(252/len(p))-1,
                max_drawdown=float((equity / np.maximum.accumulate(equity) - 1).min()),
                invested_session_fraction=float((p.position != 'CASH').mean()),
                position_changes=sum(start <= e['execution_date'] <= end for e in events)))
    # Friction must not improve the final value when targets are unchanged.
    low, _ = simulate(data, .001)
    high, _ = simulate(data, .003)
    assert high.equity.iloc[-1] <= low.equity.iloc[-1]
    report = pd.DataFrame(rows)
    report.to_csv(ROOT / 'etf_comparison.csv', index=False)
    (ROOT / 'etf_manifest.json').write_text(json.dumps(dict(
        source='Yahoo Finance via yfinance 1.5.2; auto_adjust=True',
        retrieved='2026-09-14', latest_bar=str(data.index.max().date()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameters=dict(momentum_sessions=126, sma_sessions=200, rebalance_sessions=20),
        status='research_only_not_promoted',
        checks=['historical target causality', 'warmup cash', 'higher cost reduces final equity', 'complete positive bars'],
        limitations=['One data vendor, adjusted prices are approximations to total returns',
            'No taxes, FX, account restrictions or cash interest',
            'No terminal liquidation cost; open positions marked to market',
            'Period returns include positions carried across boundaries',
            'No fresh forward evidence; universe chosen retrospectively',
            '20-session cadence is not a promise of daily entry signals']
    ), indent=2), encoding='utf-8')
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
