import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 280)

from src.accumulation import compute_ranking_history, simulate_dca_portfolio, print_portfolio

print("Loading features...")
features = pd.read_parquet("data/processed/features.parquet")
features["date"] = pd.to_datetime(features["date"])
print(f"Features: {len(features)} rows, {features['ticker'].nunique()} tickers")
print(f"Range: {features['date'].min().date()} -> {features['date'].max().date()}")

print("Building prices dict...")
prices = {}
for t, grp in features.groupby("ticker"):
    prices[t] = grp.sort_values("date")[["date", "close"]].copy()

print("Ranking (52 weeks)...")
feat = features[features["date"] >= features["date"].max() - pd.Timedelta(weeks=52)]
ranking = compute_ranking_history(feat, top_n=5)
print(f"Ranking: {len(ranking)} rows, {ranking['date'].nunique()} dates")

print("\nRun portfolio (3M/thang, top 3)...")
history = simulate_dca_portfolio(ranking, prices, monthly_amount=3_000_000, top_n=3)

print_portfolio(history)

latest = history[history["date"] == history["date"].max()]
print(f"\nLatest signals ({latest.iloc[0]['date'].date()}):")
for _, r in latest.iterrows():
    pct = r["pnl_pct"] * 100
    print(f"  {r['ticker']:<8} {r['signal']:<12} P&L: {r['pnl']:>+10,.0f} ({pct:>+5.1f}%)")
