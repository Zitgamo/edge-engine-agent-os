-- Run this in Supabase SQL Editor (https://supabase.com/dashboard/project/_/sql/new)
-- One-time setup: creates tables for Edge Engine dashboard

CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    signal_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    ensemble_score REAL,
    stop_loss REAL,
    take_profit REAL,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(signal_date, ticker)
);

CREATE TABLE IF NOT EXISTS actuals (
    id BIGSERIAL PRIMARY KEY,
    signal_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    actual_excess_return_5d REAL,
    actual_outperform INTEGER,
    realized_date DATE NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(signal_date, ticker)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    run_date TIMESTAMPTZ DEFAULT NOW(),
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1 REAL,
    roc_auc REAL,
    status TEXT
);

CREATE TABLE IF NOT EXISTS strategy_performance (
    id BIGSERIAL PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    signal_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    rank INTEGER,
    score REAL,
    actual_excess_return_5d REAL,
    actual_outperform INTEGER,
    realized INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(strategy_name, signal_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_actuals_signal ON actuals(signal_date, ticker);
CREATE INDEX IF NOT EXISTS idx_strategy_date ON strategy_performance(signal_date);
