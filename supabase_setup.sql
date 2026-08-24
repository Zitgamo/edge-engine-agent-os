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
    actual_excess_return_20d REAL,
    actual_stock_return REAL,
    benchmark_return REAL,
    gross_stock_return REAL,
    transaction_cost REAL,
    actual_outperform INTEGER,
    realized_date DATE NOT NULL,
    entry_date DATE,
    entry_price REAL,
    exit_price REAL,
    execution_status TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(signal_date, ticker)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    run_key TEXT,
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
    actual_excess_return_20d REAL,
    actual_outperform INTEGER,
    realized INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(strategy_name, signal_date, ticker)
);

ALTER TABLE actuals ADD COLUMN IF NOT EXISTS actual_excess_return_20d REAL;
ALTER TABLE actuals ADD COLUMN IF NOT EXISTS actual_stock_return REAL;
ALTER TABLE actuals ADD COLUMN IF NOT EXISTS benchmark_return REAL;
ALTER TABLE actuals ADD COLUMN IF NOT EXISTS gross_stock_return REAL;
ALTER TABLE actuals ADD COLUMN IF NOT EXISTS transaction_cost REAL;
ALTER TABLE actuals ADD COLUMN IF NOT EXISTS entry_date DATE;
ALTER TABLE actuals ADD COLUMN IF NOT EXISTS entry_price REAL;
ALTER TABLE actuals ADD COLUMN IF NOT EXISTS exit_price REAL;
ALTER TABLE actuals ADD COLUMN IF NOT EXISTS execution_status TEXT;
ALTER TABLE strategy_performance ADD COLUMN IF NOT EXISTS actual_excess_return_20d REAL;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS run_key TEXT;

-- Legacy actual_excess_return_5d stored executable T+20 outcomes.
UPDATE actuals
SET actual_excess_return_20d = actual_excess_return_5d
WHERE actual_excess_return_20d IS NULL
  AND actual_excess_return_5d IS NOT NULL;

UPDATE strategy_performance
SET actual_excess_return_20d = actual_excess_return_5d
WHERE actual_excess_return_20d IS NULL
  AND actual_excess_return_5d IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_actuals_signal ON actuals(signal_date, ticker);
CREATE INDEX IF NOT EXISTS idx_strategy_date ON strategy_performance(signal_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_runs_run_key ON pipeline_runs(run_key);

-- Security hardening: dashboards may read published analytics with anon or
-- authenticated keys, but only the service_role pipeline may write these
-- tables.  service_role bypasses RLS, so cloud sync remains unchanged.
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE actuals ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_performance ENABLE ROW LEVEL SECURITY;

GRANT SELECT ON signals, actuals, pipeline_runs, strategy_performance TO anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON signals, actuals, pipeline_runs, strategy_performance FROM anon, authenticated;

DROP POLICY IF EXISTS public_read_signals ON signals;
CREATE POLICY public_read_signals ON signals
    FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS public_read_actuals ON actuals;
CREATE POLICY public_read_actuals ON actuals
    FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS public_read_pipeline_runs ON pipeline_runs;
CREATE POLICY public_read_pipeline_runs ON pipeline_runs
    FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS public_read_strategy_performance ON strategy_performance;
CREATE POLICY public_read_strategy_performance ON strategy_performance
    FOR SELECT TO anon, authenticated USING (true);
