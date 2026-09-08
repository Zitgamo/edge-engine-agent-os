# Run diagnostics deployment — 2026-09-08

## Plan

1. Base the release on current origin/master (be51d72), preserving the latest market data and model artifacts. Copy only the reviewed diagnostics, dashboard, schema and test changes into an isolated worktree.
2. Run a separate Python 3.12 GitHub Actions test job with mock data and no production secrets. Merge only after the job passes.
3. Apply docs/migrations/20260907_run_diagnostics.sql to Supabase project avidcqqxjhttaooalexp. Verify the new JSONB column through the data API. The additive migration is compatible with the old app.
4. Publish the checked code to master; verify Streamlit reloads the updated dashboard and history page.
5. Verify a subsequent pipeline run stores diagnostics and the UI explains the actual gate result. Do not manufacture diagnostics for a historical run or change entry thresholds.

## Rollback

Revert the release code commit if the dashboard or pipeline regresses. Leave the nullable diagnostics column in place; removing it is unnecessary. Preserve all existing market data, signals and model artifacts.

## Status

- Release worktree prepared from origin/master; existing local edits and data remain intact.
- Supabase browser session currently requires login. Existing data API credentials do not grant schema migration access.
- CI, migration and cloud verification pending.
