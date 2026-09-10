# Run diagnostics deployment — 2026-09-10

## Plan

1. Base the release on current origin/master (58c08dd), preserving the latest market data and model artifacts. Copy only the reviewed diagnostics, dashboard, schema and test changes into an isolated worktree.
2. Run a separate Python 3.12 GitHub Actions test job with mock data and no production secrets. Merge only after the job passes.
3. Apply the additive diagnostics migration to Supabase project avidcqqxjhttaooalexp and verify the JSONB column through the data API. The migration is compatible with the old app.
4. Publish the checked code to master; verify Streamlit reloads the updated dashboard and history page.
5. Verify a subsequent pipeline run stores diagnostics and the UI explains the actual gate result. Do not manufacture diagnostics for a historical run or change entry thresholds.

## Rollback

Revert the release code commit if the dashboard or pipeline regresses. Leave the nullable diagnostics column in place; removing it is unnecessary. Preserve all existing market data, signals and model artifacts.

## Status

- Release worktree prepared from origin/master; existing local edits and data remain intact.
- Migration completed in Supabase: `pipeline_runs.diagnostics` is present as nullable `jsonb`; PostgREST schema reload issued and read-only API access verified.
- Release checks passed locally: Ruff and `git diff --check`; the isolated offline suite passed 179 tests.
- The reviewed changes preserve a successful publication after a later `data_failed` attempt and expose that attempt separately in the dashboard.
- PR #2 merged successfully as `c006a7c`; both PR checks passed.
- Post-deployment `daily.yml` run `34498156603` completed successfully. The main pipeline and paper-candidate jobs both passed.
- Cloud verification after that run shows `2026-09-10` as `no_trade`, with diagnostics containing `data_health` and `entry_filters`. The live Streamlit dashboard shows the same run/date and explains the 20-session breadth gate (0.3608 below 0.5; 97 symbols reduced to 0).
