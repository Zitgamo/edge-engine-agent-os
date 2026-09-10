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
- CI, push/merge and a post-deployment pipeline run remain the final verification steps.
