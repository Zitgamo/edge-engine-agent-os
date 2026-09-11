# Run-status historical-rerun fix

## Scope

A late rerun for an older market session could become the displayed and synced
status because each reader sorted `pipeline_runs` by execution time. This hid a
newer session's `no_trade` result.

## Implementation

1. Select the local sync candidate by `run_key` (market session), falling back
   to the execution date for legacy rows; use execution time only to break a
   same-session tie.
2. Choose dashboard publication status with the same session-first rule for
   cloud and local rows. Keep the newest actual execution as a separate metric
   so a historical rerun remains visible without replacing publication status.
3. Retain the existing `data_failed` safeguard: a same-session collection
   retry attaches `last_failed_attempt` diagnostics to an already successful
   publication instead of replacing it.
4. Cover the September 3 success / September 4 no-trade / late September 3
   rerun regression in cloud sync and the shared dashboard presentation.

## Verification

- Targeted dashboard and Supabase tests: `31 passed`.
- Full mock-data suite with cloud variables removed: `181 passed` in 19.91
  seconds. Ruff, Python compilation, and `git diff --check` also passed.
