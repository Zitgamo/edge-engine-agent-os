-- Additive; safe to apply before or after the code upgrade.
ALTER TABLE public.pipeline_runs ADD COLUMN IF NOT EXISTS diagnostics JSONB;
NOTIFY pgrst, 'reload schema';
