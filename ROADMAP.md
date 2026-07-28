# Roadmap

## Phase 1 — Core (done)
- [x] Pipeline fetch 126 mã VN, 23 features, ensemble 4 horizons
- [x] 5 strategies + auto-select best
- [x] GitHub Actions daily 9h VN
- [x] T+20 holding, N=3 picks, SL -3% / TP +8%
- [x] CLI: pipeline, signal, summary, history
- [x] Signal + performance in log (xem trên GitHub Actions)

## Phase 2 — Validate (running, cần ~2-4 tuần)
- [ ] Tích lũy actuals + strategy performance history
- [ ] Kiểm tra win rate thực tế qua CLI: `history`, `summary`, `strategies`
- [ ] So sánh outperform vs rs_momentum vs mean_reversion vs fundamental_value vs momentum

## Phase 3 — Expand (đang làm)
- [x] Ceiling context filter (`src/filters/ceiling_context.py`) — phân tích context 2 trần/sàn
- [x] Telegram notification (`src/notification/telegram.py`) — signal tự gửi mỗi sáng (cần set .env)
- [x] Supabase integration (`src/supabase_client.py`) — sync SQLite → Postgres cloud
- [ ] Dashboard deploy (Streamlit lên cloud) — code sẵn + Supabase, cần deploy
- [ ] Strategy mới: trend-following, RSI momentum, breakout
- [ ] Fine-tune weight / filter dựa trên actuals
