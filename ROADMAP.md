# Roadmap

## Phase 1 — Core (done)
- [x] Pipeline fetch 126 mã VN, 23 features, ensemble 4 horizons
- [x] 5 strategies + auto-select best
- [x] GitHub Actions daily 9h VN
- [x] T+20 holding, N=3 picks, SL -3% / TP +8%
- [x] CLI: pipeline, signal, summary, history
- [x] Signal + performance in log (xem trên GitHub Actions)

## Phase 2 — Validate (running, cần ~2-4 tuần từ 27/07)
- [ ] Tích lũy actuals + strategy performance history (đang chạy — cần actuals)
- [ ] Kiểm tra win rate thực tế qua CLI: `history`, `summary`, `strategies`
- [ ] So sánh outperform vs rs_momentum vs mean_reversion vs fundamental_value vs momentum

## Phase 3 — Expand (dashboard + cloud)
- [x] Ceiling context filter (`src/filters/ceiling_context.py`) — phân tích context 2 trần/sàn
- [x] Telegram notification (`src/notification/telegram.py`) — signal tự gửi (cần set .env)
- [x] Supabase cloud sync — SQLite → Postgres, pipeline sync tự động
- [x] Dashboard v2 — dark theme, signal cards, KPI, chart, ẩn pipeline
- [ ] Realtime signal tracking — tính lãi/lỗ theo giá hiện tại (SL/TP hit chưa)
- [ ] Strategy mới: trend-following, RSI momentum, breakout
- [ ] Fine-tune weight / filter dựa trên actuals

## Phase 4 — Monetization (khi có >=4 tuần actuals, win rate >60%)
- [ ] Landing page giới thiệu dịch vụ
- [ ] Auth: Supabase Auth (email + Google)
- [ ] Payment: Stripe / VNPay
- [ ] Tiers: Free (signal T+1) vs Premium (realtime + Telegram)
- [ ] Telegram bot push signal cho premium user
- [ ] Thêm dashboard realtime P&L tracking
