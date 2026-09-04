# Roadmap

## Per-ticker SL/TP research — engineering complete 2026-08-28
- [x] Rolling expanding-train/forward-validation theo mã, mặc định 3 folds
- [x] Consensus gate: cần ít nhất 2/3 folds cùng vượt baseline trước khi approve
- [x] Universe cấu hình được: VN30, toàn bộ mã có trong research cache, hoặc danh sách chỉ định
- [x] Baseline ATR/TP dùng chung giữa research, paper và profile resolver; profile cũ bị khóa bằng schema v2
- [x] Tách attribution theo `strategy_name` + `strategy_version` để không trộn cohort SL/TP
- [x] Weekly research workflow tạo artifact audit; không tự commit hoặc bật live profile
- [x] Bottom-to-now diagnostic cho toàn bộ universe: đo TP8/10/12, fixed SL, ATR×2 và nhãn HOLD/SCALP/WAIT
- [x] Dashboard hiển thị snapshot SL/TP bottom-to-now, bộ lọc HOLD/SCALP/WAIT và so sánh fixed SL với ATR×2
- [x] Paper candidate baseline đã khởi động; mỗi phiên ghi 3 pick và backfill outcome sau T+20
- [x] Apply `supabase_setup.sql` để cloud lưu dynamic SL/TP và strategy version của paper cohort (applied 2026-08-28)
- [ ] Tích lũy paper A/B tối thiểu 30 basket / 100 trade rồi mới xem xét promotion
- [ ] Chỉ bật `ENABLE_TICKER_EXIT_PROFILES=true` nếu locked holdout và paper A/B đều dương

## Engineering hardening — completed 2026-08-03
- [x] Point-in-time-safe ML feature set and missing future labels
- [x] Cloud performance contract and historical remote actuals backfill
- [x] Idempotent local actuals and timezone-aware market dates
- [x] Out-of-sample walk-forward ensemble and OHLC-based SL/TP backtest
- [x] Retry/quality gates, workflow concurrency, timeout, and smoke checks

## Self-improving production guard — implemented 2026-09-04
- [x] Daily rolling retrain on the latest regime-aware window
- [x] Adaptive strategy weights from realized local + Supabase history
- [x] Champion/challenger registry with stable model versions and atomic writes
- [x] Candidate rejection when execution top-3 return or spread regresses beyond tolerance
- [x] Fail-closed publication: quality/registry failure becomes an explicit no-trade state
- [x] GitHub Actions persists registry history and dashboard exposes the latest decision
- [ ] Add a persistent model-artifact store for physical rollback across runners
- [ ] Tune hyperparameters/entry thresholds only after a locked walk-forward experiment
- [ ] Promote paper candidates only after the existing 30-basket / 100-trade gate

## Phase 1 — Core (done)
- [x] Pipeline fetch 126 mã VN, 23 features, ensemble 4 horizons
- [x] 5 strategies + auto-select best
- [x] GitHub Actions daily 8h VN (before market open)
- [x] T+20 holding, N=3 picks, production exit baseline from config (default SL -0.5% / TP +10%); paper candidate ATR×2 / TP10
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
