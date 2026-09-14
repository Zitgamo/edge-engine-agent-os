# Review và quyết định chiến thuật — 14/09/2026

## Quyết định

Dừng hướng tối ưu để ép phát tín hiệu. Chưa có ứng viên đủ bằng chứng để thay production.
Đã mở prototype nghiên cứu thị trường ETF Mỹ độc lập, tải dữ liệu và chạy so sánh
với mua-giữ SPY. Cả đổi sang mean reversion ở VN lẫn luân chuyển ETF đơn giản
đều chưa đạt. Đây là kết quả loại ứng viên, không phải migration live thành công.

Không sửa ngưỡng production, không gửi Telegram, không ghi cloud, không đặt lệnh.
Các file mới chỉ nằm trong thư mục review; không đụng các thay đổi đang dở của dự án.

## Vì sao không có tín hiệu

Đọc trực tiếp GitHub Actions ngày 14/09, workflow daily gần nhất là 11/09.
Trạng thái workflow success không đồng nghĩa phát tín hiệu.

- 10/09: mô hình mới được chấp nhận, nhưng breadth 35/97 = 36,08% dưới 50%,
  nên entry filter chặn toàn bộ. Paper VN30 breadth 43,33% dưới 60%.
- 11/09: execution top-3 excess return trên 40 ngày kiểm định là -0,03%,
  dưới 0%; mô hình bị quality gate chặn. Paper VN30 chỉ 9/30 = 30%, dưới 60%.
- Code còn gắn việc phát tín hiệu với việc chấp nhận mô hình mới: challenger bị
  loại thì dừng, chưa đánh giá lại champion trên cùng tập kiểm định để quyết định
  có dùng tiếp hay không. Không thể chỉ bật fallback vì champion có thể cũng kém.

Nguồn trực tiếp:
- https://github.com/Zitgamo/edge-engine-agent-os/actions/runs/34498156603
- https://github.com/Zitgamo/edge-engine-agent-os/actions/runs/34605321577

Log readiness 11/09 cho thấy ensemble legacy: 44 trade, win rate 27,27%,
lợi nhuận trung bình sau phí -1,00%. Mean reversion legacy: 52 trade, win rate
50%, trung bình +2,08%. Đây là dữ liệu được hệ thống tính lại, không phải sao kê
giao dịch thật; cohort legacy_unknown, ít mẫu và không chứng minh chiến thuật mới.
Paper ATR2 mới có 3 trade hoàn tất / 1 basket, trung bình -0,34%.

## Thử đổi chiến thuật Việt Nam

Script `compare.py` dùng cache 2021–27/08/2026, danh sách VN30 trong repo,
ba mã mỗi phiên đủ 30 mã có feature, ATR2 / TP10 / tối đa 20 phiên, chi phí
0,3% vòng mua-bán theo outcome cache. Lọc label chưa hoàn tất tại cuối mỗi giai đoạn.

| Quy tắc | 2021–2023 bình quân/lệnh | 2024–2025 | 2026 đã quan sát |
|---|---:|---:|---:|
| Momentum, breadth ≥60% | -0,54% | +1,08% | +2,40% |
| Mean reversion khi breadth <50% | -0,87% | +1,21% | -1,59% |
| Momentum khi breadth ≥50%, reversion khi <50% | -0,50% | +1,05% | -0,85% |

Momentum có chuỗi 53 / 34 / 40 phiên đủ feature liên tiếp không chọn mã ở ba
giai đoạn. Như vậy chiến thuật này vốn có thời gian đứng ngoài dài. Quy tắc ghép
phát danh sách mỗi phiên nhưng mất tiền ở hai giai đoạn: nhiều tín hiệu không giải
quyết được chất lượng. Loại mean reversion yếu và cách ghép này khỏi promotion.

Không coi 2026 là holdout mới: dữ liệu đã được xem. Thành phần VN30 cố định có
survivorship bias; basket chồng lấn không độc lập; chưa mô phỏng danh mục vốn VN.
Kết quả này không so trực tiếp bình quân/lệnh VN với lợi nhuận danh mục ETF.

## Thử đổi thị trường: ETF Mỹ

Đã tải 2.940 phiên 02/01/2015–11/09/2026 cho SPY, QQQ, IWM, GLD, TLT qua
yfinance, giá điều chỉnh. Prototype `etf_rotation.py`: mỗi 20 phiên chọn ETF có
momentum 126 phiên cao nhất, dương và giá trên SMA200; không có thì tiền mặt.
Quyết định sau đóng cửa, thực hiện ở mở cửa kế tiếp. Không đòn bẩy, một vị thế,
cổ phần lẻ; mark-to-market từng phiên, không cộng chồng toàn bộ vốn cho mỗi lệnh.

| Giai đoạn | Rotation tổng lợi nhuận | MDD rotation | SPY mua-giữ | MDD SPY |
|---|---:|---:|---:|---:|
| 2016–2023 | +72,32% | -34,90% | +173,38% | -32,05% |
| 2024–2025 | +50,56% | -15,88% | +47,77% | -19,77% |
| 2026 đến 11/09 | +12,09% | -21,45% | +11,88% | -8,09% |

Phí giả định 0,10% mỗi chiều. Stress 0,30% mỗi chiều làm rotation còn
+52,49% / +45,81% / +11,20%. Rotation chỉ đổi vị thế 34 / 8 / 2 lần tương ứng,
không đáp ứng mong muốn tín hiệu mới thường xuyên. Loại khỏi promotion.

Đây là mô phỏng nghiên cứu: một nguồn giá, chưa phí thuế/FX, chưa lãi tiền mặt,
chưa chi phí thanh lý cuối kỳ; giá điều chỉnh chỉ xấp xỉ tổng lợi nhuận. Chưa xác
minh khả năng giao dịch các ETF này với tài khoản của người dùng.
Mô tả sản phẩm đã đối chiếu nguồn phát hành:
- SPY: https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy
- QQQ: https://www.invesco.com/qqq-etf/en/about.html
- TLT: https://www.ishares.com/us/literature/fact-sheet/tlt-ishares-20-year-treasury-bond-etf-fund-fact-sheet-en-us.pdf

## Hướng phát triển được đề xuất sau review

1. Ưu tiên sửa đánh giá mô hình: champion và challenger cùng cửa sổ ngoài mẫu,
   không dùng điểm champion cũ làm lý do tự động dừng mọi chiến thuật. Champion
   chỉ được dùng tiếp nếu tự vượt quality, realized và entry gates.
2. Bổ sung tiêu chí sản phẩm vào mỗi nghiên cứu: số lần đổi vị thế, số phiên không
   có cơ hội, turnover, phí, lợi nhuận và drawdown so với benchmark. Một báo cáo
   hằng ngày không đồng nghĩa một lệnh mua mới hằng ngày.
3. Tiếp tục hướng đa thị trường trong module độc lập; ETF là baseline đã có,
   chưa phải chiến thuật thắng. Đăng ký quy tắc và thời gian kiểm định tiếp theo
   trước khi chạy; không tiếp tục tối ưu trên 2026 rồi gọi đó là holdout.
4. Chỉ thay live khi có kiểm định ngoài mẫu và forward paper đủ mẫu theo gate
   hiện tại. Không hạ tiêu chuẩn vì đã chờ lâu. Không khẳng định lợi nhuận tương lai.

## Tái chạy và kiểm chứng

Từ repo root, dùng Python có pandas/numpy/pyarrow:

```powershell
python docs/strategy_review_20260914/compare.py
python docs/strategy_review_20260914/etf_rotation.py
```

Hai script đã chạy thành công. Kiểm tra dữ liệu không trùng khóa, mức phí cache,
giá ETF đầy đủ/dương, target lịch sử không đổi khi nối dữ liệu tương lai,
warmup đứng ngoài, tăng phí không làm tăng equity. Hash nguồn ghi trong hai
manifest. Không chạy test production vì không đổi code runtime.
