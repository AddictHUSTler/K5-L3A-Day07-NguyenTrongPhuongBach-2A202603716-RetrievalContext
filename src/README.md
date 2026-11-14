# Chạy bài Day 7

Python 3.11+. Chạy từ root repo; đặt `OPENAI_API_KEY` trong `.env` hoặc environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.prepare --day6 /duong/dan/Day06
.venv/bin/python -m src.benchmark_retrieval
.venv/bin/python -m src.agent --split dev
.venv/bin/python -m src.evaluate_agent --split dev --configs baseline hybrid expanded
# Chốt artifacts/selected_agent.json trên dev trước khi chạy test.
.venv/bin/python -m src.agent --split test --configs baseline expanded
.venv/bin/python -m src.evaluate_agent --split test --configs baseline expanded
.venv/bin/python -m unittest discover -s tests -v
```

Lấy [repo Day 6](https://github.com/AddictHUSTler/K5-L3A-DAY06-NguyenTrongPhuongBach-2A202603716-DataIngestion), làm theo README để tái tạo `.local/qdrant` rồi chạy `src.prepare`. Script giữ nguyên ID/vector, sao chép index sang `.local/qdrant` Day 7 và kiểm tra payload. Đóng tiến trình dùng store trước khi sao chép; không chạy đồng thời hai tiến trình trên cùng Qdrant local. Người chấm cần quyền xem cả hai repo.

`retrieval.py` cung cấp dense, TF-IDF và hybrid RRF; `agent.py` chạy model thật; `benchmark_retrieval.py` và `evaluate_agent.py` tổng hợp kết quả. Chỉ chạy lexical: `python -m src.benchmark_retrieval --methods tfidf --output results/tfidf-dev`. Có thể chỉ định `--env-file /duong/dan/.env` cho benchmark retrieval.

Gold ở `benchmark/`; agent chỉ đọc `data/queries_*.json`. Model, giá và kết nối ở `artifacts/`. Trace, nhãn và metrics ở `results/`. [Báo cáo](../report/individual/2A202603716_NguyenTrongPhuongBach.md) chứa bảng so sánh, lỗi, ví dụ trả lời và giới hạn.

Kết quả agent đã có sẽ không bị ghi đè. Khi chạy lại, lưu thư mục cũ sang tên khác, đọc/chấm câu trả lời mới vào `reviews.json` rồi tổng hợp. Chỉ tổng hợp kết quả đã nộp: chạy hai lệnh `src.evaluate_agent`, không gọi model. Index/cache/key nằm local và không commit.

## Bonus

Cần Poppler (`poppler-utils`). Corpus/index bonus riêng, dùng cùng PDF và embedding model.

```bash
.venv/bin/python -m src.prepare_bonus --index
.venv/bin/python -m src.prepare_bonus --visuals
.venv/bin/python -m src.bonus retrieval-dev
.venv/bin/python -m src.bonus run --split dev
# Đọc/chấm results/bonus/agent-dev/reviews.json trước khi freeze.
.venv/bin/python -m src.bonus evaluate --split dev
.venv/bin/python -m src.bonus freeze
.venv/bin/python -m src.bonus run --split test
.venv/bin/python -m src.bonus evaluate --split test
.venv/bin/python -m src.prepare_bonus --run-extension
```

Benchmark/gold có sẵn trong `benchmark/bonus/`; query runtime ở `data/bonus/`. Tọa độ crop ở `data/bonus/regions.json`. File chốt và các thư mục kết quả đã tồn tại được giữ nguyên; muốn chạy mới cần lưu bản cũ trước. Extension ghi riêng vào `results/bonus/agent-extension/`; đọc/chấm từng câu trước khi tổng hợp. Kết quả và phần còn thiếu nằm trong báo cáo cá nhân.
