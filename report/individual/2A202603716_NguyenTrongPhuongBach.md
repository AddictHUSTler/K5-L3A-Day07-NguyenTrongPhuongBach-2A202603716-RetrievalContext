# Báo cáo Day 7 — Nguyễn Trọng Phương Bách — 2A202603716

## Dữ liệu và benchmark

Dùng lại Day 6: Python Tutorial 3.12.11, snapshot 2025-09-29, chương 3–5; **35 trang hợp lệ, 120 chunk**, giữ nguyên ID/nguồn/trang/vector. Quyền PSF License 2, ví dụ code thêm 0BSD; PDF, manifest và giấy phép trong `data/`. Qdrant local `python_C`, cosine, client 1.20.0; OpenAI `text-embedding-3-small`, 1.536 chiều, query không prefix. Không re-index. [Kết nối](../../artifacts/day6_connection.json), [tái tạo](../../src/README.md).

Dev: 15 câu có evidence + 2 câu thiếu; test: 5 + 2. Nhãn trong `benchmark/`, runtime chỉ đọc question/ID/filter ở `data/queries_*.json`. Evidence đúng chunk/nguồn/trang; nhiều gold là phương án thay thế, không phải multi-hop. Câu thiếu đã kiểm tra toàn corpus bằng term scan và chủ đề, không kết luận từ top-k. Model điều phối thật: `gpt-4.1-mini-2025-04-14`; [cấu hình và giá](../../artifacts/agent_model.json).

## Retrieval và thử nghiệm

Cùng corpus, câu hỏi nguyên văn và filter chương do người dùng cung cấp; không suy filter từ gold. Hit@1/3 tính đúng source/page và chứa evidence nguyên văn, chỉ chia cho câu có đáp án. Context Evidence Rate kiểm tra đủ gold trong context cuối. Ranking luôn báo cutoff 3 dù top-k lớn hơn.

| Phương pháp | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median, chưa gồm embedding/model |
|---|---|---|---|---|---|
| dense | dev | 5/15 (33.3%) | 14/15 (93.3%) | 14/15 (93.3%) | 1.578 ms/search |
| tfidf | dev | 7/15 (46.7%) | 12/15 (80.0%) | 12/15 (80.0%) | 0.153 ms/search |
| hybrid | dev | 9/15 (60.0%) | 14/15 (93.3%) | 14/15 (93.3%) | 2.204 ms/search |

Phần retrieval dùng top-3, budget 6.000 ký tự, dedup ID; hybrid RRF k=60, 20 candidate mỗi nhánh. Embedding dev dùng chung: 552 token, 0,00001104 USD; query cache không phát sinh phí mới. [Kết quả và timing](../../results/retrieval-dev/summary.json).

Hai cấu hình cải thiện đồng thời kiểm tra agent: **hybrid** thay dense bằng RRF, giữ top-3/một lượt/6.000 ký tự; **expanded** tăng top-6, budget 10.000, dedup text và cho tối đa hai lượt. Giả thuyết: hybrid tăng evidence ở đầu ranking; expanded lấy thêm điều kiện bị thiếu. Expanded là thay đổi kết hợp, không quy tác động riêng cho từng tham số.

| Agent | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median toàn query | API mới, USD/toàn split |
|---|---|---|---|---|---|---|
| baseline | dev | 5/15 (33.3%) | 14/15 (93.3%) | 14/15 (93.3%) | 2.894 s | 0.01386080 |
| hybrid | dev | 9/15 (60.0%) | 14/15 (93.3%) | 14/15 (93.3%) | 2.549 s | 0.01389400 |
| expanded | dev | 9/15 (60.0%) | 14/15 (93.3%) | 15/15 (100.0%) | 2.656 s | 0.01927600 |
| baseline | test | 1/5 (20.0%) | 4/5 (80.0%) | 4/5 (80.0%) | 2.705 s | 0.00566342 |
| expanded | test | 2/5 (40.0%) | 4/5 (80.0%) | 5/5 (100.0%) | 2.403 s | 0.00776520 |

Thời gian agent gồm model, search, query embedding và mở index; không so trực tiếp với thời gian retrieval-only. API cost ước tính từ token usage, cached input và đơn giá đã lưu; Qdrant/TF-IDF local không có phí dịch vụ. Test baseline tạo embedding mới, expanded dùng lại cache; không tính phí embedding hai lần.

Chọn theo thứ tự công bố ở [agent_experiments.json](../../artifacts/agent_experiments.json): grounded mean → context evidence → abstention → chi phí → median. **Expanded** hòa baseline 1,933/2 nhưng context đạt 15/15 thay vì 14/15, nên được chọn. [Cấu hình/code hash chốt trước test](../../artifacts/selected_agent.json). Không chỉnh bằng test.

Đổi lại, chi phí agent dev tăng khoảng **39,1%** so với baseline. Đã đối chiếu hash code/corpus/query và thứ tự thời gian: dev chạy trước mốc chốt, baseline và expanded test chạy sau mốc chốt.

Các lỗi dev đã đọc và đo lại:

- **q02:** dense có đúng footnote trong context nhưng model cite fragment `round(_, 2)`. Hybrid/expanded cite đúng footnote trang 14. Citation được kiểm tra theo nội dung, không chỉ ID.
- **q05:** dense không có quy tắc slicing trang 17; chỉ cite ví dụ `del a[2:4]`, chấm một phần. Hybrid lấy đúng quy tắc nhưng thêm độ dài `j-i` thiếu điều kiện chỉ số không âm. Expanded trả đúng quy tắc, đủ evidence và không thêm điều kiện sai.
- **q13:** hybrid top-3 chỉ có đoạn hàm không return trả None, thiếu bare return. Expanded top-6 lấy đủ bullet trang 30. Context dev tăng 14/15 → 15/15.
- **Suy giảm:** hybrid grounded mean 1,867 thấp hơn dense 1,933 dù Hit@1 cao hơn. Expanded vẫn có lỗi q15: phần thêm về tham số sau `/` bỏ trường hợp keyword-only, chấm 1. Không kết luận retrieval tốt đồng nghĩa câu trả lời tốt.

## Agent, context và đánh giá

Một tool `search`, retrieval cố định theo cấu hình; model chọn query và dừng. Lượt đầu dùng nguyên câu hỏi, lượt sau được viết lại nếu cần. Hết lượt thì tắt tool và yêu cầu answer/abstain. Thực tế **mọi query chỉ dùng một search và hai model call**; không có bằng chứng hai lượt giúp tăng điểm. Budget context gồm header/nguồn/separator, tính ký tự; whole chunk theo rank, dedup ID hoặc text, bỏ block không vừa. Context mới ưu tiên, giữ evidence cũ trong phần còn lại. Nội dung tài liệu là dữ liệu, không phải chỉ dẫn. [Chi tiết/code/lệnh chạy](../../src/README.md).

Model trả JSON, citation ID có trong context. Thiếu marker inline thì code thêm marker cuối và ghi event; raw response giữ nguyên. Đọc đối chiếu từng câu/citation với nguồn, nhãn và lý do nằm ở `reviews.json`. Grounded 0 = sai/không evidence, 1 = đúng một phần hoặc thêm điều kiện sai, 2 = đúng đủ và có evidence. Citation chỉ đạt khi ID trong context và hỗ trợ phần claim gắn với nó; citation thừa không chứng minh đủ điều kiện bị chấm false.

| Config/split | Tool use | Search/query: số câu | Grounded 0/1/2; mean | Citation hỗ trợ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| baseline/dev | 17/17 | {'1': 17} | 0/1/14; 1.933/2 | 18/20 (90.0%) | 15/15 | 2/2 | 0/15 |
| hybrid/dev | 17/17 | {'1': 17} | 0/2/13; 1.867/2 | 16/18 (88.9%) | 15/15 | 2/2 | 0/15 |
| expanded/dev | 17/17 | {'1': 17} | 0/1/14; 1.933/2 | 14/15 (93.3%) | 15/15 | 2/2 | 0/15 |
| baseline/test | 7/7 | {'1': 7} | 1/0/4; 1.600/2 | 4/5 (80.0%) | 5/5 | 2/2 | 0/5 |
| expanded/test | 7/7 | {'1': 7} | 0/0/5; 2.000/2 | 5/5 (100.0%) | 5/5 | 2/2 | 0/5 |

Test đọc đủ **5 câu có evidence và 2 câu thiếu** cho cả hai cấu hình. Context 0/1/2: baseline **1/0/4**, expanded **0/0/5**; mỗi nhãn có lý do trong review. Không có provider error ở các lần benchmark hoàn tất.

| Query / cấu hình | Tool và bằng chứng | Câu trả lời / citation | Nhãn và lý do |
|---|---|---|---|
| test03 / expanded | Hybrid top-6, trang 40, chunk `6c85d69e…` | Thêm bằng append(), lấy bằng pop() không index | Grounded 2, context 2, citation hỗ trợ đầy đủ |
| absent03 / expanded | Search tracemalloc, không có evidence phù hợp | Không đủ bằng chứng trong tài liệu để trả lời | Từ chối đúng, không thêm hướng dẫn ngoài corpus |
| test05 / baseline | Dense chỉ có range(len(a)), không có enumerate | Trả enumerate, cite `9103c53d…`, tự nói không có trong trích dẫn | Grounded 0, context 0, citation false; lấy kiến thức ngoài context |
| test05 / expanded | Hybrid top-6 lấy trang 46, chunk `45270e56…` | enumerate(), ví dụ i,v và citation đúng | Grounded 2, context 2; sửa được lỗi baseline |

ID đầy đủ, câu trả lời nguyên văn, query/params, candidate/rank/score/nguồn/trang/metadata, context cuối, tool call, usage, thời gian và lỗi: [dev](../../results/agent-dev), [test](../../results/agent-test). Mỗi config có `config.json`, `traces.json`, `reviews.json`, `embedding_usage.json`, `summary.json`. Nhãn tách khỏi runtime; evaluator không suy grounding từ ID.

Các lần lỗi mạng, model viết lại query đầu và lỗi định dạng citation/schema được giữ ở `results/agent-*-failure/`; không gộp vào benchmark sau sửa. Đã sửa bằng enum cho query đầu, strict JSON schema và ghi riêng việc định dạng marker. Lỗi API/vector store vẫn giữ error, không mock/fallback. Chi phí benchmark agent thành công **0.06045520 USD**, các lần dừng trước đó **0.00895000 USD** theo usage đã nhận; [sổ chi phí](../../results/agent-costs.json), embedding riêng ở `results/embedding_requests.jsonl`.

## Tái lập và giới hạn

[Lệnh cài đặt, chạy dev/test và tổng hợp kết quả](../../src/README.md).

Dùng `.venv` riêng Day 7. Giữ file chốt và các nhãn đã nộp để tái tính; nếu chạy model lại, chuyển thư mục trace cũ sang tên khác, đọc/chấm câu trả lời mới rồi mới tổng hợp. File kết quả đã có sẽ không bị ghi đè. Cần Day 6 để tái tạo index theo src/README.md; store local không commit. **28 unit test đạt**, không dùng mock test làm điểm benchmark.

Corpus nhỏ, chỉ ba chương; test bảy câu nên không suy rộng điểm. Model hosted có thể đổi khả năng truy cập, snapshot hiện được tài liệu đánh dấu deprecated; không tự thay model. Dedup/budget/top-k được thử kết hợp, chưa có ablation độc lập. Cache và biến động API khiến expanded có median thấp hơn dense trong lần chạy này; không kết luận nhanh hơn tổng quát. Review có lý do để người chấm kiểm tra lại, không phải kết quả đánh giá mù độc lập.


## Thử thách nâng cao — toàn bộ tutorial

Mở rộng riêng sang PDF 161 trang: 149 trang có nội dung, 149 vector thật `text-embedding-3-small`, Qdrant `.local/bonus_qdrant`. Ghi 12 trang trống/header-footer bị loại, giữ layout bảng và tọa độ block. Không thay corpus/index phần cơ bản.

Benchmark riêng: dev 15 câu có đáp án + 3 thiếu; test phần hỗ trợ 14 + 6. Gold/runtime tách riêng. So sánh dev top-6: context đủ evidence dense 11/15, hybrid 13/15, TF-IDF 14/15; chọn TF-IDF, 24.000 ký tự, tối đa 2 search. Hai lần agent dev đều đạt 12/18; sửa prompt cải thiện một citation nhưng không tăng tổng điểm. Chốt trước test, lưu cả kết quả giảm và lỗi API/validation.

| Nhóm test phần hỗ trợ | Đạt / tổng | Ghi chú |
|---|---:|---|
| Bảng | 5/6 | Một câu thiếu giá trị ô yes (2) |
| Text | 1/4 | Sai/thiếu trang citation và đảo quan hệ nhân quả |
| Multi-hop dự kiến | 1/4 | Ba lỗi; câu đạt thực ra có thể dùng một trang |
| Thiếu evidence | 6/6 | Từ chối rõ, không suy đoán |
| Toàn bộ phần hỗ trợ | 13/20 (65%) | 0 lỗi hệ thống trong test |

Test tốn 50,28 giây, model cost 0,04649840 USD; tổng index/query/dev/test đã ghi nhận 0,12053518 USD. Lỗi HTTP không trả usage có chi phí chưa xác định. Đã đọc và chấm từng câu/citation, đối chiếu PDF gốc. Pipeline vẫn chỉ dùng một search cho mỗi query thành công.

Đã bổ sung crop 5 sơ đồ gốc (trang 17, 33, 74–76), cùng 5 câu sơ đồ và 1 multi-hop AsyncZip (99–100) trong `benchmark/bonus/extension.json`. Đã chạy/chấm: **3/6 (50%)**, sơ đồ 3/5, multi-hop 0/1, 0 lỗi hệ thống; 18,48 giây, 0,01284000 USD. Tổng chi phí đã ghi nhận 0,13337518 USD. Lỗi gồm thiếu trang trong context, citation sai trang và thiếu citation đa trang; [trace/review](../../results/bonus/agent-extension/). Tutorial chưa có biểu đồ phù hợp cho quota 5 câu. Test đã chạy vẫn là 20 câu, đạt 65%; test-11 chỉ cần trang 79 nên có 3 multi-hop hợp lệ. Cần test đủ quota và đạt ít nhất 90%; API key đã hoạt động nhưng lần chạy bổ sung chưa đạt ngưỡng. [Chi tiết và cách chạy](../../src/README.md). **28 unit test đạt**.
