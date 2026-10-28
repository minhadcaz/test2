# Schema Pruning & Retrieval: thiết kế, kết quả Mini-Dev và chạy Kaggle

Ngày thực nghiệm: **28/10/2026**. Mục tiêu là giảm schema đưa vào SLM nhưng giữ đủ thông tin để sinh SQL. Tính năng đã tích hợp vào pipeline hiện có; mặc định `schema_selection: false` được giữ vì kết quả BM25 chưa đủ tin cậy để bật cho mọi câu hỏi.

Hướng dẫn đánh giá chi tiết: [A–Z so sánh cùng model với full/pruned schema, metric, CI và plot](EVALUATION_A_TO_Z.md). Notebook đã có cell tạo comparison CSV/PNG/PDF và gọi EX chính thức.

## 1. Kết luận khả thi

Có thể triển khai bước retrieval trước generator trong dự án hiện tại. BM25 chạy CPU, không cần vector database hoặc API trả phí. Dense/hybrid dùng `sentence-transformers`, cosine trên ma trận vector chuẩn hóa, reciprocal-rank fusion (RRF) và cache theo fingerprint của schema/model/revision. Trong 11 DB của Mini-Dev, tập schema nhỏ nên triển khai ma trận local giúp tái lập đơn giản; dịch vụ vector database phù hợp hơn khi có nhiều database/phiên bản, cần cập nhật index, metadata filtering hoặc phục vụ đồng thời. Đây là lựa chọn kiến trúc, chưa phải benchmark so sánh vector database.

Ba nghiên cứu sơ cấp hỗ trợ hướng này:

- **RAT-SQL**, ACL 2020, mô hình hóa quan hệ schema và liên kết câu hỏi với schema: <https://aclanthology.org/2020.acl-main.677/>.
- **RESDSQL**, AAAI 2023, xếp hạng schema item trước khi sinh SQL: <https://arxiv.org/abs/2302.05965v3>.
- **CHESS**, 2024, dùng schema selector với các bước lọc cột, chọn bảng và chọn cột: <https://arxiv.org/html/2405.16755v3>.

Các paper là cơ sở thiết kế. Chúng không chứng minh mức cải thiện latency, VRAM hoặc execution accuracy cho model/cấu hình trong repository này.

Giảm prompt chủ yếu giảm công việc **prefill**, KV cache và nguy cơ vượt context; không bảo đảm số **output token** giảm. VRAM dành cho trọng số model không giảm chỉ nhờ pruning. Việc chạy encoder cũng có chi phí; mặc định encoder chạy CPU để tránh chiếm thêm VRAM. Cần đo trực tiếp cả latency retrieval, tổng request, output token và peak CUDA memory. Tài liệu kỹ thuật tham khảo về KV cache: <https://huggingface.co/docs/transformers/v4.51.3/en/kv_cache>.

## 2. Luồng đã triển khai

```text
SQLite / metadata BIRD
  → schema documents: tên bảng, tên cột, description nếu có
  → BM25 hoặc dense/hybrid(question + evidence)
  → top-k bảng
  → thêm đường FK ngắn nhất trong giới hạn bridge budget
  → top-k cột mỗi bảng + PK/FK + cột được nhắc trực tiếp
  → chiếu lại sample values theo đúng vị trí cột
  → prompt → generator → validate/execute → correction với schema đã lọc
```

Các điểm mã nguồn:

- `core/schema_retriever.py`: tokenizer lexical, BM25, cosine/RRF và cache vector. Không dùng gold SQL hoặc row values để xếp hạng.
- `core/schema_selector.py`: ngân sách bảng/cột, bounded FK path, giữ khóa, metadata reduction/latency. Tên FK được đối chiếu không phân biệt hoa thường. PK và cột JOIN có thể làm vượt **soft budget** cột; lượng vượt được ghi lại.
- `core/pipeline.py`: tái sử dụng index trong cùng instance, truyền evidence, dùng schema đã lọc cho cả sinh ban đầu và correction.
- `core/hf_model.py`: ghi input token sau chat template, output token theo EOS, context overflow, generation time, OOM retry.
- `evaluation/evaluate_schema_retrieval.py`: đánh giá retrieval không cần GPU/SQLite; xuất summary và chi tiết từng câu.
- `evaluation/evaluate.py`: so sánh model thực, logging retrieval và tùy chọn peak CUDA memory.
- `evaluation/run_sweep.py`: đã sửa route database theo từng record và áp dụng selector. Pool resume phải khớp dataset/config/pool settings; không trộn candidate full-schema và pruned-schema.
- `scripts/import_bird.py`: nhận JSON hoặc JSONL chính thức và kiểm tra SQLite từng câu.
- `scripts/export_bird_predictions.py`: xuất prediction/gold đúng thứ tự cho scorer chính thức, giữ cả câu thất bại trong mẫu số.

Bật tính năng:

```yaml
validation:
  schema: true
  schema_selection: true
retrieval:
  mode: bm25
  top_k_tables: 8
  top_k_columns: 48
  max_bridge_tables: 2
  fallback: full
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  embedding_revision: null
  local_files_only: true
  device: cpu
  cache_dir: .cache/schema_embeddings
```

Đây là cấu hình để **thử nghiệm**, chưa phải cấu hình đã chứng minh an toàn về accuracy. `fallback: full` hiện chỉ áp dụng khi không có retrieval match; điểm BM25 dương không có nghĩa là đủ schema. Nếu câu hỏi chứa từ khớp nhầm thì vẫn có thể mất bảng/cột.

Giới hạn hiện tại: ngân sách là số bảng/cột, chưa tối ưu theo hard token budget; khóa/cột nhắc trực tiếp được ưu tiên giữ nên có thể vượt ngân sách. Bridge FK chỉ có tác dụng nếu metadata định nghĩa FK; một số schema thực tế thiếu quan hệ. Tiếng Việt được chuẩn hóa dấu và có alias cho demo sinh viên, chưa có benchmark đa ngôn ngữ. Không có learned reranker, tìm kiếm giá trị ô hoặc vòng mở rộng schema tự động theo lỗi; nếu pruning mất cột, correction hiện có thể không sửa được.

## 3. Dataset chính thức và khả năng tái lập

Nguồn chính thức: <https://github.com/bird-bench/mini_dev>. README chính thức trỏ đến bản dữ liệu canonical trên HF: <https://huggingface.co/datasets/birdsql/bird_mini_dev>.

Dataset đã dùng là **Mini-Dev SQLite, 500 câu, 11 DB**, release ghi trong manifest `Mini-Dev 1024` (1024 là tên release). HF revision đã ghim:

```text
5b256b5a9c17478f06ed6912612e9a14d17065c8
```

Ba tệp được tải bởi `scripts/download_bird_minidev.py`:

| Tệp local | Nguồn trong HF | SHA-256 |
|---|---|---|
| `mini_dev_sqlite.jsonl` | `data/mini_dev_sqlite.jsonl` | `926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606` |
| `schemas.json` | `baseline_inference/schemas/sqlite.json` | `5176e324a72e80a8a37a12285596d123e0ba6c310dceaa9f13a1e413ee88ff5b` |
| `manifest.json` | `data/manifest.json` | được lưu trong `provenance.json` |

Tổng metadata khoảng **538 KiB**, gồm DDL cho 76 bảng/800 cột, không cần SQLite để đánh giá retrieval. Loader loại bỏ comment/value examples có trong DDL canonical để thí nghiệm này dùng cấu trúc thuần; không coi ví dụ trong comment là nội dung đã truy xuất từ database. Không trộn annotation ở archive 2025 với bản hiện tại theo question ID. Dataset card khai báo CC-BY-SA-4.0; khi chia sẻ lại dữ liệu cần giữ attribution theo nguồn.

Mini-Dev không có gold schema-linking annotations trong các file này. Evaluator suy ra bảng/cột từ **SQL tham chiếu**, dùng SQLGlot qualify + scope để xử lý alias, CTE, subquery, correlated query, UNION. `SELECT *` mở rộng thành các cột; `COUNT(*)` cần bảng và không tự yêu cầu mọi cột. Cột JOIN/WHERE/GROUP/ORDER đều được tính nếu tham chiếu cột gốc. Nhãn này mô tả một cách viết SQL tham chiếu; SQL tương đương có thể dùng tập tham chiếu khác. Những lỗi suy nhãn được ghi và khiến CLI thoát lỗi, không bỏ qua âm thầm. `NATURAL JOIN` hiện bị từ chối vì chưa suy nhãn khóa JOIN ngầm đầy đủ; không xuất hiện trong 500 câu đã đo. DDL có `CONSTRAINT name` được mở lớp để giữ đúng PK/FK.

Retriever nhận **db_id đã biết** và chỉ tìm trong database tương ứng; chưa đo bài toán tìm đúng database. Gold SQL chỉ vào evaluator **sau** lời gọi retrieval.

## 4. Chạy retrieval trên CPU

Tại thư mục `SLM-PRJ`, dùng Python 3.12 và dependency dự án:

```bash
python -m pip install -r requirements.txt
python scripts/download_bird_minidev.py
python -m evaluation.evaluate_schema_retrieval \
  --data data/bird_minidev/mini_dev_sqlite.jsonl \
  --schemas data/bird_minidev/schemas.json \
  --mode bm25 --top-k-tables 8 --top-k-columns 24 \
  --output results/bird_retrieval/bm25_t8_c24
```

Để đếm token thật, cài `requirements-model.txt`, tải/cache tokenizer tương ứng rồi thêm `--tokenizer <model-path-or-id>`. Mặc định chỉ dùng local files; thêm `--allow-downloads` khi cho phép kết nối. Không truyền tokenizer thì báo cáo ghi rõ `regex_proxy_not_model_tokens`, không gọi đó là token model.

Hybrid:

```bash
python -m pip install -r requirements-retrieval.txt
python -m evaluation.evaluate_schema_retrieval \
  --data data/bird_minidev/mini_dev_sqlite.jsonl \
  --schemas data/bird_minidev/schemas.json \
  --mode hybrid --top-k-tables 8 --top-k-columns 24 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --allow-downloads --output results/bird_retrieval/hybrid_t8_c24
```

Đối với thí nghiệm công bố, chỉ định thêm `--embedding-revision <commit>`; khi dùng model path local hãy gắn rõ snapshot/revision, vì thay trọng số tại cùng một đường dẫn có thể làm cache cũ không hợp lệ. Xóa/chuyển cache khi thay trọng số mà không đổi tên/revision. Lượt đầu có chi phí xây index; per-question logs ghi `index_cache_hit`. Báo cáo retrieval_ms tổng hợp gồm cả cold/warm, không so latency giữa các lượt đồng thời.

## 5. Kết quả đã chạy trên đủ 500 câu

Xem `bird_minidev_retrieval_results.json` và `bird_minidev_dense_results.json` để biết kết quả từng database, dataset hash, tokenizer/encoder provenance, SQLGlot version và độ trễ của từng arm. Tokenizer: `cycloneboy/SLM-SQL-1.5B`, revision `2cc2df91aacafdec3fef3733ac45bd53b57d54cf`; dense encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Đếm prompt thô, **chưa cộng chat template**. Tất cả 500 câu suy nhãn thành công; không dùng mock để đo retrieval.

| Phương án | Prompt token TB | Giảm prompt | Recall bảng (macro) | Recall cột (macro) | Đủ mọi tham chiếu | Retrieval TB |
|---|---:|---:|---:|---:|---:|---:|
| full | 842.6 | 0.0% | 100.0% | 100.0% | 100.0% | 0.00 ms |
| bm25_t4_c8 | 452.3 | 46.3% | 93.8% | 85.2% | 51.4% | 1.65 ms |
| bm25_t4_c12 | 488.1 | 42.1% | 93.8% | 87.6% | 59.0% | 1.67 ms |
| bm25_t6_c16 | 579.3 | 31.2% | 97.3% | 92.3% | 73.4% | 1.88 ms |
| bm25_t8_c24 | 671.2 | 20.3% | 99.2% | 95.7% | 84.8% | 1.88 ms |
| bm25_t8_c48 | 751.8 | 10.8% | 99.2% | 98.8% | 96.2% | 2.35 ms |
| dense_t8_c24 | 674.9 | 19.9% | 99.6% | 97.6% | 90.0% | 57.93 ms |
| hybrid_t8_c24 | 672.6 | 20.2% | 99.6% | 97.7% | 91.0% | 39.17 ms |
| hybrid_t6_c16 | 581.6 | 31.0% | 98.1% | 94.9% | 80.6% | 65.12 ms |
| hybrid_t4_c8 | 452.1 | 46.4% | 94.1% | 87.5% | 59.6% | 72.55 ms |
| hybrid_t8_c48 | 752.8 | 10.7% | 99.6% | 99.4% | 98.0% | 67.28 ms |

Ngân sách 8 bảng/48 cột là điểm khởi đầu ưu tiên recall: hybrid giữ đủ tham chiếu 98,0% với prompt giảm 10,7%; BM25 giữ đủ 96,2% với prompt giảm 10,8%. Chưa coi đây là bảo đảm accuracy.

Đây là sweep exploratory trên Mini-Dev; không phải kết quả held-out sau khi chọn hyperparameter. Tỷ lệ giữ đủ tham chiếu phản ánh nguy cơ loại nhầm schema, không phải cận trên tuyệt đối của execution accuracy: có thể tồn tại SQL tương đương dùng cột/bảng khác. Vì vậy chưa bật BM25 mặc định và chưa khẳng định tăng chất lượng SQL.

## 6. Hướng dẫn Kaggle

Notebook sẵn ở `notebooks/kaggle_schema_pruning_bird.ipynb`. Quy trình:

1. Tải source dự án lên một Kaggle Dataset, thêm vào notebook. Sửa `REPO_INPUT` theo vị trí thật; notebook sao chép sang `/kaggle/working/SLM-PRJ` để chạy và ghi output.
2. Chọn accelerator có sẵn trong Settings. Bật Internet để tải dependency/model/metadata, hoặc gắn sẵn cache/model/wheelhouse cho phiên offline. Không giả định loại GPU hoặc quota; ghi tên GPU/memory thực tế khi chạy.
3. Chạy phần retrieval trên 500 câu trước; xem `label_errors`, missing tables/columns, recall và prompt token reduction.
4. Gắn SQLite **đúng package cùng phiên bản** theo hướng dẫn Mini-Dev, sửa `DB_ROOT` thành thư mục có cấu trúc `<db_id>/<db_id>.sqlite`. Notebook không tự tải archive DB lớn.
5. Bật `RUN_GENERATION=True`, chọn model, smoke test 10 câu; khi ổn đổi `LIMIT=None` để chạy đủ split.
6. Chạy full/BM25/(hybrid) với greedy, batch 1, sample_rows 0, correction off và cùng giới hạn context/execution. Warm-up không tính vào báo cáo; lặp phép đo nếu muốn kết luận về latency. Khi mở lại notebook, báo rõ index/model cache cold hay warm.
7. Tải `schema_pruning_results.zip` trong output; giữ predictions, config, provenance, pip freeze và GPU/model revision cùng báo cáo.

Nguồn Kaggle: <https://www.kaggle.com/docs/notebooks>; metadata GPU/Internet: <https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md>. `/kaggle/input` là nguồn dữ liệu chỉ đọc; vì vậy notebook sao chép code sang working. Xác nhận từ Kaggle Staff: <https://www.kaggle.com/discussions/general/215281>.

Chạy model bằng CLI sau khi import dữ liệu:

```bash
python scripts/import_bird.py \
  --questions data/bird_minidev/mini_dev_sqlite.jsonl \
  --database-root /kaggle/input/bird-sqlite/dev_databases \
  --output data/benchmarks/bird_eval.json

# config_kaggle.yaml: model local_files_only=false khi cần tải;
# greedy, correction=false, sample_rows=0; cùng ngân sách cho các arm.
python -m evaluation.evaluate --config config_kaggle.yaml \
  --dataset data/benchmarks/bird_eval.json --retrieval full \
  --profile-cuda --output results/bird_full
python -m evaluation.evaluate --config config_kaggle.yaml \
  --dataset data/benchmarks/bird_eval.json --retrieval bm25 \
  --profile-cuda --output results/bird_bm25
```

Không dùng số EX nội bộ thay cho điểm BIRD chính thức. Evaluator dự án giữ duplicate rows và xét thứ tự khi gold có ORDER BY/LIMIT; comparator của Mini-Dev có quy ước riêng. Pipeline ghi input tokens sau chat template, output tokens, context overflow và peak allocated CUDA memory theo từng device. Tổng peak các device không nhất thiết là peak đồng thời của toàn hệ thống. Trọng số/model load không nằm trong thời gian mỗi request; `model_load_ms` được ghi riêng nếu evaluator tự tạo backend.

## 7. Scorer chính thức và BIRD dev

Xuất prediction (phải có **đủ đúng subset** tương ứng dataset, giữ cả thất bại):

```bash
python scripts/export_bird_predictions.py \
  --dataset data/benchmarks/bird_eval.json \
  --predictions results/bird_bm25/predictions.jsonl \
  --output results/bird_bm25/official
```

Exporter cũng ghi `mini_dev_difficulty.jsonl` vì evaluator chính thức đọc difficulty theo dòng JSONL. Prediction JSON nối SQL với db_id bằng `\t----- bird -----\t`. Gold mỗi dòng `SQL<TAB>db_id`. Exporter sắp lại theo thứ tự dataset vì scorer đọc theo vị trí, không join bằng question ID; xuất `order.json` để audit. Smoke test 10 câu cần tạo dataset subset 10 câu trước khi export, không trỏ exporter vào file 500 câu. Câu không có prediction vẫn có entry SQL thất bại, không bị xóa khỏi mẫu số.

Ghim evaluator Mini-Dev: GitHub revision `abd11b6db92a1c9f809b32f7564c7c71b34d67f0`, thư mục `evaluation/`. Nguồn script: <https://github.com/bird-bench/mini_dev/tree/abd11b6db92a1c9f809b32f7564c7c71b34d67f0/evaluation>. Dùng `evaluation_ex.py`/`evaluation_ves.py` và đối chiếu `run_evaluation.sh` của commit đó; có link cũ trong README không khớp tree hiện tại.

Ví dụ gọi EX sau khi checkout evaluator đúng commit và cài dependency `func-timeout==4.3.5 psycopg2-binary==2.9.9 PyMySQL==1.1.1`:

```bash
python /kaggle/working/mini_dev/evaluation/evaluation_ex.py \
  --predicted_sql_path results/bird_bm25/official/predict_mini_dev.json \
  --ground_truth_path results/bird_bm25/official/mini_dev_gold.sql \
  --db_root_path /kaggle/input/bird-sqlite/dev_databases/ \
  --diff_json_path results/bird_bm25/official/mini_dev_difficulty.jsonl \
  --sql_dialect SQLite --num_cpus 2 --meta_time_out 30.0 \
  --output_log_path results/bird_bm25/official/ex.txt
```

Giữ dấu `/` cuối `--db_root_path` vì evaluator ghép chuỗi đường dẫn. Bản scorer này chia cho số câu từng difficulty mà không xử lý category rỗng: smoke subset thiếu một trong ba nhóm có thể gây division by zero. Dùng đủ 500 câu với các nhóm `simple/moderate/challenging` để chấm chính thức. Script chỉ dùng exporter/scorer Mini-Dev cho SQLite; khi đổi dev hoặc dialect phải đối chiếu scorer tương ứng.

Nguồn SQLite hiện được dẫn từ GitHub Complete Package và kênh Docker trong dataset card HF. Manifest metadata đã tải **không chứa checksum SQLite**, nên không thể chứng minh archive bất kỳ khớp bản annotation mới chỉ bằng hash JSONL. Lưu checksum `.sqlite`/digest nguồn riêng và kiểm tra gold SQL. Không mặc định ZIP Mini-Dev cũ đồng nhất với release 2026.

**VES/R-VES đo hiệu quả thực thi câu SQL**, khác thời gian retrieval hoặc thời gian LLM. Giữ riêng EX/R-VES, generation latency và peak VRAM. Nguồn định nghĩa: <https://arxiv.org/html/2305.03111v3> và evaluator Mini-Dev đã ghim.

Để đánh giá BIRD dev, thay nguồn bằng `dev.json`/`dev_tables.json` và đúng SQLite package được tải từ <https://bird-bench.github.io/>. `load_schemas` hỗ trợ cả DDL Mini-Dev lẫn BIRD/Spider `tables.json`. Chạy import như trên với `--questions <dev.json>`, sau đó evaluator model. Lưu hash/version split, không coi điểm Mini-Dev 500 và dev là cùng một phép đo. Nếu tune trên Mini-Dev, chốt cấu hình rồi đánh giá dev hoặc một tập hold-out đã xác định trước.

## 8. Điều kiện quyết định đưa vào sử dụng

Đề xuất tiêu chí thực nghiệm, chưa phải kết quả đạt được: ưu tiên all-references coverage cao (ví dụ >=95%), xác định trước mức giảm EX cho phép và mức giảm latency/VRAM có ý nghĩa. Ngưỡng cần dựa trên mục tiêu ứng dụng. Đối chiếu paired outcomes trên cùng câu hỏi, báo số câu đúng→sai và sai→đúng, confidence interval khi mẫu nhỏ. Ablation nên gồm full, BM25, dense, hybrid; có/không evidence; giữ bảng toàn cột so với lọc cả cột; bridge budget; encoder cold/warm.

Máy local không có GPU hoạt động, nên hiện **chưa có phép đo EX của model thật, generation latency GPU, peak VRAM hoặc điểm scorer BIRD chính thức**. Notebook chuẩn bị các bước đó; không gọi việc kiểm tra cú pháp notebook là đã chạy Kaggle.


## 9. Kiểm chứng triển khai

- Python 3.12, đúng `requirements.txt`: **85 test đạt**, gồm Streamlit AppTest, routing nhiều DB, evidence retrieval, FK bridge budget, projection samples, gold-label scope, cache dense/hybrid, correction prompt đã lọc, token EOS/padding và exporter.
- `ruff check . --no-cache`: đạt. `git diff --check`: đạt.
- Notebook: JSON/code cell compile đạt; chưa chạy trên một session Kaggle thực tế.
- Scorer EX chính thức tại commit đã ghim chạy thành công với SQLite fixture gồm 3 câu ở 3 difficulty: 2 prediction đúng và 1 thiếu prediction, EX **66,67% đúng như dự kiến**. Đây chỉ là test exporter/scorer, **không phải điểm model trên BIRD**.
- Chạy CLI evaluation với mock để kiểm tra đường xuất artifact. Mock được gắn nhãn fixture và không dùng làm số liệu chất lượng model.
