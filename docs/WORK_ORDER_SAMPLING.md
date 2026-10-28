# WORK ORDER — Cải thiện điểm số khi sinh mẫu (sampling + prompt) cho SLM-PRJ (httm_1)

Bản này chưng cất từ dự án nghiên cứu `HTTM/SLM-PRJ` (nhánh `research/slm_sql/`, vLLM)
sang dự án này (`httm_1/SLM-PRJ`, HF-transformers). Làm **tuần tự**, không nhảy phase.
Mỗi phase có cổng nghiệm thu (gate); không đạt gate thì không sang phase sau.

Mục tiêu cuối: đo được **pass@k / SC@k / top2-coverage@k** trên **BIRD mini-dev (500 câu)**
với k đến 64, **trong một phiên làm việc duy nhất**.

---

## §0. Số liệu gốc (đã đo, không được sửa)

Nguồn: `docs/results/pool_stats.json`, `docs/results/sweep.csv`, `results/verification.md`.

| Đại lượng | Giá trị | Nguồn |
|---|---|---|
| `greedy_avg_s` (batch 1, 1 ứng viên) | **47.53 s** | pool_stats.json |
| `batch_avg_s` (**một batch 4 ứng viên**) | **57.098 s** | pool_stats.json |
| → chi phí mỗi ứng viên khi batch 4 | **14.27 s** | 57.098 / 4 |
| EX greedy | 0.43 | sweep.csv |
| EX tốt nhất (adaptive 4-4-8, t0.9, m4) | 0.5496 | sweep.csv |
| `oracle_pass@16` | **0.88** | sweep.csv |
| Khoảng cách pass@16 − SC@16 | **33 điểm** | 0.88 − 0.55 |
| Phần cứng đã đo | RTX 2050 **4 GB**, Windows, driver 528.20 | verification.md |

`batch_size` thực tế của lần sweep = **4** (suy ra từ `est_latency_s`:
k1=k2=k4=57.098, k8=114.196, k16=228.392 → `ceil(k/4) × 57.098`).

Mốc của tác giả SLM-SQL: 1534 câu × 64 mẫu = **98 176 ứng viên trong < 3 giờ**
trên **một RTX 4090D 24 GB thuê $0.26/h** → **≈ 0.11 s/ứng viên**.

**Tỷ số: 14.27 / 0.11 ≈ 130×.**

---

## §1. Vì sao latency cao 130× (chẩn đoán)

Toàn bộ 130× là **hạ tầng**, không phải thuật toán. Tác giả không có mẹo prompt nào
làm sinh mẫu nhanh hơn. Phân rã theo thứ tự độ lớn:

### 1.1 Runtime: HF `generate()` thay vì vLLM — ước ~8–15×
`grep -i vllm` toàn repo = **0 kết quả**. `core/model_backend.py:20-25` chỉ có `mock`
và `huggingface`. Hệ quả cụ thể trong `core/hf_model.py:98-141`:

- `inputs` có batch dim = **1**, còn `num_return_sequences=size`. HF gọi
  `_expand_inputs_for_generation` → `repeat_interleave` → **prompt được prefill lặp
  `size` lần**. Với prompt 7000 token và batch 4, đó là 28 000 token prefill thừa
  cho mỗi batch. vLLM prefill **một lần** rồi fork KV-cache cho cả `n` mẫu, nên chi phí
  prefill **không phụ thuộc k**.
- Batch HF chạy đến khi **chuỗi dài nhất** trong 4 chuỗi kết thúc. 3 chuỗi còn lại
  decode padding vô ích. vLLM có continuous batching: chuỗi nào xong thì rời batch,
  chỗ trống được lấp bằng request mới.
- Không có PagedAttention: KV-cache cấp phát theo `max_new_tokens`, không theo độ dài thực.

### 1.2 GPU: RTX 2050 4 GB vs 4090D 24 GB — ~9× băng thông
Decode là bài toán **bound bởi băng thông bộ nhớ** (phải đọc toàn bộ trọng số mỗi token).

| GPU | Băng thông | VRAM |
|---|---|---|
| RTX 2050 | ~112 GB/s | 4 GB |
| RTX 4090D | ~1008 GB/s | 24 GB |

**9× ngay ở băng thông.** Nhưng vấn đề nặng hơn là VRAM không đủ:

| Thành phần | Dung lượng |
|---|---|
| Trọng số 1.5B fp16 | 3.09 GB |
| CUDA context (Windows WDDM) | ~0.6–0.8 GB |
| KV-cache (Qwen2.5-1.5B: 28 lớp, 2 KV head, head_dim 128 → **28 KB/token**) × 7512 token × 4 chuỗi | ~0.84 GB |
| **Tổng** | **≈ 4.6 GB > 4.0 GB** |

Model **không vừa**. Trên Windows, driver NVIDIA không báo lỗi mà **tự tràn sang
"shared GPU memory" (RAM hệ thống qua PCIe)** → code hình dạng GPU chạy ở tốc độ PCIe.

### 1.3 Fallback thiết bị im lặng — `core/hf_model.py:27-38`
```python
self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
if self.device == "auto":
    self.device = "cpu"
...
if self.device == "cpu":
    dtype = torch.float32
```
Với `device: auto`, nếu `torch.cuda.is_available()` trả False (sai driver, sai wheel CUDA,
chạy trong môi trường khác) thì **rơi xuống CPU + cưỡng bức fp32** (~6.2 GB RAM, 2–4 tok/s)
**không một dòng cảnh báo**. Nhánh `raise RuntimeError` ở dưới chỉ chạy khi người dùng
ghi `device: cuda` tường minh — tức cấu hình mặc định là nhánh *im lặng*.

**Kiểm chứng số học:** `max_new_tokens: 512`, greedy 47.53 s → tốc độ **≤ 10.8 tok/s**
(chặn trên tuyệt đối). Nếu đầu ra thực ~150 token thì chỉ **3.2 tok/s**. RTX 2050 fp16
phải cho 15–25 tok/s. Con số đo được **thấp hơn cả mức sàn của chính GPU đó** → đang
chạy CPU-fp32 hoặc đang phân trang qua PCIe.

**Mâu thuẫn chưa giải quyết trong repo:** `results/verification.md` ghi **18.0 s/câu**
(5 câu, chạy nguội), còn sweep ghi **47.53 s**. Hai con số này là dấu vết của **hai
đường thực thi khác nhau**. Phase 1 bắt buộc phải làm rõ cái nào là thật.

### 1.4 `batch_size: 4` vs ~128–256 chuỗi đồng thời
`config.yaml` đặt `batch_size: 4`. vLLM trên 24 GB chạy 128–256 chuỗi đồng thời.
Nhưng trên 4 GB thì 4 **đã là trần** (xem 1.2) — không phải lỗi chọn tham số, mà là
hệ quả của phần cứng.

### 1.5 Không có stop string — chỉ có deadline
`core/hf_model.py:98-141` chỉ cài `DeadlineStop` (kiểm tra `time.perf_counter()`).
Prompt yêu cầu `<answer>...</answer>` (`core/prompt_builder.py:7-19`) nhưng **không có
`stop=["</answer>"]`**. Model viết xong SQL rồi vẫn decode tiếp đến EOS hoặc 512 token.
Dự án nghiên cứu đo biến thể này tường minh (`research/slm_sql/bench.py:19`,
variant `stop`) và theo dõi `pct_finish_length` để lượng hoá. Ước **1.2–2×**.

### 1.6 Dấu vết quyết định: batch gần như miễn phí
- 1 ứng viên: 47.53 s
- 4 ứng viên: 57.10 s
- **Chi phí biên mỗi ứng viên thêm vào: (57.10 − 47.53)/3 = 3.19 s**, tức **1/15 lần**
  ứng viên đầu tiên.

4× công việc chỉ tốn 1.20× thời gian. Đây là **chữ ký của chế độ bound bởi việc
stream trọng số**: gần như toàn bộ 47.53 s là chuyển trọng số, không phải tính toán.

**Kết luận: đòn bẩy lớn nhất là tăng số chuỗi đồng thời, và điều đó đòi VRAM.**
Trên 4 GB đã hết đường. Mọi tối ưu prompt/thuật toán đều là hạng thứ cấp.

### 1.7 Kiểm tra tích: 9 (GPU) × 10 (runtime) × 1.5 (stop) ≈ **135×**
Khớp với 130× đo được. **Không còn ẩn số.** Không cần đi tìm bug nào khác.

### 1.8 Khuyến nghị phần cứng
Thuê **RTX 4090 / 4090D 24 GB ở $0.26/h** (Vast.ai, RunPod…). Lý do:
- Trùng đúng kiến trúc sm89 + FlashAttention-2 với bài báo → số đo so sánh được.
- GPU cố định giữa các phiên → cache/artifact dùng lại được (vLLM **không**
  bitwise-deterministic khi đổi GPU hoặc đổi cấu hình engine).
- Không bị ngắt phiên, có root.
- Toàn bộ mini-dev k=64 ước **~15 phút ≈ $0.07**. Cả chuỗi thí nghiệm dưới $1.

**Cảnh báo:** bench trên T4 rồi chạy thật trên 4090 là **bench vô ích** — `batch_size`
tối ưu phụ thuộc chỗ trống KV-cache (16 vs 24 GB) và attention backend (xformers vs FA2).
Bench và chạy phải **cùng một GPU**.

---

## §2. Nguyên tắc chưng cất

### Port sang (liên quan trực tiếp điểm số khi sampling)
| Từ `HTTM/SLM-PRJ` | Sang | Vì sao |
|---|---|---|
| `research/oracle_analysis.py`: `pass_at_k`, `sc_at_k`, `top2_coverage_at_k`, `analyze_by_group` | `evaluation/metrics.py` | Đây là **khung ba khoảng cách**, hiện repo này hoàn toàn thiếu |
| `research/slm_sql/backend.py` (vLLM) | `core/vllm_model.py` | Đòn bẩy 8–15× |
| `research/slm_sql/bench.py`: ladder biến thể + `pct_finish_length` | `evaluation/bench.py` | Đo tốc độ có kỷ luật, tách khỏi artifact thật |
| `stop: ["</answer>"]` + `include_stop_str_in_output: true` | `config.yaml` | §1.5 |
| `research/slm_sql/artifacts.py`: `digest()`, `Journal` + `fsync` | `evaluation/journal.py` | Resume qua phiên |
| Cô lập report `_bench` / `_pilot` | `evaluation/` | Chống lẫn số bench vào kết quả |
| `core/schema_selector.py` (đã có, đang **tắt**) | bật cho BIRD | BIRD schema dài → tràn context |

### KHÔNG port
- **CSC merger / LoRA training.** Ngoài phạm vi yêu cầu (chỉ "cải thiện điểm số khi
  sinh mẫu"), và base merger `CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct` cùng dataset
  SynSQL-Think là **CC-BY-NC-4.0** (không upload công khai, không thương mại).
- **`tensor_parallel_size: 2`.** Một GPU thì TP=2 vô nghĩa; nhiều GPU thì
  **data parallelism** (shard câu hỏi) luôn thắng cho model 1.5B.
- Hạ tầng `prepare/plan/freeze/report-suite` của dự án nghiên cứu — quá nặng cho repo này.

### Đã có sẵn, KHÔNG làm lại
`evaluation/run_sweep.py` với tách `collect` / `simulate` **đã chính là thiết kế
offline-replay**: sinh một lần ở `k_max`, replay mọi policy trên pool cache với
chi phí GPU bằng 0, 200 hoán vị bootstrap. Giữ nguyên kiến trúc này, chỉ mở rộng.
`core/result_grouper.py` đã nhóm theo **kết quả thực thi** (đúng chuẩn), không theo text SQL.

---

## PHASE 1 — Xác lập lại baseline trung thực (BẮT BUỘC TRƯỚC TIÊN)

Không được tối ưu gì khi còn chưa biết code đang chạy trên thiết bị nào.

1. **Bỏ fallback im lặng.** Sửa `core/hf_model.py:27-38`: khi `device: auto` mà
   `torch.cuda.is_available()` False → **in cảnh báo rõ ràng** kèm
   `torch.__version__`, `torch.version.cuda`, `torch.cuda.device_count()`. Thêm
   `device: cuda_required` để fail-fast.
2. **Ghi thiết bị thật vào artifact.** `run_sweep.py` ghi `config.json`; bổ sung
   `device`, `dtype`, `torch.cuda.get_device_name(0)`, `total_memory`,
   `torch.cuda.max_memory_allocated()` sau mỗi câu.
3. **Giải thích mâu thuẫn 18.0 s vs 47.53 s.** Chạy lại 5 câu của
   `verification.md` bằng chính `run_sweep.py collect --limit 5` và so.
4. **Đo phân bố độ dài đầu ra.** Thêm `n_output_tokens` và `finish_reason`
   (`length` vs `eos`) vào `_record`. Nếu `pct_finish_length` cao → §1.5 là thật.

**Gate 1:** artifact nói rõ GPU/CPU + dtype; biết `%` ứng viên bị cắt vì `length`;
hai con số 18.0 s và 47.53 s đã được quy về cùng một điều kiện.

---

## PHASE 2 — Backend vLLM

1. Tạo `core/vllm_model.py` implement đúng interface `ModelBackend`
   (`generate(prompt, n, parameters, deadline) -> list[Candidate]`, `start_request(seed)`,
   `lock`, thêm `close()`).
2. Đăng ký trong `core/model_backend.py:20-25`: `backend: vllm`.
3. **Dùng `n=k` của vLLM**, tuyệt đối không vòng `while` theo batch như HF — để engine
   tự fork KV-cache từ một lần prefill.
4. Truyền thẳng **`prompt_token_ids`** (xem `research/slm_sql/backend.py`,
   `verify_prompt_tokenization`) để tokenize một lần, và để ký số prompt ổn định
   giữa các biến thể bench.
5. Allowlist request-override: `{stop, include_stop_str_in_output, max_tokens,
   temperature, top_p, logprobs}`. Không cho override khoá engine ở tầng request.
6. Khoá engine vào `config.yaml`: `gpu_memory_utilization: 0.90`, `max_model_len: 8192`,
   `dtype: float16`, `enforce_eager: true` (bench), `enable_prefix_caching: false` (mặc định).

**Gate 2:** `backend: vllm` và `backend: huggingface` cho **EX chênh ≤ 1 điểm** trên
100 câu `evaluation_questions.json` ở greedy (temperature 0). Lệch nhiều hơn là có bug
ở prompt hoặc extraction, **không phải** "sai số sampling".

---

## PHASE 3 — Stop string + prompt

1. `config.yaml`: `generation.stop: ["</answer>"]`,
   `generation.include_stop_str_in_output: true`.
   **Bắt buộc `include_stop_str_in_output: true`**, vì `core/sql_extractor.py:10`
   regex `<answer>(.*?)</answer>` cần thẻ đóng. Thiếu nó → extract rỗng 100%.
2. Thêm assert như `bench.py:237`: **`pct_extract_empty` phải bằng 0**. Lỗi này âm thầm
   biến EX thành 0 mà không crash.
3. Giữ nguyên nội dung prompt ở `core/prompt_builder.py`. **Không viết lại prompt** ở
   phase này: prompt thay đổi thì mọi số đo trước đó mất giá trị so sánh. Prompt chỉ
   được sửa sau Phase 4, và phải báo cáo như một ablation riêng.
4. `schema.sample_rows: 2` là tham số dài prompt mạnh nhất. Chưa đổi, chỉ **đo** độ dài
   prompt theo `db_id` (cần cho Phase 6).

**Gate 3:** `pct_extract_empty == 0`; `tokens_per_s` tăng và `pct_finish_length` giảm
so với Gate 1; EX **không giảm** (stop string chỉ cắt phần vô dụng sau `</answer>`).

---

## PHASE 4 — Khung ba khoảng cách (phần cốt lõi về điểm số)

Hiện `evaluation/metrics.py` có `execution_accuracy`, `exact_match`, `valid_sql_rate`,
`average_candidates`, latency p50/p95, `early_stop_rate`, correction… nhưng **thiếu hoàn
toàn** `pass@k`, `sc@k`, `top2_coverage@k`. `pass@16` chỉ tồn tại như **một dòng
`oracle_pass@16`** trong sweep.csv — không có đường cong theo k.

Bất đẳng thức: `1 ≥ pass@k ≥ top2_coverage@k ≥ SC@k`

| Khoảng cách | Công thức | Ý nghĩa | Cách đóng |
|---|---|---|---|
| **C** | `1 − pass@k` | model chưa bao giờ sinh ra đáp án đúng | tăng k, tăng đa dạng, sửa prompt, fine-tune |
| **A** | `pass@k − top2_cov@k` | đúng có trong pool nhưng **không nằm ở 2 nhóm vote lớn nhất** | **ngoài tầm với của CSC/vote** → cần reranker |
| **B** | `top2_cov@k − SC@k` | đúng nằm trong top-2 nhóm nhưng vote chọn sai | **CSC / merger / ranker thắng được phần này** |

Trên `sweep.csv` hiện tại: `0.88 − 0.55 = 33 điểm` mất ở **chọn**, không ở **sinh**.
Đo A và B tách bạch sẽ nói cho ta biết 33 điểm đó có bao nhiêu phần *lấy được*.

1. Port `pass_at_k`, `sc_at_k`, `top2_coverage_at_k` từ
   `HTTM/SLM-PRJ/research/oracle_analysis.py` vào `evaluation/metrics.py`.
   Dùng **hoán vị không hoàn lại trên pool cache** (`n_permutations=100`) → **một lần
   sinh ở k_max cho ra toàn bộ đường cong k miễn phí**.
2. **Viết mới `top3_coverage_at_k`** — bản này không tồn tại ở repo nguồn.
3. Port `analyze_by_group` để tách theo `db_id` và `difficulty`.
4. Thêm `k` vào `FIXED_K`: `[1, 2, 4, 8, 16, 32, 64]`.
5. Sửa `run_sweep.py:159`:
   `sample = dict(config["sampling"], max_candidates=len(rows[0]["pool"]))`
   đang **vô hiệu hoá** `max_candidates` trong config (ghi đè bằng kích thước pool).
   Cho phép `max_candidates < pool` để mô phỏng ngân sách thật.
6. Sửa `run_sweep.py` `_adaptive` (dòng 128-143): đang truyền `elapsed_seconds=0` nên
   nhánh `time_budget_seconds` trong `core/adaptive_sampler.py:14` **không bao giờ chạy**
   khi simulate. Đưa thời gian ước lượng tích luỹ vào để chính sách ngân sách thời gian
   được mô phỏng thật.
7. Sửa `generation_score`: `core/types.py:44` khởi tạo 0.5 và **không bao giờ cập nhật**,
   nên trọng số `ranking.generation: 0.10` là **hằng số chết** — hiện 10% điểm xếp hạng
   là nhiễu. Hoặc nối vào logprob trung bình của vLLM (`logprobs` đã trong allowlist),
   hoặc đặt trọng số về 0 và tái chuẩn hoá 3 trọng số còn lại.

**Gate 4:** báo cáo in được đường cong `pass@k`, `top2_cov@k`, `top3_cov@k`, `SC@k`
cho k ∈ {1,2,4,8,16,32,64} từ **một** lần sinh, kèm phân rã A/B/C và CI bootstrap 95%.

---

## PHASE 5 — Bench ladder có kỷ luật

Port `research/slm_sql/bench.py` thành `evaluation/bench.py`:

1. Biến thể tăng dần, **luôn bắt đầu bằng `baseline`**:
   `baseline` → `stop` → `stop+notok` → `stop+notok+prefix` → `stop+notok+batch{16,32,64}`.
2. Metric: `wall_clock_s`, `output_tokens`, `tokens_per_s`, `speedup_vs_baseline`,
   **`pct_finish_length`**, **`pct_extract_empty`**, `n_completions`.
3. **Đóng băng ký số prompt** giữa các biến thể (`bench.py:192`): token IDs của prompt
   thay đổi → raise. Nếu không có cái này, "tăng tốc" có thể chỉ là prompt ngắn đi.
4. **Tách phạm vi đo:** `engine_init_s`, `warmup_s`, `prompt_verification_s` báo cáo
   **riêng**, không tính vào `wall_clock_s`.
5. **Cô lập artifact:** thư mục output phải chứa `_bench` trong tên và phải nằm ngoài
   thư mục thí nghiệm thật (`bench.py:64-73`). Số bench không bao giờ được vào báo cáo.
6. Rung gate: rung 1 = 5 câu × k4; rung 2 = 20 câu, ≥10 DB, k16. Rung 0-2 bắt buộc
   `enforce_eager=True`.
7. Đổi khoá engine → **restart engine** và in `RESTART ENGINE: <khoá đã đổi>`.

**Gate 5:** có bảng `tokens_per_s` theo biến thể trên **đúng GPU sẽ dùng để chạy thật**;
chọn được `batch_size` / `max_num_seqs` tối ưu từ số đo, không từ phỏng đoán.

---

## PHASE 6 — BIRD mini-dev (yêu cầu CHÚ Ý)

### 6.1 Blocker phải sửa trước
`evaluation/run_sweep.py:76` là `database = args.db` — **bỏ qua hoàn toàn
`item["db_path"]`**. Mọi câu hỏi bị chạy trên **một** database duy nhất. BIRD có nhiều
`db_id`, nên harness sweep **hiện tại không thể chạy BIRD**.
`evaluation/evaluate.py:39` đã làm đúng:
```python
database = (dataset.parent / item["db_path"]).resolve() if item.get("db_path") else db_path
```
→ Áp đúng một dòng đó vào `run_sweep.py`. Đồng thời **cache schema theo `db_id`**:
hiện `schema_inspector.inspect(database)` chạy lại mỗi câu; mini-dev 500 câu / 11 DB
nghĩa là inspect lặp ~45 lần cho mỗi DB.

### 6.2 Nạp dữ liệu
`scripts/import_bird.py` đã có và đã đúng (nhận `SQL` hoặc `sql`, giữ `evidence`,
kiểm tra đường dẫn DB, ghi `db_path` tương đối). **Repo hiện không có bất kỳ file BIRD
nào** — chỉ có `databases/student_management.db` và 2×100 câu đồ chơi.

Tải gói SQLite chính thức của [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev)
(phải có **cả** câu hỏi **và** database — không lấy riêng JSON), rồi:

```bash
python scripts/import_bird.py \
  --questions data/benchmarks/mini_dev_sqlite.json \
  --database-root data/benchmarks/dev_databases \
  --output data/benchmarks/bird_minidev.json
```

### 6.3 Bẫy context — phải đo TRƯỚC khi chạy
`core/hf_model.py:116-118`: nếu `input_length > max_input_tokens` thì trả về **một
Candidate lỗi** `"Prompt vượt context"`. Không crash, không cảnh báo — pool rỗng và
**EX = 0 một cách im lặng**. Schema BIRD dài hơn `student_management.db` rất nhiều, và
`schema.sample_rows: 2` nhân thêm độ dài.

Bắt buộc: viết script đo phân bố độ dài prompt trên cả 500 câu **trước** khi sinh.
Nếu có câu vượt `max_input_tokens: 7000`, xử lý theo thứ tự:
1. Bật `validation.schema_selection: true` — `core/schema_selector.py` đã có sẵn, khớp
   tên bảng theo từ vựng câu hỏi + bao đóng khoá ngoại. Hiện bị tắt ở
   `config.yaml` / `core/pipeline.py:56-57`.
2. Giảm `schema.sample_rows` 2 → 1 hoặc 0.
3. Nâng `max_model_len` lên 8192 (vLLM) và `max_input_tokens` tương ứng.

**Báo cáo số câu bị loại vì context như một con số riêng.** Không được lặng lẽ tính
chúng là sai.

### 6.4 Chi phí pool
`_record` lưu `"result": execution.normalized_result` — **toàn bộ tập kết quả** vào
`pool.jsonl`. Với `max_result_rows: 50000` và 500 câu × 64 mẫu, file sẽ phình rất lớn.
Lưu **hash** của normalized_result để nhóm, cộng thêm một bản xem trước bị cắt ngắn.
`core/result_grouper.py` chỉ cần so sánh bằng nhau nên hash là đủ.

**Gate 6:** `run_sweep.py collect --dataset data/benchmarks/bird_minidev.json --limit 20`
chạy xong, mỗi câu dùng **đúng** database của nó, và `pct_extract_empty == 0`,
`0` câu bị loại vì context.

---

## PHASE 7 — Chạy một phiên trên mini-dev

### 7.1 Vì sao mini-dev
- BIRD dev = **1534** câu → 1534 × 64 = 98 176 ứng viên.
- BIRD **mini-dev = 500** câu → 500 × 64 = **32 000** ứng viên.
- **Giảm 3.07×** — đây chính là đòn bẩy để vừa một phiên.

### 7.2 Ngân sách thời gian (ước lượng, phải xác nhận bằng Phase 5)
Giả định với `stop=["</answer>"]`, đầu ra trung bình ~150 token
(prompt cấm reasoning nên đầu ra chỉ là SQL):
- Decode: 32 000 × 150 ≈ **4.8 M token**.
- Prefill: ~500 × 3000 ≈ **1.5 M token**, và vLLM prefill **một lần mỗi câu** (fork cho
  cả 64 mẫu) — nên prefill **không** nhân với 64.

| GPU | tokens/s (ước, batch cao) | Decode | Prefill | **Tổng** |
|---|---|---|---|---|
| T4 16 GB (Kaggle, không FA2) | ~1 100 | ~73 ph | ~8 ph | **~1.4 h** |
| L4 24 GB | ~2 000 | ~40 ph | ~3 ph | **~45 ph** |
| **4090D 24 GB** | ~6 500 | ~12 ph | ~1 ph | **~15 ph** |

Không có stop string thì nhân đôi cột decode. Cả ba đều vừa một phiên — nhưng chỉ
4090D cho dư địa chạy lại nhiều lần trong cùng phiên.

### 7.3 Lệnh chạy
```bash
# Pha 1: sinh MỘT lần ở k_max = 64. Đây là lần duy nhất tốn GPU.
python -m evaluation.run_sweep collect \
  --dataset data/benchmarks/bird_minidev.json \
  --pool 64 --batch-size 64 \
  --output results/minidev_k64

# Pha 2: replay mọi policy trên pool cache. Chi phí GPU = 0, chạy được trên CPU.
python -m evaluation.run_sweep simulate \
  --dataset data/benchmarks/bird_minidev.json \
  --pool 64 --bootstrap 200 \
  --output results/minidev_k64
```

**`collect` và `simulate` phải tách rời.** `simulate` chạy lại bao nhiêu lần cũng được,
trên máy không GPU, sau khi phiên GPU đã kết thúc. Mọi phương pháp thấy **đúng cùng
một tập mẫu** → so sánh công bằng, và toàn bộ đường cong k lấy từ một lần sinh.

### 7.4 Resume
`run_sweep.py:60-62` đã có `done` từ `pool.jsonl`. Củng cố thêm:
- `flush()` + `os.fsync()` sau **mỗi** record (mẫu `research/slm_sql/artifacts.py`),
  không chỉ dựa buffer — phiên bị ngắt giữa dòng sẽ làm hỏng JSONL.
- Ghi `digest()` = sha256 của config chuẩn hoá + `model_identity` + phiên bản runtime
  vào `pool.jsonl`; khi resume mà digest khác thì **chặn**, không trộn.
- **Đổi GPU bắt buộc thư mục thí nghiệm mới.** vLLM không bitwise-deterministic khi đổi
  GPU hoặc đổi `max_num_seqs` / `tensor_parallel_size` / phiên bản. Pool từ hai GPU
  khác nhau **không so sánh được**.

### 7.5 Latency đo được, không ngoại suy
`run_sweep.py:122-125` `_batch_cost()` **ngoại suy**: `ceil(count/batch_size) × per_batch`.
Đó là lý do k1=k2=k4 cho **cùng** 57.098 s — vô nghĩa về mặt vận hành. Với vLLM `n=k`,
mô hình này sai hẳn vì prefill được chia sẻ. Thay bằng **thời gian thật đo được mỗi câu
ở mỗi k** cho một tập con nhỏ, và nhãn rõ cột nào là đo, cột nào là ước.

**Gate 7:** một phiên duy nhất cho ra `pool.jsonl` đủ 500 câu × 64 mẫu, và `simulate`
in bảng đường cong k cùng phân rã A/B/C, kèm CI bootstrap.

---

## PHASE 8 — Đóng khoảng cách B (chỉ sau khi đã đo được)

Chỉ làm khi Phase 7 cho thấy **B đủ lớn để đáng đóng**. Nếu phần lớn 33 điểm nằm ở A
thì vote/CSC không giúp được, phải làm reranker — đó là dự án khác.

1. Xếp hạng: sửa `generation_score` (xem Phase 4.7). Cân lại
   `ranking.{vote, schema, execution, generation}` bằng **GroupKFold theo `db_id`**
   trên tập validation, **không** trên tập đánh giá cuối.
2. Quét ngưỡng adaptive (`THRESHOLDS`, `MIN_VALID`, `SCHEDULES`) **chỉ trên validation**.
   Tuyệt đối không tinh chỉnh ngưỡng trên EX của tập dev cuối.
3. Kiểm định: paired bootstrap CI 95% + McNemar exact trên các cặp đối chứng. Không
   tuyên bố cải thiện nếu CI chứa 0.

**Gate 8:** mỗi tuyên bố cải thiện đều có CI và p-value, trên tập đánh giá chưa từng
dùng để chỉnh tham số.

---

## §3. Bẫy thực thi (đọc trước khi code)

1. **Fallback CPU im lặng** — `hf_model.py:27-38`. Luôn ghi thiết bị thật vào artifact.
2. **`include_stop_str_in_output` phải True** nếu dùng `stop=["</answer>"]`, vì
   `sql_extractor.py:10` cần thẻ đóng. Sai → extract rỗng 100%, EX = 0, không crash.
3. **`pct_extract_empty` phải assert == 0.** Lỗi extraction giả dạng lỗi model.
4. **Prompt đổi thì số đo cũ chết.** Đóng băng ký số token prompt giữa các biến thể bench.
5. **`run_sweep.py:76` bỏ qua `db_path`** — blocker BIRD, xem 6.1.
6. **Tràn context im lặng** trên schema BIRD — xem 6.3.
7. **`max_candidates` bị vô hiệu** ở `run_sweep.py:159`.
8. **`elapsed_seconds=0`** ở `_adaptive` → ngân sách thời gian không bao giờ kích hoạt.
9. **`generation_score` là hằng số chết** → 10% trọng số xếp hạng là nhiễu.
10. **`est_latency_s` là ngoại suy, không phải đo** — xem 7.5.
11. **Pool.jsonl phình** vì lưu nguyên result set — xem 6.4.
12. **Đổi GPU ⇒ thư mục thí nghiệm mới.** Không resume xuyên GPU.
13. **Artifact bench không được lẫn vào report** — ép `_bench` trong tên thư mục.
14. **`num_return_sequences` của HF prefill prompt lặp k lần.** Nếu giữ đường HF cho
    đối chứng, phải biết nó đắt vì lý do này, đừng đổ cho GPU.
15. **Batch HF chạy đến chuỗi dài nhất.** Trộn câu dễ/khó vào một batch làm chi phí
    tăng theo câu tệ nhất.
16. **`float` so sánh bit-exact** khi nhóm kết quả: `AVG`/`SUM` dấu phẩy động sẽ tách
    nhóm sai. Chuẩn hoá có kiểm soát, và ghi rõ đã chuẩn hoá thế nào.
17. **Chuỗi không strip/casefold** khi so kết quả → nhóm bị tách vì khoảng trắng.
18. **Không tải database cá nhân lên môi trường cloud** nếu muốn dữ liệu chỉ ở local.
19. **`temperature: 0.7` / `top_p: 0.9`** trong repo này khác bài báo
    (`temperature 0.8`, `top_p 1.0`). Chọn một, ghi rõ, không đổi giữa các lần đo.
20. **`correction.enabled` bị `collect` tắt** (`run_sweep.py:63`) để cô lập ảnh hưởng
    của sampling. Đúng — nhưng nghĩa là EX trong sweep **không** bao gồm correction.
    Đừng so trực tiếp với EX của `evaluate.py`.

---

## §4. Thứ tự tóm gọn

| Phase | Việc | Đòn bẩy |
|---|---|---|
| 1 | Baseline trung thực, lộ thiết bị thật | điều kiện tiên quyết |
| 2 | Backend vLLM | **8–15× tốc độ** |
| 3 | Stop string | **1.2–2× tốc độ** |
| 4 | pass@k / SC@k / top2+top3 coverage, phân rã A/B/C | **toàn bộ phần điểm số** |
| 5 | Bench ladder trên đúng GPU | chọn batch bằng số đo |
| 6 | BIRD mini-dev + routing db theo câu | mở được benchmark thật |
| 7 | Một phiên, sinh một lần k=64, replay offline | **3.07× nhờ mini-dev** |
| 8 | Đóng khoảng cách B (chỉ nếu B lớn) | điểm số |

Phase 1-5 không cần dữ liệu BIRD. Phase 6-7 cần tải BIRD mini-dev.
Phần GPU chỉ ở Phase 2, 5, 7 — có thể gói vào một lần thuê 4090D ngắn.
