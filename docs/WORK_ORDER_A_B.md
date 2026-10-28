# WORK ORDER — Hướng A (Re-ranker đóng Gap A) + Hướng B (Phân bổ ngân sách liên câu)

Tài liệu đồng hành với `docs/WORK_ORDER_SAMPLING.md`.
`WORK_ORDER_SAMPLING.md` lo **tốc độ sinh** (latency). Tài liệu này lo **điểm số trên mỗi đơn vị tính toán**
(EX per unit compute) — tức là cải tiến khoa học có thể công bố.

Repo đích: `/home/mint/Documents/httm_1/SLM-PRJ` (HF transformers, không vLLM).
Commit hiện tại: `362ce9e add cav` (`main` đã khớp `origin/main`).

---

## §0. ĐÍNH CHÍNH TRẠNG THÁI THỰC TẾ — đọc trước khi làm bất cứ gì

Bốn sự thật này thay đổi hoàn toàn hình dạng kế hoạch so với bản mô tả khái quát trước đó.

### 0.1. `pool.jsonl` KHÔNG CÒN TỒN TẠI

```
docs/results/   → pool_stats.json, sweep.csv, sweep_candidates.png, sweep_latency.png
results/        → .gitkeep, verification.md
```

Chỉ còn **bản tổng hợp dẫn xuất**. Pool thô (16 ứng viên × 100 câu) đã bị xoá.

**Hệ quả:** không có bước nào của A hoặc B chạy được "ngay hôm nay với chi phí GPU = 0".
Bắt buộc phải `run_sweep.py collect` lại trước. Đây là bước đầu tiên, không thể bỏ qua.

### 0.2. Hướng A đã được implement ~70% trong `evaluation/selection_study.py` (301 dòng)

Script này **đã có sẵn**:

| Thành phần | Vị trí | Trạng thái |
|---|---|---|
| Kiểm tra nhất quán không cần gold | `core/consistency_checks.py` — `CHECKS = ("degenerate","ungrounded","bad_join")` | ✅ xong, có test |
| CAV (vote hạn chế trên ứng viên ít cờ nhất) | `selection_study._clean` + `make_selectors` | ✅ xong |
| Ablation leave-one-check-out | `make_selectors` sinh `CAV only X` / `CAV without X` | ✅ xong |
| Re-ranker logistic cấp nhóm, CV | `LogisticRanker` + `train_rankers` (numpy thuần) | ✅ xong |
| Bootstrap CI ghép đôi | `bootstrap` | ✅ xong |
| McNemar chính xác (nhị thức) | `mcnemar` | ✅ xong |
| `oracle_at_budget`, `oracle_gap_closed` | `study()` | ✅ xong |
| Độ ổn định split-half | `acc_half_odd` / `acc_half_even` | ✅ xong |
| Chẩn đoán từng cờ (fire rate / precision trên câu sai) | `check_diagnostics` | ✅ xong |
| Bảng + hình báo cáo | `evaluation/selection_report.py` | ✅ xong |

**KHÔNG viết lại những thứ này.** Kế hoạch A dưới đây chỉ *mở rộng*.

### 0.3. CAV chưa bao giờ vào pipeline thật

```
grep -rn "ConsistencyChecker" .
  → tests/test_consistency_checks.py
  → evaluation/selection_study.py
  → core/consistency_checks.py
```

`core/pipeline.py` **không import** nó. Pipeline chạy thật (Streamlit demo, `evaluation/evaluate.py`)
vẫn dùng vote thô qua `core/result_grouper.py` + `core/candidate_ranker.py`.

**Hệ quả:** dù `selection_study` có chứng minh CAV thắng bao nhiêu điểm, con số đó **chưa phải là EX của hệ thống**.
Đây là deliverable số 1 của Hướng A: đưa selector thắng vào `core/`.

### 0.4. Hướng B thiếu hoàn toàn

`run_sweep._adaptive` và `selection_study.run_budget` chỉ làm **dừng sớm trong một câu**
(within-question early stopping) — đúng như Adaptive Consistency (Aggarwal et al., 2023).
Ngân sách tiết kiệm được **bị bỏ đi**, không được chuyển sang câu khác.

Không có file nào trong repo phân bổ ngân sách *giữa các câu*. Đây là phần mới thật sự.

Thêm hai lỗi nhỏ liên quan:
- `run_sweep.py:139` → `decide(..., used, 0, sample)`
- `selection_study.py:164` → `decide(..., count, 0, sample)`

Cả hai truyền `elapsed_seconds=0` nên nhánh `time_budget` trong `core/adaptive_sampler.py` **không bao giờ cháy**.
(Lưu ý: `core/pipeline.py:110` truyền `time.perf_counter() - start` — pipeline thật thì đúng, chỉ bộ mô phỏng sai.)

---

## §1. KHUNG ĐO: PHÂN RÃ BA KHOẢNG TRỐNG

Mọi con số trong tài liệu này treo vào bất đẳng thức sau, với cùng một pool ứng viên:

```
1  ≥  pass@k  ≥  top2_coverage@k  ≥  SC@k
```

| Khoảng | Định nghĩa | Ý nghĩa | Ai đóng được |
|---|---|---|---|
| **Gap C** | `1 − pass@k` | Mô hình **không sinh ra** câu đúng lần nào | Chỉ huấn luyện / prompt tốt hơn |
| **Gap A** | `pass@k − top2_coverage@k` | Câu đúng có trong pool nhưng **không nằm trong 2 nhóm phiếu cao nhất** | **Chỉ re-ranker** — vote và CSC đều không chạm tới |
| **Gap B** | `top2_coverage@k − SC@k` | Câu đúng nằm trong top-2 nhưng **vote chọn sai nhóm** | CAV / merger / đổi trọng số |

Số đo tham chiếu:

- SLM-SQL 1.5B, BIRD dev: `82.64 − 67.08 = 15.56` điểm mất vào *lựa chọn*, không phải *sinh*.
- httm_1, toy benchmark (`sweep.csv`): `oracle_pass@16 = 0.88` vs `fixed_k16 = 0.55`
  → **33 điểm** mất vào lựa chọn. Đây là dư địa lớn nhất của dự án.

### 1.1. Thiếu hụt phải bù: `evaluation/metrics.py` chỉ có 45 dòng

Hiện có: `execution_match`, `percentile`, `summarize`.
**Không có** `pass_at_k`, `sc_at_k`, `top2_coverage_at_k`, `top3_coverage_at_k`.

→ **Hôm nay repo không tính được phân rã ba khoảng.** Phải thêm (xem §3, bước A1).

Bản tham chiếu đã có trong repo nguồn: `/home/mint/Documents/HTTM/SLM-PRJ/research/oracle_analysis.py`
(`pass_at_k:24`, `sc_at_k:49`, `top2_coverage_at_k:78`, `analyze_by_group:107`).
`top3` chưa tồn tại ở đâu cả — phải viết mới.

---

## §2. DATASET: BA TẦNG, DÙNG TẦNG NÀO CHO VIỆC GÌ

| Tầng | Dataset | Số câu | DB | Mục đích | Chi phí |
|---|---|---|---|---|---|
| **T0 — khói** | `data/evaluation_questions.json` + `student_management.db` | 100 | 1 | Kiểm tra code chạy, không crash, CSV sinh đúng cột | ~30 phút local/T4 |
| **T1 — chính** | **BIRD mini-dev (SQLite)** → `data/benchmarks/bird_eval.json` | **500** | **11** | Mọi số liệu công bố: ba khoảng, A, B, Pareto | 1 session Kaggle |
| **T2 — xác nhận** | BIRD dev | 1 534 | 11 | Chỉ chạy **một lần cuối**, cấu hình đã đóng băng | 1 session riêng |

### 2.1. Vì sao mini-dev là tầng chính

- 500 × 32 = **16 000 ứng viên**, so với dev 1 534 × 32 = 49 088 → giảm **3.07×**.
- Vẫn có 11 DB → đủ để `GroupKFold` theo `db_id` (xem §3 bước A3) có ý nghĩa.
- Có nhãn khó (`simple/moderate/challenging`) → chạy được bảng phân tầng `acc_easy/medium/hard`.

### 2.2. Vì sao KHÔNG chọn k_max = 64 trên mini-dev

Tốc độ hiện đo được của httm_1: batch 4 mất 57.1 s → **14.27 s/ứng viên**.

```
500 câu × 64 ứng viên × 14.27 s = 124 giờ
```

Không khả thi trong bất kỳ session nào. Ngay cả sau khi áp các sửa latency của
`WORK_ORDER_SAMPLING.md` (stop string `</answer>`, batch 4→32, bỏ prefill lặp do
`num_return_sequences`), mục tiêu thực tế là **0.5–1.0 s/ứng viên**:

| s/ứng viên đo được ở rung 1 | 500 câu × k_max 32 | Quyết định |
|---|---|---|
| ≤ 0.5 s | 2.2 h | ✅ Chạy đủ 500 × 32 |
| 0.5 – 1.0 s | 2.2 – 4.4 h | ✅ Chạy đủ, 1 session |
| 1.0 – 2.0 s | 4.4 – 8.9 h | ⚠️ Giảm còn 250 câu HOẶC dùng 2×T4 chia shard |
| > 2.0 s | > 8.9 h | ❌ Dừng. Quay lại `WORK_ORDER_SAMPLING.md` trước |

**Cổng bắt buộc:** đo s/ứng viên ở rung 1 (§5) *trước khi* cam kết collect toàn tập.
Không được đoán.

### 2.3. Vì sao k_max = 32 chứ không phải 16

Hướng B cần **dư địa phân bổ**: muốn so sánh các bộ phân bổ ở ngân sách *trung bình* 8–16
ứng viên/câu, phải có sẵn tới 32 để có thể dồn cho câu khó.
Nếu `k_max = 16` thì "phân bổ trung bình 16" = "cố định 16" và không còn gì để so.

### 2.4. Chặn BIRD: `run_sweep.py:74`

```python
database = args.db        # ← bỏ qua item["db_path"], mọi câu đều chạy trên 1 DB
```

BIRD có 11 DB khác nhau → mọi câu ngoài DB mặc định sẽ sai hết.
Bản sửa đúng đã có ở `evaluation/evaluate.py:39`:

```python
database = (dataset.parent / item["db_path"]).resolve() if item.get("db_path") else db_path
```

Sửa chỗ này là điều kiện tiên quyết của T1.

---

## §3. HƯỚNG A — Re-ranker đóng Gap A (mở rộng `selection_study.py`, không viết lại)

Thứ tự thực hiện bắt buộc. Mỗi bước có cổng nghiệm thu.

### A0. Tái sinh pool (bắt buộc đầu tiên — không có đường tắt)

```bash
python -m evaluation.run_sweep collect \
  --dataset data/evaluation_questions.json \
  --pool 32 --batch-size 8 \
  --output results/pool_t0_k32
```

**Cổng A0:** `results/pool_t0_k32/pool.jsonl` tồn tại, 100 dòng, mỗi dòng có đúng 32 phần tử trong `pool`.

> `collect` có resume theo id (`run_sweep.py:59-62`) và tự tắt correction (dòng 63) — chạy lại an toàn.

### A1. Thêm phân rã ba khoảng vào `evaluation/metrics.py`

Thêm (không sửa gì đang có):

```python
def pass_at_k(rows, k):              # any(correct) trong k ứng viên đầu
def sc_at_k(rows, k):                # nhóm phiếu lớn nhất có đúng không
def top_n_coverage_at_k(rows, k, n): # đúng nằm trong n nhóm phiếu lớn nhất
def gap_decomposition(rows, k):      # trả về {pass, top3, top2, sc, gap_c, gap_a, gap_b}
```

Quy ước nhóm phải **giống hệt** `core/result_grouper.py`: khoá nhóm là
`execution.normalized_result`, **không bao giờ là văn bản SQL**. Nếu lệch quy ước thì
`sc_at_k` sẽ không khớp EX thật và cả bảng vô nghĩa.

**Cổng A1:** với `k = 16`, `sc_at_k` tính từ pool phải khớp `execution_accuracy` của
`fixed_k16` trong `sweep.csv` tới ±0.01. Nếu lệch hơn → quy ước nhóm sai, dừng và sửa.

### A2. Ghi `n_rows` vào pool để khỏi phải thực thi lại

`run_sweep._record` (dòng 40-52) không lưu số dòng kết quả, mà `core/types.ExecutionResult`
**có sẵn** `row_count`. Vì thiếu nó, `selection_study.enrich` phải **thực thi lại mọi SQL khác nhau**
chỉ để lấy `rows`/`columns`.

Trên toy (100 câu) việc đó rẻ. Trên mini-dev thì không:

```
~16 000 ứng viên, giả sử 6 000 SQL khác nhau, timeout 3 s (config.yaml)
→ xấu nhất 5 giờ chỉ để chạy lại checks
```

Sửa:
- `_record` += `"n_rows": execution.row_count if execution and execution.success else -1`
- `_record` += `"n_cols": len(execution.columns) if ... else 0`
- `_candidates` (dòng 106-112) hiện **bỏ rơi** `row_count`/`columns` khi dựng lại `ExecutionResult` → truyền qua.
- `selection_study.enrich`: nếu record đã có `n_rows` thì bỏ qua thực thi lại, chỉ chạy `check_sql`
  (phần `check_result` chỉ cần `rows` rỗng/không → suy ra từ `n_rows == 0`).

**Cổng A2:** `enrich` trên pool mới ghi `check_overhead_ms_per_candidate` thấp hơn ít nhất 10× so với
pool chưa có `n_rows`.

### A3. Sửa fold của re-ranker: GroupKFold theo `db_id` (BẮT BUỘC trước mọi lần chạy BIRD)

`selection_study.train_rankers` hiện chia fold theo **chỉ số câu đã xáo**:

```python
fold_of = {q: i % folds for i, q in enumerate(order)}
```

Trên toy 1 DB thì vô hại. Trên BIRD **đây là rò rỉ**: các câu cùng `db_id` chia sẻ schema,
tên bảng, tên cột, phân bố giá trị. Ranker huấn luyện trên câu khác *cùng DB* rồi test trên câu này
sẽ học đặc trưng riêng của schema đó → điểm bị thổi phồng, không tổng quát sang DB mới.

Sửa: fold theo `db_id`, mọi câu cùng DB vào cùng một fold.
Với 11 DB → `folds = min(folds, n_db)`, nên dùng **5 fold** thay vì 10.

**Cổng A3:** `meta.json` ghi `fold_db_ids` — danh sách `db_id` trong từng fold, giao nhau phải rỗng.

### A4. Bổ sung đặc trưng cho `_group_features`

Đặc trưng hiện có (9): `vote, log_size, degenerate, ungrounded, bad_join, duplicates, log_rows, columns, sql_length`.

Thêm 5 cái sau, tất cả đều **không cần gold** và đã có sẵn dữ liệu:

| Đặc trưng | Nguồn | Lý do |
|---|---|---|
| `is_empty` = `n_rows == 0` | A2 | `log_rows = log1p(0) = 0` buộc model học phi tuyến từ một đặc trưng tuyến-tính-theo-log. Cờ nhị phân tường minh mạnh hơn và rẻ. |
| `schema_score` | `_record` **đã lưu** nhưng ranker **không dùng** | Đang bỏ không một tín hiệu miễn phí |
| `warn_count` | `_record["warnings"]` — cũng đang bị bỏ | Như trên |
| `n_groups` (số nhóm kết quả khác nhau của câu) | `_group_features` | Tín hiệu *độ khó cấp câu*; cũng là input chính cho Hướng B |
| `group_entropy` | phân bố kích thước nhóm | Phân biệt "4-4-4-4" với "13-1-1-1" ở cùng `vote` |

**Cổng A4:** `check_diagnostics` cho thấy `is_empty` có `precision_wrong > 0.8`
(tức là khi cờ cháy thì ứng viên hầu như luôn sai). Nếu không đạt thì giả thuyết
"kết quả rỗng thắng phiếu" sai trên dataset này và phải báo cáo đúng như vậy.

> Căn cứ cấu trúc cho giả thuyết này: `core/result_grouper.py` khoá nhóm bằng
> `normalized_result`, nên **mọi** kết quả 0 dòng gộp vào *một* nhóm duy nhất
> → nhóm sai đó có thể to nhất và thắng phiếu. Đồng thời `core/candidate_ranker.py`
> vẫn cộng đủ `weights["execution"] * 1` cho kết quả rỗng vì nó "thực thi thành công".

### A5. Giải phóng 10% ngân sách trọng số đang nằm chết

`core/types.py:44` → `generation_score: float = 0.5`, **không chỗ nào cập nhật**.
`config.yaml` cấp cho nó `ranking.generation = 0.10`.

Vì mọi ứng viên đều có đúng 0.5, số hạng `0.10 × 0.5 = +0.05` là **hằng số cộng vào mọi điểm**
→ nó **vô hại (inert), không phải nhiễu**: không thể làm đổi thứ tự xếp hạng.

Nhưng nó là **10% dung lượng xếp hạng chưa dùng**. Hai cách nạp tín hiệu thật, theo thứ tự ưu tiên:

1. **Điểm re-ranker đã học** (khuyến nghị): sau khi A3+A4 xong, đưa `LogisticRanker.score`
   đã chuẩn hoá vào `generation_score`. Như vậy `core/candidate_ranker.py` không cần đổi cấu trúc.
2. **Log-probability trung bình trên token** — HF `generate` trả được qua
   `output_scores=True` / `return_dict_in_generate=True`. Mạnh hơn nhưng phải sửa `core/hf_model.py`
   và làm tăng bộ nhớ; để sau.

**Cổng A5:** `python -m pytest` vẫn xanh (66 test) và `fixed_k16` EX không *giảm*.

### A6. Đưa selector thắng vào `core/` — deliverable thật của Hướng A

Đây là bước biến "nghiên cứu offline" thành "EX của hệ thống".

1. `config.yaml`, thêm khối mới:
   ```yaml
   selection:
     cav: true                  # bật vote hạn chế theo cờ
     checks: [degenerate, ungrounded, bad_join]
     ranker_weights: null       # đường dẫn tới weights đã học; null = chỉ dùng CAV
   ```
2. `core/pipeline.py`: khởi tạo `ConsistencyChecker(db)` một lần, lọc ứng viên bằng `_clean`
   **trước** `group_candidates` ở dòng 109.
3. `core/candidate_ranker.py`: cộng `- weights["degenerate_penalty"] * is_empty`
   để kết quả rỗng không còn ăn trọn `weights["execution"]`.
4. Giữ `cav: false` làm đường chạy baseline để so sánh A/B được.

**Cổng A6:** `python -m evaluation.evaluate` trên T0 cho EX ≥ EX của `cav: false`,
và `selection_study` cho `CAV (3 checks)` thắng `vote (baseline)` với `mcnemar_p < 0.05`.
**Nếu McNemar không có ý nghĩa thống kê thì không được merge** — báo cáo kết quả âm, đúng như nó là.

---

## §4. HƯỚNG B — Phân bổ ngân sách liên câu (phần mới hoàn toàn)

### 4.1. Phát biểu vấn đề

`fixed_k16` cấp **16 ứng viên cho mọi câu**. Nhưng:
- Câu dễ hội tụ ở k=2 (`sweep.csv`: `acc_easy` đã 0.808 ngay ở greedy) → 14 ứng viên bị đốt vô ích.
- Câu khó ở k=16 vẫn chỉ `acc_hard = 0.086`, trong khi `oracle_pass@16` của nhóm khó là **0.714**
  → chúng cần nhiều hơn 16, và dư địa thật sự là ở đây.

Với tổng ngân sách cố định `C = 16 × N` ứng viên, **phân bổ lại** có nâng được EX không?

Adaptive Consistency chỉ *dừng sớm* — tiết kiệm được thì bỏ đi. Hướng B **tiêu lại phần tiết kiệm đó**
vào câu khó. Đó là khác biệt so với prior art.

### 4.2. Thiết kế (replay offline, chi phí GPU = 0 sau khi collect)

File mới: `evaluation/budget_allocator.py`

**Pha 1 — thăm dò.** Cấp `k₀ = 4` ứng viên cho mọi câu. Chi phí `4N`.

**Pha 2 — tính điểm bất định `u(q)` chỉ từ pha 1** (tất cả đều gold-free):

| Tín hiệu | Nguồn |
|---|---|
| `confidence` = tỉ lệ nhóm lớn nhất | `core/adaptive_sampler.decide` đã tính |
| `margin` = (top1 − top2)/valid | như trên |
| `n_groups`, `group_entropy` | A4 |
| `valid_rate` = valid/k₀ | pool |
| `flag_rate` = cờ trung bình trên ứng viên pha 1 | `enrich` |
| **điểm re-ranker của nhóm dẫn đầu** | **Hướng A** ← đây là điểm nối A→B |

**Pha 3 — phân bổ phần còn lại** `R = C − 4N`. Bốn bộ phân bổ để so sánh:

| Tên | Cơ chế | Vai trò |
|---|---|---|
| `uniform` | chia đều → tương đương `fixed_k16` | **baseline** |
| `threshold` | dừng sớm trong câu nếu `confidence ≥ τ`, **không** tái phân bổ | = adaptive hiện có; EX ngang baseline, chi phí thấp hơn |
| `greedy_uncertainty` | xếp câu theo `u(q)` giảm dần, cấp thêm +4 lần lượt tới khi hết `R` | **Hướng B thật sự** |
| `learned_gain` | model nhỏ dự đoán `P(vote đổi sang đúng \| +Δ)` từ đặc trưng pha 1, phân bổ theo lợi ích kỳ vọng | mở rộng, dùng `GroupKFold` theo `db_id` như A3 |
| `oracle_alloc` | cấp đủ `k_max` đúng cho các câu mà việc đó làm đổi kết quả | **trần trên** của B |

**Ràng buộc trung thực của replay:** chỉ phân bổ được tới `k_max` đã thực sự sinh ra.
Vì thế phải `collect` ở `k_max = 32` rồi replay các bộ phân bổ ở ngân sách **trung bình 8 / 12 / 16**.
Không bao giờ được "phân bổ" nhiều hơn số ứng viên có trong pool.

### 4.3. Chỉ số của B

Chỉ số duy nhất có ý nghĩa: **EX tại cùng số ứng viên trung bình**.

```
EX vs average_candidates  →  đường Pareto
```

Báo cáo cho từng bộ phân bổ, tại từng ngân sách trung bình:
`execution_accuracy`, `ci95_low/high`, `average_candidates` (phải khớp nhau giữa các bộ ±0.3),
`delta_vs_uniform` + CI bootstrap ghép đôi, `mcnemar_p`, `oracle_alloc_gap_closed`.

**So sánh ở ngân sách lệch nhau là vô nghĩa** — nếu `greedy_uncertainty` dùng trung bình 17.2
ứng viên mà `uniform` dùng 16.0 thì không được kết luận gì. Phải hiệu chỉnh để khớp.

### 4.4. Sửa `elapsed_seconds = 0`

- `run_sweep.py:139` và `selection_study.py:164`: truyền thời gian thật.
- Trong replay offline không có thời gian thật → dùng `_batch_cost` tích luỹ làm proxy
  và **ghi rõ trong `meta.json`** rằng đó là proxy, không phải đồng hồ.
  Nếu không muốn proxy thì bỏ hẳn nhánh `time_budget` trong replay và nói rõ, chứ không để `0` âm thầm.

### 4.5. `_batch_cost` không dùng được làm trục chi phí của Pareto

```python
def _batch_cost(row, count):                      # run_sweep.py:122-125
    per_batch = mean(row["batch_ms"])
    return ceil(count / row["batch_size"]) * per_batch
```

Nó tính tiền **cả một batch** cho dù chỉ dùng 1 ứng viên. Với `batch_size = 32` và câu chỉ cần 4
ứng viên, nó báo chi phí gấp 8 lần thực tế → `threshold` và `greedy_uncertainty` bị phạt oan.

→ Trục chi phí chính của mọi hình Pareto phải là **`average_candidates`** (độc lập phần cứng).
Latency chỉ là trục phụ, và phải **đo thật** (§5), không ngoại suy.

### 4.6. File cần tạo

| File | Nội dung |
|---|---|
| `evaluation/budget_allocator.py` | 5 bộ phân bổ + hàm tính `u(q)` |
| `evaluation/run_allocation.py` | replay trên `pool.jsonl`, xuất `allocation.csv` + `per_question.csv` |
| `evaluation/gap_report.py` | bảng ba khoảng + hình Pareto (theo khuôn `selection_report.py`) |

---

## §5. KẾ HOẠCH BENCH SO SÁNH CHỈ SỐ

### 5.1. Hai loại bench, không được trộn

| | **Bench sinh** (latency) | **Bench chọn** (chất lượng) |
|---|---|---|
| Hỏi gì | Bao nhiêu giây một ứng viên? | Bao nhiêu điểm EX ở ngân sách này? |
| Cần GPU | Có | **Không** — replay trên pool |
| Cần gold | Không | Có |
| Công cụ | `evaluation/bench_generation.py` (tạo mới) | `selection_study.py` + `run_allocation.py` |
| Khi nào | Trước collect, để chốt cấu hình | Sau collect, chạy bao nhiêu lần cũng được |

### 5.2. Bench sinh — port kỷ luật của `research/slm_sql/bench.py` sang HF

File nguồn tham chiếu: `/home/mint/Documents/HTTM/SLM-PRJ/research/slm_sql/bench.py` (248 dòng).
Sáu biện pháp bảo vệ sau **phải giữ nguyên**, vì chúng là lý do bench đáng tin:

1. **Cô lập thư mục kết quả** (`_output_directory`): tên thư mục phải chứa `_bench`, phải nằm ngoài
   thư mục kết quả thật, phải mới hoặc rỗng. Ngăn số liệu bench lẫn vào số liệu công bố.
2. **Đóng băng chữ ký prompt**: hash token-id của prompt; nếu biến thể sau có token khác
   → `ValueError`. Nếu không có cái này, "tăng tốc 2×" có thể chỉ là prompt bị ngắn đi.
3. **Assert cứng `pct_extract_empty == 0`**: nếu stop string phá `<answer>...</answer>` thì
   `core/sql_extractor.py:10` trả rỗng và bench "nhanh" chỉ vì nó không sinh gì. Phải fail ngay.
   → Nhắc lại: thêm stop string **bắt buộc** kèm `include_stop_str_in_output = True`.
4. **`pct_finish_length`**: tỉ lệ completion bị cắt vì `max_new_tokens`. Nếu nó cao thì `512`
   đang cắt câu SQL → EX tụt mà không rõ nguyên nhân.
5. **Rung gate**: rung 1 = 5 câu × k4; rung 2 = 20 câu / ≥10 DB / k16. Không cho nhảy rung.
6. **Warmup và init tính riêng**: `engine_init_s`, `warmup_s` báo tách; `wall_clock_s` chỉ là thời gian sinh.

Điều chỉnh cho HF (khác vLLM):
- Không có `enable_prefix_caching` / `enforce_eager` → biến thể thay bằng:
  `baseline` → `+stop` → `+stop+batch16` → `+stop+batch32` → `+stop+sorted_by_db`
- Thêm `peak_vram_gb` qua `torch.cuda.max_memory_allocated()` — T4 chỉ 16 GB, batch 32 có thể OOM.
  `core/hf_model.py:143-150` có handler tự giảm batch; nếu nó cháy thì batch báo cáo **không phải**
  batch đã chạy → phải ghi `batch_size_effective` riêng.
- Thêm `s_per_candidate` = `wall_clock_s / n_completions` — đây là con số ra quyết định ở §2.2.

### 5.3. Bảng so sánh chính (deliverable của bench chọn)

Một bảng, mọi dòng chạy trên **cùng một `pool.jsonl`**:

| method | avg_cand | EX | CI95 | pass@k | top3@k | top2@k | SC@k | gap_C | gap_A | gap_B | Δ vs base | McNemar p | gap_closed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 1 | | | | | | | | | | — | — | — |
| fixed k=1,2,4,8,16,32 | | | | | | | | | | | | | |
| vote (baseline) @16 | 16 | | | | | | | | | | **base** | — | — |
| CAV-3 @16 | 16 | | | | | | | | | | | | |
| CAV without degenerate | 16 | | | | | | | | | | | | |
| CAV only degenerate | 16 | | | | | | | | | | | | |
| learned re-ranker (CV) | 16 | | | | | | | | | | | | |
| CAV + re-ranker | 16 | | | | | | | | | | | | |
| **alloc: uniform** | 16 | | | | | | | | | | **base** | — | — |
| **alloc: threshold τ** | <16 | | | | | | | | | | | | |
| **alloc: greedy_uncertainty** | 16 | | | | | | | | | | | | |
| **alloc: learned_gain** | 16 | | | | | | | | | | | | |
| oracle_pass@k_max | 32 | | | | | | | | | | — | — | **trần** |
| oracle_alloc | 16 | | | | | | | | | | — | — | **trần của B** |

Mỗi dòng kèm `acc_simple / acc_moderate / acc_challenging` và `acc_half_odd / acc_half_even`
(split-half: nếu hai nửa lệch nhau nhiều hơn độ rộng CI thì kết quả không ổn định, đừng tin).

### 5.4. Hai hình

1. **Pareto**: `EX` (trục y) vs `average_candidates` (trục x). Mỗi họ phương pháp một đường.
   Đường nào nằm trên-trái thì thắng. Đây là hình chính của bài.
2. **Thác ba khoảng**: cột xếp tầng `SC@k → +Gap B → +Gap A → +Gap C → 1.0`, cho `k = 1..32`.
   Cho thấy trực quan phần nào đã đóng và phần nào còn lại.

Khuôn mã: theo `evaluation/selection_report.py` (đã có `matplotlib.use("Agg")`,
bảng màu `COLORS`, `CHECK_NAMES` tiếng Việt).

### 5.5. Chuỗi lệnh đầy đủ

```bash
# ── Tầng T0: khói, local hoặc T4, ~30 phút ─────────────────────────
python -m pytest -q                                    # cổng: 66 test xanh
python -m evaluation.bench_generation --rung 1 --questions 5 --k 4 \
       --output results/t0_bench_r1                    # cổng: s_per_candidate
python -m evaluation.run_sweep collect --pool 32 --batch-size 8 \
       --output results/pool_t0_k32
python -m evaluation.run_sweep simulate --output results/pool_t0_k32
python -m evaluation.selection_study results/pool_t0_k32 --folds 5
python -m evaluation.run_allocation results/pool_t0_k32
python -m evaluation.gap_report results/pool_t0_k32

# ── Tầng T1: BIRD mini-dev, 1 session Kaggle ───────────────────────
python scripts/import_bird.py \
       --questions <mini_dev_sqlite.json> \
       --database-root <dev_databases> \
       --output data/benchmarks/bird_eval.json
python -m evaluation.bench_generation --rung 2 --questions 20 --min-dbs 10 --k 16 \
       --output results/bird_bench_r2                  # CỔNG QUYẾT ĐỊNH §2.2
python -m evaluation.run_sweep collect \
       --dataset data/benchmarks/bird_eval.json \
       --pool 32 --batch-size 32 --sort-by-db \
       --output results/pool_bird_k32
python -m evaluation.run_sweep simulate --output results/pool_bird_k32
python -m evaluation.selection_study results/pool_bird_k32 --folds 5
python -m evaluation.run_allocation results/pool_bird_k32
python -m evaluation.gap_report results/pool_bird_k32
```

---

## §6. LỊCH CHẠY MỘT SESSION KAGGLE (12 h, quota 30 h/tuần)

| Giờ | Việc | Cổng |
|---|---|---|
| 0.0 – 0.3 | Cài đặt, tải model, `pytest` | 66 test xanh |
| 0.3 – 0.6 | `bench_generation` rung 1 (5 câu × k4) | `pct_extract_empty == 0` |
| 0.6 – 1.0 | `bench_generation` rung 2 (20 câu / 10 DB / k16), quét batch 8/16/32 | **chốt `s_per_candidate` → quyết định theo bảng §2.2** |
| 1.0 – 5.5 | `collect` 500 câu × k32 (có resume; checkpoint `pool.jsonl` liên tục) | 500 dòng × 32 |
| 5.5 – 6.0 | `simulate` + `selection_study` + `run_allocation` + `gap_report` (CPU, không GPU) | CSV + PNG đầy đủ |
| 6.0 – 6.5 | Lưu artifact ra Kaggle Dataset / tải về | pool + mọi CSV đã an toàn ngoài session |

Dư ~5.5 h làm đệm. Nếu rung 2 báo > 2.0 s/ứng viên thì **đừng vào bước collect** — dừng session,
quay lại `WORK_ORDER_SAMPLING.md`, đừng đốt quota.

Lưu ý hạ tầng:
- **2×T4: chia shard câu hỏi (data parallel), KHÔNG tensor parallel.** Model 1.5B nằm thừa trong
  16 GB; chia tensor chỉ thêm chi phí giao tiếp. Thêm `--shard i/n` và gộp `pool.jsonl` sau.
- Upload BIRD thành **Kaggle Dataset bền vững**, đừng tải lại mỗi session.
- `pool.jsonl` lưu nguyên chuỗi `normalized_result` → trên BIRD sẽ phình. Cân nhắc lưu hash
  và giữ bảng tra riêng, nếu dung lượng session thành vấn đề.

---

## §7. BẪY — đọc trước khi chạy, không phải sau khi lỗi

| # | Bẫy | Hậu quả | Phòng |
|---|---|---|---|
| 1 | `pool.jsonl` đã mất | Không bước nào của A/B chạy được | A0 trước mọi thứ |
| 2 | `run_sweep.py:74` `database = args.db` | Mọi câu BIRD ngoài DB mặc định sai hết, EX sụp mà không báo lỗi | Theo `evaluate.py:39` |
| 3 | `hf_model.py:116-118` guard context | Schema BIRD dài → trả **một** Candidate lỗi → **EX = 0 âm thầm** | Bật `validation.schema_selection`, giảm `schema.sample_rows`, hoặc nâng `max_input_tokens`. Phải **đếm và báo** số câu bị chặn. |
| 4 | Fold theo chỉ số câu (A3) | Điểm re-ranker thổi phồng trên BIRD | GroupKFold theo `db_id` |
| 5 | `enrich` thực thi lại mọi SQL, timeout 3 s | Xấu nhất ~5 h CPU trên mini-dev | A2: ghi `n_rows` lúc collect |
| 6 | `_batch_cost` tính cả batch cho ứng viên lẻ | Phạt oan mọi phương pháp tiết kiệm | Trục Pareto dùng `average_candidates` |
| 7 | `elapsed_seconds = 0` ở 2 chỗ | Nhánh `time_budget` không bao giờ cháy, kết quả mô phỏng khác thực tế | §4.4 |
| 8 | So sánh ở ngân sách lệch nhau | Kết luận sai hoàn toàn | Khớp `average_candidates` ±0.3 |
| 9 | Stop string phá `<answer>` | `extract_sql` rỗng, bench "nhanh" giả | `include_stop_str_in_output = True` + assert `pct_extract_empty == 0` |
| 10 | `numpy` chưa khai báo | `selection_study.py:19` import nó mà `requirements.txt` không có (chỉ vào gián tiếp qua pandas) | Thêm `numpy` tường minh. **Không** thêm sklearn/lightgbm/scipy — `LogisticRanker` numpy thuần đã chạy tốt. |
| 11 | `oracle_pass@16` hardcode nhãn (`run_sweep.py:156`) | Với `--pool 32` nhãn nói 16 mà số là 32 | `f"oracle_pass@{pool}"` |
| 12 | `max_candidates` bị ghi đè (`run_sweep.py:159`) | Giá trị trong `config.yaml` vô hiệu, dễ tưởng mình đã cấu hình | Ghi rõ trong `meta.json` |
| 13 | Tinh chỉnh ngưỡng trên EX cuối cùng | Overfit, số liệu không công bố được | Chọn τ và trọng số **chỉ** trên T0/validation. Đóng băng rồi mới chạy T2. |
| 14 | Bench trên T4 rồi chạy trên GPU khác | `batch_size` tối ưu phụ thuộc dung lượng KV-cache (16 vs 24 GB) và backend attention → bench vô giá trị | Bench và chạy trên **cùng** loại GPU; đổi GPU thì bench lại |
| 15 | Đẩy database cá nhân lên cloud | README đã cảnh báo | Chỉ upload BIRD công khai |
| 16 | `enforce` giấy phép | `CscSQL-Merge-*` và `SynSQL-*Think` là **CC-BY-NC-4.0** | Không upload công khai, không dùng thương mại |

---

## §8. CHECKLIST NGHIỆM THU

Hướng A xong khi:
- [ ] `gap_decomposition` chạy được, `sc_at_k` khớp `fixed_k16` ±0.01 (cổng A1)
- [ ] `selection.csv` cho `CAV-3` thắng `vote (baseline)`, `mcnemar_p < 0.05`, CI của delta không chứa 0
- [ ] `meta.json` ghi `fold_db_ids` rời nhau (cổng A3)
- [ ] `check_diagnostics` báo `precision_wrong` của từng cờ
- [ ] `oracle_gap_closed` được báo cáo (bao nhiêu % Gap A+B đã đóng)
- [ ] Selector thắng đã vào `core/pipeline.py`, `evaluate.py` cho EX cao hơn baseline
- [ ] Split-half lệch nhau trong độ rộng CI

Hướng B xong khi:
- [ ] `allocation.csv` có đủ 5 bộ phân bổ ở ≥3 ngân sách trung bình
- [ ] `greedy_uncertainty` thắng `uniform` **ở cùng `average_candidates`**, có McNemar
- [ ] Hình Pareto cho thấy đường của B nằm trên-trái đường `fixed_k`
- [ ] `oracle_alloc` được báo cáo làm trần trên

Nếu một trong hai hướng cho **kết quả âm** (không thắng có ý nghĩa thống kê):
báo cáo đúng như vậy, kèm `oracle_gap_closed` để chỉ ra dư địa còn lại ở đâu.
Phân rã ba khoảng trên BIRD mini-dev tự nó đã là một đóng góp đo lường — nó không phụ thuộc
vào việc A hay B có thắng.

---

## §9. VIỆC CẦN BẠN QUYẾT / LÀM — tôi không tự làm

1. **Tải BIRD mini-dev** (gói SQLite chính thức, gồm *cả* câu hỏi *và* database).
   `scripts/import_bird.py` đã sẵn sàng và đúng, nhưng repo hiện **không có file BIRD nào**.
   Không dùng file dự đoán của tác giả làm gold SQL.
2. **Chạy training / chạy GPU**: tôi không tự chạy. Mọi lệnh ở §5.5 cần bạn thực thi.
3. **Push lên GitHub**: `httm_1/SLM-PRJ` đang khớp `origin/main` tại `362ce9e`, nhưng
   `docs/WORK_ORDER_SAMPLING.md` và tài liệu này **chưa được commit**.
   Kaggle/Colab chỉ pull được từ remote → phải commit + push trước khi chạy cloud.
   Repo nguồn `/home/mint/Documents/HTTM/SLM-PRJ` cũng đang `[ahead 1]` chưa push.
