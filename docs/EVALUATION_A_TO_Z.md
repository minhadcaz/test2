# Đánh giá Schema Pruning từ A–Z: cùng mô hình, full schema và pruned schema

Tài liệu này hướng dẫn đo tác động của pruning lên một mô hình Text-to-SQL cụ thể. Baseline là **mô hình gốc với toàn bộ schema của database đang được hỏi**. Nhánh thử nghiệm dùng **chính trọng số, tokenizer và cấu hình sinh đó**, thay schema đầu vào bằng kết quả truy xuất. Không cần fine-tune để làm thí nghiệm này.

Notebook chạy được theo từng bước: [`notebooks/kaggle_schema_pruning_bird.ipynb`](../notebooks/kaggle_schema_pruning_bird.ipynb). Hướng dẫn dữ liệu và thiết kế pruning: [`SCHEMA_PRUNING_BIRD_KAGGLE.md`](SCHEMA_PRUNING_BIRD_KAGGLE.md).

## A. Câu hỏi thí nghiệm và tiêu chí chấp nhận

Đặt mục tiêu trước khi xem kết quả cuối. Ví dụ: “giảm input token ít nhất 10%, giảm latency đầu cuối, và không giảm EX quá 1 điểm phần trăm”. Các con số này là **tiêu chí bạn chọn**, không phải kết quả đã đạt.

So sánh ba arm nếu đủ tài nguyên:

| Arm | Model | Schema vào prompt | Vai trò |
|---|---|---|---|
| full | Cùng checkpoint | Full schema | Baseline mô hình gốc |
| bm25 | Cùng checkpoint | BM25 + PK/FK/bridge | Phương pháp nhẹ |
| hybrid | Cùng checkpoint | BM25 + embeddings + PK/FK/bridge | Đánh đổi thêm chi phí retrieval |

Bắt đầu với `greedy`, batch 1, correction tắt, sample rows 0. Sau khi biết riêng tác động pruning, làm thí nghiệm khác bật correction/adaptive sampling cho **tất cả** arm. Thay cùng lúc model hoặc sampling sẽ không còn cô lập được tác động pruning.

## B. Đo những metric nào?

### B1. Truy xuất có giữ đúng schema không?

Với câu hỏi i, gọi `G_i` là tập tham chiếu bảng/cột từ gold SQL và `S_i` là tập được giữ lại:

- `Recall_i = |G_i ∩ S_i| / |G_i|`. Tính riêng cho bảng và cột.
- Macro recall: trung bình recall trên các câu có nhãn hợp lệ.
- Micro recall: tổng số tham chiếu giữ đúng / tổng số tham chiếu gold.
- All-reference coverage: tỷ lệ câu giữ **đủ tất cả bảng và cột** mà gold SQL tham chiếu.
- Prompt token reduction: `1 − sum(tokens_pruned) / sum(tokens_full)` trên cùng tập câu; nhân 100 để ra %.
- Retrieval latency mean/p50/p95, fallback count, số lần các cột bắt buộc vượt soft budget.

Nhãn ở đây được suy ra từ gold SQL, không phải annotation schema linking thủ công của BIRD. Gold chỉ vào evaluator **sau retrieval**; không dùng gold để chọn schema. Coverage không tính cột dư là lỗi, vì vậy phải đọc cùng token reduction. Có thể giữ mọi cột để đạt coverage 100% nhưng không tiết kiệm gì.

Coverage thấp là tín hiệu đáng lo: model có thể mất cột JOIN, lọc hoặc aggregate. Coverage 98% **không có nghĩa EX 98%**; model vẫn có thể viết SQL sai. Gold dùng một cách giải nên một SQL tương đương cũng có thể dùng tham chiếu khác.

### B2. SQL của mô hình có đúng không?

| Metric | Công thức/ý nghĩa | Khi báo cáo |
|---|---|---|
| Execution accuracy (EX) | Số câu kết quả SQL khớp gold / toàn bộ N câu | Metric chất lượng chính |
| ΔEX (pp) | `100 × (EX_pruned − EX_full)` nếu EX ở thang 0–1 | Điểm phần trăm, không phải % tương đối |
| Gained | Full sai, pruning đúng | Đếm từng cặp cùng câu |
| Lost | Full đúng, pruning sai | Tổn thất do thay schema/cách sinh |
| Valid SQL rate | SQL đáp ứng điều kiện valid của evaluator | SQL chạy được chưa chắc đúng |
| Exact match | SQL chuẩn hóa giống chuỗi gold | Metric phụ; SQL tương đương có thể khác chuỗi |
| Error breakdown | wrong_result, syntax/validation, timeout, overflow... | Chẩn đoán nguyên nhân |

**Giữ tất cả thất bại trong mẫu số N**: không có prediction, SQL lỗi, timeout, overflow đều không được loại để tăng điểm. Nếu gold SQL không chạy được, cần giải quyết dataset/DB/giới hạn execution và chạy lại cả hai arm. Không gán lỗi gold cho model.

Dự án có `execution_metric=project_execution_match_not_official_BIRD_scorer`: comparator nội bộ giữ duplicate và xử lý ORDER BY/LIMIT. Scorer Mini-Dev chính thức có quy tắc khác; phải ghi rõ hai loại điểm và dùng scorer chính thức để công bố BIRD EX. Không thay tên điểm nội bộ thành BIRD EX.

### B3. Tốc độ, token và bộ nhớ

| Trường/metric | Đơn vị/phạm vi | Cách hiểu |
|---|---|---|
| `input_tokens_first` trong comparison | Token sau chat template, call đầu của mỗi câu | So sánh độ dài prompt thực sự vào model |
| `input_tokens_call_sum` | Tổng input token trên các backend call của câu | Bao gồm call sửa lỗi/lặp nếu dùng chung backend; không nhân batch/sequence, không phải tổng FLOPs |
| `output_tokens_total` | Tổng token sinh của các sequence trong các call | Có EOS, bỏ padding sau EOS; không chỉ token của SQL đã parse |
| `generation_ms` | Tổng batch generation thành công trong câu | Có CUDA synchronize; không gồm tokenize/decode hoặc thời gian batch OOM thất bại |
| `elapsed_ms` | Thời gian đầu cuối mỗi câu | Schema inspection + retrieval + prompt + generation + validate/execute/rank/correction |
| Mean/p50/p95 latency | ms | Mean cho chi phí trung bình, p95 để xem đuôi chậm |
| End-to-end speedup | `sum(T_full) / sum(T_pruned)` trên các cặp có đủ telemetry | >1 là pruning nhanh hơn; đã gồm retrieval |
| Peak CUDA allocated | MiB, đo từng GPU/từng câu | Bao gồm model đang nằm trên GPU; không phải NVML hay peak reserved |

`elapsed_ms` không gồm load model hoặc thực thi gold; evaluator chạy gold trước request nên có thể làm ấm SQLite/OS cache. Notebook tái sử dụng backend, vì vậy `model_load_ms` trong các arm không đại diện chi phí load model ban đầu.

Đồ thị VRAM của script dùng **peak lớn nhất trên một device** trong mỗi câu. Với nhiều GPU, xem thêm mapping `cuda_peak_allocated_mb_by_device` trong prediction; không cộng peak từng GPU rồi gọi là peak đồng thời. Metric này không bao quát bộ nhớ ngoài PyTorch allocator.

Pruning chủ yếu giảm công việc xử lý input và nhu cầu KV cache. Trọng số mô hình vẫn chiếm bộ nhớ; giảm 20% token không suy ra giảm 20% VRAM. Số output token cũng không được bảo đảm giảm. Nếu model output dài hơn hoặc retrieval chậm, end-to-end có thể tăng.

Hiện chưa đo TTFT, prefill/decode tách riêng hoặc inter-token latency. Không đổi tên `generation_ms` thành TTFT. Nếu có correction backend riêng, telemetry hiện chỉ lấy backend chính; dùng cấu hình correction tắt ở thí nghiệm đầu.

## C. Chuẩn bị source và dữ liệu trên Kaggle

1. Tạo Notebook, chọn GPU có sẵn; bật Internet nếu cần tải dependency/model. Phiên offline cần source, wheels, model và dataset đã gắn sẵn.
2. Thêm source repository dưới dạng Kaggle Dataset. Mở notebook trong `notebooks/` và sửa `REPO_INPUT` tới thư mục chứa `core/pipeline.py`.
3. Chạy cell copy source và install. Giữ PyTorch GPU của Kaggle; requirements của dự án đã có matplotlib và pandas.
4. Chạy cell tải metadata: script ghim phiên bản và kiểm SHA-256 của 500 câu Mini-Dev cùng 11 schema. Tên release “Mini-Dev 1024” không phải số câu.
5. Để sinh SQL, gắn đúng SQLite package: `<DB_ROOT>/<db_id>/<db_id>.sqlite`, rồi sửa `DB_ROOT`. Metadata không bao gồm database; không dùng tùy tiện archive cũ.
6. Nếu dùng model mặc định, giữ `MODEL_ID` và `MODEL_REVISION` đã ghim. Nếu thay model, thay revision theo model đó. Notebook tải snapshot cụ thể cho generation và ghi hash từng tệp.

Lệnh tương đương khi đã ở thư mục dự án:

```bash
python -m pip install -r requirements.txt -r requirements-model.txt
python scripts/download_bird_minidev.py --output data/bird_minidev
python scripts/import_bird.py \
  --questions data/bird_minidev/mini_dev_sqlite.jsonl \
  --database-root /kaggle/input/bird-sqlite/dev_databases \
  --output data/benchmarks/bird_eval.json
```

Kiểm tra import thành công không thay thế kiểm tra gold SQL trên SQLite. Evaluator sẽ dừng nếu gold lỗi; xử lý trước full run. Với BIRD dev, dùng `dev.json` và đúng DB tương ứng, `dev_tables.json` cho bước retrieval. Không trộn câu hỏi, schema và SQLite thuộc phiên bản khác nhau.

## D. Đánh giá retrieval trước, không cần GPU

Chạy các cell “Baseline và BM25” rồi “Đếm token thật”. Lượt đầu dùng proxy regex chỉ để kiểm pipeline. Muốn báo số token thực, dùng tokenizer của đúng model. Notebook ghi tokenizer tag gồm model ID, revision và fingerprint vocab/special tokens/backend vào report để tránh so nhầm tokenizer. Token retrieval là **raw prompt trước chat template**; telemetry generation là token thực sau template, nên hai con số không bắt buộc bằng nhau.

Để thử hybrid, bật `RUN_SEMANTIC=True`; encoder được ghim revision, mặc định chạy CPU để tránh tranh VRAM với model sinh SQL. Không cần dựng một vector DB service cho 11 DB nhỏ.

Tệp `results/retrieval*/<arm>/summary.json` có macro/micro recall, coverage và token reduction. `details.jsonl` có từng câu, các bảng/cột thiếu và lỗi gán nhãn. Nếu `label_errors > 0`, giải quyết nhãn trước khi kết luận.

Ngân sách 8 bảng/48 cột là điểm bắt đầu có recall tốt trong exploratory run đã lưu, không phải optimum đã được chứng minh. Cột khóa/cột bắt buộc và bảng bridge có thể làm vượt ngân sách; đó là thiết kế soft budget.

## E. Chạy baseline và pruning công bằng

Ở cell generation của notebook:

```python
RUN_GENERATION = True       # Đặt ở cell trước khi import dataset
LIMIT = 10                  # Smoke test; sau đó None
REPEATS = 1                 # Sau smoke: 3 nếu đủ thời gian
```

Mở lại cell generation sau khi đổi biến; cell có gán `LIMIT`/`REPEATS` bên trong nên sửa ngay các dòng đó. Chạy smoke để kiểm DB, tokenizer, GPU, output và biểu đồ. Sau đó chạy lại với `LIMIT=None` để đánh giá toàn bộ 500 câu; giữ thư mục smoke riêng nếu cần so lịch sử, vì chạy lại cùng đường dẫn sẽ ghi đè artifact.

Notebook tự thực hiện:

1. Dùng cùng model snapshot cho mọi arm; greedy, batch 1, correction off, sample rows 0, cùng seed và execution limits.
2. Tạo `bird_eval_subset.json` đúng subset được chấm. Full và pruning nhận đúng cùng câu/evidence/DB.
3. Hash SQLite, subset và model files; lưu GPU, dtype/quantization thực tế, dependency vào `runtime_manifest.json`.
4. Warm-up GPU mỗi arm, loại warm-up khỏi báo cáo. Evaluator tạo selector mới nên chi phí khởi tạo theo DB vẫn nằm trong đo; cache embedding disk có thể được dùng lại.
5. Xoay thứ tự arm giữa các repeat để giảm ảnh hưởng chạy full trước/pruning sau. Cùng seed qua repeat giúp quan sát runtime; không biến 500 câu thành 1.500 quan sát độc lập.
6. Gọi evaluator với `profile_cuda=True` và lưu `predictions.jsonl`, `summary.json`, `summary.csv`, `config.json`.

Output mặc định:

```text
results/generation/repeat_00/full/
results/generation/repeat_00/bm25/
results/generation/repeat_00/hybrid/   # nếu bật semantic
results/generation/repeat_01/...      # nếu REPEATS > 1
```

Dùng cùng GPU/session càng tốt. Không vừa benchmark vừa chạy workload GPU khác. Hash và warm-up ở ngoài timed request. So sánh latency theo từng repeat, báo mean/spread giữa repeat; đừng bootstrap trên các bản lặp của cùng câu như các câu độc lập. Chi phí xây index cold và các request dùng cache nên được báo riêng nếu nghiên cứu production; các timing mặc định là mixture first access + cache theo chính sách ghi trong manifest.

## F. Sinh bảng so sánh và biểu đồ tự động

Cell plotting trong notebook đã gọi script dưới. Nếu muốn chạy lại:

```bash
python -m evaluation.compare_schema_runs \
  --run full=results/generation/repeat_00/full \
  --run bm25=results/generation/repeat_00/bm25 \
  --output results/plots/repeat_00 \
  --bootstrap-samples 2000 --bootstrap-unit question --seed 42 --pdf
```

Nếu có hybrid, thêm `--run hybrid=results/generation/repeat_00/hybrid` trước `--output`. Luôn đặt full đầu tiên. Script kiểm số prediction bằng `summary.questions` để phát hiện file bị cắt; từ chối khác tập `(id, db_id)`, difficulty, dataset hash, seed/backend/mode, metric hoặc cấu hình model/generation/execution. Chỉ loại các trường intervention retrieval và `validation.schema_selection` khỏi kiểm config. Manifest runtime có ở cả hai bên thì phải khớp; thiếu manifest sẽ được ghi cảnh báo vì cùng model_id chưa chứng minh cùng trọng số.

Artifact:

| File | Dùng làm gì |
|---|---|
| `comparison.csv`, `comparison.json` | EX nội bộ, delta, CI, gained/lost, token, mean/p50/p95 latency, speedup, số quan sát có telemetry |
| `paired_questions.csv` | Từng cặp câu, transition, delta token/latency và baseline |
| `by_difficulty.csv` | Điểm từng difficulty kèm số câu N |
| `accuracy_and_transitions.png` | EX và số câu gained/lost |
| `accuracy_delta_ci.png` | ΔEX và khoảng bootstrap 95% |
| `input_tokens_first.png` | Boxplot và ECDF input token thực |
| `elapsed_ms.png` | Phân phối latency đầu cuối |
| `paired_latency.png` | Mỗi chấm là cùng câu ở full/pruning; dưới đường chéo là nhanh hơn |
| `cuda_peak_allocated_mb_device_max.png` | Phân phối peak allocated của device lớn nhất |
| `error_breakdown.png` | Kết quả đúng và các nhóm lỗi |
| `plot_notes.json` | Biểu đồ bỏ qua vì không có telemetry |

Các metric khác cũng có plot nếu đã đo. `--pdf` xuất thêm vector PDF. Không có GPU profile thì không có hình VRAM; ô trống trong CSV là thiếu phép đo, không phải 0. Các thống kê ghi `_n` để biết bao nhiêu câu thực sự có metric; chỉ kết luận toàn tập nếu telemetry đủ. Script đánh dấu mock trong tiêu đề hình.

Bootstrap mặc định lấy lại mẫu **cặp câu hỏi**, 2.000 lần, giữ nguyên full/pruning của mỗi câu; CI ở đơn vị điểm phần trăm. Chọn `--bootstrap-unit db` để resample cả nhóm database. Mini-Dev chỉ có 11 DB nên CI theo DB vẫn có giới hạn. CI không bao gồm nhiễu do đổi model seed, GPU khác hoặc chọn hyperparameter trên tập đánh giá.

Nếu mọi cặp đều có delta bằng 0, percentile bootstrap có thể trả [0; 0]; điều đó không chứng minh rủi ro mất điểm trên câu mới bằng 0.

Nếu ΔEX = −0,4 pp và CI = [−1,6; +0,8], chưa đủ cơ sở kết luận mất tối đa 1 pp: biên dưới vẫn thấp hơn −1. Muốn phát biểu non-inferiority cần margin chốt trước, dữ liệu đủ và thiết kế thống kê phù hợp. CI hẹp trên một model greedy không chứng minh tổng quát hóa sang model/dataset khác.

## G. Chấm BIRD EX bằng evaluator chính thức

Notebook có cell riêng, mặc định `RUN_OFFICIAL_EX=False`. Gắn mã nguồn evaluator chính thức tại commit được ghim rồi bật sau khi full run hoàn tất. Có thể chạy thủ công:

```bash
python scripts/export_bird_predictions.py \
  --dataset data/benchmarks/bird_eval_subset.json \
  --predictions results/generation/repeat_00/full/predictions.jsonl \
  --output results/generation/repeat_00/full/official

python -m pip install func-timeout==4.3.5 psycopg2-binary==2.9.9 PyMySQL==1.1.1

python /path/mini_dev/evaluation/evaluation_ex.py \
  --predicted_sql_path results/generation/repeat_00/full/official/predict_mini_dev.json \
  --ground_truth_path results/generation/repeat_00/full/official/mini_dev_gold.sql \
  --db_root_path /kaggle/input/bird-sqlite/dev_databases/ \
  --diff_json_path results/generation/repeat_00/full/official/mini_dev_difficulty.jsonl \
  --sql_dialect SQLite --num_cpus 2 --meta_time_out 30.0 \
  --output_log_path results/generation/repeat_00/full/official/ex.txt
```

Lặp lại cho bm25/hybrid. Đường dẫn DB root phải có `/` cuối vì scorer ghép chuỗi. Difficulty là **JSONL**. Scorer append vào log; notebook xóa `ex.txt` cũ trước mỗi lần chấm. Khi chạy lệnh thủ công, dùng log mới hoặc lưu bản cũ trước khi chạy lại. Scorer ghép theo **vị trí dòng**, exporter kiểm ID/DB và thứ tự, giữ prediction thiếu thành query sai. `--dataset` phải là đúng subset; không export 10 predictions với file 500 câu. Scorer có breakdown ba difficulty và có thể chia cho 0 nếu smoke subset thiếu nhóm, vì vậy dùng full run cho bước này.

Commit evaluator: `abd11b6db92a1c9f809b32f7564c7c71b34d67f0`. Đọc hướng dẫn của chính phiên bản source bạn dùng. VES/R-VES là hiệu quả **thực thi SQL**, không phải tốc độ inference của LLM; EX, inference latency và peak VRAM cần các phép đo riêng.

`comparison.csv` vẫn chứa điểm nội bộ; lưu `ex.txt` của mỗi arm để báo BIRD EX ở cột riêng. Không lấy CI nội bộ và gắn cho điểm chính thức nếu hai comparator chưa cho cùng kết quả từng câu.

## H. Vẽ dữ liệu retrieval thật có sẵn, ngay cả khi chưa chạy GPU

Hai JSON trong docs đã chứa retrieval chạy thật trên 500 câu, dùng tokenizer và encoder được ghim revision. Tạo lại bốn biểu đồ:

```bash
python -m evaluation.plot_schema_retrieval \
  --input docs/bird_minidev_retrieval_results.json \
  --input docs/bird_minidev_dense_results.json \
  --output results/plots/retrieval_reference --pdf
```

Các hình sẵn để xem: [`figures/schema_retrieval/coverage_vs_tokens.png`](figures/schema_retrieval/coverage_vs_tokens.png), [`raw_prompt_tokens.png`](figures/schema_retrieval/raw_prompt_tokens.png), [`schema_recall.png`](figures/schema_retrieval/schema_recall.png), [`retrieval_latency.png`](figures/schema_retrieval/retrieval_latency.png).

| Arm, budget | Mean raw prompt tokens | Giảm so full | All-reference coverage |
|---|---:|---:|---:|
| Full | 842,62 | 0% | 100% |
| BM25 8/24 | 671,25 | 20,34% | 84,8% |
| Hybrid 8/24 | 672,58 | 20,18% | 91,0% |
| BM25 8/48 | 751,81 | 10,78% | 96,2% |
| Hybrid 8/48 | 752,80 | 10,66% | 98,0% |

Đây là số retrieval, **chưa có EX của model hoặc VRAM GPU**. CPU timing ở các JSON đến từ lượt chạy khác nhau/có trạng thái cache khác nhau; hình latency chỉ mô tả số đã ghi, không dùng để kết luận hybrid nhanh/chậm hơn bao nhiêu trong thí nghiệm được kiểm soát.

`plot_schema_retrieval` hiện nhận wrapper JSON có `reports` như hai file docs. Muốn plot các summary mới của bạn, ghép chúng thành wrapper có cùng provenance, giữ `label` duy nhất và cùng tokenizer/dataset/schema/evidence. Ví dụ trong notebook:

```python
# Sau cell retrieval_tokens, báo cáo đã được lưu cho từng arm.
from pathlib import Path
import json
from evaluation.plot_schema_retrieval import plot_reports
reports = []
modes = ["full", "bm25"]  # thêm "hybrid" nếu đã chạy retrieval cho hybrid
for mode in modes:
    path = Path("results/retrieval_tokens") / mode / "summary.json"
    report = json.loads(path.read_text())
    report["label"] = mode
    reports.append(report)
plot_reports(reports, Path("results/plots/retrieval_this_session"), pdf=True)
```

Chỉ dùng đoạn này khi mọi report có cùng tập câu, tokenizer, evidence và không có lỗi nhãn. Nếu muốn so nhiều file khác nhau, dùng CLI `load_reports` để kiểm provenance trước.

## I. Một vài đoạn code để phân tích thêm

Tạo bảng gọn để đưa vào báo cáo; kiểm cột `_n` trước khi dùng metric:

```python
import pandas as pd
from IPython.display import display
comparison = pd.read_csv("results/plots/repeat_00/comparison.csv")
display(comparison[["run", "questions", "project_ex_pct", "ex_delta_pp", "gained", "lost",
    "input_tokens_first_reduction_pct", "input_tokens_first_reduction_n",
    "elapsed_ms_mean", "elapsed_ms_p95", "end_to_end_speedup_paired",
    "cuda_peak_allocated_mb_device_max_mean", "cuda_peak_allocated_mb_device_max_n"]])
```

Xem những câu full đúng nhưng pruning sai để sửa retrieval:

```python
pairs = pd.read_csv("results/plots/repeat_00/paired_questions.csv")
lost = pairs[(pairs["run"] == "bm25") & (pairs["transition"] == "lost")]
display(lost[["id", "db_id", "difficulty", "error_type", "delta_input_tokens_first", "delta_elapsed_ms"]])
# Tra id ở predictions.jsonl để xem SQL; retrieval details dùng question_id,
# còn importer generation đặt id thành bird_<question_id>.
```

Vẽ điểm theo difficulty từ CSV có sẵn:

```python
import matplotlib.pyplot as plt
scores = pd.read_csv("results/plots/repeat_00/by_difficulty.csv")
ax = scores.pivot(index="difficulty", columns="run", values="project_ex_pct").plot.bar(figsize=(8, 4))
ax.set(ylabel="Project execution match (%)", ylim=(0, 100), xlabel="Difficulty")
ax.set_title("Chất lượng theo độ khó — đọc cùng số câu n trong CSV")
plt.tight_layout()
plt.savefig("results/plots/repeat_00/ex_by_difficulty.png", dpi=180)
plt.show()
```

Không điền số EX/GPU giả để hoàn thiện biểu đồ. Trước generation, hãy dùng biểu đồ retrieval thật ở bước H.

## J. Đọc kết quả và viết báo cáo cuối

Một bảng kết quả cần có N, model snapshot, ngân sách pruning, schema coverage, BIRD EX chính thức, EX nội bộ nếu dùng, ΔEX, gained/lost, input token reduction, latency mean/p50/p95, speedup, peak allocated từng device và cấu hình đo.

Nếu token giảm nhưng latency tăng: xem retrieval và output token, độ dài output có thay đổi không, số call/correction, cache và thứ tự arm. Nếu coverage cao mà EX giảm: kiểm tra các cột JOIN bị mất, bằng chứng câu hỏi, khả năng SQL tương đương, và ca model vốn dùng thông tin từ schema dư. Nếu VRAM gần như không đổi: so footprint trọng số với phần KV cache và độ dài context; đây là kết quả có thể xảy ra.

Chọn budget trên một tập tuning rồi đánh giá cuối trên tập chưa dùng để chọn. Không tự coi BIRD dev là holdout độc lập với Mini-Dev: kiểm tra giao nhau theo câu hỏi/SQL/database, và nêu rõ nếu dùng cùng database. Những kết quả trong docs hiện là exploratory retrieval, không phải tuyên bố chất lượng benchmark cuối.

Khi hoàn tất, chạy cell ZIP, lưu predictions, summary/config, manifest, thứ tự chạy, pip freeze, official logs, CSV và PNG/PDF. Không chỉ giữ hình vì sẽ khó truy nguyên số liệu.

Nguồn chính để đối chiếu dữ liệu và scorer:

- BIRD: https://bird-bench.github.io/
- Mini-Dev source/evaluation: https://github.com/bird-bench/mini_dev
- Metadata: https://huggingface.co/datasets/birdsql/bird_mini_dev
- Hướng dẫn bootstrap paired: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

Trạng thái kiểm chứng: dữ liệu retrieval trong docs đã chạy thật. Notebook được kiểm cấu trúc/cú pháp và bộ so sánh có unit tests; chưa có lần chạy GPU Kaggle của notebook này để báo EX/latency/VRAM thực tế.
