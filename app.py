"""Vietnamese Streamlit interface for local Adaptive SLM SQL."""
import hashlib
import json
import logging
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from core.config import ROOT, load_config
from core.model_backend import create_backend
from core.pipeline import Pipeline
from core.schema_inspector import SchemaInspector
from databases.create_student_db import create_database

st.set_page_config(page_title="Adaptive SLM SQL", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Đang tải model…")
def cached_backend(config_json: str):
    return create_backend({"model": json.loads(config_json)})


@st.cache_resource
def demo_database():
    return create_database()


def show_result(record: dict):
    selected = record["selected"]
    decision = record["decision"]
    candidates = record["candidates"]
    valid = sum(bool(c["execution"] and c["execution"]["success"]) for c in candidates)
    cols = st.columns(5)
    cols[0].metric("Đồng thuận", f"{decision['confidence']:.0%}")
    cols[1].metric("Ứng viên", len(candidates))
    cols[2].metric("Thực thi hợp lệ", f"{valid / len(candidates):.0%}" if candidates else "0%")
    cols[3].metric("Thời gian", f"{record['elapsed_ms'] / 1000:.2f} s")
    cols[4].metric("Dừng", decision["reason"])
    st.caption(record["explanation"])
    if selected:
        st.subheader("SQL được chọn")
        st.code(selected["sql"], language="sql")
        execution = selected["execution"]
        frame = pd.DataFrame(execution["rows"], columns=execution["columns"])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if execution["truncated"]:
            st.caption(f"Hiển thị {len(frame)} / {execution['row_count']} dòng; bỏ phiếu dùng toàn bộ kết quả.")
        st.download_button("Tải bảng kết quả CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                           "query_result.csv", "text/csv")
    else:
        st.warning(record["explanation"])
    with st.expander("Ứng viên và kiểm tra", expanded=not bool(selected)):
        rows = []
        for c in candidates:
            validation, execution = c["validation"] or {}, c["execution"] or {}
            rows.append({"ID": c["id"], "SQL": c["sql"], "An toàn": validation.get("is_safe"),
                         "Schema / cú pháp": validation.get("is_valid"),
                         "Chạy được": execution.get("success", False),
                         "Nhóm": (c["group"] or "")[:10], "Phiếu": c["vote_score"],
                         "Điểm": c["score"], "Lượt sửa": len(c["corrections"]),
                         "Lỗi": c["error"] or execution.get("error") or "; ".join(validation.get("errors", []))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Các vòng sinh SQL"):
        st.dataframe(pd.DataFrame(record["trace"]), use_container_width=True, hide_index=True)
    st.download_button("Tải trace JSON", json.dumps(record, ensure_ascii=False, indent=2),
                       "query_trace.json", "application/json")


def main():
    config = load_config()
    logging.basicConfig(level=config["logging_level"])
    st.title("Hỏi dữ liệu bằng ngôn ngữ tự nhiên")
    st.write("Adaptive SLM SQL · Chuyển câu hỏi thành SQL, kiểm tra và đối chiếu các kết quả.")
    with st.sidebar:
        st.header("Cấu hình")
        backends = ["mock", "huggingface"]
        config["model"]["backend"] = st.selectbox("Backend", backends, index=backends.index(config["model"]["backend"]))
        config["model"]["model_id"] = st.text_input("Model ID", config["model"]["model_id"])
        devices, quantizations = ["auto", "cpu", "cuda"], ["none", "4bit", "8bit"]
        config["model"]["device"] = st.selectbox("Thiết bị", devices, index=devices.index(config["model"]["device"]))
        config["model"]["quantization"] = st.selectbox(
            "Quantization", quantizations, index=quantizations.index(str(config["model"]["quantization"])))
        modes = ["adaptive_sc", "greedy", "fixed_sc"]
        config["sampling"]["mode"] = st.selectbox("Phương pháp", modes, index=modes.index(config["sampling"]["mode"]))
        # Sidebar defaults come from config.yaml so editing the file takes effect in the UI.
        sample, gen = config["sampling"], config["generation"]
        sample["fixed_k"] = st.number_input("K cố định", 1, 32, sample["fixed_k"])
        rounds = st.text_input("Ứng viên mỗi vòng", ",".join(map(str, sample["rounds"])))
        gen["temperature"] = st.slider("Temperature", 0.0, 1.5, float(gen["temperature"]), 0.05)
        gen["top_p"] = st.slider("Top p", 0.05, 1.0, float(gen["top_p"]), 0.05)
        gen["max_new_tokens"] = st.number_input("Max new tokens", 32, 4096, gen["max_new_tokens"], 32)
        config["generation"]["batch_size"] = st.number_input("Batch size", 1, 8, config["generation"]["batch_size"])
        config["validation"]["schema_selection"] = st.checkbox("Chọn schema theo câu hỏi")
        config["correction"]["enabled"] = st.checkbox("Sửa lỗi thực thi", value=True)
        sample["confidence_threshold"] = st.slider(
            "Ngưỡng đồng thuận", 0.5, 1.0, float(sample["confidence_threshold"]), 0.05)
        sample["time_budget_seconds"] = st.number_input(
            "Ngân sách thời gian (giây)", 5, 1800, int(sample["time_budget_seconds"]))
    if config["model"]["backend"] == "mock":
        st.info("Chế độ mock: chọn câu hỏi mẫu để xem cách hoạt động. Kết quả không đo năng lực AI.")
    source = st.radio("Database", ["Dữ liệu sinh viên mẫu", "Tải SQLite lên"], horizontal=True)
    upload = st.file_uploader("File .db hoặc .sqlite", type=["db", "sqlite"]) if source == "Tải SQLite lên" else None
    if source == "Tải SQLite lên" and upload is None:
        st.caption("Chọn một database SQLite để bắt đầu.")
        return
    questions = json.loads((ROOT / "data/demo_questions.json").read_text(encoding="utf-8"))
    if source == "Dữ liệu sinh viên mẫu":
        picked = st.selectbox("Câu hỏi mẫu", [item["question"] for item in questions])
        if st.button("Dùng câu hỏi mẫu"):
            st.session_state["question"] = picked
    st.text_area("Câu hỏi", key="question", placeholder="Ví dụ: Có bao nhiêu sinh viên?", max_chars=4000)
    evidence = st.text_input("Thông tin nghiệp vụ bổ sung (tùy chọn)", max_chars=8000)
    with tempfile.TemporaryDirectory(prefix="adaptive_sql_") as directory:
        if upload is not None:
            payload = upload.getvalue()
            if len(payload) > config["upload"]["max_mb"] * 1024 * 1024:
                st.error("File vượt giới hạn 20 MB.")
                return
            if not payload.startswith(b"SQLite format 3\x00"):
                st.error("Nội dung file không phải SQLite hợp lệ.")
                return
            db_path = Path(directory) / "uploaded.sqlite"
            db_path.write_bytes(payload)
            signature = hashlib.sha256(payload).hexdigest()
        else:
            db_path = demo_database()
            signature = "demo"
        with st.expander("Cấu trúc database"):
            st.code(SchemaInspector.format(SchemaInspector(**config["schema"]).inspect(db_path)), language="text")
        if st.button("Generate SQL", type="primary"):
            if not st.session_state.get("question", "").strip():
                st.error("Vui lòng nhập câu hỏi trước khi sinh SQL.")
                return
            config["sampling"]["rounds"] = [int(value.strip()) for value in rounds.split(",")]
            backend = cached_backend(json.dumps(config["model"], sort_keys=True))
            for warning in getattr(backend, "warnings", []):
                st.warning(warning)
            with st.spinner("Đang sinh và kiểm tra SQL…"):
                result = Pipeline(config, backend).run(db_path, st.session_state["question"], evidence)
            st.session_state["last_result"] = result.public_dict()
            st.session_state["last_database"] = signature
    if st.session_state.get("last_database") == signature and st.session_state.get("last_result"):
        st.caption("Kết quả lần chạy: " + st.session_state["last_result"]["question"])
        show_result(st.session_state["last_result"])


try:
    main()
except (ValueError, RuntimeError) as error:
    st.error(str(error))
except Exception:
    st.error("Không xử lý được yêu cầu. Kiểm tra file SQLite, cấu hình và hướng dẫn trong README.")
