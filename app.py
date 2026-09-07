"""
Streamlit app: NLP Exam Answer Grader

Teacher workflow:
  1. Upload the question paper + answer key (image or typed text)
  2. Review/edit the auto-detected questions: max marks, type
     (theory / numeric / diagram), keywords, expected numeric value
  3. Add as many students as you like, each with their own answer-sheet
     photo(s)/text, and review/edit the OCR'd, per-question answers
  4. Pick a base marking strictness, then grade the whole batch at once.
     Grading is "dynamic": a paper just under the pass mark gets a
     benefit-of-the-doubt boost, a paper scoring very high gets
     double-checked against a stricter benchmark, and — once every
     student is graded — anyone at the very top of the class's bell
     curve is rounded up to full marks.
  5. Review a class bell-curve chart, a summary table, and per-student
     breakdowns, and download everything as CSV.

Diagram-type questions are never auto-marked — they're flagged for the
teacher to grade by hand. Everything else (Maths/Physics/Chemistry/any
subject's text answers) is graded by the NLP engine in src/grader.py.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.grader import QuestionSpec, grade_paper_dynamic, total_marks
from src.ocr import extract_text
from src.parser import split_into_questions, merge_specs_with_student_answers
from src.analytics import StudentTotal, apply_top_of_curve_bonus, fit_bell_curve, percentile_rank, plot_bell_curve, MIN_STUDENTS_FOR_CURVE

st.set_page_config(page_title="NLP Exam Grader", page_icon="📝", layout="wide")

if "question_specs" not in st.session_state:
    st.session_state.question_specs = None  # list[dict] editable
if "students" not in st.session_state:
    st.session_state.students = []  # list[dict]: id, name, chunks
if "student_counter" not in st.session_state:
    st.session_state.student_counter = 0
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


def add_student(name: str = "") -> None:
    st.session_state.student_counter += 1
    sid = st.session_state.student_counter
    st.session_state.students.append({
        "id": sid,
        "name": name or f"Student {len(st.session_state.students) + 1}",
        "chunks": None,
    })


if not st.session_state.students:
    add_student()


def save_upload_to_tmp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def get_text_from_upload(uploaded_file, ocr_backend: str, api_key: str | None) -> str:
    """
    Reads text out of one uploaded file. Never raises — a broken photo or a
    missing OCR engine surfaces as a visible st.error() plus an empty-string
    result for that one file, instead of crashing the whole grading run and
    losing every other student's already-reviewed work.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        tmp_path = save_upload_to_tmp(uploaded_file)
        try:
            return extract_text(tmp_path, backend=ocr_backend, api_key=api_key)
        except Exception as e:
            st.error(f"Couldn't read **{uploaded_file.name}**: {e}")
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        try:
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Couldn't read **{uploaded_file.name}**: {e}")
            return ""


st.title("📝 NLP Exam Answer Grader")
st.caption(
    "Auto-marks the **text** portions of student answers (Maths, Physics, Chemistry, "
    "or any subject) against a teacher-supplied answer key. Diagram-based questions "
    "are always flagged for manual review, never auto-graded. Grades an entire "
    "class of students in one batch and shows where each student lands on the "
    "class bell curve."
)

with st.sidebar:
    st.header("⚙️ Settings")
    ocr_backend = st.radio(
        "OCR engine for photo uploads",
        options=["tesseract", "claude_vision"],
        format_func=lambda x: "Tesseract (offline, free)" if x == "tesseract"
        else "Claude Vision (needs API key, best on handwriting)",
    )
    api_key = None
    if ocr_backend == "claude_vision":
        api_key = st.text_input("Anthropic API key", type="password",
                                 help="Or set the ANTHROPIC_API_KEY environment variable instead.")

    st.divider()
    st.subheader("Marking strictness")
    difficulty = st.select_slider(
        "Base strictness", options=["easy", "medium", "hard"], value="easy",
        help="Easy = generous partial credit, key points matter more than exact wording. "
             "Hard = only close, well-covered answers score high.",
    )
    st.caption(
        "If a question has **required keywords/key-points** set in the answer key, "
        "covering most of them earns full marks for that question even if the "
        "wording is different from the model answer."
    )

    st.divider()
    st.subheader("🎯 Dynamic auto-adjustment")
    enable_dynamic = st.checkbox(
        "Enable smart pass-margin boost & high-score strict recheck", value=True,
        help="Papers just under the pass mark get partial credit rounded up to help "
             "them pass. Papers scoring very high get double-checked against a "
             "stricter benchmark, so full marks are only kept when genuinely earned.",
    )
    pass_threshold = st.slider("Pass mark (%)", min_value=0, max_value=100, value=40, step=1,
                                disabled=not enable_dynamic)
    boost_margin = st.slider("Boost margin below pass mark (percentage points)",
                              min_value=0, max_value=40, value=15, step=1,
                              disabled=not enable_dynamic,
                              help="A paper scoring between (pass mark − margin) and the pass "
                                   "mark is eligible for the benefit-of-the-doubt boost.")
    high_score_cutoff = st.slider("Re-verify strictly at/above (%)", min_value=50, max_value=100,
                                   value=90, step=1, disabled=not enable_dynamic)

    st.divider()
    st.subheader("📈 Class bell curve")
    enable_bell_bonus = st.checkbox(
        "Auto full-marks for top of the class curve (≥95th percentile)", value=True,
        help=f"Needs at least {MIN_STUDENTS_FOR_CURVE} students. A student whose score sits "
             "at/beyond the 95th percentile of the class's fitted normal distribution is "
             "rounded up to full marks.",
    )

    st.divider()
    st.caption(
        "⚠️ Diagrams, graphs and figures are never interpreted by this tool — "
        "those questions are always flagged for the teacher to mark by hand."
    )

step1, step2, step3 = st.tabs(
    ["1️⃣ Question Paper & Answer Key", "2️⃣ Students' Answer Sheets", "3️⃣ Grade the Class"]
)

# ---------------------------------------------------------------------------
# STEP 1: Question paper + answer key -> editable question specs
# ---------------------------------------------------------------------------
with step1:
    st.subheader("Upload the answer key")
    st.write(
        "Upload the teacher's answer key as a photo or a text file. "
        "It should list each question number with the model/expected answer "
        "(e.g. `Q1) Model answer text...`)."
    )
    key_file = st.file_uploader(
        "Answer key (image or .txt)", type=["png", "jpg", "jpeg", "webp", "txt"], key="key_upload"
    )

    if key_file and st.button("Extract & parse questions", key="parse_key_btn"):
        with st.spinner("Reading answer key..."):
            raw_text = get_text_from_upload(key_file, ocr_backend, api_key)
            chunks = split_into_questions(raw_text)
        if not chunks:
            st.warning(
                "Couldn't auto-detect question numbers. You can still add questions manually below."
            )
            chunks = {}
        st.session_state.question_specs = [
            {
                "qno": q, "max_marks": 5.0, "q_type": "theory",
                "model_answer": text, "keywords": "", "expected_value": None,
            }
            for q, text in sorted(chunks.items(), key=lambda kv: (len(kv[0]), kv[0]))
        ]
        st.success(f"Detected {len(chunks)} question(s). Review and correct below.")

    st.divider()
    st.subheader("Review & edit the answer key")
    st.caption(
        "OCR/parsing is a best-effort first pass — fix question numbers, marks, "
        "type, model answers and keywords here before grading. "
        "For **numeric** questions (common in Maths/Physics/Chemistry), set the "
        "expected numeric value. For **diagram** questions, marking is skipped entirely. "
        "**Tip:** filling in 'Required keywords' lets the grader give full marks whenever "
        "a student hits the key points, regardless of exact phrasing."
    )

    if st.session_state.question_specs is None:
        st.session_state.question_specs = [
            {"qno": "1", "max_marks": 5.0, "q_type": "theory",
             "model_answer": "", "keywords": "", "expected_value": None}
        ]

    edited_df = st.data_editor(
        pd.DataFrame(st.session_state.question_specs),
        num_rows="dynamic",
        column_config={
            "qno": st.column_config.TextColumn("Q No."),
            "max_marks": st.column_config.NumberColumn("Max Marks", min_value=0.0, step=0.5),
            "q_type": st.column_config.SelectboxColumn("Type", options=["theory", "numeric", "diagram"]),
            "model_answer": st.column_config.TextColumn("Model / Expected Answer", width="large"),
            "keywords": st.column_config.TextColumn("Required keywords (comma-separated)", width="medium"),
            "expected_value": st.column_config.NumberColumn("Expected numeric value (if numeric)"),
        },
        use_container_width=True,
        key="key_editor",
    )
    st.session_state.question_specs = edited_df.to_dict("records")

# ---------------------------------------------------------------------------
# STEP 2: Students' answer sheets -> editable per-question answers, per student
# ---------------------------------------------------------------------------
with step2:
    st.subheader("Add students and upload each one's answer sheet")
    st.write(
        "Add as many students as you need. For each one, upload their answer sheet "
        "(one or more photos, or a text file), extract the text, then review/correct "
        "it in the table before grading."
    )

    expected_qnos = [str(s["qno"]) for s in (st.session_state.question_specs or [])]
    remove_ids: list[int] = []

    for student in st.session_state.students:
        sid = student["id"]
        with st.expander(f"🧑‍🎓 {student['name']}", expanded=len(st.session_state.students) <= 3):
            top_cols = st.columns([3, 1])
            with top_cols[0]:
                new_name = st.text_input("Student name", value=student["name"], key=f"name_{sid}")
                student["name"] = new_name or student["name"]
            with top_cols[1]:
                st.write("")
                st.write("")
                if st.button("🗑️ Remove", key=f"remove_{sid}"):
                    remove_ids.append(sid)

            student_files = st.file_uploader(
                "Answer sheet (image(s) or .txt)",
                type=["png", "jpg", "jpeg", "webp", "txt"],
                accept_multiple_files=True,
                key=f"files_{sid}",
            )

            if student_files and st.button("Extract & parse this student's answers", key=f"extract_{sid}"):
                combined_text = []
                with st.spinner(f"Reading {student['name']}'s answers..."):
                    for f in student_files:
                        combined_text.append(get_text_from_upload(f, ocr_backend, api_key))
                raw_text = "\n\n".join(t for t in combined_text if t)
                chunks = split_into_questions(raw_text)
                student["chunks"] = merge_specs_with_student_answers(expected_qnos, chunks)
                st.success(f"Extracted answers for {sum(1 for v in chunks.values() if v)} question(s). Review below.")

            if student["chunks"] is None:
                student["chunks"] = {q: "" for q in expected_qnos}
            else:
                # keep the table in sync if questions were added/removed in Step 1
                student["chunks"] = {q: student["chunks"].get(q, "") for q in expected_qnos}

            student_df = pd.DataFrame(
                [{"qno": q, "student_answer": txt} for q, txt in student["chunks"].items()]
            )
            edited_student_df = st.data_editor(
                student_df,
                num_rows="dynamic",
                column_config={
                    "qno": st.column_config.TextColumn("Q No."),
                    "student_answer": st.column_config.TextColumn("Student's Answer", width="large"),
                },
                use_container_width=True,
                key=f"editor_{sid}",
            )
            student["chunks"] = dict(
                zip(edited_student_df["qno"].astype(str), edited_student_df["student_answer"])
            )

    if remove_ids:
        st.session_state.students = [s for s in st.session_state.students if s["id"] not in remove_ids]
        if not st.session_state.students:
            add_student()
        st.rerun()

    st.button("➕ Add another student", on_click=add_student)

# ---------------------------------------------------------------------------
# STEP 3: Grade the whole class at once
# ---------------------------------------------------------------------------
with step3:
    st.subheader(f"Grade **{len(st.session_state.students)}** student(s) at **{difficulty.upper()}** base strictness")

    if st.button("▶️ Run grading for the whole class", type="primary"):
        specs = []
        for row in st.session_state.question_specs:
            kw = [k.strip() for k in str(row.get("keywords") or "").split(",") if k.strip()]
            specs.append(QuestionSpec(
                qno=str(row["qno"]),
                max_marks=float(row["max_marks"] or 0),
                model_answer=str(row.get("model_answer") or ""),
                q_type=row.get("q_type") or "theory",
                keywords=kw,
                expected_value=(float(row["expected_value"])
                                if row.get("expected_value") not in (None, "", "None") else None),
            ))

        batch: dict[str, dict] = {}
        with st.spinner("Grading every student..."):
            for student in st.session_state.students:
                answers = {str(k): v for k, v in (student.get("chunks") or {}).items()}
                results, note = grade_paper_dynamic(
                    answers, specs,
                    base_difficulty=difficulty,
                    enable_dynamic=enable_dynamic,
                    pass_threshold_pct=float(pass_threshold),
                    boost_margin_pct=float(boost_margin),
                    high_score_pct=float(high_score_cutoff),
                )
                awarded, maxm = total_marks(results)
                batch[student["name"]] = {"results": results, "awarded": awarded, "maxm": maxm, "note": note}
        st.session_state.batch_results = batch

    if st.session_state.batch_results:
        batch = st.session_state.batch_results

        totals = [StudentTotal(name, d["awarded"], d["maxm"]) for name, d in batch.items()]
        if enable_bell_bonus:
            bell_info = apply_top_of_curve_bonus(totals)
            for t in totals:
                batch[t.name]["awarded"] = t.awarded  # sync any bump back into batch
        else:
            mean, std = fit_bell_curve(totals)
            bell_info = {t.name: {"percentile": round(percentile_rank(t.pct, mean, std), 1), "boosted": False}
                         for t in totals}

        maxm_ref = totals[0].max_marks if totals else 0
        col1, col2, col3 = st.columns(3)
        class_awarded = sum(t.awarded for t in totals)
        class_pct = (class_awarded / (maxm_ref * len(totals)) * 100) if maxm_ref and totals else 0
        col1.metric("Students graded", f"{len(totals)}")
        col2.metric("Class average", f"{class_pct:.1f}%")
        col3.metric("Passing (≥ pass mark)", f"{sum(1 for t in totals if t.pct >= pass_threshold)} / {len(totals)}")

        st.divider()
        st.subheader("📈 Class bell curve")
        if len(totals) >= MIN_STUDENTS_FOR_CURVE:
            fig = plot_bell_curve(totals, pass_threshold_pct=float(pass_threshold))
            st.pyplot(fig)
        else:
            st.info(
                f"Add at least {MIN_STUDENTS_FOR_CURVE} students to see the class bell curve "
                "and enable the top-of-the-curve auto full-marks rule."
            )

        st.divider()
        st.subheader("Class summary")
        manual_review_by_student = {
            name: [r for r in d["results"] if r.method == "diagram-skipped"]
            for name, d in batch.items()
        }
        if any(manual_review_by_student.values()):
            st.warning("⚠️ Some students have diagram-based question(s) needing manual grading.")

        rows = []
        for name, d in batch.items():
            pct = (d["awarded"] / d["maxm"] * 100) if d["maxm"] else 0
            rows.append({
                "Student": name,
                "Marks": d["awarded"],
                "Max Marks": d["maxm"],
                "%": round(pct, 1),
                "Result": "PASS" if pct >= pass_threshold else "FAIL",
                "Class percentile": bell_info[name]["percentile"],
                "Top-of-curve full marks": "✅" if bell_info[name]["boosted"] else "",
                "Adjustment": d["note"],
            })
        summary_df = pd.DataFrame(rows).sort_values("%", ascending=False)
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            "⬇️ Download class summary (CSV)", summary_df.to_csv(index=False),
            file_name="class_summary.csv", mime="text/csv",
        )

        st.divider()
        st.subheader("Per-student breakdown")
        detail_rows = []
        for name, d in batch.items():
            with st.expander(f"📄 {name}"):
                if d["note"]:
                    st.info(d["note"])
                if manual_review_by_student[name]:
                    st.warning(
                        "⚠️ Needs manual grading: "
                        + ", ".join(f"Q{r.qno}" for r in manual_review_by_student[name])
                    )
                q_rows = [{
                    "Q No.": r.qno,
                    "Marks Awarded": r.awarded_marks,
                    "Max Marks": r.max_marks,
                    "Method": r.method,
                    "Match Score": r.similarity_score,
                    "Keyword Coverage": r.keyword_coverage,
                    "Missing Keywords": ", ".join(r.missing_keywords),
                    "Feedback": r.feedback,
                    "Adjustment": r.adjustment_note,
                } for r in d["results"]]
                st.dataframe(pd.DataFrame(q_rows), use_container_width=True)
                for row in q_rows:
                    detail_rows.append({"Student": name, **row})

        if detail_rows:
            detail_df = pd.DataFrame(detail_rows)
            st.download_button(
                "⬇️ Download full per-question results (CSV)", detail_df.to_csv(index=False),
                file_name="detailed_grading_results.csv", mime="text/csv",
            )
    else:
        st.info("Fill in Steps 1 & 2, then click **Run grading for the whole class**.")