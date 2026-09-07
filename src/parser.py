"""
parser.py
---------
Splits raw text (from OCR or typed input) into a dict of
{question_number: text}.

This is a heuristic, regex-based segmenter — it is meant to give the teacher
a fast first pass in the Streamlit UI, which they can then correct in an
editable table before grading runs. It is NOT expected to be perfect on
messy OCR output, which is why the UI always shows an editable review step
rather than grading blind.

Recognised question-number patterns (start of line, case-insensitive):
    Q1, Q1), Q1., Q.1, Question 1, 1), 1., (1)
"""

from __future__ import annotations

import re

QNO_PATTERN = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*[-:]?\s*|)"      # optional 'Q' / 'Question' prefix
    r"\(?(\d{1,2})\)?"                            # the number itself
    r"\s*[).:\-]?\s*",
    re.IGNORECASE,
)

# A line counts as a "new question start" if it begins with a short number
# (1-2 digits) followed by a separator. OCR output is noisy — a ')' can come
# out as '}', '.', or even get dropped entirely — so the separator set is
# deliberately permissive (including plain whitespace). This is a heuristic
# first pass only; the Streamlit UI always shows an editable review step
# before grading, so occasional false positives/negatives here are expected
# and correctable, not silently trusted.
LINE_START_QNO = re.compile(r"^\s*Q?\.?\s*\(?(\d{1,2})\)?[\s).:\-\}]+", re.IGNORECASE)


def split_into_questions(raw_text: str) -> dict[str, str]:
    """Best-effort split of raw text into {qno: text}. qno keys are strings like '1', '2'."""
    lines = raw_text.splitlines()
    chunks: dict[str, list[str]] = {}
    current_q: str | None = None

    for line in lines:
        m = LINE_START_QNO.match(line)
        if m:
            current_q = m.group(1)
            remainder = line[m.end():]
            chunks.setdefault(current_q, [])
            if remainder.strip():
                chunks[current_q].append(remainder.strip())
        else:
            if current_q is not None and line.strip():
                chunks[current_q].append(line.strip())
            # lines before the first recognised question number are dropped
            # (usually header info: name, roll no, subject, date, etc.)

    return {q: " ".join(parts).strip() for q, parts in chunks.items()}


def merge_specs_with_student_answers(
    question_numbers: list[str],
    student_chunks: dict[str, str],
) -> dict[str, str]:
    """Ensure every expected question number has an entry (possibly empty)."""
    return {q: student_chunks.get(q, "") for q in question_numbers}
