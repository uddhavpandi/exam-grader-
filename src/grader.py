"""
grader.py
---------
The NLP grading engine.

Given a student's answer text and the teacher's model (reference) answer,
this module scores how well the student answered — WITHOUT any diagram,
handwriting-neatness, or image-based judgement. Pure text/NLP comparison.

Two signals are combined:
  1. Semantic / lexical similarity  -> TF-IDF cosine similarity (always available,
     no internet / model download needed). If sentence-transformers is
     installed AND a model is already cached locally, it is used instead for
     a stronger semantic signal (see semantic_backend.py).
  2. Keyword / concept coverage     -> what fraction of the teacher's required
     key terms (definitions, named laws, formulas, numeric results) actually
     appear in the student's answer.

A third, independent path handles NUMERIC / FORMULA questions (common in
Maths, Physics, Chemistry): the final numeric answer is extracted from both
texts and compared with a tolerance that depends on the marking strictness.

Marking "difficulty" (easy / medium / hard) does NOT change what is measured —
it changes how forgiving the mark-conversion curve is. This mirrors how a
human examiner marks more leniently or strictly while using the same rubric.

KEY-POINT SUFFICIENCY (new): if the teacher has supplied required keywords/
key-points for a question and the student's answer covers most of them
(>= KEYWORD_FULL_MARKS_COVERAGE), the answer is treated as "sufficient" and
pushed to full marks for that question even if the wording/phrasing doesn't
closely mirror the model answer. The idea: an examiner cares whether the
key point was made, not whether it was made in the exact same words.

DYNAMIC / ADAPTIVE MARKING (new): grade_paper_dynamic() wraps the base
grading pass with two class-fairness adjustments a human moderator would
often make by hand:
  - "Pass-margin boost": a paper that lands just below the pass mark has
    its partial-credit answers given the benefit of the doubt until it
    either clears the pass mark or every attempted answer is fully
    credited (blank/zero-effort answers are never boosted).
  - "High-score strict recheck": a paper that already looks close to full
    marks gets re-verified against a stricter benchmark, so that "full
    marks" is only kept where it's genuinely earned rather than an
    artifact of a lenient strictness setting.
The strictness applied to any single paper is therefore not fixed — it
adapts to where that paper landed, which is what makes it "dynamic".
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .semantic_backend import get_semantic_similarity, semantic_backend_available

Difficulty = Literal["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Difficulty curves
# ---------------------------------------------------------------------------
# Each curve maps a raw match score in [0, 1] to a fraction of max marks.
# "easy"   -> generous: decent overlap already earns full marks
# "medium" -> standard/balanced curve
# "hard"   -> strict: only close, well-covered answers earn full marks;
#             low-overlap answers are pushed towards zero faster.
# These have been loosened slightly across the board (per teacher feedback
# that the tool was under-crediting reasonable answers) — "hard" is kept
# meaningfully strict on purpose, since it now also does double duty as the
# benchmark used for the "high score -> re-verify" dynamic recheck below.
DIFFICULTY_CURVES: dict[Difficulty, dict] = {
    "easy":   {"full_marks_at": 0.45, "floor_below": 0.10},
    "medium": {"full_marks_at": 0.65, "floor_below": 0.20},
    "hard":   {"full_marks_at": 0.88, "floor_below": 0.40},
}

# Numeric-answer tolerance (relative error) by difficulty
NUMERIC_TOLERANCE: dict[Difficulty, float] = {
    "easy": 0.10,    # within 10% counts full marks
    "medium": 0.05,  # within 5%
    "hard": 0.01,    # within 1%
}

# If the student's answer covers at least this fraction of the teacher's
# required keywords/key-points, treat the answer as "sufficient" and grade
# it as a full-marks match regardless of how differently it's worded.
KEYWORD_FULL_MARKS_COVERAGE = 0.8

# --- Dynamic/adaptive marking defaults (used by grade_paper_dynamic) -------
DEFAULT_PASS_THRESHOLD_PCT = 40.0   # a paper needs this % to "pass"
DEFAULT_BOOST_MARGIN_PCT = 15.0     # papers within this many points below
                                     # the pass mark are eligible for a boost
DEFAULT_HIGH_SCORE_PCT = 90.0       # papers at/above this % get re-verified
                                     # against the "hard" curve

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:\s*[eE][-+]?\d+)?")


@dataclass
class QuestionSpec:
    """Everything the teacher defines for one question via the answer key."""
    qno: str
    max_marks: float
    model_answer: str
    q_type: Literal["theory", "numeric", "diagram"] = "theory"
    keywords: list[str] = field(default_factory=list)
    # for numeric/formula questions, the teacher can give the accepted
    # numeric result directly instead of relying on extraction:
    expected_value: float | None = None


@dataclass
class GradeResult:
    qno: str
    max_marks: float
    awarded_marks: float
    similarity_score: float
    keyword_coverage: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    method: str          # "theory" | "numeric" | "diagram-skipped"
    feedback: str
    # Set by grade_paper_dynamic() when a mark was nudged up (pass-margin
    # boost) or re-checked down (high-score strict recheck). Empty string
    # means the mark is exactly what the base difficulty curve produced.
    adjustment_note: str = ""


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.+\-*/=%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_numbers(text: str) -> list[float]:
    out = []
    for m in NUMBER_RE.findall(text or ""):
        m = m.replace(",", "")
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


def _tfidf_similarity(a: str, b: str) -> float:
    """Cosine similarity between two short texts via TF-IDF. Offline, no downloads."""
    a, b = _clean(a), _clean(b)
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words="english").fit([a, b])
        matrix = vec.transform([a, b])
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
        return float(max(0.0, min(1.0, sim)))
    except ValueError:
        # e.g. both texts were entirely stopwords -> fall back to token overlap
        ta, tb = set(a.split()), set(b.split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


def _keyword_coverage(student_text: str, keywords: list[str]) -> tuple[float, list[str], list[str]]:
    if not keywords:
        return 1.0, [], []
    student_clean = _clean(student_text)
    matched, missing = [], []
    for kw in keywords:
        kw_clean = _clean(kw)
        if kw_clean and kw_clean in student_clean:
            matched.append(kw)
        else:
            missing.append(kw)
    coverage = len(matched) / len(keywords)
    return coverage, matched, missing


def _apply_difficulty_curve(raw_score: float, difficulty: Difficulty) -> float:
    """Map a raw [0,1] match score to a fraction of full marks, per strictness."""
    curve = DIFFICULTY_CURVES[difficulty]
    full_at = curve["full_marks_at"]
    floor_below = curve["floor_below"]

    if raw_score <= floor_below:
        return 0.0
    if raw_score >= full_at:
        return 1.0
    # linear ramp between floor and full-marks point
    return (raw_score - floor_below) / (full_at - floor_below)


def grade_theory_answer(
    student_text: str,
    spec: QuestionSpec,
    difficulty: Difficulty,
) -> GradeResult:
    student_text = student_text or ""

    if semantic_backend_available():
        sim = get_semantic_similarity(student_text, spec.model_answer)
    else:
        sim = _tfidf_similarity(student_text, spec.model_answer)

    coverage, matched, missing = _keyword_coverage(student_text, spec.keywords)

    # Blend: semantic/lexical similarity carries more weight for flowing prose;
    # keyword coverage matters more when the teacher supplied explicit keywords.
    key_point_override = False
    if spec.keywords:
        raw_score = 0.5 * sim + 0.5 * coverage
        # Key-point sufficiency: if the student hit (most of) the required
        # key points, don't penalise them just for phrasing it differently
        # from the model answer — push the raw score up to this curve's
        # full-marks threshold so the answer scores full marks.
        if coverage >= KEYWORD_FULL_MARKS_COVERAGE:
            full_at = DIFFICULTY_CURVES[difficulty]["full_marks_at"]
            if raw_score < full_at:
                raw_score = full_at
                key_point_override = True
    else:
        raw_score = sim

    fraction = _apply_difficulty_curve(raw_score, difficulty)
    awarded = round(fraction * spec.max_marks * 2) / 2  # round to nearest 0.5

    if not student_text.strip():
        feedback = "No answer text detected for this question."
        awarded = 0.0
    elif key_point_override:
        feedback = "All the key points expected in the answer are present — full marks."
    elif fraction >= 0.95:
        feedback = "Strong match with the model answer."
    elif fraction >= 0.6:
        feedback = "Partially correct. " + (
            f"Missing: {', '.join(missing)}." if missing else "Some key detail is missing or underdeveloped."
        )
    elif fraction > 0:
        feedback = "Weak match. " + (
            f"Missing key points: {', '.join(missing)}." if missing else "Answer diverges substantially from the expected response."
        )
    else:
        feedback = "Does not match the expected answer."

    return GradeResult(
        qno=spec.qno,
        max_marks=spec.max_marks,
        awarded_marks=awarded,
        similarity_score=round(sim, 3),
        keyword_coverage=round(coverage, 3),
        matched_keywords=matched,
        missing_keywords=missing,
        method="theory",
        feedback=feedback,
    )


def grade_numeric_answer(
    student_text: str,
    spec: QuestionSpec,
    difficulty: Difficulty,
) -> GradeResult:
    tol = NUMERIC_TOLERANCE[difficulty]
    student_nums = _extract_numbers(student_text)

    expected = spec.expected_value
    if expected is None:
        model_nums = _extract_numbers(spec.model_answer)
        expected = model_nums[-1] if model_nums else None  # final answer usually last number

    if expected is None or not student_nums:
        # fall back to a light text/formula comparison (covers "=" steps, units, etc.)
        sim = _tfidf_similarity(student_text, spec.model_answer)
        fraction = _apply_difficulty_curve(sim, difficulty)
        awarded = round(fraction * spec.max_marks * 2) / 2
        feedback = "No clear final numeric value found — graded on formula/working similarity."
        return GradeResult(
            qno=spec.qno, max_marks=spec.max_marks, awarded_marks=awarded,
            similarity_score=round(sim, 3), keyword_coverage=0.0,
            matched_keywords=[], missing_keywords=[], method="numeric", feedback=feedback,
        )

    # pick the student number closest to the expected value (handles extra numbers in working)
    best = min(student_nums, key=lambda x: abs(x - expected))
    rel_error = abs(best - expected) / (abs(expected) if expected != 0 else 1)

    if rel_error <= tol:
        fraction = 1.0
        feedback = f"Final answer {best:g} matches expected {expected:g} within tolerance."
    elif rel_error <= tol * 4:
        fraction = max(0.0, 1 - (rel_error - tol) / (tol * 3))
        feedback = f"Close but outside tolerance: got {best:g}, expected {expected:g}."
    else:
        fraction = 0.0
        feedback = f"Incorrect: got {best:g}, expected {expected:g}."

    awarded = round(fraction * spec.max_marks * 2) / 2
    return GradeResult(
        qno=spec.qno, max_marks=spec.max_marks, awarded_marks=awarded,
        similarity_score=round(1 - min(rel_error, 1), 3), keyword_coverage=0.0,
        matched_keywords=[], missing_keywords=[], method="numeric", feedback=feedback,
    )


def grade_question(student_text: str, spec: QuestionSpec, difficulty: Difficulty) -> GradeResult:
    if spec.q_type == "diagram":
        return GradeResult(
            qno=spec.qno, max_marks=spec.max_marks, awarded_marks=0.0,
            similarity_score=0.0, keyword_coverage=0.0, matched_keywords=[],
            missing_keywords=[], method="diagram-skipped",
            feedback="Diagram-based question — not auto-graded. Needs manual review by the teacher.",
        )
    if spec.q_type == "numeric":
        return grade_numeric_answer(student_text, spec, difficulty)
    return grade_theory_answer(student_text, spec, difficulty)


def grade_paper(
    student_answers: dict[str, str],
    question_specs: list[QuestionSpec],
    difficulty: Difficulty,
) -> list[GradeResult]:
    """student_answers maps question number -> student's OCR'd/typed answer text."""
    results = []
    for spec in question_specs:
        student_text = student_answers.get(spec.qno, "")
        results.append(grade_question(student_text, spec, difficulty))
    return results


def total_marks(results: list[GradeResult]) -> tuple[float, float]:
    awarded = sum(r.awarded_marks for r in results)
    maxm = sum(r.max_marks for r in results)
    return awarded, maxm


def paper_percentage(results: list[GradeResult]) -> float:
    awarded, maxm = total_marks(results)
    return (awarded / maxm * 100.0) if maxm else 0.0


# ---------------------------------------------------------------------------
# Dynamic / adaptive marking: pass-margin boost + high-score strict recheck
# ---------------------------------------------------------------------------
def _boost_towards_pass(
    results: list[GradeResult],
    pass_threshold_pct: float,
) -> tuple[list[GradeResult], str]:
    """
    If the paper is just below the pass mark, nudge up partial-credit
    answers (never blank/zero-effort ones) until it clears the pass mark
    or every attempted answer is maxed out. Mutates and returns `results`.
    """
    awarded, maxm = total_marks(results)
    if maxm <= 0:
        return results, ""
    target = pass_threshold_pct / 100.0 * maxm
    if awarded >= target:
        return results, ""

    # Candidates: attempted, gradeable answers with room left to grow.
    # "Attempted" = they scored something, or showed real overlap/coverage
    # with the expected answer (so a genuinely blank answer is never boosted).
    eligible = [
        r for r in results
        if r.method in ("theory", "numeric")
        and r.awarded_marks < r.max_marks
        and (r.awarded_marks > 0 or r.similarity_score > 0.1 or r.keyword_coverage > 0.1)
    ]
    # Boost the closest-to-full answers first — that's how a human moderator
    # typically finds a couple of marks: rounding up near-misses, not
    # inventing credit for answers that missed the point entirely.
    eligible.sort(key=lambda r: -r.awarded_marks)

    boosted_any = False
    guard = 0
    while awarded < target and eligible and guard < 500:
        guard += 1
        progressed = False
        for r in eligible:
            if r.awarded_marks >= r.max_marks:
                continue
            step = min(0.5, r.max_marks - r.awarded_marks)
            r.awarded_marks = round((r.awarded_marks + step) * 2) / 2
            awarded += step
            progressed = True
            boosted_any = True
            if not r.adjustment_note:
                r.adjustment_note = "Nudged up — benefit of the doubt to help this paper reach the pass mark."
            if awarded >= target:
                break
        if not progressed:
            break

    note = ""
    if boosted_any:
        note = (
            f"This paper was just below the {pass_threshold_pct:.0f}% pass mark. "
            "Partial-credit answers were given the benefit of the doubt to help it reach a pass."
        )
    return results, note


def _strict_recheck(
    student_answers: dict[str, str],
    question_specs: list[QuestionSpec],
    results: list[GradeResult],
) -> tuple[list[GradeResult], str]:
    """
    If the paper is already scoring very high, re-grade every non-diagram
    question against the "hard" curve and keep whichever score is stricter
    (lower), so a high score under an easy/medium setting isn't just an
    artifact of that setting being lenient.
    """
    strict_results = grade_paper(student_answers, question_specs, "hard")
    merged: list[GradeResult] = []
    changed = False
    for original, strict in zip(results, strict_results):
        if original.method == "diagram-skipped":
            merged.append(original)
            continue
        if strict.awarded_marks < original.awarded_marks:
            strict.adjustment_note = "Re-checked at a stricter benchmark since this paper scored very high."
            merged.append(strict)
            changed = True
        else:
            merged.append(original)
    note = ""
    if changed:
        note = (
            "This paper scored very high, so its answers were re-verified against a stricter "
            "benchmark to make sure full marks are genuinely earned, not just leniently graded."
        )
    return merged, note


def grade_paper_dynamic(
    student_answers: dict[str, str],
    question_specs: list[QuestionSpec],
    base_difficulty: Difficulty = "medium",
    enable_dynamic: bool = True,
    pass_threshold_pct: float = DEFAULT_PASS_THRESHOLD_PCT,
    boost_margin_pct: float = DEFAULT_BOOST_MARGIN_PCT,
    high_score_pct: float = DEFAULT_HIGH_SCORE_PCT,
) -> tuple[list[GradeResult], str]:
    """
    Main entry point for grading one student's paper with adaptive
    strictness. Returns (results, note) where `note` is a short explanation
    shown to the teacher if any class-fairness adjustment was applied
    (empty string if the base grading was used as-is).

    The "correction level" is dynamic in the sense that the *effective*
    strictness for a given paper depends on where that paper lands:
      - within `boost_margin_pct` below the pass mark -> boosted upward
      - at/above `high_score_pct`                      -> re-verified strictly
      - otherwise                                       -> base_difficulty as chosen
    """
    results = grade_paper(student_answers, question_specs, base_difficulty)
    if not enable_dynamic:
        return results, ""

    pct = paper_percentage(results)
    if pct < pass_threshold_pct and pct >= (pass_threshold_pct - boost_margin_pct):
        return _boost_towards_pass(results, pass_threshold_pct)
    if pct >= high_score_pct:
        return _strict_recheck(student_answers, question_specs, results)
    return results, ""