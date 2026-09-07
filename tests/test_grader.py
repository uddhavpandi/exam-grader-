import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.grader import QuestionSpec, grade_question, grade_paper, total_marks
from src.parser import split_into_questions


def test_theory_good_answer_easy_vs_hard():
    spec = QuestionSpec(
        qno="1",
        max_marks=5,
        model_answer="Photosynthesis is the process by which green plants use sunlight, "
                      "water and carbon dioxide to produce glucose and release oxygen, "
                      "using chlorophyll in chloroplasts.",
        q_type="theory",
        keywords=["chlorophyll", "sunlight", "carbon dioxide", "glucose", "oxygen"],
    )
    good_answer = ("Plants make food using sunlight, water and carbon dioxide. "
                   "Chlorophyll in the chloroplast absorbs light and the plant "
                   "produces glucose and releases oxygen.")

    easy = grade_question(good_answer, spec, "easy")
    hard = grade_question(good_answer, spec, "hard")
    print("EASY:", easy)
    print("HARD:", hard)
    assert easy.awarded_marks >= hard.awarded_marks
    assert easy.awarded_marks > 0


def test_theory_blank_answer_gets_zero():
    spec = QuestionSpec(qno="2", max_marks=5, model_answer="Newton's second law: F = ma.")
    result = grade_question("", spec, "medium")
    assert result.awarded_marks == 0.0


def test_numeric_within_tolerance():
    spec = QuestionSpec(
        qno="3", max_marks=4, model_answer="Using v=u+at, final velocity = 29.4 m/s",
        q_type="numeric", expected_value=29.4,
    )
    close = grade_question("v = u + at = 29.5 m/s", spec, "medium")
    far = grade_question("v = u + at = 50 m/s", spec, "medium")
    assert close.awarded_marks == 4.0
    assert far.awarded_marks == 0.0


def test_numeric_strictness_affects_grading():
    spec = QuestionSpec(qno="4", max_marks=10, model_answer="", q_type="numeric", expected_value=100.0)
    student = "98"
    easy = grade_question(student, spec, "easy")     # 2% error -> within 10% tol
    hard = grade_question(student, spec, "hard")      # 2% error -> outside 1% tol
    print("NUM EASY:", easy.awarded_marks, "NUM HARD:", hard.awarded_marks)
    assert easy.awarded_marks == 10.0
    assert hard.awarded_marks < 10.0


def test_diagram_question_is_skipped_not_zeroed_silently():
    spec = QuestionSpec(qno="5", max_marks=5, model_answer="", q_type="diagram")
    result = grade_question("some text near the diagram", spec, "medium")
    assert result.method == "diagram-skipped"
    assert "manual review" in result.feedback.lower()


def test_full_paper_totals():
    specs = [
        QuestionSpec(qno="1", max_marks=5, model_answer="Water boils at 100 degrees Celsius at sea level.",
                     keywords=["100", "celsius"]),
        QuestionSpec(qno="2", max_marks=5, model_answer="", q_type="numeric", expected_value=9.8),
    ]
    student_answers = {"1": "Water boils at 100 C at sea level.", "2": "g = 9.8 m/s^2"}
    results = grade_paper(student_answers, specs, "medium")
    awarded, maxm = total_marks(results)
    print("TOTAL:", awarded, "/", maxm)
    assert maxm == 10
    assert awarded > 0


def test_parser_splits_questions():
    raw = """Name: Uddhav
Roll No: 12

Q1) Explain photosynthesis in short.
It is the process plants use to make food.

Q2) What is Newton's first law?
An object stays at rest or in motion unless acted on by a force.
"""
    chunks = split_into_questions(raw)
    assert "1" in chunks and "2" in chunks
    assert "process plants use" in chunks["1"].lower()
    assert "object" in chunks["2"].lower()


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
