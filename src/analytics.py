"""
analytics.py
------------
Class-level analytics: fit a bell curve (normal distribution) to all
students' percentage scores for this paper, and apply the
"top-of-the-curve gets full marks" rule.

Rationale: a single student's raw % doesn't say whether they genuinely
did the best in the class. Fitting a normal distribution to the whole
class's totals and plotting it lets the teacher see, at a glance, who is
actually at the top of the pack — and it's shown in the app so the
teacher can sanity-check grading progress across the batch, not just
per-student.

Students whose score sits at or beyond the 95th percentile of the class
distribution are rounded up to full marks: they are already ahead of
essentially the whole class, so further nitpicking their wording doesn't
add real information. This needs a handful of students with genuine
spread to mean anything, so it only kicks in with >= 3 students and a
non-trivial standard deviation.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # headless rendering for Streamlit
import matplotlib.pyplot as plt
import numpy as np

TOP_OF_CURVE_PERCENTILE = 95.0
MIN_STUDENTS_FOR_CURVE = 3


@dataclass
class StudentTotal:
    name: str
    awarded: float
    max_marks: float

    @property
    def pct(self) -> float:
        return (self.awarded / self.max_marks * 100.0) if self.max_marks else 0.0


def _normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 1e-9:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def percentile_rank(value: float, mean: float, std: float) -> float:
    """Approx percentile (0-100) of `value` under a fitted normal(mean, std)."""
    return _normal_cdf(value, mean, std) * 100.0


def fit_bell_curve(totals: list[StudentTotal]) -> tuple[float, float]:
    """Mean & population std-dev of the class's percentage scores."""
    pcts = [t.pct for t in totals]
    if not pcts:
        return 0.0, 0.0
    if len(pcts) < 2:
        return pcts[0], 0.0
    return statistics.mean(pcts), statistics.pstdev(pcts)


def apply_top_of_curve_bonus(
    totals: list[StudentTotal],
    percentile_cutoff: float = TOP_OF_CURVE_PERCENTILE,
) -> dict[str, dict]:
    """
    Bump every student at/above `percentile_cutoff` of the class's normal
    fit up to full marks (mutates `totals` in place).
    Returns {student_name: {"percentile": float, "boosted": bool}}.
    """
    mean, std = fit_bell_curve(totals)
    info: dict[str, dict] = {}
    enough_data = len(totals) >= MIN_STUDENTS_FOR_CURVE and std > 1e-5
    for t in totals:
        pct_rank = percentile_rank(t.pct, mean, std) if enough_data else 0.0
        boosted = False
        if enough_data and pct_rank >= percentile_cutoff and t.awarded < t.max_marks:
            t.awarded = t.max_marks
            boosted = True
        info[t.name] = {"percentile": round(pct_rank, 1), "boosted": boosted}
    return info


def plot_bell_curve(totals: list[StudentTotal], pass_threshold_pct: float | None = None):
    """Return a matplotlib Figure: fitted normal curve with each student marked on it."""
    mean, std = fit_bell_curve(totals)
    # A degenerate/zero spread still needs *some* width to draw a curve.
    std_plot = std if std > 1e-5 else max(mean * 0.1, 5.0)

    lo = max(0.0, mean - 4 * std_plot)
    hi = min(100.0, mean + 4 * std_plot)
    if hi <= lo:
        lo, hi = 0.0, 100.0
    xs = np.linspace(lo, hi, 300)
    ys = (1.0 / (std_plot * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((xs - mean) / std_plot) ** 2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, color="#4C72B0", lw=2, label="Class distribution (fitted)")
    ax.fill_between(xs, ys, color="#4C72B0", alpha=0.12)

    if pass_threshold_pct is not None:
        ax.axvline(pass_threshold_pct, color="#C44E52", ls="--", lw=1.5,
                   label=f"Pass mark ({pass_threshold_pct:.0f}%)")

    cutoff_x = mean + 1.645 * std_plot  # ~95th percentile of a normal distribution
    if len(totals) >= MIN_STUDENTS_FOR_CURVE and std > 1e-5:
        ax.axvline(cutoff_x, color="#55A868", ls=":", lw=1.5, label="~95th percentile (auto full marks)")

    for t in totals:
        ax.plot([t.pct], [0], marker="v", markersize=9, color="#333333", clip_on=False, zorder=5)
        ax.annotate(t.name, (t.pct, 0), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, rotation=45)

    ax.set_xlabel("Score (%)")
    ax.set_ylabel("Density")
    ax.set_title("Class score distribution (bell curve)")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig