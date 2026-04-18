"""Weighted scoring and strict category capping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ScoreOutput:
    overall_score: int
    overall_category: str
    flags: List[str]


def score_report(
    centering: int,
    corners: int,
    edges: int,
    surface: int,
    image_confidence: int,
    severe_surface_glare: bool,
) -> ScoreOutput:
    weighted = (
        centering * 0.35 + corners * 0.25 + edges * 0.20 + surface * 0.20
    )
    overall = int(round(weighted))
    category = category_from_score(overall)
    flags: List[str] = []

    if image_confidence < 70 and category in {"Gem Candidate", "Strong Raw"}:
        category = "Borderline"
        flags.append("Image confidence below 70: final category capped at Borderline.")

    if severe_surface_glare:
        flags.append("Surface uncertain due to glare in critical area.")

    return ScoreOutput(overall_score=overall, overall_category=category, flags=flags)


def category_from_score(score: int) -> str:
    if score >= 90:
        return "Gem Candidate"
    if score >= 78:
        return "Strong Raw"
    if score >= 65:
        return "Borderline"
    return "Visible Risk"
