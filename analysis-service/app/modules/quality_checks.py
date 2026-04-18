from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityMetrics:
    blur_score: float
    glare_score: float
    brightness_score: float
    alignment_score: float
    edges_visible: bool
    accepted: bool
    prompts: list[str]


def run_quality_checks(image: np.ndarray) -> QualityMetrics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Higher variance means sharper.
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Ratio of near-white pixels is a practical glare proxy.
    glare_mask = gray >= 245
    glare_score = float(np.mean(glare_mask) * 100.0)

    brightness_mean = float(np.mean(gray))
    brightness_score = max(0.0, min(100.0, (brightness_mean / 255.0) * 100.0))

    edges_visible, alignment_score = _edge_coverage_score(gray)

    prompts: list[str] = []
    if blur_score < 85:
        prompts.append("too blurry")
    if glare_score > 8:
        prompts.append("reduce glare")
    if brightness_score < 25:
        prompts.append("increase lighting")
    if not edges_visible:
        prompts.append("card edges not fully visible")
    if alignment_score < 60:
        prompts.append("center card in frame")

    accepted = (
        blur_score >= 85
        and glare_score <= 8
        and brightness_score >= 25
        and edges_visible
        and alignment_score >= 60
    )

    return QualityMetrics(
        blur_score=round(blur_score, 2),
        glare_score=round(glare_score, 2),
        brightness_score=round(brightness_score, 2),
        alignment_score=round(alignment_score, 2),
        edges_visible=edges_visible,
        accepted=accepted,
        prompts=prompts,
    )


def _edge_coverage_score(gray: np.ndarray) -> tuple[bool, float]:
    h, w = gray.shape
    margin_y = max(2, h // 25)
    margin_x = max(2, w // 25)

    top = gray[:margin_y, :]
    bottom = gray[h - margin_y :, :]
    left = gray[:, :margin_x]
    right = gray[:, w - margin_x :]

    # If border strips are too dark/flat we assume edge visibility is weak.
    strips = [top, bottom, left, right]
    variances = [float(np.var(s)) for s in strips]
    means = [float(np.mean(s)) for s in strips]
    variance_score = min(100.0, (sum(variances) / len(variances)) / 3.0)
    brightness_penalty = max(0.0, 35.0 - min(means))
    alignment_score = max(0.0, min(100.0, variance_score - brightness_penalty))
    edges_visible = alignment_score >= 45.0
    return edges_visible, alignment_score
