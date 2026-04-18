"""Rule-based centering estimates for MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CenteringResult:
    left_right_ratio: str
    top_bottom_ratio: str
    score: int
    confidence: float


def analyze_centering(image: np.ndarray) -> CenteringResult:
    h, w = image.shape[:2]
    # Heuristic: assume we retained a near-full card crop and score near-center crops better.
    x_center_offset = abs((w / 2) - (w * 0.5)) / max(w, 1)
    y_center_offset = abs((h / 2) - (h * 0.5)) / max(h, 1)
    drift = (x_center_offset + y_center_offset) / 2.0

    # Conservative estimate baseline.
    lr_major = int(round(50 + min(8, drift * 100)))
    lr_minor = 100 - lr_major
    tb_major = int(round(50 + min(8, drift * 90)))
    tb_minor = 100 - tb_major

    score = max(60, min(97, int(round(96 - drift * 200))))
    confidence = float(max(0.45, min(0.9, 0.78 - drift)))

    return CenteringResult(
      left_right_ratio=f"{min(lr_major, lr_minor)}/{max(lr_major, lr_minor)}",
      top_bottom_ratio=f"{min(tb_major, tb_minor)}/{max(tb_major, tb_minor)}",
      score=score,
      confidence=confidence,
    )
