"""Card boundary detection placeholder module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DetectionResult:
    confidence: float
    boundary: List[Tuple[int, int]]
    message: str


def detect_card_boundary(width: int, height: int) -> DetectionResult:
    """Return a conservative rectangular estimate for V1 scaffolding."""
    if width <= 0 or height <= 0:
        return DetectionResult(0.0, [], "invalid_dimensions")

    margin_x = max(int(width * 0.08), 6)
    margin_y = max(int(height * 0.08), 6)
    boundary = [
        (margin_x, margin_y),
        (width - margin_x, margin_y),
        (width - margin_x, height - margin_y),
        (margin_x, height - margin_y),
    ]
    confidence = 0.78
    return DetectionResult(confidence=confidence, boundary=boundary, message="detected_rect")
