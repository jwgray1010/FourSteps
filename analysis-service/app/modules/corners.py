from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CornerResult:
    score: int
    details: dict[str, int]
    flags: list[str]


def analyze_corners(image: np.ndarray) -> CornerResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    patch_h = max(8, h // 12)
    patch_w = max(8, w // 12)

    patches = {
        "topLeft": gray[:patch_h, :patch_w],
        "topRight": gray[:patch_h, w - patch_w :],
        "bottomLeft": gray[h - patch_h :, :patch_w],
        "bottomRight": gray[h - patch_h :, w - patch_w :],
    }

    details: dict[str, int] = {}
    flags: list[str] = []
    for label, patch in patches.items():
        stddev = float(np.std(patch))
        bright_ratio = float(np.mean(patch > 220))
        whitening_penalty = bright_ratio * 28.0
        roughness_penalty = max(0.0, (stddev - 26.0) * 0.9)
        score = int(round(max(45.0, min(99.0, 96.0 - whitening_penalty - roughness_penalty))))
        details[label] = score
        if score < 74:
            flags.append(f"Possible corner wear at {label}")

    avg_score = int(round(sum(details.values()) / max(1, len(details))))
    return CornerResult(score=avg_score, details=details, flags=flags)
