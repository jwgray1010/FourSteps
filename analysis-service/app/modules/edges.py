from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def analyze_edges(image: np.ndarray) -> Dict[str, object]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    strips = {
        "top": gray[: max(4, h // 30), :],
        "bottom": gray[h - max(4, h // 30) :, :],
        "left": gray[:, : max(4, w // 30)],
        "right": gray[:, w - max(4, w // 30) :],
    }

    detail = {}
    scores = []
    findings = []
    for side, strip in strips.items():
        stddev = float(np.std(strip))
        roughness = max(0.0, min(1.0, (stddev - 8.0) / 40.0))
        score = int(round((1.0 - roughness) * 100))
        detail[side] = score
        scores.append(score)
        if score < 75:
            findings.append(f"Possible {side}-edge whitening or roughness")

    return {
        "score": int(round(float(np.mean(scores)))),
        "details": detail,
        "findings": findings,
    }
