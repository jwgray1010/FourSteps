from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class SurfaceResult:
    status: str
    confidence: float


def surface_module(glare_values: list[float], angled_count: int) -> tuple[int, SurfaceResult, list[str]]:
    avg_glare = mean(glare_values) if glare_values else 0.65
    flags: list[str] = []

    if angled_count < 2:
        flags.append("Surface uncertain: missing required angled captures.")
        score = 58
        status = "Surface uncertain due to missing angled captures."
        confidence = 0.5
    elif avg_glare < 0.42:
        flags.append("Surface uncertain due to glare.")
        score = 62
        status = "Surface uncertain due to glare."
        confidence = 0.56
    else:
        score = int(70 + avg_glare * 24)
        status = (
            "No major defects detected"
            if score >= 80
            else "Possible faint line or subtle surface concern."
        )
        confidence = min(0.91, 0.62 + avg_glare * 0.3)
        if score < 80:
            flags.append("Possible faint surface line detected.")

    return score, SurfaceResult(status=status, confidence=confidence), flags
