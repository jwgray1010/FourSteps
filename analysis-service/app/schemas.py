from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class ImageInput(BaseModel):
    type: str
    url: HttpUrl
    width: int = 1200
    height: int = 1800
    blur_score: float = 80.0
    glare_score: float = 6.0
    alignment_score: float = 85.0

    @property
    def image(self):  # pragma: no cover - deterministic placeholder
        import numpy as np

        h = max(200, int(self.height))
        w = max(200, int(self.width))
        return np.full((h, w, 3), 200, dtype=np.uint8)


class AnalyzeRequest(BaseModel):
    scanId: str = Field(min_length=1)
    images: list[ImageInput] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    success: bool
    imageConfidence: int
    overallScore: int
    overallCategory: str
    flags: list[str]
    subscores: dict[str, int]
    details: dict[str, Any]
    disclaimer: str
