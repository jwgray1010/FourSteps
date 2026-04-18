from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NormalizedCard:
  image: np.ndarray
  width: int
  height: int


def normalize_card(image: np.ndarray, target_width: int = 750, target_height: int = 1050) -> NormalizedCard:
  normalized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
  return NormalizedCard(image=normalized, width=target_width, height=target_height)
