"""Centering-only grading using OpenAI Vision via Responses API."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


ALLOWED_REJECT_REASONS = {
    "missing_front",
    "missing_back",
    "cannot_judge_centering",
    "front_centering_bad",
    "back_centering_bad",
    "miscut_or_tilt",
}

ALLOWED_LANE = {"psa10_centering_ok", "reject"}
ALLOWED_CENTERING_CALL = {"ok", "bad", "cannot_judge"}


SYSTEM_PROMPT = """You are a strict PSA 10 centering pre-screen grader for raw trading cards.

TASK:
- Evaluate ONLY centering on front and back images.
- Do not evaluate corners, edges, surface, color, or print defects.
- Use conservative thresholds:
  * Front centering must clearly beat PSA 10 tolerance (55/45 max). Treat borderline as FAIL.
  * Back centering must clearly beat PSA 10 tolerance (75/25 max). Treat borderline as FAIL.
- If image perspective is angled, card is tilted, edges are cropped, or the card is not flat enough to judge centering, reject with reason "miscut_or_tilt".
- You MUST identify both a front and back image from the provided set.
  * If no confident front image: reject reason "missing_front".
  * If no confident back image: reject reason "missing_back".
- For borderless designs, estimate centering using anchor points and negative space:
  * text boxes, logos, serial-number box, color blocks, frames, and mirrored design spacing.
  * if still unreliable, reject with "cannot_judge_centering".

OUTPUT:
- Return JSON only, no markdown.
- grade_lane is "psa10_centering_ok" or "reject".
- If grade_lane is reject, reject_reason MUST be one of the allowed reject reasons.
- If grade_lane is psa10_centering_ok, reject_reason must be null.
"""


JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "grade_lane": {"type": "string", "enum": ["psa10_centering_ok", "reject"]},
        "reject_reason": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "string",
                    "enum": sorted(ALLOWED_REJECT_REASONS),
                },
            ]
        },
        "centering_front_lr": {
            "type": "string",
            "enum": ["ok", "bad", "cannot_judge"],
        },
        "centering_front_tb": {
            "type": "string",
            "enum": ["ok", "bad", "cannot_judge"],
        },
        "centering_back_lr": {"type": "string", "enum": ["ok", "bad", "cannot_judge"]},
        "centering_back_tb": {"type": "string", "enum": ["ok", "bad", "cannot_judge"]},
        "est_front_lr_ratio": {"type": "string"},
        "est_front_tb_ratio": {"type": "string"},
        "est_back_lr_ratio": {"type": "string"},
        "est_back_tb_ratio": {"type": "string"},
        "front_image_index": {"type": "integer"},
        "back_image_index": {"type": "integer"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": [
        "grade_lane",
        "reject_reason",
        "centering_front_lr",
        "centering_front_tb",
        "centering_back_lr",
        "centering_back_tb",
        "est_front_lr_ratio",
        "est_front_tb_ratio",
        "est_back_lr_ratio",
        "est_back_tb_ratio",
        "front_image_index",
        "back_image_index",
        "confidence",
        "notes",
    ],
}


class PhotoGrader:
    """Grades card centering only from listing photos."""

    def __init__(
        self,
        model: str,
        require_two_images: bool = True,
        api_key: Optional[str] = None,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        self.model = model
        self.require_two_images = require_two_images
        self.client = openai_client or OpenAI(api_key=api_key)

    def grade_centering(
        self,
        item_id: str,
        title: str,
        image_urls: List[str],
    ) -> Dict[str, Any]:
        """Grade centering and return normalized JSON-safe output."""
        cleaned_images = _dedupe_http_urls(image_urls)
        if self.require_two_images and len(cleaned_images) < 2:
            if not cleaned_images:
                return _reject_template(
                    reason="missing_front",
                    notes="Listing has no usable images.",
                    confidence=0.0,
                )
            return _reject_template(
                reason="missing_back",
                notes="Listing has fewer than 2 images; cannot verify both sides.",
                confidence=0.05,
            )

        # Keep prompt size practical while still supporting multiple listing photos.
        image_inputs = cleaned_images[:8]
        prompt_lines = [
            f"Item ID: {item_id}",
            f"Listing title: {title or ''}",
            f"Image count provided: {len(image_inputs)}",
            (
                "Decide if this listing should be kept strictly for centering. "
                "Require both front and back card images."
            ),
            (
                "Pass only when front and back are clearly inside PSA 10 centering "
                "tolerance and not borderline."
            ),
        ]

        user_content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": "\n".join(prompt_lines)}
        ]
        for idx, img_url in enumerate(image_inputs):
            user_content.append(
                {
                    "type": "input_text",
                    "text": f"Image index {idx}",
                }
            )
            user_content.append({"type": "input_image", "image_url": img_url})

        try:
            base_input = [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {"role": "user", "content": user_content},
            ]
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=base_input,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "centering_grade",
                            "schema": JSON_SCHEMA,
                            "strict": True,
                        }
                    },
                    temperature=0,
                    max_output_tokens=350,
                )
            except Exception:
                # Compatibility fallback for SDK/model combinations that ignore json_schema.
                response = self.client.responses.create(
                    model=self.model,
                    input=base_input,
                    temperature=0,
                    max_output_tokens=350,
                )
            raw_text = getattr(response, "output_text", "") or _extract_text_from_response(
                response
            )
            parsed = _safe_parse_json(raw_text)
            normalized = _normalize(parsed)
            return normalized
        except Exception as exc:  # noqa: BLE001
            return _reject_template(
                reason="cannot_judge_centering",
                notes=f"OpenAI grading error: {exc}",
                confidence=0.0,
            )


def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {}

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Last-resort extraction if model wrapped JSON in extra text.
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def _dedupe_http_urls(image_urls: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for raw in image_urls:
        url = str(raw).strip()
        if not url.lower().startswith("http"):
            continue
        if url in seen:
            continue
        deduped.append(url)
        seen.add(url)
    return deduped


def _normalize(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return _reject_template(
            reason="cannot_judge_centering",
            notes="Model response was not valid JSON object.",
            confidence=0.0,
        )

    lane = candidate.get("grade_lane")
    if lane not in ALLOWED_LANE:
        lane = "reject"

    front_lr = _as_centering_call(candidate.get("centering_front_lr"))
    front_tb = _as_centering_call(candidate.get("centering_front_tb"))
    back_lr = _as_centering_call(candidate.get("centering_back_lr"))
    back_tb = _as_centering_call(candidate.get("centering_back_tb"))

    result: Dict[str, Any] = {
        "grade_lane": lane,
        "reject_reason": candidate.get("reject_reason"),
        "centering_front_lr": front_lr,
        "centering_front_tb": front_tb,
        "centering_back_lr": back_lr,
        "centering_back_tb": back_tb,
        "est_front_lr_ratio": _ratio_string(candidate.get("est_front_lr_ratio")),
        "est_front_tb_ratio": _ratio_string(candidate.get("est_front_tb_ratio")),
        "est_back_lr_ratio": _ratio_string(candidate.get("est_back_lr_ratio")),
        "est_back_tb_ratio": _ratio_string(candidate.get("est_back_tb_ratio")),
        "front_image_index": _as_int(candidate.get("front_image_index"), default=-1),
        "back_image_index": _as_int(candidate.get("back_image_index"), default=-1),
        "confidence": _as_confidence(candidate.get("confidence")),
        "notes": str(candidate.get("notes", ""))[:1000],
    }

    reject_reason = result["reject_reason"]
    if reject_reason not in ALLOWED_REJECT_REASONS:
        reject_reason = None

    # Enforce lane/reason consistency and prevent unknown reject reasons.
    if result["grade_lane"] == "psa10_centering_ok":
        front_bad = "bad" in (result["centering_front_lr"], result["centering_front_tb"])
        back_bad = "bad" in (result["centering_back_lr"], result["centering_back_tb"])
        has_cannot = any(
            v == "cannot_judge"
            for v in (
                result["centering_front_lr"],
                result["centering_front_tb"],
                result["centering_back_lr"],
                result["centering_back_tb"],
            )
        )
        if front_bad:
            result["grade_lane"] = "reject"
            reject_reason = "front_centering_bad"
        elif back_bad:
            result["grade_lane"] = "reject"
            reject_reason = "back_centering_bad"
        elif has_cannot:
            result["grade_lane"] = "reject"
            reject_reason = "cannot_judge_centering"
        else:
            reject_reason = None
    else:
        if reject_reason is None:
            if "bad" in (result["centering_front_lr"], result["centering_front_tb"]):
                reject_reason = "front_centering_bad"
            elif "bad" in (result["centering_back_lr"], result["centering_back_tb"]):
                reject_reason = "back_centering_bad"
            else:
                reject_reason = "cannot_judge_centering"

    result["reject_reason"] = reject_reason
    return result


def _reject_template(reason: str, notes: str, confidence: float) -> Dict[str, Any]:
    safe_reason = reason if reason in ALLOWED_REJECT_REASONS else "cannot_judge_centering"
    return {
        "grade_lane": "reject",
        "reject_reason": safe_reason,
        "centering_front_lr": "cannot_judge",
        "centering_front_tb": "cannot_judge",
        "centering_back_lr": "cannot_judge",
        "centering_back_tb": "cannot_judge",
        "est_front_lr_ratio": "cannot_judge",
        "est_front_tb_ratio": "cannot_judge",
        "est_back_lr_ratio": "cannot_judge",
        "est_back_tb_ratio": "cannot_judge",
        "front_image_index": -1,
        "back_image_index": -1,
        "confidence": _as_confidence(confidence),
        "notes": str(notes)[:1000],
    }


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _as_centering_call(value: Any) -> str:
    val = str(value).strip().lower()
    if val in ALLOWED_CENTERING_CALL:
        return val
    return "cannot_judge"


def _extract_text_from_response(response: Any) -> str:
    try:
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return ""
        collected: List[str] = []
        for block in output:
            contents = getattr(block, "content", None)
            if not isinstance(contents, list):
                continue
            for part in contents:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    collected.append(text)
        return "\n".join(collected)
    except Exception:  # noqa: BLE001
        return ""


def _ratio_string(value: Any) -> str:
    if value is None:
        return "cannot_judge"
    text = str(value).strip()
    if not text:
        return "cannot_judge"
    return text[:40]
