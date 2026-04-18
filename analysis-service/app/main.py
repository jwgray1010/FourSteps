from __future__ import annotations

from fastapi import FastAPI

from app.modules.centering import analyze_centering
from app.modules.corners import analyze_corners
from app.modules.detect_card import detect_card_boundary
from app.modules.edges import analyze_edges
from app.modules.quality_checks import run_quality_checks
from app.modules.scoring import score_report
from app.modules.surface import surface_module
from app.schemas import AnalyzeRequest, AnalyzeResponse

DISCLAIMER = (
    "AI-assisted evaluation based on submitted images. "
    "This is not an official grade and does not guarantee any grading outcome."
)

app = FastAPI(title="RAWIFY Analysis Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    img = payload.images[0]
    quality = run_quality_checks(img.image)
    detection = detect_card_boundary(img.width, img.height)
    centering = analyze_centering(img.image)
    corners = analyze_corners()
    edges = analyze_edges(img.image)

    glare_values = [max(0.0, 1.0 - (i.glare_score or 0.5)) for i in payload.images]
    angled_count = len([i for i in payload.images if "angle" in i.type])
    surface_score, surface_details, surface_flags = surface_module(glare_values, angled_count)

    corner_avg = int(round(sum(corners.values()) / max(1, len(corners))))
    edge_avg = int(edges["score"])
    score = score_report(
        centering=centering.score,
        corners=corner_avg,
        edges=edge_avg,
        surface=surface_score,
        image_confidence=int(round(75 if quality.accepted else 62)),
        severe_surface_glare=quality.glare_score > 8.0,
    )

    flags: list[str] = []
    flags.extend([f"Capture quality: {p}" for p in quality.prompts])
    flags.extend(surface_flags)
    flags.extend(edges.get("findings", []))
    flags = list(dict.fromkeys(flags))

    details = {
        "centering": {
            "frontLeftRight": centering.left_right_ratio,
            "frontTopBottom": centering.top_bottom_ratio,
            "confidence": round(centering.confidence, 2),
        },
        "corners": corners,
        "edges": edges["details"],
        "surface": {
            "status": surface_details.status,
            "confidence": round(surface_details.confidence, 2),
        },
        "detection": {
            "confidence": round(detection.confidence, 2),
            "boundary": detection.boundary,
            "message": detection.message,
        },
    }

    return AnalyzeResponse(
        success=True,
        imageConfidence=int(round(75 if quality.accepted else 62)),
        overallScore=score.overall_score,
        overallCategory=score.overall_category,
        flags=flags,
        subscores={
            "centering": centering.score,
            "corners": corner_avg,
            "edges": edge_avg,
            "surface": surface_score,
        },
        details=details,
        disclaimer=DISCLAIMER,
    )
