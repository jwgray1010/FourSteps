# RAWIFY Analysis Service

FastAPI-based analysis stub for RAWIFY MVP.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

## Endpoint

- `POST /analyze`

Accepts card metadata and image payloads, returns strict JSON with:
- image confidence
- subscores (centering/corners/edges/surface)
- overall score/category
- findings flags
- mandatory disclaimer

> AI-assisted evaluation based on submitted images. This is not an official grade and does not guarantee any grading outcome.
