# RAWIFY MVP (Monorepo Scaffold)

RAWIFY is a mobile-first AI-assisted verification experience for **raw** sports cards.

> AI-assisted evaluation based on submitted images. This is not an official grade and does not guarantee any grading outcome.

## Folder structure

```text
/workspace
├── web/                  # Next.js App Router MVP
│   ├── prisma/
│   │   ├── schema.prisma
│   │   └── seed.ts
│   └── src/
│       ├── app/
│       │   ├── (auth)/...
│       │   ├── (app)/...
│       │   ├── api/...
│       │   ├── report/[id]/page.tsx
│       │   └── sellers/[username]/page.tsx
│       ├── components/
│       │   ├── capture/
│       │   ├── listings/
│       │   ├── report/
│       │   └── layout/
│       ├── lib/
│       └── types/
├── analysis-service/      # FastAPI modular analysis service
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       └── modules/
│           ├── detect_card.py
│           ├── normalize_card.py
│           ├── quality_checks.py
│           ├── centering.py
│           ├── corners.py
│           ├── edges.py
│           ├── surface.py
│           └── scoring.py
└── .env.rawify.example
```

## MVP features scaffolded

- Auth (email/password)
- Dashboard + scan list
- Guided mobile-first capture flow
- Client-side image quality gate (blur/glare/lighting/alignment heuristics)
- Scan image upload metadata endpoint
- Analyze endpoint (calls Python service if configured; otherwise strict deterministic stub)
- Report generation + shareable public report page
- Marketplace create/browse/detail + seller profile page
- Prisma schema + seed demo data

## Legal/trust copy included

- “AI-assisted verification”
- “Not an official grade”
- “Results depend on image quality”
- “No guarantee of third-party grading outcome”

## Run instructions

### 1) Web app

```bash
cd /workspace/web
cp .env.example .env
```

Populate `.env`:

- `DATABASE_URL` (PostgreSQL)
- `ANALYSIS_SERVICE_URL` (optional; defaults to `http://127.0.0.1:8000`)

Then:

```bash
npm install
npm run prisma:generate
npm run prisma:migrate -- --name init_rawify
npm run db:seed
npm run dev
```

Open: `http://localhost:3000`

### 2) Analysis service

```bash
cd /workspace/analysis-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## API endpoints included

### Web route handlers

- `POST /api/auth/signup`
- `POST /api/auth/signin`
- `POST /api/auth/signout`
- `POST /api/scans`
- `GET /api/scans/:id`
- `POST /api/scans/:id/images`
- `POST /api/scans/:id/analyze`
- `GET /api/scans/:id/report`
- `POST /api/listings`
- `GET /api/listings`
- `GET /api/listings/:id`

### Analysis service

- `GET /health`
- `POST /analyze`

## Notes

- Architecture is intentionally modular so deterministic CV modules can be replaced by ML components later.
- Storage URLs are stubbed as object-like paths (`rawify://...`) for MVP speed.
- This is a strict-first implementation; quality failures are surfaced instead of optimistic grading.
