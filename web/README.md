# RAWIFY Web App

Phone-first web frontend for RAWIFY MVP.

## Stack

- Next.js 16 App Router
- TypeScript
- Tailwind CSS
- Prisma + PostgreSQL
- Cookie/session auth (email/password)

## Quick start

1. Create env file:

```bash
cp .env.example .env
```

2. Set your `DATABASE_URL` and optional `ANALYSIS_SERVICE_URL`.

3. Install and set up database:

```bash
npm install
npm run prisma:generate
npm run prisma:migrate -- --name init
npm run db:seed
```

4. Run app:

```bash
npm run dev
```

Open <http://localhost:3000>.

## MVP capabilities

- Auth (signup/signin/signout)
- Dashboard and scan history
- Guided mobile capture flow (required angles + quality gate)
- Analysis trigger route
- Shareable report page
- Marketplace listing creation + browse + detail

## Mobile backend endpoints (scaffolded)

Namespace: `/api/mobile`

- `POST /api/mobile/scans` - create scan
- `POST /api/mobile/scans/:id/images` - upload image metadata
- `POST /api/mobile/scans/:id/analyze` - trigger analysis (mock mode by default)
- `GET /api/mobile/scans/:id/report` - fetch report payload
- `POST /api/mobile/listings` - create marketplace listing
- `GET /api/mobile/listings` - fetch listings

Mock analysis controls:

- `USE_MOCK_ANALYSIS=1` (default) -> returns deterministic mock analysis
- set `USE_MOCK_ANALYSIS=0` to call the Python analysis service via `ANALYSIS_SERVICE_URL`

## Important trust language

Reports include:

> AI-assisted evaluation based on submitted images. This is not an official grade and does not guarantee any grading outcome.
