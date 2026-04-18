import { RAWIFY_TRUST_COPY } from "@/lib/constants";
import { categoryDescription, formatCategoryLabel } from "@/lib/score";
import type { PublicReportPayload } from "@/types/domain";

type Props = {
  report: PublicReportPayload;
};

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-zinc-300">
        <span>{label}</span>
        <span>{value}</span>
      </div>
      <div className="h-2 rounded-full bg-zinc-800">
        <div
          className="h-2 rounded-full bg-emerald-400"
          style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        />
      </div>
    </div>
  );
}

export function PublicReport({ report }: Props) {
  const category = report.overallCategory ?? "visible_risk";
  return (
    <div className="space-y-6">
      <header className="rounded-2xl border border-zinc-800 bg-zinc-950 p-5 sm:p-6">
        <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">RAWIFY Report</p>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100">{report.title}</h1>
        <div className="mt-4 grid gap-3 text-sm text-zinc-300 sm:grid-cols-2 lg:grid-cols-4">
          <p>
            <span className="text-zinc-500">Sport</span>
            <br />
            {report.sport}
          </p>
          <p>
            <span className="text-zinc-500">Player</span>
            <br />
            {report.playerName || "Unknown"}
          </p>
          <p>
            <span className="text-zinc-500">Scan date</span>
            <br />
            {new Date(report.scanDate).toLocaleDateString()}
          </p>
          <p>
            <span className="text-zinc-500">Image confidence</span>
            <br />
            {report.imageConfidence ?? "N/A"}
          </p>
        </div>
      </header>

      <section className="grid gap-4 sm:grid-cols-2">
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950 p-5">
          <p className="text-xs uppercase tracking-[0.15em] text-zinc-500">Overall label</p>
          <h2 className="mt-2 text-2xl font-semibold text-zinc-100">
            {formatCategoryLabel(category)}
          </h2>
          <p className="mt-2 text-sm text-zinc-400">{categoryDescription(category)}</p>
          <p className="mt-4 text-sm text-zinc-300">Overall score: {report.overallScore ?? "N/A"}</p>
        </article>
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950 p-5">
          <p className="text-xs uppercase tracking-[0.15em] text-zinc-500">Trust notes</p>
          <ul className="mt-2 space-y-2 text-sm text-zinc-300">
            {RAWIFY_TRUST_COPY.map((line) => (
              <li key={line}>- {line}</li>
            ))}
          </ul>
        </article>
      </section>

      {report.subscores ? (
        <section className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-950 p-5 sm:p-6">
          <h3 className="text-lg font-semibold text-zinc-100">Sub-scores</h3>
          <div className="grid gap-4 sm:grid-cols-2">
            <ScoreBar label="Centering" value={report.subscores.centering} />
            <ScoreBar label="Corners" value={report.subscores.corners} />
            <ScoreBar label="Edges" value={report.subscores.edges} />
            <ScoreBar label="Surface" value={report.subscores.surface} />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <h4 className="font-medium text-zinc-200">Findings</h4>
              <ul className="mt-2 space-y-2 text-sm text-zinc-300">
                {(report.flags ?? []).map((flag) => (
                  <li key={flag}>- {flag}</li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-zinc-200">Uncertainty disclosure</h4>
              <p className="mt-2 text-sm text-zinc-400">
                RAWIFY rejects low-quality photos and applies strict confidence penalties.
              </p>
            </div>
          </div>
        </section>
      ) : null}

      <section className="space-y-3 rounded-2xl border border-zinc-800 bg-zinc-950 p-5 sm:p-6">
        <h3 className="text-lg font-semibold text-zinc-100">Submitted images</h3>
        <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
          {report.images.map((image) => (
            <div key={image.id} className="rounded-xl border border-zinc-800 bg-zinc-900 p-3">
              <p className="text-xs uppercase tracking-[0.12em] text-zinc-500">{image.type}</p>
              <div className="mt-2 aspect-[3/4] rounded-lg border border-dashed border-zinc-700 bg-zinc-950 p-2 text-xs text-zinc-500">
                {image.originalUrl}
              </div>
              <p className="mt-2 text-xs text-zinc-500">
                Quality gate: {image.accepted ? "accepted" : "rejected/warned"}
              </p>
            </div>
          ))}
        </div>
      </section>

      <footer className="rounded-2xl border border-amber-800/70 bg-amber-950/30 p-5 text-sm text-amber-100">
        <p className="font-semibold">Disclaimer</p>
        <p className="mt-2">{report.disclaimer}</p>
      </footer>
    </div>
  );
}
