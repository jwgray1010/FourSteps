import Link from "next/link";
import { redirect } from "next/navigation";

import { getCurrentUser } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { formatCategory } from "@/lib/score";

export default async function DashboardPage() {
  const user = await getCurrentUser();
  if (!user) {
    redirect("/signin");
  }

  const scans = await prisma.cardScan.findMany({
    where: { userId: user.id },
    orderBy: { updatedAt: "desc" },
    include: { listing: true },
    take: 12,
  });

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5">
        <h1 className="text-2xl font-semibold text-zinc-50">Welcome back, {user.username}</h1>
        <p className="mt-2 text-sm text-zinc-300">
          AI-assisted verification workflow for raw cards. Not an official grade.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Link href="/scans/new" className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium">
            Create new scan
          </Link>
          <Link
            href="/marketplace"
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium"
          >
            Browse marketplace
          </Link>
          <Link
            href="/listings/new"
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium"
          >
            Create listing
          </Link>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5">
        <h2 className="text-lg font-semibold text-zinc-100">Recent scans</h2>
        <div className="mt-4 space-y-3">
          {scans.map((scan) => (
            <article key={scan.id} className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-4">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <h3 className="font-medium text-zinc-50">{scan.title}</h3>
                  <p className="text-xs text-zinc-400">
                    {scan.sport} • {scan.playerName} • {scan.status}
                  </p>
                </div>
                <div className="text-right text-xs text-zinc-300">
                  <div>{scan.overallScore ?? "--"} score</div>
                  <div>{scan.imageConfidence ?? "--"} image confidence</div>
                  <div>{scan.overallCategory ? formatCategory(scan.overallCategory) : "Pending"}</div>
                </div>
              </div>
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                <Link className="rounded-md border border-zinc-700 px-2 py-1" href={`/scans/${scan.id}`}>
                  Open scan
                </Link>
                <Link
                  className="rounded-md border border-zinc-700 px-2 py-1"
                  href={`/report/${scan.shareToken}`}
                >
                  Public report
                </Link>
                {!scan.listing ? (
                  <Link
                    className="rounded-md border border-zinc-700 px-2 py-1"
                    href={`/listings/new?scanId=${scan.id}`}
                  >
                    Create listing
                  </Link>
                ) : null}
              </div>
            </article>
          ))}
          {scans.length === 0 ? (
            <p className="text-sm text-zinc-400">No scans yet. Start by creating a new scan.</p>
          ) : null}
        </div>
      </section>
    </div>
  );
}
