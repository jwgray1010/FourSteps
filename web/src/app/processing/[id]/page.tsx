import Link from "next/link";
import { redirect } from "next/navigation";

import { getCurrentUser } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { RAWIFY_REPORT_DISCLAIMER } from "@/lib/constants";

export default async function ProcessingPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const user = await getCurrentUser();
  if (!user) redirect("/signin");

  const { id } = await params;
  const scan = await prisma.cardScan.findFirst({
    where: { id, userId: user.id },
    include: { analysis: true },
  });

  if (!scan) {
    return <main className="p-6 text-zinc-300">Scan not found.</main>;
  }

  return (
    <main className="space-y-4">
      <h1 className="text-2xl font-semibold">Processing report...</h1>
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4 text-sm">
        <p>Status: {scan.status}</p>
        <p>Analysis ready: {scan.analysis ? "yes" : "not yet"}</p>
      </div>
      <p className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-3 text-xs text-zinc-400">
        {RAWIFY_REPORT_DISCLAIMER}
      </p>
      <div className="flex gap-2">
        <Link href={`/scans/${scan.id}`} className="rounded-lg border border-zinc-700 px-3 py-2 text-sm">
          Back to scan
        </Link>
        <Link href={`/report/${scan.shareToken}`} className="rounded-lg bg-indigo-500 px-3 py-2 text-sm font-medium">
          Open report
        </Link>
      </div>
    </main>
  );
}
