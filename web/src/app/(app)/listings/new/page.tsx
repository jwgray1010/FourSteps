import { redirect } from "next/navigation";

import { NewListingForm } from "@/components/listings/new-listing-form";
import { getCurrentUser } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export default async function NewListingPage() {
  const user = await getCurrentUser();
  if (!user) {
    redirect("/signin");
  }

  const scans = await prisma.cardScan.findMany({
    where: {
      userId: user.id,
      status: { in: ["completed", "listed"] },
    },
    orderBy: { updatedAt: "desc" },
    select: {
      id: true,
      title: true,
      playerName: true,
      overallCategory: true,
      overallScore: true,
      imageConfidence: true,
    },
  });

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-semibold">Create marketplace listing</h1>
      <p className="text-zinc-400">
        Publish a report-backed listing in RAWIFY&apos;s internal marketplace.
      </p>
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5">
        <NewListingForm scans={scans} />
      </div>
    </div>
  );
}
