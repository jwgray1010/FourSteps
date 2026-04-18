import Link from "next/link";
import { notFound } from "next/navigation";

import { prisma } from "@/lib/prisma";
import { formatCategoryLabel } from "@/lib/score";
import { toCurrency } from "@/lib/utils";

export default async function SellerProfilePage({
  params,
}: {
  params: Promise<{ username: string }>;
}) {
  const { username } = await params;
  const seller = await prisma.user.findUnique({
    where: { username },
    include: {
      listings: {
        where: { status: { in: ["active", "sold"] } },
        orderBy: { createdAt: "desc" },
        include: { cardScan: true },
      },
    },
  });

  if (!seller) notFound();

  return (
    <main className="space-y-6">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5">
        <h1 className="text-2xl font-semibold">@{seller.username}</h1>
        <p className="mt-2 text-sm text-zinc-300">
          Seller profile with report-backed raw card listings.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {seller.listings.map((listing) => (
          <article key={listing.id} className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
            <p className="text-xs text-zinc-400">{listing.cardScan.sport}</p>
            <h2 className="mt-1 text-lg font-semibold">{listing.cardScan.title}</h2>
            <p className="text-sm text-zinc-300">{listing.cardScan.playerName}</p>
            <p className="mt-2 text-sm">{toCurrency(Number(listing.askingPrice))}</p>
            <p className="text-xs text-zinc-400">
              {listing.cardScan.overallCategory
                ? formatCategoryLabel(listing.cardScan.overallCategory)
                : "Pending"}
            </p>
            <Link
              href={`/report/${listing.cardScan.shareToken}`}
              className="mt-3 inline-flex rounded-md border border-zinc-700 px-2 py-1 text-xs"
            >
              View report
            </Link>
          </article>
        ))}
      </section>
    </main>
  );
}
