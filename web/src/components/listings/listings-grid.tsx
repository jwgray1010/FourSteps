"use client";

import Link from "next/link";

import { formatCategoryLabel } from "@/lib/score";
import { toCurrency } from "@/lib/utils";
import type { PublicListingSummary } from "@/types/domain";

export function ListingsGrid({ listings }: { listings: PublicListingSummary[] }) {
  if (listings.length === 0) {
    return (
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 text-sm text-zinc-300">
        No active listings yet.
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {listings.map((listing) => (
        <article key={listing.id} className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-400">{listing.scan.sport}</p>
            <p className="text-sm font-semibold">{toCurrency(listing.askingPrice)}</p>
          </div>
          <h3 className="mt-1 text-lg font-semibold">{listing.scan.title}</h3>
          <p className="text-sm text-zinc-300">{listing.scan.playerName}</p>
          <p className="mt-2 text-xs text-zinc-400">
            Category:{" "}
            {listing.scan.overallCategory
              ? formatCategoryLabel(listing.scan.overallCategory)
              : "Pending"}
            {" · "}Confidence: {listing.scan.imageConfidence ?? "--"}
          </p>
          <div className="mt-4 flex gap-2">
            <Link
              href={`/report/${listing.scan.shareToken}`}
              className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs"
            >
              Report
            </Link>
            <Link
              href={`/sellers/${listing.seller.username}`}
              className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs"
            >
              Seller
            </Link>
          </div>
        </article>
      ))}
    </div>
  );
}
