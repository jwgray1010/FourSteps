"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

type ScanOption = {
  id: string;
  title: string;
  playerName: string;
  overallCategory: string | null;
};

type Props = {
  scans: ScanOption[];
  initialScanId?: string;
};

export function NewListingForm({ scans, initialScanId }: Props) {
  const router = useRouter();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!scans.length) {
    return (
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 text-sm text-zinc-300">
        You need at least one completed scan before creating a listing.
      </div>
    );
  }

  return (
    <form
      className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5"
      onSubmit={async (event) => {
        event.preventDefault();
        const formData = new FormData(event.currentTarget);
        setSubmitting(true);
        setError(null);

        try {
          const response = await fetch("/api/listings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              cardScanId: formData.get("cardScanId"),
              askingPrice: Number(formData.get("askingPrice")),
              description: formData.get("description"),
              status: "active",
            }),
          });

          if (!response.ok) {
            const payload = (await response.json()) as { error?: string };
            throw new Error(payload.error ?? "Unable to create listing.");
          }

          router.push("/marketplace");
          router.refresh();
        } catch (err) {
          setError(err instanceof Error ? err.message : "Unable to create listing.");
        } finally {
          setSubmitting(false);
        }
      }}
    >
      <div className="space-y-2">
        <label className="text-sm text-zinc-300" htmlFor="cardScanId">
          Scan
        </label>
        <select
          id="cardScanId"
          name="cardScanId"
          className="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100"
          defaultValue={initialScanId ?? scans[0]?.id}
        >
          {scans.map((scan) => (
            <option key={scan.id} value={scan.id}>
              {scan.title} - {scan.playerName} ({scan.overallCategory ?? "not categorized"})
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-2">
        <label className="text-sm text-zinc-300" htmlFor="askingPrice">
          Asking price (USD)
        </label>
        <input
          id="askingPrice"
          name="askingPrice"
          type="number"
          min={1}
          step={1}
          required
          className="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100"
          placeholder="125"
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm text-zinc-300" htmlFor="description">
          Description
        </label>
        <textarea
          id="description"
          name="description"
          required
          rows={4}
          className="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100"
          placeholder="Add shipping terms, condition notes, and payment preferences."
        />
      </div>

      {error ? <p className="text-sm text-rose-400">{error}</p> : null}

      <button
        type="submit"
        disabled={submitting}
        className="rounded-full bg-cyan-500 px-5 py-2 text-sm font-semibold text-zinc-950 disabled:opacity-60"
      >
        {submitting ? "Publishing..." : "Create listing"}
      </button>
    </form>
  );
}
