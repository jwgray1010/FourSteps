"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { MobileCaptureFlow } from "@/components/capture/mobile-capture-flow";
import type { ImageType, ScanDetailDto } from "@/types/domain";

type Props = {
  scan: ScanDetailDto;
};

export function ScanDetailClient({ scan }: Props) {
  const router = useRouter();
  const [working, setWorking] = useState(false);
  const [message, setMessage] = useState<string>("");
  const [existingTypes, setExistingTypes] = useState<ImageType[]>(
    scan.images.map((img) => img.type),
  );

  const missingRequired = useMemo(() => {
    const required: ImageType[] = [
      "front_straight",
      "back_straight",
      "front_angle_left",
      "front_angle_right",
    ];
    const set = new Set(existingTypes);
    return required.filter((type) => !set.has(type));
  }, [existingTypes]);

  async function handleCaptureUpload(payload: {
    type: ImageType;
    imageDataUrl: string;
    width: number;
    height: number;
  }) {
    setWorking(true);
    setMessage(`Uploading ${payload.type}...`);
    try {
      const response = await fetch(`/api/scans/${scan.id}/images`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = (await response.json()) as {
        error?: string;
        image?: { type: ImageType };
      };
      if (!response.ok) {
        setMessage(data.error ?? "Image upload failed.");
        return;
      }
      if (data.image?.type && !existingTypes.includes(data.image.type)) {
        setExistingTypes((prev) => [...prev, data.image!.type]);
      }
      setMessage(`Saved ${payload.type}.`);
    } finally {
      setWorking(false);
    }
  }

  async function handleRunAnalysis() {
    if (missingRequired.length > 0) {
      setMessage(`Missing required captures: ${missingRequired.join(", ")}`);
      return;
    }

    setWorking(true);
    setMessage("Running analysis...");
    try {
      const response = await fetch(`/api/scans/${scan.id}/analyze`, {
        method: "POST",
      });
      const data = (await response.json()) as { error?: string };
      if (!response.ok) {
        setMessage(data.error ?? "Analysis failed.");
        return;
      }
      setMessage("Analysis complete. Opening report...");
      router.push(`/report/${scan.shareToken}`);
    } finally {
      setWorking(false);
    }
  }

  return (
    <main className="space-y-6">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5">
        <h1 className="text-2xl font-semibold text-zinc-50">{scan.title}</h1>
        <p className="mt-2 text-sm text-zinc-300">
          Guided capture flow for strict AI-assisted verification. Not an official grade.
        </p>
      </section>

      <MobileCaptureFlow
        existingTypes={new Set(existingTypes)}
        onCapture={handleCaptureUpload}
      />

      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5">
        <h2 className="text-lg font-semibold text-zinc-100">Readiness</h2>
        {missingRequired.length === 0 ? (
          <p className="mt-2 text-sm text-emerald-300">All required captures are present.</p>
        ) : (
          <ul className="mt-2 list-disc pl-5 text-sm text-amber-300">
            {missingRequired.map((missing) => (
              <li key={missing}>Missing: {missing}</li>
            ))}
          </ul>
        )}

        <button
          type="button"
          onClick={handleRunAnalysis}
          disabled={working || missingRequired.length > 0}
          className="mt-4 rounded-lg bg-indigo-500 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
        >
          {working ? "Working..." : "Run analysis"}
        </button>

        {message ? <p className="mt-3 text-sm text-zinc-300">{message}</p> : null}
      </section>
    </main>
  );
}
