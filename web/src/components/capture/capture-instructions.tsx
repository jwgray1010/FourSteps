import { DISCLAIMER_FULL } from "@/lib/constants";

export function CaptureInstructions() {
  const tips = [
    "Use a plain dark background.",
    "Use bright indirect light and avoid harsh shadows.",
    "Remove sleeve/toploader if possible for scanning.",
    "Fill the frame with the card and keep all edges visible.",
    "Capture required straight + angled shots.",
  ];

  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4">
      <h2 className="text-lg font-semibold text-zinc-100">Step A: Capture setup</h2>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-zinc-300">
        {tips.map((tip) => (
          <li key={tip}>{tip}</li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-zinc-500">{DISCLAIMER_FULL}</p>
    </section>
  );
}
