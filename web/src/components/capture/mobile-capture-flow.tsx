"use client";

import { useMemo, useState } from "react";

import type { ImageType } from "@/types/domain";
import { CAPTURE_IMAGE_TYPES, CAPTURE_STEPS, RAWIFY_REPORT_DISCLAIMER } from "@/lib/constants";
import { qualityGateFromClientImage } from "@/lib/quality-gate";

type CapturePayload = {
  type: ImageType;
  imageDataUrl: string;
  width: number;
  height: number;
};

type Props = {
  existingTypes: Set<ImageType>;
  onCapture: (payload: CapturePayload) => Promise<void>;
};

type GateState = {
  passed: boolean;
  prompts: string[];
};

export function MobileCaptureFlow({ existingTypes, onCapture }: Props) {
  const [activeType, setActiveType] = useState<ImageType>("front_straight");
  const [busy, setBusy] = useState(false);
  const [gate, setGate] = useState<GateState | null>(null);

  const requiredDone = useMemo(
    () =>
      ["front_straight", "back_straight", "front_angle_left", "front_angle_right"].every((t) =>
        existingTypes.has(t as ImageType),
      ),
    [existingTypes],
  );

  async function simulateCapture() {
    const probe = qualityGateFromClientImage({
      width: 1200,
      height: 1800,
      brightness: 0.65,
      blurScore: 0.8,
      glareScore: 0.2,
      alignmentScore: 0.85,
      edgeCoverage: 0.9,
      perspectiveDistortion: 0.85,
      sleeveReflectionScore: 0.15,
    });
    setGate({ passed: probe.passed, prompts: probe.prompts });
    if (!probe.passed) {
      return;
    }

    setBusy(true);
    try {
      await onCapture({
        type: activeType,
        imageDataUrl: `https://storage.rawify.app/demo/${activeType}.jpg`,
        width: 1200,
        height: 1800,
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-4">
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-4">
        <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Step A: Setup checklist</p>
        <ul className="mt-2 space-y-1 text-sm text-zinc-300">
          {CAPTURE_STEPS.map((step) => (
            <li key={step}>- {step}</li>
          ))}
        </ul>
      </div>

      <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-4">
        <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Step B/C: Guided capture</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {CAPTURE_IMAGE_TYPES.map((type) => {
            const done = existingTypes.has(type);
            return (
              <button
                key={type}
                type="button"
                onClick={() => setActiveType(type)}
                className={`rounded-full border px-3 py-1 text-xs ${
                  activeType === type
                    ? "border-indigo-400 bg-indigo-500/20 text-indigo-100"
                    : "border-zinc-700 text-zinc-300"
                }`}
              >
                {type}
                {done ? " ✓" : ""}
              </button>
            );
          })}
        </div>

        <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-sm text-zinc-300">Active: {activeType}</p>
          <div className="mt-3 aspect-[3/4] max-w-[260px] rounded-lg border border-dashed border-indigo-300/70 bg-zinc-950">
            <div className="flex h-full items-center justify-center text-xs text-zinc-500">
              Camera overlay target
            </div>
          </div>

          <button
            type="button"
            onClick={simulateCapture}
            disabled={busy}
            className="mt-3 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs hover:bg-zinc-800 disabled:opacity-60"
          >
            {busy ? "Uploading..." : "Simulate capture + upload"}
          </button>

          {gate ? (
            <div className="mt-3 text-xs">
              <p className={gate.passed ? "text-emerald-300" : "text-amber-300"}>
                Quality gate: {gate.passed ? "accepted" : "rejected - rescan required"}
              </p>
              {gate.prompts.length > 0 ? (
                <ul className="mt-1 list-disc pl-4 text-zinc-300">
                  {gate.prompts.map((prompt) => (
                    <li key={prompt}>{prompt}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>

      <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-4 text-xs text-zinc-400">
        <p>Required captures complete: {requiredDone ? "yes" : "not yet"}</p>
        <p className="mt-1">{RAWIFY_REPORT_DISCLAIMER}</p>
      </div>
    </section>
  );
}
