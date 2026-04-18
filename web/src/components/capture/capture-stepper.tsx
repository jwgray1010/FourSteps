"use client";

import { Camera, CheckCircle2, ImagePlus, Sparkles } from "lucide-react";

const STEPS = [
  {
    key: "instructions",
    title: "Instructions",
    description: "Lighting, background, and setup tips",
    icon: Sparkles,
  },
  {
    key: "capture",
    title: "Guided capture",
    description: "Align card in frame and monitor quality",
    icon: Camera,
  },
  {
    key: "upload",
    title: "Upload required shots",
    description: "Front, back, and front angled views",
    icon: ImagePlus,
  },
  {
    key: "review",
    title: "Ready for analysis",
    description: "Pass quality gate and run strict scoring",
    icon: CheckCircle2,
  },
] as const;

type CaptureStep = (typeof STEPS)[number]["key"];

export function CaptureStepper({ current }: { current: CaptureStep }) {
  return (
    <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {STEPS.map((step) => {
        const Icon = step.icon;
        const isActive = step.key === current;
        const isCompleted = STEPS.findIndex((s) => s.key === current) > STEPS.findIndex((s) => s.key === step.key);

        return (
          <li
            key={step.key}
            className={[
              "rounded-xl border p-3",
              isActive
                ? "border-indigo-400 bg-indigo-500/10"
                : isCompleted
                  ? "border-emerald-500/50 bg-emerald-500/10"
                  : "border-zinc-800 bg-zinc-950/80",
            ].join(" ")}
          >
            <div className="mb-2 flex items-center gap-2">
              <Icon size={16} className={isActive ? "text-indigo-300" : "text-zinc-400"} />
              <p className="text-sm font-semibold">{step.title}</p>
            </div>
            <p className="text-xs text-zinc-400">{step.description}</p>
          </li>
        );
      })}
    </ol>
  );
}
