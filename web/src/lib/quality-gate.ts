import type { ImageQualityResult } from "@/types/domain";

type QualityInput = {
  blurScore?: number;
  glareScore?: number;
  alignmentScore?: number;
  width?: number;
  height?: number;
  accepted?: boolean;
};

const THRESHOLDS = {
  blur: 55,
  glare: 45,
  alignment: 60,
  minWidth: 900,
  minHeight: 1200,
};

function clampScore(value: number | undefined, fallback: number): number {
  const v = Number.isFinite(value) ? Number(value) : fallback;
  return Math.max(0, Math.min(100, v));
}

export function runQualityGate(input: QualityInput): ImageQualityResult {
  const blurScore = clampScore(input.blurScore, 70);
  const glareScore = clampScore(input.glareScore, 70);
  const alignmentScore = clampScore(input.alignmentScore, 70);
  const width = input.width ?? THRESHOLDS.minWidth;
  const height = input.height ?? THRESHOLDS.minHeight;

  const prompts: string[] = [];
  if (blurScore < THRESHOLDS.blur) prompts.push("Too blurry");
  if (glareScore < THRESHOLDS.glare) prompts.push("Reduce glare");
  if (alignmentScore < THRESHOLDS.alignment) prompts.push("Center card in frame");
  if (width < THRESHOLDS.minWidth || height < THRESHOLDS.minHeight) {
    prompts.push("Capture a higher resolution image");
  }

  const accepted = prompts.length === 0 && input.accepted !== false;
  return {
    blurScore,
    glareScore,
    alignmentScore,
    lightingScore: Math.round((glareScore + blurScore) / 2),
    edgesVisible: width >= THRESHOLDS.minWidth && height >= THRESHOLDS.minHeight,
    accepted,
    prompts,
  };
}

export function qualityGateFromClientImage(input: {
  width: number;
  height: number;
  brightness: number;
  blurScore: number;
  glareScore: number;
  alignmentScore: number;
  edgeCoverage: number;
  perspectiveDistortion: number;
  sleeveReflectionScore: number;
}): {
  passed: boolean;
  prompts: string[];
  metrics: {
    blurScore: number;
    glareScore: number;
    alignmentScore: number;
  };
} {
  const blurScore = Math.round(input.blurScore * 100);
  const glareScore = Math.round((1 - input.glareScore) * 100);
  const alignmentScore = Math.round(input.alignmentScore * 100);

  const result = runQualityGate({
    blurScore,
    glareScore,
    alignmentScore,
    width: input.width,
    height: input.height,
  });

  if (input.perspectiveDistortion < 0.55) result.prompts.push("Rotate slightly");
  if (input.sleeveReflectionScore > 0.35) result.prompts.push("Remove sleeve reflections");
  if (input.edgeCoverage < 0.7) result.prompts.push("Ensure full card edges are visible");
  if (input.brightness < 0.4) result.prompts.push("Increase lighting");

  return {
    passed: result.accepted && result.prompts.length <= 1,
    prompts: result.prompts,
    metrics: {
      blurScore: result.blurScore,
      glareScore: result.glareScore,
      alignmentScore: result.alignmentScore,
    },
  };
}
