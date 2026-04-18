import type { ImageType, OverallCategory } from "@/types/domain";

export const RAWIFY_REPORT_DISCLAIMER =
  "AI-assisted evaluation based on submitted images. This is not an official grade and does not guarantee any grading outcome.";

export const RAWIFY_NOT_OFFICIAL_GRADE =
  "AI-assisted verification only. RAWIFY is not an official grader.";

export const RAWIFY_TRUST_COPY = [
  "AI-assisted verification",
  "Not an official grade",
  "Results depend on image quality",
  "No guarantee of third-party grading outcome",
] as const;

export const CATEGORY_DESCRIPTIONS: Record<OverallCategory, string> = {
  gem_candidate:
    "Strong visual candidate based on submitted images, with no major issues detected. Not an official grade.",
  strong_raw: "Presents well with minor risk factors or uncertainty.",
  borderline:
    "Usable raw copy with visible or possible concerns that may affect premium grade outcomes.",
  visible_risk: "One or more visible concerns detected from submitted images.",
};

export const CATEGORY_COPY = CATEGORY_DESCRIPTIONS;

export const CAPTURE_IMAGE_TYPES: ImageType[] = [
  "front_straight",
  "back_straight",
  "front_angle_left",
  "front_angle_right",
  "back_angle_left",
  "back_angle_right",
];

export const CAPTURE_STEPS = [
  "Use a plain dark background.",
  "Use strong diffuse light, avoid direct glare.",
  "Remove sleeve/toploader if possible.",
  "Fill frame with full card edges visible.",
  "Keep camera steady and parallel to card.",
];

export const CAPTURE_MINIMUMS = {
  blur: 55,
  glare: 55,
  brightness: 45,
  alignment: 60,
  edgesVisible: 70,
  perspective: 55,
};

export const DISCLAIMER_TEXT = RAWIFY_REPORT_DISCLAIMER;
export const DISCLAIMER_FULL = RAWIFY_REPORT_DISCLAIMER;

export function categoryLabel(category: string): string {
  return category
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function categoryDescription(category: string): string {
  const key = (category || "visible_risk") as OverallCategory;
  return CATEGORY_DESCRIPTIONS[key] ?? CATEGORY_DESCRIPTIONS.visible_risk;
}
