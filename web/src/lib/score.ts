import { CATEGORY_COPY } from "@/lib/constants";
import { OverallCategory } from "@/types/domain";

export function deriveOverallCategory(score: number): OverallCategory {
  if (score >= 90) {
    return "gem_candidate";
  }
  if (score >= 78) {
    return "strong_raw";
  }
  if (score >= 65) {
    return "borderline";
  }
  return "visible_risk";
}

export function clampCategoryByImageConfidence(
  category: OverallCategory,
  imageConfidence: number,
): OverallCategory {
  if (imageConfidence >= 70) {
    return category;
  }
  if (category === "gem_candidate" || category === "strong_raw") {
    return "borderline";
  }
  return category;
}

export function categoryCopy(category: OverallCategory) {
  return CATEGORY_COPY[category];
}

export function formatCategory(category: string): string {
  return category
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatCategoryLabel(category: string): string {
  return formatCategory(category);
}

export function categoryDescription(category: string): string {
  const normalized = (category || "visible_risk").toLowerCase();
  const key = normalized
    .replace(/\s+/g, "_")
    .replace("gem_candidate", "gem_candidate")
    .replace("strong_raw", "strong_raw")
    .replace("visible_risk", "visible_risk")
    .replace("borderline", "borderline") as OverallCategory;
  return CATEGORY_COPY[key] ?? CATEGORY_COPY.visible_risk;
}

export function toOverallCategory(label: string): OverallCategory {
  const normalized = (label || "").toLowerCase().replace(/\s+/g, "_");
  if (normalized.includes("gem")) {
    return "gem_candidate";
  }
  if (normalized.includes("strong")) {
    return "strong_raw";
  }
  if (normalized.includes("border")) {
    return "borderline";
  }
  return "visible_risk";
}
