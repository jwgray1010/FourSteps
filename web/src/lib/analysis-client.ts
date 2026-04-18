import type { AnalysisResult } from "@/types/domain";

const DEFAULT_ANALYSIS_URL = "http://127.0.0.1:8000";

export type AnalyzeScanInput = {
  scanId: string;
  title?: string;
  sport?: string;
  imageUrls: Record<string, string>;
};

export async function analyzeScan(input: AnalyzeScanInput): Promise<AnalysisResult> {
  const baseUrl = process.env.ANALYSIS_SERVICE_URL ?? DEFAULT_ANALYSIS_URL;
  const response = await fetch(`${baseUrl}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Analysis service error: ${response.status}`);
  }

  return (await response.json()) as AnalysisResult;
}
