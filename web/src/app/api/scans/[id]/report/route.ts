import { RAWIFY_REPORT_DISCLAIMER } from "@/lib/constants";
import { notFound, ok } from "@/lib/http";
import { prisma } from "@/lib/prisma";
import { categoryDescription, formatCategoryLabel } from "@/lib/score";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const scan = await prisma.cardScan.findUnique({
    where: { id },
    include: { analysis: true, images: true },
  });
  if (!scan) {
    return notFound("Scan not found.");
  }

  const overallCategory = scan.overallCategory ?? "visible_risk";
  return ok({
    report: {
      title: scan.title,
      sport: scan.sport,
      playerName: scan.playerName,
      overallCategory,
      overallScore: scan.overallScore,
      imageConfidence: scan.imageConfidence,
      scanDate: scan.updatedAt.toISOString(),
      categoryLabel: formatCategoryLabel(overallCategory),
      categoryDescription: categoryDescription(overallCategory),
      disclaimer: RAWIFY_REPORT_DISCLAIMER,
      trustCopy: [
        "AI-assisted verification",
        "Not an official grade",
        "Results depend on image quality",
        "No guarantee of third-party grading outcome",
      ],
      flags: scan.analysis
        ? (Array.isArray(scan.analysis.flags) ? (scan.analysis.flags as string[]) : [])
        : [],
      subscores: scan.analysis
        ? {
            centering: scan.analysis.centeringScore,
            corners: scan.analysis.cornersScore,
            edges: scan.analysis.edgesScore,
            surface: scan.analysis.surfaceScore,
          }
        : null,
      images: scan.images.map((image) => ({
        id: image.id,
        type: image.type,
        originalUrl: image.originalUrl,
        accepted: image.accepted,
      })),
    },
  });
}
