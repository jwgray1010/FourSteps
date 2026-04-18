import { analyzeScan } from "@/lib/analysis-client";
import { requireUserOrResponse } from "@/lib/auth";
import { notFound, ok, serverError } from "@/lib/http";
import { prisma } from "@/lib/prisma";
import { toOverallCategory } from "@/lib/score";
import type { AnalysisResult } from "@/types/domain";

export async function POST(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const auth = await requireUserOrResponse();
  if (!auth.ok) {
    return auth.response;
  }

  const { id } = await params;
  const scan = await prisma.cardScan.findFirst({
    where: { id, userId: auth.user.id },
    include: { images: true },
  });

  if (!scan) {
    return notFound("Scan not found.");
  }

  await prisma.cardScan.update({
    where: { id: scan.id },
    data: { status: "processing" },
  });

  try {
    const result: AnalysisResult = await analyzeScan({
      scanId: scan.id,
      title: scan.title,
      sport: scan.sport,
      imageUrls: Object.fromEntries(scan.images.map((img) => [img.type, img.originalUrl])),
    });

    const category = toOverallCategory(result.overallCategory);
    await prisma.cardAnalysis.upsert({
      where: { cardScanId: scan.id },
      update: {
        centeringScore: result.subscores.centering,
        cornersScore: result.subscores.corners,
        edgesScore: result.subscores.edges,
        surfaceScore: result.subscores.surface,
        centeringDetails: result.details.centering,
        cornersDetails: result.details.corners,
        edgesDetails: result.details.edges,
        surfaceDetails: result.details.surface,
        flags: result.flags,
        reasoning: {
          imageConfidence: result.imageConfidence,
          disclaimer: result.disclaimer,
        },
      },
      create: {
        cardScanId: scan.id,
        centeringScore: result.subscores.centering,
        cornersScore: result.subscores.corners,
        edgesScore: result.subscores.edges,
        surfaceScore: result.subscores.surface,
        centeringDetails: result.details.centering,
        cornersDetails: result.details.corners,
        edgesDetails: result.details.edges,
        surfaceDetails: result.details.surface,
        flags: result.flags,
        reasoning: {
          imageConfidence: result.imageConfidence,
          disclaimer: result.disclaimer,
        },
      },
    });

    await prisma.cardScan.update({
      where: { id: scan.id },
      data: {
        status: "completed",
        overallCategory: category,
        overallScore: result.overallScore,
        imageConfidence: result.imageConfidence,
      },
    });

    return ok({ success: true, result });
  } catch (error) {
    await prisma.cardScan.update({
      where: { id: scan.id },
      data: { status: "failed" },
    });
    return serverError(error instanceof Error ? error.message : "Analysis failed.");
  }
}
