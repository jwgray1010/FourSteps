import { getSessionUserId } from "@/lib/auth";
import { badRequest, notFound, ok, unauthorized } from "@/lib/http";
import { triggerMockAnalysis } from "@/lib/mobile-api";
import { prisma } from "@/lib/prisma";

export async function POST(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const userId = await getSessionUserId();
  if (!userId) {
    return unauthorized();
  }

  const { id } = await params;
  const scan = await prisma.cardScan.findFirst({
    where: { id, userId },
    select: { id: true },
  });
  if (!scan) {
    return notFound("Scan not found.");
  }

  const result = await triggerMockAnalysis(userId, id);
  if (!result.ok) {
    return badRequest("Unable to run analysis for this scan.");
  }

  return ok({
    success: true,
    scanId: id,
    analysis: result.result,
    integrationMode: "mock",
    analysisServiceReadyFor: "python_opencv_fastapi",
  });
}
