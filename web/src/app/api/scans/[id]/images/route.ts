import type { NextRequest } from "next/server";

import { getSessionUserId } from "@/lib/auth";
import { badRequest, notFound, ok, unauthorized } from "@/lib/http";
import { prisma } from "@/lib/prisma";
import { runQualityGate } from "@/lib/quality-gate";
import { uploadImageSchema } from "@/lib/validation";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const userId = await getSessionUserId();
  if (!userId) {
    return unauthorized();
  }

  const { id } = await params;
  const scan = await prisma.cardScan.findFirst({
    where: { id, userId },
  });
  if (!scan) {
    return notFound("Scan not found.");
  }

  const body = await request.json().catch(() => null);
  const parsed = uploadImageSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest("Invalid image payload.", parsed.error.flatten());
  }

  const quality = runQualityGate({
    blurScore: parsed.data.blurScore,
    glareScore: parsed.data.glareScore,
    alignmentScore: parsed.data.alignmentScore,
    width: parsed.data.width,
    height: parsed.data.height,
    accepted: parsed.data.accepted,
  });

  const image = await prisma.cardImage.create({
    data: {
      cardScanId: id,
      type: parsed.data.type,
      originalUrl: parsed.data.originalUrl,
      width: parsed.data.width,
      height: parsed.data.height,
      blurScore: quality.blurScore,
      glareScore: quality.glareScore,
      alignmentScore: quality.alignmentScore,
      accepted: quality.accepted,
    },
  });

  return ok({ image, qualityGate: quality });
}
