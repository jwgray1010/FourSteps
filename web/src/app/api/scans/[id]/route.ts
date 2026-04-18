import { prisma } from "@/lib/prisma";
import { notFound, ok, unauthorized } from "@/lib/http";
import { getSessionUserId } from "@/lib/auth";

export async function GET(
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
    include: {
      images: true,
      analysis: true,
      listing: true,
      user: { select: { username: true } },
    },
  });

  if (!scan) {
    return notFound("Scan not found.");
  }

  return ok({ scan });
}

export async function DELETE(
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

  await prisma.cardScan.delete({ where: { id: scan.id } });
  return ok({ deleted: true });
}
