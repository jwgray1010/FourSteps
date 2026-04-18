import { ok, notFound } from "@/lib/http";
import { prisma } from "@/lib/prisma";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const listing = await prisma.listing.findUnique({
    where: { id },
    include: {
      user: { select: { username: true, profileImage: true } },
      cardScan: { include: { analysis: true, images: true } },
    },
  });

  if (!listing) {
    return notFound("Listing not found.");
  }
  return ok({ listing });
}
