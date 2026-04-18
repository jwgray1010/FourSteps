import { notFound, redirect } from "next/navigation";

import { ScanDetailClient } from "@/components/capture/scan-detail-client";
import { getCurrentUser } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export default async function ScanDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const user = await getCurrentUser();
  if (!user) {
    redirect("/signin");
  }

  const { id } = await params;
  const scan = await prisma.cardScan.findFirst({
    where: { id, userId: user.id },
    include: { images: true, analysis: true, listing: true },
  });

  if (!scan) {
    notFound();
  }

  return (
    <ScanDetailClient
      scan={{
        id: scan.id,
        shareToken: scan.shareToken,
        title: scan.title,
        status: scan.status,
        overallCategory: scan.overallCategory,
        overallScore: scan.overallScore,
        imageConfidence: scan.imageConfidence,
        images: scan.images.map((img) => ({
          id: img.id,
          type: img.type,
          accepted: img.accepted,
          originalUrl: img.originalUrl,
        })),
      }}
    />
  );
}
