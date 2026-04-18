import { notFound } from "next/navigation";

import { PublicReport } from "@/components/report/public-report";
import { RAWIFY_REPORT_DISCLAIMER, RAWIFY_TRUST_COPY } from "@/lib/constants";
import { prisma } from "@/lib/prisma";

export default async function ReportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const scan = await prisma.cardScan.findUnique({
    where: { shareToken: id },
    include: { analysis: true, images: true },
  });
  if (!scan) notFound();

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <PublicReport
        report={{
          title: scan.title,
          sport: scan.sport,
          playerName: scan.playerName,
          scanDate: scan.updatedAt.toISOString(),
          overallCategory: scan.overallCategory,
          overallScore: scan.overallScore,
          imageConfidence: scan.imageConfidence,
          subscores: scan.analysis
            ? {
                centering: scan.analysis.centeringScore,
                corners: scan.analysis.cornersScore,
                edges: scan.analysis.edgesScore,
                surface: scan.analysis.surfaceScore,
              }
            : undefined,
          flags: scan.analysis
            ? (Array.isArray(scan.analysis.flags) ? (scan.analysis.flags as string[]) : [])
            : [],
          images: scan.images.map((image) => ({
            id: image.id,
            type: image.type,
            originalUrl: image.originalUrl,
            accepted: image.accepted,
          })),
          disclaimer: RAWIFY_REPORT_DISCLAIMER,
          trustCopy: [...RAWIFY_TRUST_COPY],
        }}
      />
    </main>
  );
}
