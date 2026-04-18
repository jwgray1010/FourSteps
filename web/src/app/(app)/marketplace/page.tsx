import { ListingsGrid } from "@/components/listings/listings-grid";
import { prisma } from "@/lib/prisma";
import type { PublicListingSummary } from "@/types/domain";

export default async function MarketplacePage() {
  const listings = await prisma.listing.findMany({
    where: { status: { in: ["active", "sold"] } },
    include: {
      user: { select: { username: true, profileImage: true } },
      cardScan: {
        select: {
          id: true,
          title: true,
          sport: true,
          playerName: true,
          overallCategory: true,
          overallScore: true,
          imageConfidence: true,
          shareToken: true,
        },
      },
    },
    orderBy: { createdAt: "desc" },
  });

  const mapped: PublicListingSummary[] = listings.map((listing) => ({
    id: listing.id,
    askingPrice: Number(listing.askingPrice),
    description: listing.description,
    status: listing.status,
    createdAt: listing.createdAt.toISOString(),
    scan: {
      id: listing.cardScan.id,
      title: listing.cardScan.title,
      sport: listing.cardScan.sport,
      playerName: listing.cardScan.playerName,
      overallCategory: listing.cardScan.overallCategory,
      overallScore: listing.cardScan.overallScore,
      imageConfidence: listing.cardScan.imageConfidence,
      shareToken: listing.cardScan.shareToken,
    },
    seller: {
      username: listing.user.username,
      profileImage: listing.user.profileImage,
    },
  }));

  return (
    <main className="space-y-6">
      <h1 className="text-3xl font-semibold">Marketplace</h1>
      <p className="text-zinc-400">
        Browse report-backed raw listings. RAWIFY provides AI-assisted verification only.
      </p>
      <ListingsGrid listings={mapped} />
    </main>
  );
}
