import { NextRequest } from "next/server";

import { requireUserOrResponse } from "@/lib/auth";
import { badRequest, ok, serverError } from "@/lib/http";
import { prisma } from "@/lib/prisma";
import { createListingSchema } from "@/lib/validation";

export async function POST(request: NextRequest) {
  const auth = await requireUserOrResponse();
  if (!auth.ok) {
    return auth.response;
  }

  try {
    const payload = await request.json();
    const parsed = createListingSchema.safeParse(payload);
    if (!parsed.success) {
      return badRequest("Invalid listing payload.", parsed.error.flatten());
    }

    const scan = await prisma.cardScan.findFirst({
      where: {
        id: parsed.data.cardScanId,
        userId: auth.user.id,
      },
      include: {
        analysis: true,
        listing: true,
      },
    });

    if (!scan) {
      return badRequest("Scan not found or not owned by user.");
    }
    if (!scan.analysis) {
      return badRequest("Run analysis before creating a listing.");
    }
    if (scan.listing) {
      return badRequest("Listing already exists for this scan.");
    }

    const listing = await prisma.listing.create({
      data: {
        userId: auth.user.id,
        cardScanId: scan.id,
        askingPrice: parsed.data.askingPrice,
        description: parsed.data.description,
        status: parsed.data.status,
      },
    });

    await prisma.cardScan.update({
      where: { id: scan.id },
      data: { status: "listed" },
    });

    return ok({ listing }, 201);
  } catch (error) {
    return serverError(error instanceof Error ? error.message : "Failed to create listing.");
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const sport = searchParams.get("sport");
    const player = searchParams.get("player");
    const category = searchParams.get("category");
    const minConfidence = searchParams.get("minConfidence");
    const maxPrice = searchParams.get("maxPrice");

    const listings = await prisma.listing.findMany({
      where: {
        status: { in: ["active", "sold"] },
        ...(maxPrice ? { askingPrice: { lte: Number(maxPrice) } } : {}),
        cardScan: {
          ...(sport ? { sport: { contains: sport, mode: "insensitive" } } : {}),
          ...(player ? { playerName: { contains: player, mode: "insensitive" } } : {}),
          ...(category ? { overallCategory: category as never } : {}),
          ...(minConfidence ? { imageConfidence: { gte: Number(minConfidence) } } : {}),
        },
      },
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
      take: 100,
    });

    return ok({ listings });
  } catch (error) {
    return serverError(error instanceof Error ? error.message : "Failed to fetch listings.");
  }
}
