import type { CardImage, Listing, User, CardScan } from "@prisma/client";

import { getSessionUserId } from "@/lib/auth";
import { RAWIFY_REPORT_DISCLAIMER, RAWIFY_TRUST_COPY } from "@/lib/constants";
import { runQualityGate } from "@/lib/quality-gate";
import { prisma } from "@/lib/prisma";
import { toOverallCategory } from "@/lib/score";
import { createListingSchema, createScanSchema, uploadImageSchema } from "@/lib/validation";
import type { AnalysisResult } from "@/types/domain";

export async function getMobileAuthUserId(): Promise<string | null> {
  return getSessionUserId();
}

export async function createScanRecord(userId: string, payload: unknown) {
  const parsed = createScanSchema.safeParse(payload);
  if (!parsed.success) {
    return {
      ok: false as const,
      error: "invalid_scan_payload",
      issues: parsed.error.flatten(),
    };
  }

  const data = parsed.data;
  const scan = await prisma.cardScan.create({
    data: {
      userId,
      title: data.title,
      sport: data.sport,
      year: data.year ?? null,
      brand: data.brand ?? null,
      setName: data.setName ?? null,
      playerName: data.playerName,
      cardNumber: data.cardNumber ?? null,
      serialNumber: data.serialNumber ?? null,
      disclaimerAccepted: data.disclaimerAccepted,
      status: "draft",
    },
  });
  return { ok: true as const, scan };
}

export async function saveImageMetadata(userId: string, scanId: string, payload: unknown) {
  const parsed = uploadImageSchema.safeParse(payload);
  if (!parsed.success) {
    return {
      ok: false as const,
      error: "invalid_image_payload",
      issues: parsed.error.flatten(),
    };
  }

  const scan = await prisma.cardScan.findFirst({
    where: { id: scanId, userId },
    select: { id: true },
  });
  if (!scan) {
    return {
      ok: false as const,
      error: "scan_not_found",
      issues: null,
    };
  }

  const qualityGate = runQualityGate({
    blurScore: parsed.data.blurScore,
    glareScore: parsed.data.glareScore,
    alignmentScore: parsed.data.alignmentScore,
    width: parsed.data.width,
    height: parsed.data.height,
    accepted: parsed.data.accepted,
  });

  const image = await prisma.cardImage.create({
    data: {
      cardScanId: scanId,
      type: parsed.data.type,
      originalUrl: parsed.data.originalUrl,
      width: parsed.data.width,
      height: parsed.data.height,
      blurScore: qualityGate.blurScore,
      glareScore: qualityGate.glareScore,
      alignmentScore: qualityGate.alignmentScore,
      accepted: qualityGate.accepted,
    },
  });

  return {
    ok: true as const,
    image,
    qualityGate,
  };
}

export async function triggerMockAnalysis(userId: string, scanId: string) {
  const scan = await prisma.cardScan.findFirst({
    where: { id: scanId, userId },
    include: { images: true },
  });
  if (!scan) {
    return { ok: false as const, error: "scan_not_found" };
  }

  await prisma.cardScan.update({
    where: { id: scan.id },
    data: { status: "processing" },
  });

  const imageCount = scan.images.length;
  const hasBack = scan.images.some((img) => img.type === "back_straight");
  const hasAngles =
    scan.images.some((img) => img.type === "front_angle_left") &&
    scan.images.some((img) => img.type === "front_angle_right");
  const imageConfidence = Math.max(58, Math.min(90, 72 + (imageCount >= 4 ? 8 : -6)));
  const centering = Math.max(62, Math.min(95, 88 + (hasBack ? 3 : -6)));
  const corners = Math.max(60, Math.min(92, 83 + (imageCount >= 4 ? 2 : -5)));
  const edges = Math.max(58, Math.min(90, 80 + (hasAngles ? 4 : -7)));
  const surface = Math.max(56, Math.min(90, 81 + (hasAngles ? 3 : -8)));
  const overallScore = Math.round(
    centering * 0.35 + corners * 0.25 + edges * 0.2 + surface * 0.2,
  );

  const overallCategory = toOverallCategory(
    overallScore >= 90
      ? "Gem Candidate"
      : overallScore >= 78
        ? "Strong Raw"
        : overallScore >= 65
          ? "Borderline"
          : "Visible Risk",
  );

  const result: AnalysisResult = {
    success: true,
    imageConfidence,
    overallScore,
    overallCategory:
      overallCategory
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ") as AnalysisResult["overallCategory"],
    flags: [
      ...(hasBack ? [] : ["Back image missing; confidence reduced"]),
      ...(hasAngles ? [] : ["Surface uncertain due to missing angled captures"]),
      "Mock analysis mode enabled; integrate Python OpenCV service later",
    ],
    subscores: {
      centering,
      corners,
      edges,
      surface,
    },
    details: {
      centering: {
        frontLeftRight: "49/51",
        frontTopBottom: "48/52",
        backLeftRight: hasBack ? "50/50" : "unknown",
        confidence: Number((0.64 + imageConfidence / 500).toFixed(2)),
      },
      corners: {
        topLeft: corners + 1,
        topRight: corners - 1,
        bottomLeft: corners,
        bottomRight: corners - 2,
      },
      edges: {
        top: edges - 1,
        bottom: edges + 1,
        left: edges,
        right: edges - 2,
      },
      surface: {
        status: hasAngles
          ? "No major defects detected"
          : "Surface uncertain due to missing angled captures",
        confidence: Number((0.58 + imageConfidence / 500).toFixed(2)),
      },
    },
    disclaimer: RAWIFY_REPORT_DISCLAIMER,
  };

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
        mode: "mock",
        futureIntegration: "python_opencv_fastapi",
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
        mode: "mock",
        futureIntegration: "python_opencv_fastapi",
        disclaimer: result.disclaimer,
      },
    },
  });

  await prisma.cardScan.update({
    where: { id: scan.id },
    data: {
      status: "completed",
      overallCategory,
      overallScore: result.overallScore,
      imageConfidence: result.imageConfidence,
    },
  });

  return {
    ok: true as const,
    result,
  };
}

export async function getScanReport(userId: string, scanId: string) {
  const scan = await prisma.cardScan.findFirst({
    where: { id: scanId, userId },
    include: {
      analysis: true,
      images: true,
    },
  });
  if (!scan) {
    return null;
  }

  const category = scan.overallCategory ?? "visible_risk";
  return {
    id: scan.id,
    title: scan.title,
    sport: scan.sport,
    year: scan.year,
    brand: scan.brand,
    setName: scan.setName,
    playerName: scan.playerName,
    cardNumber: scan.cardNumber,
    serialNumber: scan.serialNumber,
    scanDate: scan.updatedAt.toISOString(),
    overallCategory: category,
    overallScore: scan.overallScore,
    imageConfidence: scan.imageConfidence,
    trustCopy: [...RAWIFY_TRUST_COPY],
    disclaimer: RAWIFY_REPORT_DISCLAIMER,
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
    details: scan.analysis
      ? {
          centering: scan.analysis.centeringDetails,
          corners: scan.analysis.cornersDetails,
          edges: scan.analysis.edgesDetails,
          surface: scan.analysis.surfaceDetails,
        }
      : null,
    images: scan.images.map((img) => mapCardImage(img)),
  };
}

type CreateListingInput = {
  cardScanId: string;
  askingPrice: number;
  description: string;
  status: "draft" | "active" | "sold" | "archived";
};

export async function createMarketplaceListingForUser(userId: string, payload: unknown) {
  const parsed = createListingSchema.safeParse(payload);
  if (!parsed.success) {
    return {
      ok: false as const,
      error: "invalid_listing_payload",
      issues: parsed.error.flatten(),
    };
  }

  const data = parsed.data as CreateListingInput;
  const scan = await prisma.cardScan.findFirst({
    where: { id: data.cardScanId, userId },
    include: { analysis: true, listing: true },
  });
  if (!scan) {
    return { ok: false as const, error: "scan_not_found", issues: null };
  }
  if (!scan.analysis) {
    return { ok: false as const, error: "scan_missing_analysis", issues: null };
  }
  if (scan.listing) {
    return { ok: false as const, error: "scan_already_listed", issues: null };
  }

  const listing = await prisma.listing.create({
    data: {
      userId,
      cardScanId: data.cardScanId,
      askingPrice: data.askingPrice,
      description: data.description,
      status: data.status,
    },
  });
  await prisma.cardScan.update({
    where: { id: data.cardScanId },
    data: { status: "listed" },
  });

  return { ok: true as const, listing };
}

export async function listMarketplaceListingsForMobile(filters?: {
  sport?: string | null;
  player?: string | null;
  category?: string | null;
  minConfidence?: string | null;
  maxPrice?: string | null;
}) {
  const minConfidence = filters?.minConfidence
    ? Number.parseInt(filters.minConfidence, 10)
    : null;
  const maxPrice = filters?.maxPrice ? Number.parseFloat(filters.maxPrice) : null;

  return prisma.listing.findMany({
    where: {
      status: { in: ["active", "sold"] },
      ...(maxPrice != null && Number.isFinite(maxPrice) ? { askingPrice: { lte: maxPrice } } : {}),
      cardScan: {
        ...(filters?.sport ? { sport: { contains: filters.sport, mode: "insensitive" } } : {}),
        ...(filters?.player
          ? { playerName: { contains: filters.player, mode: "insensitive" } }
          : {}),
        ...(filters?.category ? { overallCategory: filters.category as never } : {}),
        ...(minConfidence != null && Number.isFinite(minConfidence)
          ? { imageConfidence: { gte: minConfidence } }
          : {}),
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
}

export function mapCardImage(image: CardImage) {
  return {
    id: image.id,
    type: image.type,
    originalUrl: image.originalUrl,
    processedUrl: image.processedUrl,
    width: image.width,
    height: image.height,
    blurScore: image.blurScore,
    glareScore: image.glareScore,
    alignmentScore: image.alignmentScore,
    accepted: image.accepted,
    createdAt: image.createdAt.toISOString(),
  };
}

export function mapListingForMobile(
  listing: Listing & {
    cardScan: Pick<
      CardScan,
      | "id"
      | "title"
      | "sport"
      | "playerName"
      | "overallCategory"
      | "overallScore"
      | "imageConfidence"
      | "shareToken"
    >;
    user: Pick<User, "username" | "profileImage">;
  },
) {
  return {
    id: listing.id,
    askingPrice: Number(listing.askingPrice),
    description: listing.description,
    status: listing.status,
    createdAt: listing.createdAt.toISOString(),
    updatedAt: listing.updatedAt.toISOString(),
    seller: {
      username: listing.user.username,
      profileImage: listing.user.profileImage,
    },
    card: {
      scanId: listing.cardScan.id,
      title: listing.cardScan.title,
      sport: listing.cardScan.sport,
      playerName: listing.cardScan.playerName,
      overallCategory: listing.cardScan.overallCategory,
      overallScore: listing.cardScan.overallScore,
      imageConfidence: listing.cardScan.imageConfidence,
      shareToken: listing.cardScan.shareToken,
    },
  };
}
