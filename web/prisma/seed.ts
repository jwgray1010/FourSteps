import { PrismaClient } from "@prisma/client";
import { hash } from "bcryptjs";

const prisma = new PrismaClient();

async function main() {
  const passwordHash = await hash("demo1234", 10);

  const user = await prisma.user.upsert({
    where: { email: "demo@rawify.app" },
    create: {
      email: "demo@rawify.app",
      username: "rawifydemo",
      passwordHash,
      profileImage: "https://picsum.photos/seed/rawify-user/128/128",
    },
    update: {
      username: "rawifydemo",
      passwordHash,
    },
  });

  const now = new Date();
  const scanOne = await prisma.cardScan.upsert({
    where: { id: "scan_demo_1" },
    create: {
      id: "scan_demo_1",
      userId: user.id,
      title: "2023 Topps Chrome Sapphire Victor Wembanyama RC",
      sport: "Basketball",
      year: "2023",
      brand: "Topps",
      setName: "Chrome Sapphire",
      playerName: "Victor Wembanyama",
      cardNumber: "SW-1",
      status: "completed",
      overallCategory: "strong_raw",
      overallScore: 87,
      imageConfidence: 84,
      disclaimerAccepted: true,
      shareToken: "demo-share-token-1",
      createdAt: now,
    },
    update: {
      status: "completed",
      overallCategory: "strong_raw",
      overallScore: 87,
      imageConfidence: 84,
      disclaimerAccepted: true,
    },
  });

  await prisma.cardImage.createMany({
    data: [
      {
        cardScanId: scanOne.id,
        type: "front_straight",
        originalUrl: "https://picsum.photos/seed/rawify-front-1/1200/1700",
        processedUrl: "https://picsum.photos/seed/rawify-front-1-overlay/1200/1700",
        accepted: true,
        blurScore: 0.18,
        glareScore: 0.21,
        alignmentScore: 0.92,
        width: 1200,
        height: 1700,
      },
      {
        cardScanId: scanOne.id,
        type: "back_straight",
        originalUrl: "https://picsum.photos/seed/rawify-back-1/1200/1700",
        processedUrl: "https://picsum.photos/seed/rawify-back-1-overlay/1200/1700",
        accepted: true,
        blurScore: 0.2,
        glareScore: 0.25,
        alignmentScore: 0.89,
        width: 1200,
        height: 1700,
      },
      {
        cardScanId: scanOne.id,
        type: "front_angle_left",
        originalUrl: "https://picsum.photos/seed/rawify-angle-left-1/1200/1700",
        accepted: true,
        blurScore: 0.22,
        glareScore: 0.3,
        alignmentScore: 0.85,
        width: 1200,
        height: 1700,
      },
      {
        cardScanId: scanOne.id,
        type: "front_angle_right",
        originalUrl: "https://picsum.photos/seed/rawify-angle-right-1/1200/1700",
        accepted: true,
        blurScore: 0.24,
        glareScore: 0.31,
        alignmentScore: 0.84,
        width: 1200,
        height: 1700,
      },
    ],
    skipDuplicates: true,
  });

  await prisma.cardAnalysis.upsert({
    where: { cardScanId: scanOne.id },
    create: {
      cardScanId: scanOne.id,
      centeringScore: 92,
      cornersScore: 88,
      edgesScore: 80,
      surfaceScore: 86,
      centeringDetails: {
        frontLeftRight: "48/52",
        frontTopBottom: "49/51",
        backLeftRight: "50/50",
        backTopBottom: "49/51",
        confidence: 0.82,
      },
      cornersDetails: {
        topLeft: 90,
        topRight: 87,
        bottomLeft: 88,
        bottomRight: 89,
      },
      edgesDetails: {
        top: 78,
        bottom: 82,
        left: 84,
        right: 80,
      },
      surfaceDetails: {
        status: "No major defects detected",
        confidence: 0.71,
      },
      flags: [
        "Possible minor top-edge whitening",
        "Surface mostly clear; angled scans acceptable",
      ],
      reasoning: {
        notes: [
          "Strict model reduced edge score due to small top-edge whitening pattern.",
          "Image quality acceptable for report confidence above 80.",
        ],
      },
    },
    update: {
      centeringScore: 92,
      cornersScore: 88,
      edgesScore: 80,
      surfaceScore: 86,
    },
  });

  await prisma.listing.upsert({
    where: { cardScanId: scanOne.id },
    create: {
      userId: user.id,
      cardScanId: scanOne.id,
      askingPrice: 149.99,
      description:
        "Strong visual copy with RAWIFY report. AI-assisted verification only, not an official grade.",
      status: "active",
    },
    update: {
      askingPrice: 149.99,
      status: "active",
    },
  });

  const scanTwo = await prisma.cardScan.upsert({
    where: { id: "scan_demo_2" },
    create: {
      id: "scan_demo_2",
      userId: user.id,
      title: "2018 Bowman Chrome Shohei Ohtani RC",
      sport: "Baseball",
      year: "2018",
      brand: "Bowman",
      setName: "Bowman Chrome",
      playerName: "Shohei Ohtani",
      cardNumber: "1",
      serialNumber: "N/A",
      status: "completed",
      overallCategory: "borderline",
      overallScore: 74,
      imageConfidence: 66,
      disclaimerAccepted: true,
      shareToken: "demo-share-token-2",
    },
    update: {
      overallCategory: "borderline",
      overallScore: 74,
      imageConfidence: 66,
    },
  });

  await prisma.cardAnalysis.upsert({
    where: { cardScanId: scanTwo.id },
    create: {
      cardScanId: scanTwo.id,
      centeringScore: 84,
      cornersScore: 72,
      edgesScore: 74,
      surfaceScore: 70,
      centeringDetails: {
        frontLeftRight: "46/54",
        frontTopBottom: "47/53",
        backLeftRight: "55/45",
        backTopBottom: "53/47",
        confidence: 0.66,
      },
      cornersDetails: {
        topLeft: 70,
        topRight: 74,
        bottomLeft: 73,
        bottomRight: 71,
      },
      edgesDetails: {
        top: 72,
        bottom: 76,
        left: 75,
        right: 73,
      },
      surfaceDetails: {
        status: "Surface uncertain due to glare in right half.",
        confidence: 0.54,
      },
      flags: [
        "Surface uncertain due to glare",
        "Image confidence below 70 caps overall category",
      ],
      reasoning: {
        capReason: "image_confidence_below_threshold",
      },
    },
    update: {
      centeringScore: 84,
      cornersScore: 72,
      edgesScore: 74,
      surfaceScore: 70,
    },
  });

  console.log("Seed complete:", { userId: user.id, scans: [scanOne.id, scanTwo.id] });
}

main()
  .catch((error) => {
    console.error("Seed failed", error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
