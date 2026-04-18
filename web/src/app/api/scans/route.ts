import { getCurrentUser } from "@/lib/auth";
import { badRequest, ok, unauthorized } from "@/lib/http";
import { prisma } from "@/lib/prisma";
import { createScanSchema } from "@/lib/validation";

export async function POST(request: Request) {
  const user = await getCurrentUser();
  if (!user) {
    return unauthorized();
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return badRequest("Invalid JSON payload.");
  }

  const parsed = createScanSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest("Invalid scan payload.", parsed.error.flatten());
  }

  const data = parsed.data;
  const scan = await prisma.cardScan.create({
    data: {
      userId: user.id,
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

  return ok({ scanId: scan.id, scan });
}
