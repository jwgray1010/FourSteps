import { getCurrentUser } from "@/lib/auth";
import { badRequest, ok, unauthorized } from "@/lib/http";
import { createScanRecord } from "@/lib/mobile-api";
import { createScanSchema } from "@/lib/validation";

export async function POST(request: Request) {
  const user = await getCurrentUser();
  if (!user) {
    return unauthorized();
  }

  const payload = await request.json().catch(() => null);
  const parsed = createScanSchema.safeParse(payload);
  if (!parsed.success) {
    return badRequest("Invalid scan payload.", parsed.error.flatten());
  }

  const result = await createScanRecord(user.id, parsed.data);
  if (!result.ok) {
    return badRequest("Invalid scan payload.", result.issues);
  }

  return ok({
    scanId: result.scan.id,
    status: result.scan.status,
    createdAt: result.scan.createdAt.toISOString(),
    scan: {
      id: result.scan.id,
      title: result.scan.title,
      sport: result.scan.sport,
      playerName: result.scan.playerName,
      status: result.scan.status,
    },
  });
}
