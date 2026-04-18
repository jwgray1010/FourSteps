import { getSessionUserId } from "@/lib/auth";
import { badRequest, notFound, ok, unauthorized } from "@/lib/http";
import { saveImageMetadata } from "@/lib/mobile-api";

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const userId = await getSessionUserId();
  if (!userId) {
    return unauthorized();
  }

  const { id } = await params;
  const payload = await request.json().catch(() => null);
  if (!payload) {
    return badRequest("Invalid JSON payload.");
  }

  const result = await saveImageMetadata(userId, id, payload);
  if (!result.ok) {
    if (result.error === "scan_not_found") {
      return notFound("Scan not found.");
    }
    return badRequest("Invalid image payload.", result.issues);
  }

  return ok({
    image: result.image,
    qualityGate: result.qualityGate,
  });
}
