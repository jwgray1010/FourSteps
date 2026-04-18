import { getSessionUserId } from "@/lib/auth";
import { notFound, ok, unauthorized } from "@/lib/http";
import { getScanReport } from "@/lib/mobile-api";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const userId = await getSessionUserId();
  if (!userId) {
    return unauthorized();
  }

  const { id } = await params;
  const report = await getScanReport(userId, id);
  if (!report) {
    return notFound("Scan report not found.");
  }
  return ok({ report });
}
