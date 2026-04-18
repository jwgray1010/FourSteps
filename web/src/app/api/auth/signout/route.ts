import { revokeSessionByCookie } from "@/lib/auth";
import { ok } from "@/lib/http";

export async function POST() {
  await revokeSessionByCookie();
  return ok({ success: true });
}
