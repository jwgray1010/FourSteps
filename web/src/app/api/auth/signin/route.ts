import { verifyPassword, setUserSession } from "@/lib/auth";
import { badRequest, unauthorized, ok } from "@/lib/http";
import { prisma } from "@/lib/prisma";
import { signInSchema } from "@/lib/validation";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const parsed = signInSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest("Invalid sign-in payload.", parsed.error.flatten());
  }

  const user = await prisma.user.findUnique({
    where: { email: parsed.data.email },
  });
  if (!user) {
    return unauthorized();
  }

  const matches = await verifyPassword(parsed.data.password, user.passwordHash);
  if (!matches) {
    return unauthorized();
  }

  await setUserSession(user.id);
  return ok({
    user: {
      id: user.id,
      email: user.email,
      username: user.username,
    },
  });
}
