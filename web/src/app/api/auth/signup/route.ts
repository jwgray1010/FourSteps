import { hashPassword, setUserSession } from "@/lib/auth";
import { badRequest, conflict } from "@/lib/http";
import { prisma } from "@/lib/prisma";
import { signUpSchema } from "@/lib/validation";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const parsed = signUpSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest("Invalid sign-up payload.", parsed.error.flatten());
  }

  const { email, username, password } = parsed.data;
  const existing = await prisma.user.findFirst({
    where: { OR: [{ email }, { username }] },
    select: { id: true },
  });
  if (existing) {
    return conflict("Email or username already in use.");
  }

  const passwordHash = await hashPassword(password);
  const user = await prisma.user.create({
    data: { email, username, passwordHash },
    select: { id: true, email: true, username: true },
  });

  await setUserSession(user.id);
  return Response.json({ user }, { status: 201 });
}
