import { cookies } from "next/headers";
import { randomBytes } from "node:crypto";

import bcrypt from "bcryptjs";

import { unauthorized } from "@/lib/http";
import { prisma } from "@/lib/prisma";

export const SESSION_COOKIE = "rawify_session";
const SESSION_DURATION_MS = 1000 * 60 * 60 * 24 * 30;

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 10);
}

export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

export async function createSession(userId: string): Promise<string> {
  const token = randomBytes(32).toString("hex");
  await prisma.session.create({
    data: {
      userId,
      sessionToken: token,
      expiresAt: new Date(Date.now() + SESSION_DURATION_MS),
    },
  });
  return token;
}

export async function setUserSession(userId: string): Promise<void> {
  const token = await createSession(userId);
  await setSessionCookie(token);
}

export async function setSessionCookie(token: string): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.set(SESSION_COOKIE, token, {
    path: "/",
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: 60 * 60 * 24 * 30,
  });
}

export async function clearSessionCookie(): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.delete(SESSION_COOKIE);
}

export async function getCurrentUser() {
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE)?.value;
  if (!token) {
    return null;
  }

  const session = await prisma.session.findUnique({
    where: { sessionToken: token },
    include: { user: true },
  });

  if (!session) {
    return null;
  }

  if (session.expiresAt.getTime() < Date.now()) {
    await prisma.session.delete({ where: { id: session.id } }).catch(() => undefined);
    return null;
  }

  return session.user;
}

export async function requireUser() {
  const user = await getCurrentUser();
  if (!user) {
    throw new Error("UNAUTHORIZED");
  }
  return user;
}

export async function requireUserResponse() {
  const user = await getCurrentUser();
  if (!user) {
    return { ok: false as const, response: unauthorized() };
  }
  return { ok: true as const, user };
}

export async function requireUserOrResponse() {
  return requireUserResponse();
}

export async function getSessionUserId() {
  const user = await getCurrentUser();
  return user?.id ?? null;
}

export async function revokeSessionByCookie(): Promise<void> {
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE)?.value;
  if (!token) {
    return;
  }
  await prisma.session.deleteMany({ where: { sessionToken: token } });
  cookieStore.delete(SESSION_COOKIE);
}
