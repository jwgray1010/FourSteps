"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

import { signUpSchema, type SignUpInput } from "@/lib/validation";

export function SignUpForm() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const form = useForm<SignUpInput>({
    resolver: zodResolver(signUpSchema),
    defaultValues: {
      email: "",
      username: "",
      password: "",
    },
  });

  const onSubmit = form.handleSubmit(async (values) => {
    setLoading(true);
    setError(null);
    const response = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const payload = (await response.json().catch(() => ({}))) as {
      error?: string;
    };
    if (!response.ok) {
      setError(payload.error ?? "Could not create account.");
      setLoading(false);
      return;
    }
    router.push("/dashboard");
    router.refresh();
  });

  return (
    <form
      onSubmit={onSubmit}
      className="mt-6 space-y-4 rounded-2xl border border-zinc-800 bg-zinc-900/70 p-6"
    >
      <label className="block space-y-1 text-sm">
        <span>Email</span>
        <input
          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
          type="email"
          {...form.register("email")}
        />
      </label>
      <label className="block space-y-1 text-sm">
        <span>Username</span>
        <input
          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
          {...form.register("username")}
        />
      </label>
      <label className="block space-y-1 text-sm">
        <span>Password</span>
        <input
          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
          type="password"
          {...form.register("password")}
        />
      </label>
      {error ? <p className="text-sm text-red-400">{error}</p> : null}
      <button
        disabled={loading}
        className="w-full rounded-lg bg-emerald-500 px-4 py-2 font-medium text-black disabled:opacity-60"
        type="submit"
      >
        {loading ? "Creating account..." : "Sign up"}
      </button>

      <p className="text-sm text-zinc-400">
        Already have an account?{" "}
        <Link className="text-emerald-400 underline" href="/signin">
          Sign in
        </Link>
      </p>
    </form>
  );
}
