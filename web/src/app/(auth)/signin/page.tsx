import Link from "next/link";

import { SignInForm } from "@/components/auth/sign-in-form";

export default function SignInPage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-md items-center px-4 py-10">
      <div className="w-full rounded-2xl border border-zinc-800 bg-zinc-900/70 p-6">
        <div className="mb-6 space-y-2">
          <h1 className="text-2xl font-semibold text-zinc-100">Welcome back</h1>
          <p className="text-sm text-zinc-400">
            Sign in to continue scanning and sharing raw card verification reports.
          </p>
        </div>
        <SignInForm />
        <p className="mt-6 text-sm text-zinc-400">
          No account yet?{" "}
          <Link className="text-emerald-300 hover:text-emerald-200" href="/signup">
            Create one
          </Link>
          .
        </p>
      </div>
    </main>
  );
}
