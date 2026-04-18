import Link from "next/link";

import { SignUpForm } from "@/components/auth/sign-up-form";

export default function SignUpPage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-md items-center px-4 py-10">
      <div className="w-full rounded-2xl border border-zinc-800 bg-zinc-900/70 p-6">
        <div className="mb-6 space-y-2">
          <h1 className="text-2xl font-semibold text-zinc-100">Create your RAWIFY account</h1>
          <p className="text-sm text-zinc-400">
            Start scanning raw cards with guided capture and AI-assisted verification.
          </p>
        </div>
        <SignUpForm />
        <p className="mt-6 text-sm text-zinc-400">
          Already have an account?{" "}
          <Link className="text-emerald-300 hover:text-emerald-200" href="/signin">
            Sign in
          </Link>
          .
        </p>
      </div>
    </main>
  );
}
