import Link from "next/link";

import { getCurrentUser } from "@/lib/auth";
import { CATEGORY_DESCRIPTIONS, RAWIFY_REPORT_DISCLAIMER } from "@/lib/constants";

export default async function HomePage() {
  const user = await getCurrentUser();
  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-4">
        <Link href="/" className="text-xl font-semibold tracking-wide">
          RAWIFY
        </Link>
        <div className="flex items-center gap-2 text-sm">
          {user ? (
            <>
              <Link
                href="/dashboard"
                className="rounded-md border border-zinc-700 px-3 py-2 text-zinc-100 hover:bg-zinc-800"
              >
                Dashboard
              </Link>
              <form action="/api/auth/signout" method="post">
                <button
                  type="submit"
                  className="rounded-md border border-zinc-700 px-3 py-2 text-zinc-100 hover:bg-zinc-800"
                >
                  Sign out
                </button>
              </form>
            </>
          ) : (
            <>
              <Link
                href="/signin"
                className="rounded-md border border-zinc-700 px-3 py-2 text-zinc-100 hover:bg-zinc-800"
              >
                Sign in
              </Link>
              <Link
                href="/signup"
                className="rounded-md bg-indigo-500 px-3 py-2 font-medium text-white hover:bg-indigo-400"
              >
                Sign up
              </Link>
            </>
          )}
        </div>
      </header>

      <section className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-10 md:grid-cols-2">
        <div className="space-y-5">
          <p className="text-xs uppercase tracking-[0.2em] text-emerald-300">AI-assisted verification</p>
          <h1 className="text-4xl font-bold leading-tight sm:text-5xl">
            Sell raw cards with more trust.
          </h1>
          <p className="text-zinc-300">
            RAWIFY uses guided photo capture and AI-assisted inspection to generate a shareable
            raw card verification report.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/scans/new"
              className="rounded-xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-zinc-950"
            >
              Scan a card
            </Link>
            <Link
              href="/marketplace"
              className="rounded-xl border border-zinc-700 px-4 py-2 text-sm font-semibold"
            >
              Browse verified listings
            </Link>
          </div>
        </div>

        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
          <h2 className="text-lg font-semibold">Category descriptions</h2>
          <div className="mt-4 space-y-3 text-sm text-zinc-300">
            {Object.entries(CATEGORY_DESCRIPTIONS).map(([key, value]) => (
              <div key={key} className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
                <p className="font-semibold text-zinc-100">
                  {key
                    .split("_")
                    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
                    .join(" ")}
                </p>
                <p className="mt-1">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto w-full max-w-6xl px-4 pb-12">
        <p className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-sm text-zinc-300">
          {RAWIFY_REPORT_DISCLAIMER}
        </p>
      </section>
    </main>
  );
}
