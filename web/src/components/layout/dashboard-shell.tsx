"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

type DashboardShellProps = {
  children: ReactNode;
};

const LINKS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/scans/new", label: "New scan" },
  { href: "/marketplace", label: "Marketplace" },
  { href: "/listings/new", label: "Create listing" },
];

export function DashboardShell({ children }: DashboardShellProps) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="sticky top-0 z-30 border-b border-zinc-800/80 bg-zinc-950/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
          <Link href="/" className="text-lg font-semibold tracking-wide text-white">
            RAWIFY
          </Link>
          <nav className="flex items-center gap-2">
            {LINKS.map((link) => {
              const active = pathname?.startsWith(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={cn(
                    "rounded-full px-3 py-1.5 text-sm transition",
                    active
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "text-zinc-300 hover:bg-zinc-800 hover:text-white",
                  )}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-8 sm:px-6">
        {children}
      </main>
    </div>
  );
}
