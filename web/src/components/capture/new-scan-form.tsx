"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import type { UseFormRegisterReturn } from "react-hook-form";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";

import { createScanFormSchema, type CreateScanFormInput } from "@/lib/validation";

export function NewScanForm() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const form = useForm<CreateScanFormInput>({
    resolver: zodResolver(createScanFormSchema),
    defaultValues: {
      title: "",
      sport: "Baseball",
      year: "",
      brand: "",
      setName: "",
      playerName: "",
      cardNumber: "",
      serialNumber: "",
      disclaimerAccepted: false,
    },
  });

  const onSubmit = form.handleSubmit(async (values) => {
    setLoading(true);
    setError(null);
    const response = await fetch("/api/scans", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const payload = (await response.json().catch(() => ({}))) as {
      error?: string;
      scanId?: string;
    };

    if (!response.ok || !payload.scanId) {
      setError(payload.error ?? "Unable to create scan.");
      setLoading(false);
      return;
    }

    router.push(`/scans/${payload.scanId}`);
  });

  return (
    <form onSubmit={onSubmit} className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
      <div className="grid gap-4 sm:grid-cols-2">
        <TextField
          label="Card title"
          registration={form.register("title")}
          placeholder="2023 Topps Chrome Corbin Carroll RC"
        />
        <TextField label="Sport" registration={form.register("sport")} placeholder="Baseball" />
        <TextField label="Year" registration={form.register("year")} placeholder="2023" />
        <TextField label="Brand" registration={form.register("brand")} placeholder="Topps" />
        <TextField label="Set name" registration={form.register("setName")} placeholder="Chrome Update" />
        <TextField
          label="Player name"
          registration={form.register("playerName")}
          placeholder="Corbin Carroll"
        />
        <TextField
          label="Card number"
          registration={form.register("cardNumber")}
          placeholder="#US123"
        />
        <TextField
          label="Serial number"
          registration={form.register("serialNumber")}
          placeholder="025/150"
        />
      </div>

      <label className="flex gap-3 rounded-xl border border-zinc-700 bg-zinc-950 p-3 text-xs text-zinc-300">
        <input type="checkbox" {...form.register("disclaimerAccepted")} />
        <span>
          AI-assisted evaluation based on submitted images. This is not an official grade and does not
          guarantee any grading outcome.
        </span>
      </label>
      {form.formState.errors.disclaimerAccepted ? (
        <p className="text-xs text-rose-300">{form.formState.errors.disclaimerAccepted.message}</p>
      ) : null}

      {error ? <p className="text-sm text-rose-300">{error}</p> : null}

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-xl bg-indigo-500 px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
      >
        {loading ? "Creating..." : "Create scan"}
      </button>
    </form>
  );
}

function TextField({
  label,
  placeholder,
  registration,
}: {
  label: string;
  placeholder: string;
  registration: UseFormRegisterReturn;
}) {
  return (
    <label className="space-y-1 text-sm">
      <span className="text-zinc-400">{label}</span>
      <input
        {...registration}
        placeholder={placeholder}
        className="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100"
      />
    </label>
  );
}
