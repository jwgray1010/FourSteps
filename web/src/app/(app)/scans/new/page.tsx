import { redirect } from "next/navigation";

import { NewScanForm } from "@/components/capture/new-scan-form";
import { getCurrentUser } from "@/lib/auth";

export default async function NewScanPage() {
  const user = await getCurrentUser();
  if (!user) {
    redirect("/signin");
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Create new scan</h1>
      <p className="text-sm text-zinc-400">
        Start by entering card details and accepting RAWIFY disclaimer language.
      </p>
      <NewScanForm />
    </div>
  );
}
