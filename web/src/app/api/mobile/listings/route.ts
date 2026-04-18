import { NextRequest } from "next/server";

import { requireUserOrResponse } from "@/lib/auth";
import { badRequest, ok, serverError } from "@/lib/http";
import {
  createMarketplaceListingForUser,
  listMarketplaceListingsForMobile,
  mapListingForMobile,
} from "@/lib/mobile-api";

export async function POST(request: NextRequest) {
  const auth = await requireUserOrResponse();
  if (!auth.ok) {
    return auth.response;
  }

  try {
    const payload = await request.json();
    const result = await createMarketplaceListingForUser(auth.user.id, payload);
    if (!result.ok) {
      if (result.error === "invalid_listing_payload") {
        return badRequest("Invalid listing payload.", result.issues);
      }
      if (result.error === "scan_not_found") {
        return badRequest("Scan not found or not owned by user.");
      }
      if (result.error === "scan_missing_analysis") {
        return badRequest("Run analysis before creating a listing.");
      }
      return badRequest("Listing already exists for this scan.");
    }

    return ok({ listing: mapListingForMobile(result.listing as never) }, 201);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to create listing.";
    return serverError(message);
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const listings = await listMarketplaceListingsForMobile({
      sport: searchParams.get("sport"),
      player: searchParams.get("player"),
      category: searchParams.get("category"),
      minConfidence: searchParams.get("minConfidence"),
      maxPrice: searchParams.get("maxPrice"),
    });
    return ok({
      listings: listings.map((listing) => mapListingForMobile(listing as never)),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch listings.";
    return serverError(message);
  }
}
