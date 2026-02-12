"""Minimal eBay Browse API client for Buy It Now card scans."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests


EBAY_IDENTITY_BASE = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_BASE = "https://api.ebay.com"
EBAY_SANDBOX_BROWSE_BASE = "https://api.sandbox.ebay.com"


@dataclass
class EbayConfig:
    client_id: str
    client_secret: str
    env: str = "production"
    timeout_seconds: int = 30


class EbayBrowseClient:
    """Simple Browse API wrapper with app-token auth and search helpers."""

    def __init__(self, config: EbayConfig) -> None:
        self.config = config
        self._app_token: Optional[str] = None
        self._token_expiry_epoch: float = 0.0
        self._browse_base = (
            EBAY_BROWSE_BASE
            if config.env.lower() == "production"
            else EBAY_SANDBOX_BROWSE_BASE
        )

    def _token_is_valid(self) -> bool:
        return bool(self._app_token) and (time.time() < self._token_expiry_epoch - 60)

    def _build_basic_auth_header(self) -> str:
        creds = f"{self.config.client_id}:{self.config.client_secret}".encode("utf-8")
        return base64.b64encode(creds).decode("ascii")

    def get_app_token(self) -> str:
        if self._token_is_valid():
            return self._app_token or ""

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {self._build_basic_auth_header()}",
        }
        data = {
            "grant_type": "client_credentials",
            "scope": (
                "https://api.ebay.com/oauth/api_scope "
                "https://api.ebay.com/oauth/api_scope/buy.browse"
            ),
        }

        resp = requests.post(
            EBAY_IDENTITY_BASE,
            headers=headers,
            data=data,
            timeout=self.config.timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"eBay token request failed ({resp.status_code}): {resp.text[:500]}"
            )

        payload = resp.json()
        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 7200))
        if not token:
            raise RuntimeError("eBay token response missing access_token.")

        self._app_token = token
        self._token_expiry_epoch = time.time() + expires_in
        return token

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_app_token()}",
            "Content-Type": "application/json",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        }

    def search_buy_it_now(
        self,
        query: str,
        max_price: float,
        limit: int = 30,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search Buy It Now listings in US with a max item price."""
        safe_limit = max(1, min(limit, 200))
        filter_fragments = [
            "buyingOptions:{FIXED_PRICE}",
            f"price:[..{max_price:.2f}]",
            "itemLocationCountry:US",
        ]
        params = {
            "q": query,
            "limit": safe_limit,
            "offset": max(offset, 0),
            "sort": "newlyListed",
            "filter": ",".join(filter_fragments),
        }
        url = f"{self._browse_base}/buy/browse/v1/item_summary/search"
        resp = requests.get(
            url,
            headers=self._auth_headers(),
            params=params,
            timeout=self.config.timeout_seconds,
        )

        if resp.status_code >= 400:
            raise RuntimeError(
                f"eBay search failed ({resp.status_code}) for '{query}': "
                f"{resp.text[:500]}"
            )

        payload = resp.json()
        summaries = payload.get("itemSummaries", []) or []
        if not isinstance(summaries, list):
            return []
        return [self._normalize_summary(item) for item in summaries if isinstance(item, dict)]

    def get_item(self, item_id: str) -> Dict[str, Any]:
        """Fetch full listing details for richer image coverage."""
        encoded_item_id = quote(item_id, safe="")
        url = f"{self._browse_base}/buy/browse/v1/item/{encoded_item_id}"
        resp = requests.get(
            url,
            headers=self._auth_headers(),
            timeout=self.config.timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"eBay item lookup failed ({resp.status_code}) for {item_id}: "
                f"{resp.text[:500]}"
            )
        payload = resp.json()
        if not isinstance(payload, dict):
            return {}
        return payload

    def extract_image_urls(
        self,
        item_summary: Optional[Dict[str, Any]] = None,
        item_detail: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Collect candidate listing image URLs from summary + detail payloads."""
        urls: List[str] = []

        for payload in (item_summary or {}, item_detail or {}):
            # Known common fields first so we preserve practical ordering.
            image = payload.get("image")
            if isinstance(image, dict) and isinstance(image.get("imageUrl"), str):
                urls.append(image["imageUrl"])

            for key in ("additionalImages", "thumbnailImages"):
                raw = payload.get(key)
                if isinstance(raw, list):
                    for entry in raw:
                        if isinstance(entry, dict) and isinstance(
                            entry.get("imageUrl"), str
                        ):
                            urls.append(entry["imageUrl"])

            self._collect_image_urls(payload, urls)

        deduped: List[str] = []
        seen = set()
        for raw_url in urls:
            url = str(raw_url).strip()
            if not url.lower().startswith("http"):
                continue
            if url in seen:
                continue
            deduped.append(url)
            seen.add(url)
        return deduped

    def _collect_image_urls(self, node: Any, sink: List[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() == "imageurl" and isinstance(value, str):
                    sink.append(value)
                else:
                    self._collect_image_urls(value, sink)
        elif isinstance(node, list):
            for value in node:
                self._collect_image_urls(value, sink)

    def _normalize_summary(self, item: Dict[str, Any]) -> Dict[str, Any]:
        price_obj = item.get("price", {}) if isinstance(item.get("price"), dict) else {}
        price_value_raw = price_obj.get("value")
        try:
            price_value = float(price_value_raw) if price_value_raw is not None else None
        except (TypeError, ValueError):
            price_value = None

        return {
            "itemId": item.get("itemId"),
            "title": item.get("title"),
            "itemWebUrl": item.get("itemWebUrl"),
            "buyingOptions": item.get("buyingOptions", []),
            "price": {
                "value": price_value,
                "currency": price_obj.get("currency"),
            },
            "raw": item,
        }
