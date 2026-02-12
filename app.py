"""PSA 10 centering spotter scan CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

from ebay import EbayBrowseClient, EbayConfig
from photo_grader import ALLOWED_REJECT_REASONS, PhotoGrader
from seen_store import SeenItemStore


EXCLUDED_TITLE_KEYWORDS = {
    "psa",
    "bgs",
    "sgc",
    "graded",
    "slab",
    "lot",
    "lots",
    "break",
    "breaks",
    "pack",
    "packs",
    "box",
    "boxes",
    "digital",
    "reprint",
    "custom",
    "repack",
}


@dataclass
class ScanConfig:
    ebay_client_id: str
    ebay_client_secret: str
    ebay_env: str
    openai_api_key: str
    openai_vision_model: str
    max_price: float
    scan_limit_per_query: int
    grade_cap: int
    require_2_images: bool
    targets_path: Path
    seen_ids_path: Path
    candidates_out_path: Path
    sendgrid_api_key: str
    sendgrid_from_email: str
    sendgrid_to_email: str = "jwgray165@gmail.com"


def load_config() -> ScanConfig:
    load_dotenv()
    ebay_client_id = os.getenv("EBAY_CLIENT_ID", "").strip()
    ebay_client_secret = os.getenv("EBAY_CLIENT_SECRET", "").strip()
    ebay_env = os.getenv("EBAY_ENV", "production").strip() or "production"

    return ScanConfig(
        ebay_client_id=ebay_client_id,
        ebay_client_secret=ebay_client_secret,
        ebay_env=ebay_env,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini").strip()
        or "gpt-4.1-mini",
        max_price=_env_float("MAX_PRICE", 100.0),
        scan_limit_per_query=_env_int("SCAN_LIMIT_PER_QUERY", 30),
        grade_cap=_env_int("GRADE_CAP", 40),
        require_2_images=_env_bool("REQUIRE_2_IMAGES", True),
        targets_path=Path("data/targets.json"),
        seen_ids_path=Path("data/seen_item_ids.json"),
        candidates_out_path=Path("data/candidates.json"),
        sendgrid_api_key=os.getenv("SENDGRID_API_KEY", "").strip(),
        sendgrid_from_email=os.getenv("SENDGRID_FROM_EMAIL", "").strip(),
        sendgrid_to_email=os.getenv("SENDGRID_TO_EMAIL", "jwgray165@gmail.com").strip()
        or "jwgray165@gmail.com",
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def load_targets(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Targets config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    buckets = payload.get("buckets", [])
    if not isinstance(buckets, list):
        raise ValueError("targets.json must contain a 'buckets' array.")
    return [b for b in buckets if isinstance(b, dict)]


def title_has_excluded_keyword(title: str) -> bool:
    title_tokens = _tokenize(title)
    if not title_tokens:
        return False
    if any(token.startswith("psa") for token in title_tokens):
        return True
    for token in EXCLUDED_TITLE_KEYWORDS:
        if token in title_tokens:
            return True
    return False


def _tokenize(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in (text or ""))
    return [token for token in cleaned.split() if token]


def extract_price(summary: Dict[str, Any]) -> Tuple[float | None, str | None]:
    price = summary.get("price") if isinstance(summary.get("price"), dict) else {}
    value = price.get("value")
    currency = price.get("currency")
    return value, currency


def has_fixed_price_buying_option(summary: Dict[str, Any]) -> bool:
    options = summary.get("buyingOptions")
    if not isinstance(options, list):
        return False
    normalized = {str(o).upper() for o in options}
    return "FIXED_PRICE" in normalized


def run_scan(config: ScanConfig) -> int:
    _validate_config(config)

    ebay = EbayBrowseClient(
        EbayConfig(
            client_id=config.ebay_client_id,
            client_secret=config.ebay_client_secret,
            env=config.ebay_env,
        )
    )
    grader = PhotoGrader(
        model=config.openai_vision_model,
        require_two_images=config.require_2_images,
        api_key=config.openai_api_key,
    )

    seen_store = SeenItemStore(str(config.seen_ids_path))
    previously_seen = seen_store.load()

    buckets = load_targets(config.targets_path)
    if not buckets:
        raise RuntimeError("No target buckets configured in data/targets.json.")

    query_stats: List[Dict[str, Any]] = []
    unique_candidates: Dict[str, Dict[str, Any]] = {}
    query_errors: List[Dict[str, Any]] = []

    for bucket in buckets:
        bucket_id = str(bucket.get("bucket_id", "unknown_bucket"))
        player = str(bucket.get("player", "unknown_player"))
        queries = bucket.get("queries", [])
        if not isinstance(queries, list) or not queries:
            query_stats.append(
                {
                    "bucket_id": bucket_id,
                    "player": player,
                    "query": "<none>",
                    "returned": 0,
                    "error": "missing_queries",
                }
            )
            continue

        for query in queries:
            safe_query = str(query).strip()
            if not safe_query:
                continue
            lowered = f" {safe_query.lower()} "
            if "(" in safe_query or ")" in safe_query or " or " in lowered:
                query_stats.append(
                    {
                        "bucket_id": bucket_id,
                        "player": player,
                        "query": safe_query,
                        "returned": 0,
                        "error": "query_contains_disallowed_operators",
                    }
                )
                continue
            try:
                items = ebay.search_buy_it_now(
                    query=safe_query,
                    max_price=config.max_price,
                    limit=config.scan_limit_per_query,
                )
            except Exception as exc:  # noqa: BLE001
                query_stats.append(
                    {
                        "bucket_id": bucket_id,
                        "player": player,
                        "query": safe_query,
                        "returned": 0,
                        "error": str(exc),
                    }
                )
                query_errors.append(
                    {
                        "bucket_id": bucket_id,
                        "query": safe_query,
                        "error": str(exc),
                    }
                )
                continue

            query_stats.append(
                {
                    "bucket_id": bucket_id,
                    "player": player,
                    "query": safe_query,
                    "returned": len(items),
                }
            )

            for item in items:
                item_id = str(item.get("itemId") or "").strip()
                if not item_id:
                    continue
                existing = unique_candidates.get(item_id)
                if existing:
                    existing.setdefault("matched_queries", []).append(safe_query)
                    continue

                unique_candidates[item_id] = {
                    "itemId": item_id,
                    "title": item.get("title") or "",
                    "itemWebUrl": item.get("itemWebUrl") or "",
                    "price": item.get("price") or {},
                    "buyingOptions": item.get("buyingOptions") or [],
                    "raw_summary": item.get("raw") or {},
                    "bucket_id": bucket_id,
                    "player": player,
                    "matched_queries": [safe_query],
                }

    reject_counts: Counter[str] = Counter()
    decisions: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    stats = defaultdict(int)
    stats["total_scanned"] = len(unique_candidates)

    for item_id, listing in unique_candidates.items():
        if item_id in previously_seen:
            stats["already_seen_skipped"] += 1
            continue

        if not has_fixed_price_buying_option(listing):
            stats["non_bin_skipped"] += 1
            seen_store.add(item_id)
            continue

        price_value, _ = extract_price(listing)
        if price_value is None or float(price_value) > config.max_price:
            stats["price_skipped"] += 1
            seen_store.add(item_id)
            continue

        title = str(listing.get("title") or "")
        if title_has_excluded_keyword(title):
            stats["filtered_by_title"] += 1
            decisions.append(
                {
                    "itemId": listing.get("itemId"),
                    "title": listing.get("title"),
                    "url": listing.get("itemWebUrl"),
                    "bucket_id": listing.get("bucket_id"),
                    "player": listing.get("player"),
                    "matched_queries": listing.get("matched_queries") or [],
                    "decision_type": "filtered_before_grading",
                    "skip_reason": "excluded_title_keyword",
                }
            )
            seen_store.add(item_id)
            continue

        if stats["graded_count"] >= config.grade_cap:
            stats["grade_cap_skipped"] += 1
            continue

        detail_payload: Dict[str, Any] = {}
        try:
            detail_payload = ebay.get_item(item_id)
        except Exception as exc:  # noqa: BLE001
            detail_payload = {}
            stats["detail_fetch_errors"] += 1
            listing.setdefault("debug_notes", []).append(f"detail_fetch_failed: {exc}")

        image_urls = ebay.extract_image_urls(
            item_summary=listing.get("raw_summary"),
            item_detail=detail_payload,
        )

        if config.require_2_images and len(image_urls) < 2:
            reason = "missing_front" if len(image_urls) == 0 else "missing_back"
            reject_counts[reason] += 1
            decisions.append(
                _build_decision_record(
                    listing=listing,
                    grade={
                        "grade_lane": "reject",
                        "reject_reason": reason,
                        "confidence": 0.05,
                        "notes": "Listing does not provide both front and back photos.",
                    },
                    image_urls=image_urls,
                )
            )
            seen_store.add(item_id)
            continue

        grade = grader.grade_centering(
            item_id=item_id,
            title=title,
            image_urls=image_urls,
        )
        stats["graded_count"] += 1
        decision = _build_decision_record(listing=listing, grade=grade, image_urls=image_urls)
        decisions.append(decision)
        seen_store.add(item_id)

        if grade.get("grade_lane") == "psa10_centering_ok":
            kept.append(_build_kept_record(listing=listing, grade=grade))
        else:
            reason = str(grade.get("reject_reason") or "cannot_judge_centering")
            if reason not in ALLOWED_REJECT_REASONS:
                reason = "cannot_judge_centering"
            reject_counts[reason] += 1

    stats["kept_count"] = len(kept)
    stats["rejected_count"] = len(decisions) - len(kept)
    stats["seen_store_size"] = len(seen_store.seen)
    stats["queries_run"] = len(query_stats)
    stats["query_results_total"] = sum(
        int(row.get("returned", 0) or 0) for row in query_stats
    )

    seen_store.save()
    top_reject_reasons = reject_counts.most_common(10)

    output_payload = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_price": config.max_price,
            "scan_limit_per_query": config.scan_limit_per_query,
            "grade_cap": config.grade_cap,
            "require_2_images": config.require_2_images,
            "openai_vision_model": config.openai_vision_model,
            "ebay_env": config.ebay_env,
        },
        "stats": dict(stats),
        "query_stats": query_stats,
        "query_errors": query_errors,
        "top_reject_reasons": [
            {"reason": reason, "count": count} for reason, count in top_reject_reasons
        ],
        "kept": kept,
        "decisions": decisions,
    }
    config.candidates_out_path.parent.mkdir(parents=True, exist_ok=True)
    config.candidates_out_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    _print_summary(output_payload, config)
    _maybe_send_sendgrid_summary(
        config=config,
        kept=kept,
        top_reject_reasons=top_reject_reasons,
    )
    return 0


def _build_decision_record(
    listing: Dict[str, Any],
    grade: Dict[str, Any],
    image_urls: List[str] | None = None,
) -> Dict[str, Any]:
    price_value, currency = extract_price(listing)
    return {
        "itemId": listing.get("itemId"),
        "title": listing.get("title"),
        "url": listing.get("itemWebUrl"),
        "price": price_value,
        "currency": currency,
        "bucket_id": listing.get("bucket_id"),
        "player": listing.get("player"),
        "matched_queries": listing.get("matched_queries") or [],
        "image_count": len(image_urls or []),
        "grade": grade,
    }


def _build_kept_record(listing: Dict[str, Any], grade: Dict[str, Any]) -> Dict[str, Any]:
    price_value, currency = extract_price(listing)
    return {
        "itemId": listing.get("itemId"),
        "title": listing.get("title"),
        "url": listing.get("itemWebUrl"),
        "price": price_value,
        "currency": currency,
        "bucket_id": listing.get("bucket_id"),
        "player": listing.get("player"),
        "matched_queries": listing.get("matched_queries") or [],
        "confidence": grade.get("confidence"),
        "centering": {
            "front_lr": grade.get("est_front_lr_ratio"),
            "front_tb": grade.get("est_front_tb_ratio"),
            "back_lr": grade.get("est_back_lr_ratio"),
            "back_tb": grade.get("est_back_tb_ratio"),
            "front_lr_call": grade.get("centering_front_lr"),
            "front_tb_call": grade.get("centering_front_tb"),
            "back_lr_call": grade.get("centering_back_lr"),
            "back_tb_call": grade.get("centering_back_tb"),
        },
        "notes": grade.get("notes"),
    }


def _print_summary(output_payload: Dict[str, Any], config: ScanConfig) -> None:
    stats = output_payload.get("stats", {})
    top = output_payload.get("top_reject_reasons", [])
    query_stats = output_payload.get("query_stats", [])

    print("=== PSA 10 Centering Spotter Run Summary ===")
    print(f"Total scanned: {stats.get('total_scanned', 0)}")
    print(f"Graded count: {stats.get('graded_count', 0)}")
    print(f"Kept count: {stats.get('kept_count', 0)}")
    print(f"Already seen skipped: {stats.get('already_seen_skipped', 0)}")
    print(f"Non-BIN skipped: {stats.get('non_bin_skipped', 0)}")
    print(f"Price skipped: {stats.get('price_skipped', 0)}")
    print(f"Filtered by title keywords: {stats.get('filtered_by_title', 0)}")
    print(f"Grade-cap skipped: {stats.get('grade_cap_skipped', 0)}")
    print(f"Saved candidates JSON: {config.candidates_out_path}")

    print("\nTop reject reasons:")
    if top:
        for row in top:
            print(f"  - {row.get('reason')}: {row.get('count')}")
    else:
        print("  - none")

    print("\nQuery returns:")
    for row in query_stats:
        bucket_id = row.get("bucket_id")
        query = row.get("query")
        returned = row.get("returned", 0)
        error = row.get("error")
        if error:
            print(f"  - [{bucket_id}] {query}: ERROR {error}")
        else:
            print(f"  - [{bucket_id}] {query}: {returned}")

    bucket_rollup: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"queries": 0, "nonzero_queries": 0, "items_returned": 0}
    )
    for row in query_stats:
        bucket_id = str(row.get("bucket_id"))
        bucket_rollup[bucket_id]["queries"] += 1
        returned = int(row.get("returned", 0) or 0)
        bucket_rollup[bucket_id]["items_returned"] += returned
        if returned > 0:
            bucket_rollup[bucket_id]["nonzero_queries"] += 1

    print("\nBucket query coverage:")
    for bucket_id, rollup in bucket_rollup.items():
        print(
            f"  - {bucket_id}: queries={rollup['queries']}, "
            f"nonzero_queries={rollup['nonzero_queries']}, "
            f"items_returned={rollup['items_returned']}"
        )

    if stats.get("total_scanned", 0) == 0:
        print(
            "\nNo listings scanned. Broaden query breadth in data/targets.json, "
            "increase SCAN_LIMIT_PER_QUERY, and verify EBAY_ENV=production."
        )
    elif stats.get("kept_count", 0) == 0:
        print(
            "\nNo kept candidates this run. To widen the net, add more safe keyword "
            "variants per bucket in data/targets.json, increase SCAN_LIMIT_PER_QUERY, "
            "and raise GRADE_CAP."
        )


def _validate_config(config: ScanConfig) -> None:
    missing = []
    if not config.ebay_client_id:
        missing.append("EBAY_CLIENT_ID")
    if not config.ebay_client_secret:
        missing.append("EBAY_CLIENT_SECRET")
    if not config.openai_api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Set them in .env."
        )


def _maybe_send_sendgrid_summary(
    config: ScanConfig,
    kept: List[Dict[str, Any]],
    top_reject_reasons: List[Tuple[str, int]],
) -> None:
    if not config.sendgrid_api_key:
        return
    if not kept:
        return

    sender = config.sendgrid_from_email or config.sendgrid_to_email
    subject = f"PSA 10 centering spotter: {len(kept)} kept candidates"
    lines = [
        f"Kept candidates: {len(kept)}",
        "",
    ]
    for idx, row in enumerate(kept, start=1):
        lines.append(f"{idx}. {row.get('title')}")
        lines.append(
            f"   Price: {row.get('price')} {row.get('currency')} | "
            f"Confidence: {row.get('confidence')}"
        )
        c = row.get("centering", {})
        lines.append(
            "   Centering est: "
            f"front LR {c.get('front_lr')}, front TB {c.get('front_tb')}, "
            f"back LR {c.get('back_lr')}, back TB {c.get('back_tb')}"
        )
        lines.append(f"   URL: {row.get('url')}")
        lines.append("")

    if top_reject_reasons:
        lines.append("Top reject reasons:")
        for reason, count in top_reject_reasons[:5]:
            lines.append(f"- {reason}: {count}")

    payload = {
        "personalizations": [{"to": [{"email": config.sendgrid_to_email}]}],
        "from": {"email": sender},
        "subject": subject,
        "content": [{"type": "text/plain", "value": "\n".join(lines)}],
    }
    headers = {
        "Authorization": f"Bearer {config.sendgrid_api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers=headers,
            data=json.dumps(payload),
            timeout=20,
        )
        if resp.status_code >= 400:
            print(
                "SendGrid email failed: "
                f"{resp.status_code} {resp.text[:400]}",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"SendGrid email error: {exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSA 10 centering spotter")
    parser.add_argument("--mode", choices=["scan"], default="scan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()

    if args.mode == "scan":
        return run_scan(config)
    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
