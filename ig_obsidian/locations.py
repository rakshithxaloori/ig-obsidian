from __future__ import annotations

from collections import OrderedDict
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .models import InstagramPost, LocationRecord


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify_address(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return ", ".join(str(part).strip() for part in value.values() if str(part).strip())
    if isinstance(value, list):
        return ", ".join(str(part).strip() for part in value if str(part).strip())
    return str(value).strip()


def _record_from_payload(payload: Dict[str, Any], source: str) -> Optional[LocationRecord]:
    name = str(
        payload.get("name")
        or payload.get("title")
        or payload.get("place_name")
        or payload.get("location_name")
        or ""
    ).strip()
    description = str(payload.get("description") or payload.get("notes") or "").strip()
    address = _stringify_address(
        payload.get("address")
        or payload.get("address_json")
        or payload.get("formatted_address")
    )
    latitude = _coerce_float(payload.get("latitude") or payload.get("lat"))
    longitude = _coerce_float(payload.get("longitude") or payload.get("lng") or payload.get("lon"))

    if not any([name, address, latitude is not None and longitude is not None]):
        return None

    return LocationRecord(
        name=name,
        description=description,
        address=address,
        latitude=latitude,
        longitude=longitude,
        source=source,
    )


def locations_from_metadata(metadata: Dict[str, Any]) -> list[LocationRecord]:
    candidates: list[Any] = []
    for key in ("location", "locations", "venue", "place"):
        value = metadata.get(key)
        if value:
            candidates.append(value)

    records: list[LocationRecord] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            record = _record_from_payload(candidate, source="metadata")
            if record is not None:
                records.append(record)
            continue
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict):
                    record = _record_from_payload(item, source="metadata")
                    if record is not None:
                        records.append(record)
    return merge_location_records(records)


def load_location_overrides(path: Optional[Path]) -> Dict[str, list[LocationRecord]]:
    if path is None or not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("locations_file must contain a JSON object keyed by shortcode.")

    overrides: Dict[str, list[LocationRecord]] = {}
    for shortcode, raw_value in payload.items():
        if isinstance(raw_value, dict):
            items = [raw_value]
        elif isinstance(raw_value, list):
            items = raw_value
        else:
            raise ValueError("locations_file values must be objects or arrays of objects.")

        parsed: list[LocationRecord] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each locations_file entry must be a JSON object.")
            record = _record_from_payload(item, source="override")
            if record is not None:
                parsed.append(record)

        if parsed:
            overrides[str(shortcode).strip()] = merge_location_records(parsed)

    return overrides


def merge_location_records(records: Iterable[LocationRecord]) -> list[LocationRecord]:
    merged: "OrderedDict[tuple[str, ...], LocationRecord]" = OrderedDict()
    for record in records:
        key = record.map_key
        existing = merged.get(key)
        if existing is None:
            merged[key] = LocationRecord(
                name=record.name,
                description=record.description,
                address=record.address,
                latitude=record.latitude,
                longitude=record.longitude,
                source=record.source,
            )
            continue

        if record.source == "override":
            existing.source = "override"
        if record.name and (not existing.name or record.source == "override"):
            existing.name = record.name
        if record.description and (not existing.description or record.source == "override"):
            existing.description = record.description
        if record.address and (not existing.address or record.source == "override"):
            existing.address = record.address
        if record.latitude is not None and record.longitude is not None:
            existing.latitude = record.latitude
            existing.longitude = record.longitude
    return list(merged.values())


def attach_locations(
    posts: list[InstagramPost], location_overrides: Dict[str, list[LocationRecord]]
) -> list[InstagramPost]:
    for post in posts:
        metadata_records = locations_from_metadata(post.metadata)
        override_records = location_overrides.get(post.shortcode, [])
        post.locations = merge_location_records([*metadata_records, *override_records])
    return posts


def _snippet(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def export_google_maps_csv(
    posts: list[InstagramPost],
    output_path: Path,
    *,
    aggregate_by_location: bool = True,
    caption_snippet_length: int = 240,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, str]] = []

    aggregates: "OrderedDict[tuple[str, ...], dict[str, Any]]" = OrderedDict()
    for post in posts:
        for index, location in enumerate(post.locations):
            if aggregate_by_location:
                key = location.map_key
            else:
                key = (*location.map_key, post.shortcode, str(index))

            aggregate = aggregates.setdefault(
                key,
                {
                    "name": location.name,
                    "description_parts": [],
                    "address": location.address,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "instagram_urls": [],
                    "shortcodes": [],
                    "collections": [],
                },
            )

            if location.name and not aggregate["name"]:
                aggregate["name"] = location.name
            if location.address and not aggregate["address"]:
                aggregate["address"] = location.address
            if location.latitude is not None and location.longitude is not None:
                aggregate["latitude"] = location.latitude
                aggregate["longitude"] = location.longitude

            if location.description:
                aggregate["description_parts"].append(location.description)

            caption = _snippet(post.caption, caption_snippet_length) if post.caption else ""
            post_line = f"{post.shortcode} by {post.author}"
            if caption:
                post_line = f"{post_line}: {caption}"
            aggregate["description_parts"].append(post_line)
            aggregate["instagram_urls"].append(post.url)
            aggregate["shortcodes"].append(post.shortcode)
            aggregate["collections"].extend(post.collections)

    for aggregate in aggregates.values():
        unique_description_parts = list(OrderedDict.fromkeys(part for part in aggregate["description_parts"] if part))
        unique_urls = list(OrderedDict.fromkeys(aggregate["instagram_urls"]))
        unique_shortcodes = list(OrderedDict.fromkeys(aggregate["shortcodes"]))
        unique_collections = sorted(set(aggregate["collections"]))

        description_lines = unique_description_parts[:]
        if unique_collections:
            description_lines.append(f"Collections: {', '.join(unique_collections)}")
        if unique_urls:
            description_lines.append("Instagram URLs:")
            description_lines.extend(unique_urls)

        rows.append(
            {
                "Name": aggregate["name"] or aggregate["address"] or unique_shortcodes[0],
                "Description": "\n".join(description_lines),
                "Latitude": "" if aggregate["latitude"] is None else f"{aggregate['latitude']:.6f}",
                "Longitude": "" if aggregate["longitude"] is None else f"{aggregate['longitude']:.6f}",
                "Address": aggregate["address"] or "",
                "Instagram URL": unique_urls[0] if unique_urls else "",
                "Shortcodes": ", ".join(unique_shortcodes),
                "Collections": ", ".join(unique_collections),
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Name",
                "Description",
                "Latitude",
                "Longitude",
                "Address",
                "Instagram URL",
                "Shortcodes",
                "Collections",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
