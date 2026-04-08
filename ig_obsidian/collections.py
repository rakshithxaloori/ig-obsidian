from __future__ import annotations

from collections import defaultdict
import json
import re
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Optional, Set


INSTAGRAM_URL_RE = re.compile(r"instagram\.com/(?:reel|p|tv)/(?P<shortcode>[A-Za-z0-9_-]{5,32})")
PLAIN_SHORTCODE_RE = re.compile(r"^[A-Za-z0-9_-]{5,32}$")


def normalize_shortcode(value: str) -> Optional[str]:
    candidate = value.strip()
    match = INSTAGRAM_URL_RE.search(candidate)
    if match:
        return match.group("shortcode")
    if PLAIN_SHORTCODE_RE.fullmatch(candidate):
        return candidate
    return None


def _add_values(
    mapping: DefaultDict[str, Set[str]], shortcode_values: Iterable[str], collection_name: str
) -> None:
    for value in shortcode_values:
        shortcode = normalize_shortcode(value)
        if shortcode:
            mapping[shortcode].add(collection_name.strip())


def load_collection_map(path: Optional[Path]) -> Dict[str, list[str]]:
    if path is None or not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: DefaultDict[str, Set[str]] = defaultdict(set)

    if not isinstance(payload, dict):
        raise ValueError("collections_file must contain a JSON object.")

    values_are_strings = all(isinstance(value, str) for value in payload.values())
    if values_are_strings:
        for shortcode_like, collection_name in payload.items():
            shortcode = normalize_shortcode(shortcode_like)
            if shortcode and collection_name.strip():
                mapping[shortcode].add(collection_name.strip())
        return {shortcode: sorted(values) for shortcode, values in mapping.items()}

    for collection_name, values in payload.items():
        if isinstance(values, str):
            _add_values(mapping, [values], collection_name)
            continue
        if isinstance(values, list):
            _add_values(mapping, [str(value) for value in values], collection_name)
            continue
        raise ValueError(
            "collections_file values must be strings or arrays of reel/post URLs or shortcodes."
        )

    return {shortcode: sorted(values) for shortcode, values in mapping.items()}
