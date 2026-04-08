from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from .config import CategorizationConfig
from .models import CategoryResult, InstagramPost


CATEGORY_SUFFIX = ".ai_categories.json"


@dataclass
class Taxonomy:
    categories: Dict[str, str]
    signature: str

    @property
    def names(self) -> list[str]:
        return list(self.categories.keys())


def _emit(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)


def _normalize_taxonomy_payload(payload: object) -> Dict[str, str]:
    if isinstance(payload, list):
        normalized: Dict[str, str] = {}
        for item in payload:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("taxonomy_file list items must be non-empty strings.")
            normalized[item.strip()] = ""
        return normalized

    if isinstance(payload, dict):
        normalized = {}
        for raw_name, raw_value in payload.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("taxonomy_file category names must be non-empty.")
            if isinstance(raw_value, str):
                normalized[name] = raw_value.strip()
                continue
            if isinstance(raw_value, dict):
                description = raw_value.get("description", "")
                normalized[name] = str(description).strip()
                continue
            raise ValueError("taxonomy_file object values must be strings or objects.")
        return normalized

    raise ValueError("taxonomy_file must contain either a JSON object or an array of category names.")


def load_taxonomy(path: Optional[Path], fallback_category: str) -> Taxonomy:
    if path is None:
        raise ValueError("categorization.taxonomy_file is required when categorization is enabled.")
    if not path.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found: {path}. Copy taxonomy.example.json to taxonomy.json first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = _normalize_taxonomy_payload(payload)
    fallback = fallback_category.strip()
    if fallback and fallback not in categories:
        categories[fallback] = "Fallback bucket when no configured category is a good fit."

    if not categories:
        raise ValueError("taxonomy_file must define at least one category.")

    signature = hashlib.sha256(
        json.dumps(categories, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return Taxonomy(categories=categories, signature=signature)


def _category_anchor_path(post: InstagramPost) -> Optional[Path]:
    for candidate in (post.category_file, post.metadata_file, post.caption_file, *post.media_files):
        if candidate is not None:
            return candidate
    return None


def category_output_path(post: InstagramPost) -> Optional[Path]:
    anchor = _category_anchor_path(post)
    if anchor is None:
        return None
    if anchor.name.endswith(CATEGORY_SUFFIX):
        return anchor
    base_name = anchor.name
    for suffix in (".json.xz", ".json", ".txt", ".mp4", ".mov", ".m4v", ".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    return anchor.with_name(f"{base_name}{CATEGORY_SUFFIX}")


def load_category_result_from_path(path: Path) -> Optional[CategoryResult]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    primary = str(payload.get("primary_category", "")).strip()
    if not primary:
        return None

    secondary = [str(value).strip() for value in payload.get("secondary_categories", []) if str(value).strip()]
    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None

    return CategoryResult(
        primary_category=primary,
        secondary_categories=secondary,
        confidence=confidence,
        reasoning=str(payload.get("reasoning", "")).strip(),
        model=str(payload.get("model", "")).strip(),
        provider=str(payload.get("provider", "openai")).strip() or "openai",
        taxonomy_signature=str(payload.get("taxonomy_signature", "")).strip(),
        categorized_at=str(payload.get("categorized_at", "")).strip(),
    )


def write_category_result(path: Path, result: CategoryResult) -> None:
    payload = {
        "primary_category": result.primary_category,
        "secondary_categories": result.secondary_categories,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "model": result.model,
        "provider": result.provider,
        "taxonomy_signature": result.taxonomy_signature,
        "categorized_at": result.categorized_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_post_text(post: InstagramPost, max_input_chars: int) -> str:
    sections = [
        f"Author: {post.author}",
        f"Shortcode: {post.shortcode}",
        f"URL: {post.url}",
    ]
    if post.collections:
        sections.append(f"Collections: {', '.join(post.collections)}")
    if post.locations:
        sections.append(
            "Locations: "
            + "; ".join(
                " | ".join(
                    part
                    for part in [
                        location.name,
                        location.address,
                        (
                            f"{location.latitude:.6f}, {location.longitude:.6f}"
                            if location.latitude is not None and location.longitude is not None
                            else ""
                        ),
                    ]
                    if part
                )
                for location in post.locations
            )
        )
    sections.append(f"Caption:\n{post.caption or '(none)'}")
    if post.transcript.strip():
        sections.append(f"Transcript:\n{post.transcript.strip()}")

    joined = "\n\n".join(sections).strip()
    return joined[:max_input_chars]


def _build_taxonomy_prompt(taxonomy: Taxonomy) -> str:
    lines = ["Use only these categories:"]
    for name, description in taxonomy.categories.items():
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def _classify_post_with_openai(
    post: InstagramPost,
    config: CategorizationConfig,
    taxonomy: Taxonomy,
    *,
    client: object,
) -> CategoryResult:
    prompt = _build_post_text(post, config.max_input_chars)
    taxonomy_prompt = _build_taxonomy_prompt(taxonomy)
    schema = {
        "type": "object",
        "properties": {
            "primary_category": {
                "type": "string",
                "enum": taxonomy.names,
            },
            "secondary_categories": {
                "type": "array",
                "items": {"type": "string", "enum": taxonomy.names},
            },
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["primary_category", "secondary_categories", "confidence", "reasoning"],
        "additionalProperties": False,
    }

    response = client.responses.create(
        model=config.model,
        input=[
            {
                "role": "system",
                "content": (
                    "You classify saved Instagram posts into a user-defined taxonomy. "
                    "Use only the provided categories. Prefer the fallback category when uncertain."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{taxonomy_prompt}\n\n"
                    "Return the best primary category, any secondary categories, "
                    "a confidence between 0 and 1, and a short reasoning.\n\n"
                    f"{prompt}"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "post_categorization",
                "schema": schema,
                "strict": True,
            }
        },
    )
    payload = json.loads(response.output_text)
    return CategoryResult(
        primary_category=str(payload["primary_category"]).strip(),
        secondary_categories=[
            str(value).strip()
            for value in payload.get("secondary_categories", [])
            if str(value).strip() and str(value).strip() != str(payload["primary_category"]).strip()
        ],
        confidence=float(payload["confidence"]),
        reasoning=str(payload["reasoning"]).strip(),
        model=config.model,
        provider=config.provider,
        taxonomy_signature=taxonomy.signature,
        categorized_at=datetime.now(timezone.utc).isoformat(),
    )


def categorize_posts(
    posts: list[InstagramPost],
    config: CategorizationConfig,
    taxonomy: Taxonomy,
    *,
    progress: Optional[Callable[[str], None]] = None,
    client: object | None = None,
) -> int:
    if not config.enabled:
        _emit(progress, "AI categorization disabled; skipping.")
        return 0

    if config.provider != "openai":
        raise RuntimeError(f"Unsupported categorization provider: {config.provider}")

    pending: list[tuple[InstagramPost, Path]] = []
    skipped_existing = 0
    for post in posts:
        output_path = category_output_path(post)
        if output_path is None:
            continue
        existing = post.category_result
        if (
            existing is not None
            and existing.taxonomy_signature == taxonomy.signature
            and not config.overwrite
        ):
            skipped_existing += 1
            continue
        pending.append((post, output_path))

    if not pending:
        _emit(
            progress,
            f"No posts need AI categorization. Existing category sidecars skipped: {skipped_existing}.",
        )
        return 0

    api_key = os.environ.get(config.api_key_env)
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK is not installed. Reinstall with `pip install -e '.[categorize]'` "
                "or `pip install -r requirements.txt`."
            ) from exc

        if not api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env} is required for AI categorization."
            )
        client = OpenAI(api_key=api_key)

    _emit(
        progress,
        (
            f"Starting AI categorization with {config.provider}:{config.model}. "
            f"Pending posts: {len(pending)}. Existing category sidecars skipped: {skipped_existing}."
        ),
    )
    written = 0
    for index, (post, output_path) in enumerate(pending, start=1):
        _emit(progress, f"[{index}/{len(pending)}] Categorizing {post.shortcode}")
        result = _classify_post_with_openai(post, config, taxonomy, client=client)
        post.category_result = result
        post.category_file = output_path
        write_category_result(output_path, result)
        written += 1
        _emit(
            progress,
            f"[{index}/{len(pending)}] Saved {output_path.name} as {result.primary_category}",
        )

    return written
