from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Optional
from urllib import error, request

from .config import CategorizationConfig
from .models import CategoryResult, InstagramPost


CATEGORY_SUFFIX = ".ai_categories.json"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"


@dataclass
class Taxonomy:
    categories: Dict[str, str]
    signature: str
    dynamic_location_categories: tuple[str, ...] = ()

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


def _normalize_dynamic_location_categories(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def load_taxonomy(
    path: Optional[Path],
    fallback_category: str,
    dynamic_location_categories: Optional[list[str]] = None,
) -> Taxonomy:
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

    dynamic_roots = tuple(
        category
        for category in _normalize_dynamic_location_categories(dynamic_location_categories or [])
        if category in categories
    )
    signature = hashlib.sha256(
        json.dumps(
            {
                "categories": categories,
                "dynamic_location_categories": dynamic_roots,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return Taxonomy(
        categories=categories,
        signature=signature,
        dynamic_location_categories=dynamic_roots,
    )


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
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        confidence = None

    return CategoryResult(
        primary_category=primary,
        secondary_categories=secondary,
        confidence=confidence,
        reasoning=str(payload.get("reasoning", "")).strip(),
        model=str(payload.get("model", "")).strip(),
        provider=str(payload.get("provider", "ollama")).strip() or "ollama",
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
    lines = ["Generic categories:"]
    for name, description in taxonomy.categories.items():
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    if taxonomy.dynamic_location_categories:
        lines.append("")
        lines.append("Location-based category rule:")
        lines.append(
            "If the best generic fit is one of the location-aware categories below and the post has a clear place,"
            " prefer `<category>/<location>` instead of the bare category. Use exactly one slash and do not invent"
            " locations. Fall back to the bare generic category when the place is unclear."
        )
        for name in taxonomy.dynamic_location_categories:
            description = taxonomy.categories.get(name, "")
            if description:
                lines.append(f"- {name}/<location>: {description}")
            else:
                lines.append(f"- {name}/<location>")
    return "\n".join(lines)


def _dynamic_category_pattern(taxonomy: Taxonomy) -> str:
    roots = "|".join(re.escape(category) for category in taxonomy.dynamic_location_categories)
    if not roots:
        return ""
    return rf"^(?:{roots})/[^/\n]+$"


def _category_schema(taxonomy: Taxonomy) -> Dict[str, Any]:
    options: list[Dict[str, Any]] = [
        {
            "type": "string",
            "enum": taxonomy.names,
        }
    ]
    dynamic_pattern = _dynamic_category_pattern(taxonomy)
    if dynamic_pattern:
        options.append(
            {
                "type": "string",
                "pattern": dynamic_pattern,
            }
        )
    if len(options) == 1:
        return options[0]
    return {"anyOf": options}


def _categorization_schema(taxonomy: Taxonomy) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "primary_category": _category_schema(taxonomy),
            "secondary_categories": {
                "type": "array",
                "items": _category_schema(taxonomy),
                "uniqueItems": True,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["primary_category", "secondary_categories", "confidence", "reasoning"],
        "additionalProperties": False,
    }


def _categorization_tool(taxonomy: Taxonomy) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "categorize_saved_post",
            "description": "Return the best categories for the provided saved Instagram post.",
            "parameters": _categorization_schema(taxonomy),
        },
    }


def _resolved_model(config: CategorizationConfig) -> str:
    model = os.environ.get(OLLAMA_MODEL_ENV, "").strip() or config.model.strip()
    if not model:
        raise RuntimeError(
            "No Ollama model configured. Set categorization.model or the OLLAMA_MODEL environment variable."
        )
    return model


def _resolved_base_url(config: CategorizationConfig) -> str:
    base_url = os.environ.get(OLLAMA_HOST_ENV, "").strip() or config.base_url.strip()
    if not base_url:
        base_url = DEFAULT_OLLAMA_BASE_URL
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


def _chat_endpoint(config: CategorizationConfig) -> str:
    return f"{_resolved_base_url(config)}/api/chat"


def _send_ollama_chat(
    payload: Dict[str, Any],
    config: CategorizationConfig,
    *,
    client: object | None = None,
) -> Dict[str, Any]:
    if client is not None:
        chat = getattr(client, "chat", None)
        if callable(chat):
            response = chat(payload)
        elif callable(client):
            response = client(payload)
        else:
            raise TypeError("categorization client must be callable or expose a .chat(payload) method.")
        if not isinstance(response, dict):
            raise TypeError("categorization client must return a dict-shaped Ollama response.")
        return response

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        _chat_endpoint(config),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = detail or exc.reason
        raise RuntimeError(f"Ollama categorization request failed with HTTP {exc.code}: {message}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            "Could not reach the Ollama server. Start Ollama locally or set OLLAMA_HOST "
            f"(current endpoint: {_chat_endpoint(config)})."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON for the categorization request.") from exc


def _parse_tool_arguments(arguments: object) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        payload = json.loads(arguments)
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("Ollama returned invalid function-call arguments for categorization.")


def _extract_categorization_payload(response: Dict[str, Any]) -> Dict[str, Any]:
    message = response.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama categorization response did not include a message payload.")

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            if function.get("name") != "categorize_saved_post":
                continue
            return _parse_tool_arguments(function.get("arguments"))

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload

    raise RuntimeError("Ollama did not return a categorization function call or JSON response.")


def _normalize_category_name(value: object) -> str:
    raw = str(value).strip()
    if "/" not in raw:
        return raw
    prefix, suffix = raw.split("/", 1)
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix or not suffix:
        return raw
    return f"{prefix}/{suffix}"


def _is_dynamic_location_category(category: str, taxonomy: Taxonomy) -> bool:
    if "/" not in category:
        return False
    prefix, suffix = category.split("/", 1)
    if prefix not in taxonomy.dynamic_location_categories:
        return False
    return bool(suffix and "/" not in suffix)


def _is_allowed_category(category: str, taxonomy: Taxonomy) -> bool:
    return category in taxonomy.categories or _is_dynamic_location_category(category, taxonomy)


def _result_from_payload(
    payload: Dict[str, Any],
    taxonomy: Taxonomy,
    *,
    model_name: str,
) -> CategoryResult:
    primary_category = _normalize_category_name(payload.get("primary_category", ""))
    if not _is_allowed_category(primary_category, taxonomy):
        raise RuntimeError(
            f"Ollama returned an unknown primary category: {primary_category or '<empty>'}."
        )

    seen_secondary = set()
    secondary_categories = []
    for raw_value in payload.get("secondary_categories", []):
        category = _normalize_category_name(raw_value)
        if not category or category == primary_category or category in seen_secondary:
            continue
        if not _is_allowed_category(category, taxonomy):
            continue
        seen_secondary.add(category)
        secondary_categories.append(category)

    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        confidence = None

    return CategoryResult(
        primary_category=primary_category,
        secondary_categories=secondary_categories,
        confidence=confidence,
        reasoning=str(payload.get("reasoning", "")).strip(),
        model=model_name,
        provider="ollama",
        taxonomy_signature=taxonomy.signature,
        categorized_at=datetime.now(timezone.utc).isoformat(),
    )


def _classify_post_with_ollama(
    post: InstagramPost,
    config: CategorizationConfig,
    taxonomy: Taxonomy,
    *,
    client: object | None = None,
) -> CategoryResult:
    model_name = _resolved_model(config)
    schema = _categorization_schema(taxonomy)
    prompt = _build_post_text(post, config.max_input_chars)
    taxonomy_prompt = _build_taxonomy_prompt(taxonomy)
    request_payload = {
        "model": model_name,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You classify saved Instagram posts into a user-defined taxonomy. "
                    "Call the categorize_saved_post function exactly once. "
                    "If tool calling is unavailable, return raw JSON that matches the provided schema exactly. "
                    "When a configured location-aware category fits and the post includes a clear place, "
                    "prefer `<category>/<location>` over the bare category."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{taxonomy_prompt}\n\n"
                    "Choose one primary category, optional secondary categories, "
                    "a confidence between 0 and 1, and a short reasoning.\n\n"
                    f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                    f"{prompt}"
                ),
            },
        ],
        "tools": [_categorization_tool(taxonomy)],
        "stream": False,
        "options": {
            "temperature": config.temperature,
        },
    }
    response = _send_ollama_chat(request_payload, config, client=client)
    payload = _extract_categorization_payload(response)
    return _result_from_payload(payload, taxonomy, model_name=model_name)


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

    model_name = _resolved_model(config)
    _emit(
        progress,
        (
            f"Starting AI categorization with ollama:{model_name}. "
            f"Pending posts: {len(pending)}. Existing category sidecars skipped: {skipped_existing}."
        ),
    )
    written = 0
    for index, (post, output_path) in enumerate(pending, start=1):
        _emit(progress, f"[{index}/{len(pending)}] Categorizing {post.shortcode}")
        result = _classify_post_with_ollama(post, config, taxonomy, client=client)
        post.category_result = result
        post.category_file = output_path
        write_category_result(output_path, result)
        written += 1
        _emit(
            progress,
            f"[{index}/{len(pending)}] Saved {output_path.name} as {result.primary_category}",
        )

    return written
