from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Iterable

from .config import AppConfig
from .models import InstagramPost


def _slugify_tag(value: str) -> str:
    lowered = value.strip().lower()
    collapsed = re.sub(r"[^a-z0-9]+", "-", lowered)
    return collapsed.strip("-")


def _category_tag(value: str) -> str:
    parts = [segment for segment in value.split("/") if segment.strip()]
    if not parts:
        return ""
    return "category/" + "/".join(_slugify_tag(segment) for segment in parts)


def _yaml_lines(key: str, value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []

    if isinstance(value, bool):
        return [f"{key}: {'true' if value else 'false'}"]

    if isinstance(value, (int, float)):
        return [f"{key}: {value}"]

    if isinstance(value, str):
        if "\n" in value:
            lines = [f"{key}: |-"]
            lines.extend(f"  {line}" for line in value.splitlines())
            return lines
        return [f"{key}: {json.dumps(value, ensure_ascii=False)}"]

    if isinstance(value, Iterable):
        lines = [f"{key}:"]
        for item in value:
            lines.append(f"  - {json.dumps(str(item), ensure_ascii=False)}")
        return lines

    raise TypeError(f"Unsupported frontmatter value for {key}: {type(value)!r}")


def _frontmatter(post: InstagramPost, media_links: list[str], tags: list[str]) -> str:
    location_names = [location.name for location in post.locations if location.name]
    location_addresses = [location.address for location in post.locations if location.address]
    category_result = post.category_result
    fields: Dict[str, Any] = {
        "type": "instagram-post",
        "instagram_kind": post.kind,
        "shortcode": post.shortcode,
        "author": post.author,
        "url": post.url,
        "post_date": post.date.isoformat() if post.date else None,
        "saved_collection": post.collections[0] if len(post.collections) == 1 else None,
        "saved_collections": post.collections if len(post.collections) > 1 else None,
        "ai_primary_category": category_result.primary_category if category_result else None,
        "ai_secondary_categories": (
            category_result.secondary_categories if category_result and category_result.secondary_categories else None
        ),
        "ai_category_confidence": category_result.confidence if category_result else None,
        "ai_category_model": category_result.model if category_result else None,
        "location_names": location_names,
        "location_addresses": location_addresses,
        "media_files": media_links,
        "has_transcript": bool(post.transcript.strip()),
        "tags": tags,
    }
    lines = ["---"]
    for key, value in fields.items():
        lines.extend(_yaml_lines(key, value))
    lines.append("---")
    return "\n".join(lines)


def _note_body(post: InstagramPost, media_links: list[str], embed_media: bool) -> str:
    sections: list[str] = []
    if embed_media and media_links:
        sections.extend(f"![[{link}]]" for link in media_links)

    sections.append("## Caption")
    sections.append(post.caption or "_No caption saved._")

    if post.transcript.strip():
        sections.append("## Transcript")
        sections.append(post.transcript.strip())

    if post.collections:
        sections.append("## Collections")
        sections.append(", ".join(post.collections))

    if post.category_result is not None:
        sections.append("## AI Categories")
        sections.append(f"Primary: {post.category_result.primary_category}")
        if post.category_result.secondary_categories:
            sections.append(
                "Secondary: " + ", ".join(post.category_result.secondary_categories)
            )
        if post.category_result.confidence is not None:
            sections.append(f"Confidence: {post.category_result.confidence:.2f}")
        if post.category_result.reasoning:
            sections.append(post.category_result.reasoning)

    if post.locations:
        sections.append("## Locations")
        for location in post.locations:
            parts = []
            if location.name:
                parts.append(location.name)
            if location.address:
                parts.append(location.address)
            if location.latitude is not None and location.longitude is not None:
                parts.append(f"{location.latitude:.6f}, {location.longitude:.6f}")
            sections.append(" | ".join(parts) if parts else "_Unnamed location_")
            if location.description:
                sections.append(location.description)

    sections.append("## Source")
    sections.append(post.url)
    return "\n\n".join(sections).strip() + "\n"


def _materialize_media(source: Path, target: Path, use_symlinks: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    if use_symlinks:
        if target.is_symlink():
            existing = os.readlink(target)
            if Path(existing) == source:
                return
            target.unlink()
        elif target.exists():
            target.unlink()
        target.symlink_to(source)
        return

    if target.is_symlink():
        target.unlink()
    if target.exists():
        source_stat = source.stat()
        target_stat = target.stat()
        if (
            int(source_stat.st_size) == int(target_stat.st_size)
            and int(source_stat.st_mtime) <= int(target_stat.st_mtime)
        ):
            return
    shutil.copy2(source, target)


def write_notes(posts: list[InstagramPost], config: AppConfig) -> int:
    notes_dir = config.paths.notes_path
    media_dir = config.paths.media_path
    notes_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for post in posts:
        note_path = notes_dir / f"{post.note_slug}.md"
        linked_media: list[str] = []

        for source_path in post.media_files:
            relative_source = source_path.relative_to(config.paths.archive_dir)
            target_path = media_dir / relative_source
            _materialize_media(source_path, target_path, config.obsidian.use_symlinks)
            linked_media.append(os.path.relpath(target_path, note_path.parent))

        tags = set(config.obsidian.base_tags)
        tags.update(post.collections)
        if config.categorization.attach_category_tags and post.category_result is not None:
            primary_tag = _category_tag(post.category_result.primary_category)
            if primary_tag:
                tags.add(primary_tag)
            for category in post.category_result.secondary_categories:
                secondary_tag = _category_tag(category)
                if secondary_tag:
                    tags.add(secondary_tag)
        content = "\n\n".join(
            [
                _frontmatter(post, linked_media, sorted(tags)),
                _note_body(post, linked_media, config.obsidian.embed_media),
            ]
        ).strip()
        note_path.write_text(content + "\n", encoding="utf-8")
        written += 1

    return written
