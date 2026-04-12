from __future__ import annotations

from datetime import datetime, timezone
import json
import lzma
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional

from .categorize import CATEGORY_ERROR_SUFFIX, CATEGORY_SUFFIX, load_category_result_from_path
from .models import InstagramPost


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
TRANSCRIPT_SUFFIX = ".transcript.txt"
METADATA_SUFFIXES = (".json.xz", ".json")
SHORTCODE_CANDIDATE_RE = re.compile(r"^[A-Za-z0-9_-]{5,32}$")
DATE_PREFIX_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_"
)


def _is_ignored_archive_path(root: Path, path: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts)


def _strip_known_suffixes(filename: str) -> str:
    suffixes = [
        CATEGORY_ERROR_SUFFIX,
        CATEGORY_SUFFIX,
        TRANSCRIPT_SUFFIX,
        *METADATA_SUFFIXES,
        ".txt",
        *sorted(MEDIA_EXTENSIONS),
    ]
    for suffix in sorted(suffixes, key=len, reverse=True):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def _looks_like_shortcode(value: str) -> bool:
    if not SHORTCODE_CANDIDATE_RE.fullmatch(value):
        return False
    if DATE_PREFIX_RE.match(value):
        return False
    return True


def extract_shortcode_from_name(filename: str) -> Optional[str]:
    stem = _strip_known_suffixes(filename)
    parts = stem.split("_")

    candidates = []
    if len(parts) >= 2 and parts[-1].isdigit():
        candidates.append(parts[-2])
    if parts:
        candidates.append(parts[-1])
    candidates.append(stem)

    for candidate in candidates:
        if _looks_like_shortcode(candidate):
            return candidate
    return None


def _author_from_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return "unknown"


def _prefer_author(current: str, candidate: str) -> str:
    if current == "unknown" and candidate != "unknown":
        return candidate
    return current


def _parse_datetime_from_filename(filename: str) -> Optional[datetime]:
    match = DATE_PREFIX_RE.match(_strip_known_suffixes(filename))
    if not match:
        return None
    return datetime.strptime(
        f"{match.group('date')} {match.group('time').replace('-', ':')}",
        "%Y-%m-%d %H:%M:%S",
    )


def _load_metadata(path: Path) -> Dict[str, Any]:
    try:
        if path.name.endswith(".xz"):
            with lzma.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, lzma.LZMAError):
        return {}


def _caption_from_metadata(metadata: Dict[str, Any]) -> str:
    if isinstance(metadata.get("caption"), str):
        return metadata["caption"].strip()

    edges = metadata.get("edge_media_to_caption", {}).get("edges", [])
    if edges and isinstance(edges[0], dict):
        node = edges[0].get("node", {})
        if isinstance(node.get("text"), str):
            return node["text"].strip()
    return ""


def _author_from_metadata(metadata: Dict[str, Any]) -> str:
    if isinstance(metadata.get("owner_username"), str):
        return metadata["owner_username"].strip()
    owner = metadata.get("owner")
    if isinstance(owner, dict):
        username = owner.get("username") or owner.get("owner_username")
        if isinstance(username, str):
            return username.strip()
    return ""


def _datetime_from_metadata(metadata: Dict[str, Any]) -> Optional[datetime]:
    value = metadata.get("date_utc")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            pass

    timestamp = metadata.get("taken_at_timestamp")
    if isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)
    return None


def _kind_from_metadata(metadata: Dict[str, Any], has_video: bool, has_multiple_media: bool) -> str:
    product_type = metadata.get("product_type")
    if product_type == "clips":
        return "reel"

    typename = metadata.get("__typename") or metadata.get("typename")
    if typename == "GraphSidecar" or has_multiple_media:
        return "carousel"
    if typename == "GraphVideo" and has_video:
        return "video"
    if has_video:
        return "reel"
    return "post"


def _split_caption_and_sidecar_metadata(raw_text: str) -> tuple[str, Dict[str, str]]:
    lines = raw_text.splitlines()
    metadata: Dict[str, str] = {}
    cursor = len(lines)

    while cursor > 0:
        line = lines[cursor - 1].strip()
        if not line:
            cursor -= 1
            continue
        if "=" not in line:
            break
        key, value = line.split("=", 1)
        normalized_key = key.strip().lower()
        if normalized_key not in {"url", "owner", "date", "permalink"}:
            break
        metadata[normalized_key] = value.strip()
        cursor -= 1

    while cursor > 0 and not lines[cursor - 1].strip():
        cursor -= 1

    caption = "\n".join(lines[:cursor]).strip()
    return caption, metadata


def _join_text_files(paths: Iterable[Path]) -> str:
    chunks = []
    seen = set()
    for path in sorted(paths):
        text = _read_text_file(path).strip()
        if text and text not in seen:
            chunks.append(text)
            seen.add(text)
    return "\n\n".join(chunks)


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _preferred_metadata_path(current: Optional[Path], candidate: Path) -> Path:
    if current is None:
        return candidate
    if current.name.endswith(".json") and candidate.name.endswith(".json.xz"):
        return candidate
    return current


def _path_quality_key(path: Path) -> tuple[int, float, str]:
    stat = path.stat()
    return (int(stat.st_size), float(stat.st_mtime), str(path))


def _prefer_path(current: Optional[Path], candidate: Path) -> Path:
    if current is None:
        return candidate
    if _path_quality_key(candidate) > _path_quality_key(current):
        return candidate
    return current


def _media_variant_key(path: Path) -> tuple[str, str]:
    stem = _strip_known_suffixes(path.name)
    parts = stem.split("_")
    slot = "0"
    shortcode = parts[-1] if parts else stem
    if len(parts) >= 2 and parts[-1].isdigit():
        slot = parts[-1]
        shortcode = parts[-2]
    return (shortcode, f"{slot}{path.suffix.lower()}")


def _dedupe_media_files(paths: Iterable[Path]) -> list[Path]:
    selected: Dict[tuple[str, str], Path] = {}
    for path in paths:
        key = _media_variant_key(path)
        selected[key] = _prefer_path(selected.get(key), path)
    return sorted(selected.values())


def build_post_url(shortcode: str, kind: str, metadata: Dict[str, Any], footer_meta: Dict[str, str]) -> str:
    for key in ("permalink", "url", "link"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("permalink", "url"):
        value = footer_meta.get(key)
        if value:
            return value
    path_kind = "reel" if kind == "reel" else "p"
    return f"https://www.instagram.com/{path_kind}/{shortcode}/"


def discover_posts(root: Path, collection_map: Dict[str, list[str]]) -> list[InstagramPost]:
    posts: Dict[str, InstagramPost] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _is_ignored_archive_path(root, path):
            continue

        shortcode = extract_shortcode_from_name(path.name)
        if not shortcode:
            continue

        author = _author_from_path(root, path)
        post = posts.setdefault(shortcode, InstagramPost(shortcode=shortcode, author=author))
        post.author = _prefer_author(post.author, author)

        if path.suffix.lower() in MEDIA_EXTENSIONS:
            post.media_files.append(path)
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                post.video_files.append(path)
            else:
                post.image_files.append(path)
            if post.date is None:
                post.date = _parse_datetime_from_filename(path.name)
            continue

        if path.name.endswith(TRANSCRIPT_SUFFIX):
            post.transcript_files.append(path)
            continue

        if path.name.endswith(CATEGORY_ERROR_SUFFIX):
            continue

        if path.name.endswith(CATEGORY_SUFFIX):
            post.category_file = _prefer_path(post.category_file, path)
            continue

        if any(path.name.endswith(suffix) for suffix in METADATA_SUFFIXES):
            post.metadata_file = _preferred_metadata_path(post.metadata_file, path)
            continue

        if path.suffix.lower() == ".txt":
            post.caption_file = _prefer_path(post.caption_file, path)
            continue

    hydrated_posts = []
    for post in posts.values():
        footer_meta: Dict[str, str] = {}
        if post.metadata_file is not None:
            post.metadata = _load_metadata(post.metadata_file)
        if post.category_file is not None:
            post.category_result = load_category_result_from_path(post.category_file)

        if post.caption_file is not None:
            raw_caption = _read_text_file(post.caption_file)
            post.caption, footer_meta = _split_caption_and_sidecar_metadata(raw_caption)

        if not post.caption:
            post.caption = _caption_from_metadata(post.metadata)

        post.transcript = _join_text_files(post.transcript_files)

        metadata_author = _author_from_metadata(post.metadata)
        if metadata_author:
            post.author = metadata_author

        if post.date is None:
            post.date = _datetime_from_metadata(post.metadata)

        post.media_files = _dedupe_media_files(post.media_files)
        post.video_files = [path for path in post.media_files if path.suffix.lower() in VIDEO_EXTENSIONS]
        post.image_files = [path for path in post.media_files if path.suffix.lower() in IMAGE_EXTENSIONS]
        post.transcript_files.sort()

        post.kind = _kind_from_metadata(
            post.metadata,
            has_video=bool(post.video_files),
            has_multiple_media=len(post.media_files) > 1,
        )
        post.url = build_post_url(post.shortcode, post.kind, post.metadata, footer_meta)
        post.collections = collection_map.get(post.shortcode, [])

        if post.media_files:
            hydrated_posts.append(post)

    return sorted(
        hydrated_posts,
        key=lambda post: (post.date or datetime.min, post.author.lower(), post.shortcode),
    )
