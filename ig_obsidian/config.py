from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_path(raw: Optional[str], base_dir: Path) -> Optional[Path]:
    if raw in (None, ""):
        return None
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded
    return (base_dir / expanded).resolve()


@dataclass
class PathsConfig:
    archive_dir: Path
    vault_dir: Path
    notes_dir: str = "notes"
    media_dir: str = "media"
    collections_file: Optional[Path] = None
    locations_file: Optional[Path] = None
    exports_dir: str = "exports"

    @property
    def notes_path(self) -> Path:
        return self.vault_dir / self.notes_dir

    @property
    def media_path(self) -> Path:
        return self.vault_dir / self.media_dir

    @property
    def exports_path(self) -> Path:
        return self.vault_dir / self.exports_dir


@dataclass
class DownloadConfig:
    enabled: bool = False
    instagram_username: str = ""
    instaloader_bin: str = "instaloader"
    dirname_pattern: str = "saved"
    filename_pattern: str = "{profile}/{date_utc:%Y-%m-%d_%H-%M-%S}_{shortcode}"
    metadata_template: str = "{caption}"
    extra_args: list[str] = field(default_factory=list)
    session_file: Optional[Path] = None


@dataclass
class TranscriptionConfig:
    enabled: bool = False
    backend: str = "faster-whisper"
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    language: Optional[str] = None
    overwrite: bool = False


@dataclass
class ObsidianConfig:
    use_symlinks: bool = True
    embed_media: bool = True
    base_tags: list[str] = field(default_factory=lambda: ["instagram"])


@dataclass
class MapsConfig:
    enabled: bool = False
    export_filename: str = "google-my-maps.csv"
    aggregate_by_location: bool = True
    caption_snippet_length: int = 240


@dataclass
class CategorizationConfig:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    taxonomy_file: Optional[Path] = None
    overwrite: bool = False
    max_input_chars: int = 8000
    fallback_category: str = "uncategorized"
    attach_category_tags: bool = True


@dataclass
class AppConfig:
    paths: PathsConfig
    download: DownloadConfig = field(default_factory=DownloadConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    obsidian: ObsidianConfig = field(default_factory=ObsidianConfig)
    maps: MapsConfig = field(default_factory=MapsConfig)
    categorization: CategorizationConfig = field(default_factory=CategorizationConfig)


def load_config(config_path: Path) -> AppConfig:
    resolved_path = config_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved_path}. Copy config.example.json to config.json first."
        )

    raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    base_dir = resolved_path.parent

    paths_raw: Dict[str, Any] = raw.get("paths", {})
    archive_dir = _resolve_path(paths_raw.get("archive_dir"), base_dir)
    vault_dir = _resolve_path(paths_raw.get("vault_dir"), base_dir)
    if archive_dir is None or vault_dir is None:
        raise ValueError("Config requires paths.archive_dir and paths.vault_dir.")

    paths = PathsConfig(
        archive_dir=archive_dir,
        vault_dir=vault_dir,
        notes_dir=paths_raw.get("notes_dir", "notes"),
        media_dir=paths_raw.get("media_dir", "media"),
        collections_file=_resolve_path(paths_raw.get("collections_file"), base_dir),
        locations_file=_resolve_path(paths_raw.get("locations_file"), base_dir),
        exports_dir=paths_raw.get("exports_dir", "exports"),
    )

    download_raw: Dict[str, Any] = raw.get("download", {})
    download = DownloadConfig(
        enabled=bool(download_raw.get("enabled", False)),
        instagram_username=download_raw.get("instagram_username", ""),
        instaloader_bin=download_raw.get("instaloader_bin", "instaloader"),
        dirname_pattern=download_raw.get("dirname_pattern", "saved"),
        filename_pattern=download_raw.get(
            "filename_pattern", "{profile}/{date_utc:%Y-%m-%d_%H-%M-%S}_{shortcode}"
        ),
        metadata_template=download_raw.get("metadata_template", "{caption}"),
        extra_args=list(download_raw.get("extra_args", [])),
        session_file=_resolve_path(download_raw.get("session_file"), base_dir),
    )

    transcription_raw: Dict[str, Any] = raw.get("transcription", {})
    transcription = TranscriptionConfig(
        enabled=bool(transcription_raw.get("enabled", False)),
        backend=transcription_raw.get("backend", "faster-whisper"),
        model=transcription_raw.get("model", "small"),
        device=transcription_raw.get("device", "cpu"),
        compute_type=transcription_raw.get("compute_type", "int8"),
        beam_size=int(transcription_raw.get("beam_size", 5)),
        language=transcription_raw.get("language"),
        overwrite=bool(transcription_raw.get("overwrite", False)),
    )

    obsidian_raw: Dict[str, Any] = raw.get("obsidian", {})
    obsidian = ObsidianConfig(
        use_symlinks=bool(obsidian_raw.get("use_symlinks", True)),
        embed_media=bool(obsidian_raw.get("embed_media", True)),
        base_tags=list(obsidian_raw.get("base_tags", ["instagram"])),
    )

    maps_raw: Dict[str, Any] = raw.get("maps", {})
    maps = MapsConfig(
        enabled=bool(maps_raw.get("enabled", False)),
        export_filename=maps_raw.get("export_filename", "google-my-maps.csv"),
        aggregate_by_location=bool(maps_raw.get("aggregate_by_location", True)),
        caption_snippet_length=int(maps_raw.get("caption_snippet_length", 240)),
    )

    categorization_raw: Dict[str, Any] = raw.get("categorization", {})
    categorization = CategorizationConfig(
        enabled=bool(categorization_raw.get("enabled", False)),
        provider=categorization_raw.get("provider", "openai"),
        model=categorization_raw.get("model", "gpt-4o-mini"),
        api_key_env=categorization_raw.get("api_key_env", "OPENAI_API_KEY"),
        taxonomy_file=_resolve_path(categorization_raw.get("taxonomy_file"), base_dir),
        overwrite=bool(categorization_raw.get("overwrite", False)),
        max_input_chars=int(categorization_raw.get("max_input_chars", 8000)),
        fallback_category=categorization_raw.get("fallback_category", "uncategorized"),
        attach_category_tags=bool(categorization_raw.get("attach_category_tags", True)),
    )

    return AppConfig(
        paths=paths,
        download=download,
        transcription=transcription,
        obsidian=obsidian,
        maps=maps,
        categorization=categorization,
    )
