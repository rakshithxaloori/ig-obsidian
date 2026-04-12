from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from typing import Optional

from .categorize import categorize_posts, load_taxonomy
from .collections import load_collection_map
from .config import AppConfig, load_config
from .locations import attach_locations, export_google_maps_csv, load_location_overrides
from .instagram import discover_posts
from .obsidian import write_notes
from .transcribe import transcribe_posts


def _format_log_message(message: str) -> str:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    return f"[{timestamp}] [ig-obsidian] {message}"


def _log(message: str) -> None:
    print(_format_log_message(message), flush=True)


def _ensure_directories(config: AppConfig) -> None:
    directories = [
        config.paths.archive_dir,
        config.paths.vault_dir,
        config.paths.notes_path,
        config.paths.media_path,
        config.paths.exports_path,
    ]
    created = []
    for path in directories:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)

    if created:
        _log("Created missing directories:")
        for path in created:
            _log(f"  - {path}")


def _build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the JSON config file. Defaults to ./config.json.",
    )

    parser = argparse.ArgumentParser(
        prog="ig-obsidian",
        description="Sync an Instaloader export into an Obsidian-friendly archive.",
        parents=[shared],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "download",
        help="Run Instaloader against :saved using the configured paths.",
        parents=[shared],
    )

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Generate transcript files for downloaded videos.",
        parents=[shared],
    )
    transcribe_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing transcript files."
    )

    subparsers.add_parser(
        "build",
        help="Generate or refresh Obsidian notes and media links.",
        parents=[shared],
    )
    categorize_parser = subparsers.add_parser(
        "categorize",
        help="Run AI categorization against caption/transcript content using the configured taxonomy.",
        parents=[shared],
    )
    categorize_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing AI category sidecars."
    )
    subparsers.add_parser(
        "export-maps",
        help="Export detected/manual locations to a Google My Maps compatible CSV.",
        parents=[shared],
    )

    sync_parser = subparsers.add_parser(
        "sync",
        help="Run download, transcription, and note generation.",
        parents=[shared],
    )
    sync_parser.add_argument(
        "--download",
        action="store_true",
        help="Force the download step even if download.enabled is false in the config.",
    )
    sync_parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Skip transcription even if it is enabled in the config.",
    )
    sync_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing transcript files during sync.",
    )

    return parser


def _run_instaloader(config: AppConfig) -> None:
    _ensure_directories(config)
    archive_parent = config.paths.archive_dir.parent

    command = [config.download.instaloader_bin]
    if config.download.instagram_username:
        command.append(f"--login={config.download.instagram_username}")
    if config.download.session_file:
        command.append(f"--sessionfile={config.download.session_file}")
    command.extend(
        [
            f"--dirname-pattern={config.download.dirname_pattern}",
            f"--filename-pattern={config.download.filename_pattern}",
            f"--post-metadata-txt={config.download.metadata_template}",
        ]
    )
    command.extend(config.download.extra_args)
    command.append(":saved")

    if shutil.which(config.download.instaloader_bin) is None:
        raise RuntimeError(
            f"Could not find `{config.download.instaloader_bin}` on PATH. Install the package first."
        )

    _log(
        "Starting Instaloader download for saved posts. "
        f"Archive root: {config.paths.archive_dir}"
    )
    subprocess.run(command, cwd=archive_parent, check=True)
    _log("Instaloader download step finished.")


def _load_posts(config: AppConfig):
    _ensure_directories(config)
    _log(f"Scanning archive: {config.paths.archive_dir}")
    collection_map = load_collection_map(config.paths.collections_file)
    location_overrides = load_location_overrides(config.paths.locations_file)
    posts = discover_posts(config.paths.archive_dir, collection_map)
    posts = attach_locations(posts, location_overrides)
    video_count = sum(len(post.video_files) for post in posts)
    location_count = sum(len(post.locations) for post in posts)
    _log(
        f"Discovered {len(posts)} post(s), {video_count} video file(s), "
        f"{location_count} location record(s)."
    )
    if not posts:
        _log(
            "Archive scan found no matching Instaloader media files yet. "
            "If this is a fresh setup, run with --download or copy existing files into the archive directory."
        )
    return posts


def _handle_download(config: AppConfig) -> None:
    _run_instaloader(config)
    _log(f"Download finished in {config.paths.archive_dir}")


def _handle_transcribe(config: AppConfig, force: bool) -> None:
    posts = _load_posts(config)
    config.transcription.overwrite = force or config.transcription.overwrite
    created = transcribe_posts(posts, config.transcription, progress=_log)
    _log(f"Created {created} transcript file(s).")


def _handle_build(config: AppConfig) -> None:
    posts = _load_posts(config)
    _log(f"Writing notes into {config.paths.notes_path}")
    written = write_notes(posts, config)
    _log(f"Wrote {written} note(s) into {config.paths.notes_path}")


def _handle_categorize(config: AppConfig, force: bool) -> None:
    posts = _load_posts(config)
    config.categorization.overwrite = force or config.categorization.overwrite
    taxonomy = load_taxonomy(
        config.categorization.taxonomy_file,
        config.categorization.fallback_category,
        config.categorization.dynamic_location_categories,
    )
    _log(
        f"Loaded taxonomy with {len(taxonomy.names)} categories from "
        f"{config.categorization.taxonomy_file}"
    )
    written = categorize_posts(posts, config.categorization, taxonomy, progress=_log)
    _log(f"Wrote {written} AI category sidecar file(s).")


def _handle_export_maps(config: AppConfig) -> None:
    posts = _load_posts(config)
    output_path = config.paths.exports_path / config.maps.export_filename
    _log(f"Exporting Google My Maps CSV to {output_path}")
    written = export_google_maps_csv(
        posts,
        output_path,
        aggregate_by_location=config.maps.aggregate_by_location,
        caption_snippet_length=config.maps.caption_snippet_length,
    )
    _log(f"Wrote {written} map row(s) into {output_path}")


def _handle_sync(config: AppConfig, force_download: bool, skip_transcribe: bool, force: bool) -> None:
    _log("Starting sync run.")
    if force_download or config.download.enabled:
        _run_instaloader(config)
    else:
        _log("Download step disabled; using existing archive files.")

    posts = _load_posts(config)
    if not skip_transcribe:
        config.transcription.overwrite = force or config.transcription.overwrite
        created = transcribe_posts(posts, config.transcription, progress=_log)
        if created:
            _log(f"Created {created} transcript file(s). Reloading archive metadata.")
            posts = _load_posts(config)
        else:
            _log("No new transcript files were created.")
    else:
        _log("Transcription step skipped by flag.")

    if config.categorization.enabled:
        taxonomy = load_taxonomy(
            config.categorization.taxonomy_file,
            config.categorization.fallback_category,
            config.categorization.dynamic_location_categories,
        )
        _log(
            f"Loaded taxonomy with {len(taxonomy.names)} categories from "
            f"{config.categorization.taxonomy_file}"
        )
        categorized = categorize_posts(posts, config.categorization, taxonomy, progress=_log)
        if categorized:
            _log(f"Wrote {categorized} AI category sidecar file(s).")
        else:
            _log("No new AI category sidecars were created.")
    else:
        _log("AI categorization disabled.")

    _log(f"Writing notes into {config.paths.notes_path}")
    written = write_notes(posts, config)
    if config.maps.enabled:
        output_path = config.paths.exports_path / config.maps.export_filename
        _log(f"Exporting Google My Maps CSV to {output_path}")
        map_rows = export_google_maps_csv(
            posts,
            output_path,
            aggregate_by_location=config.maps.aggregate_by_location,
            caption_snippet_length=config.maps.caption_snippet_length,
        )
        _log(f"Wrote {map_rows} map row(s) into {output_path}")
    else:
        _log("Map export disabled.")
    _log(f"Sync complete. Processed {len(posts)} post(s) and wrote {written} note(s).")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _log(f"Loading config from {args.config}")
        config = load_config(args.config)
        if args.command == "download":
            _handle_download(config)
        elif args.command == "transcribe":
            _handle_transcribe(config, force=args.force)
        elif args.command == "build":
            _handle_build(config)
        elif args.command == "categorize":
            _handle_categorize(config, force=args.force)
        elif args.command == "export-maps":
            _handle_export_maps(config)
        elif args.command == "sync":
            _handle_sync(
                config,
                force_download=args.download,
                skip_transcribe=args.skip_transcribe,
                force=args.force,
            )
        else:
            parser.error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        parser.exit(status=130, message=_format_log_message("Cancelled by user.") + "\n")
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(status=1, message=_format_log_message(f"error: {exc}") + "\n")

    return 0
