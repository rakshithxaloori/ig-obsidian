from __future__ import annotations

import contextlib
from datetime import datetime
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from ig_obsidian.categorize import category_error_path, categorize_posts, load_taxonomy
from ig_obsidian.collections import load_collection_map, normalize_shortcode
from ig_obsidian.config import (
    AppConfig,
    CategorizationConfig,
    ObsidianConfig,
    PathsConfig,
    TranscriptionConfig,
)
from ig_obsidian.cli import _build_parser, _format_log_message, _load_posts, main
from ig_obsidian.instagram import discover_posts, extract_shortcode_from_name
from ig_obsidian.locations import attach_locations, export_google_maps_csv, load_location_overrides
from ig_obsidian.models import CategoryResult, InstagramPost
from ig_obsidian.obsidian import write_notes
from ig_obsidian.transcribe import (
    transcript_error_path_for_video,
    transcript_path_for_video,
    transcribe_posts,
)


class CliParsingTest(unittest.TestCase):
    def test_config_is_accepted_after_subcommand(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["sync", "--config", "config.json"])
        self.assertEqual(args.command, "sync")
        self.assertEqual(args.config, Path("config.json"))

    def test_missing_archive_dir_is_created_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            config = AppConfig(
                paths=PathsConfig(
                    archive_dir=base / "missing-archive",
                    vault_dir=base / "vault",
                ),
                transcription=TranscriptionConfig(enabled=False),
                obsidian=ObsidianConfig(),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                posts = _load_posts(config)
            self.assertEqual(posts, [])
            self.assertTrue(config.paths.archive_dir.exists())
            self.assertTrue(config.paths.vault_dir.exists())
            self.assertTrue(config.paths.notes_path.exists())
            self.assertTrue(config.paths.media_path.exists())
            self.assertTrue(config.paths.exports_path.exists())


class CliLoggingTest(unittest.TestCase):
    def test_format_log_message_includes_timestamp(self) -> None:
        formatted = _format_log_message("hello")
        self.assertRegex(
            formatted,
            r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\] \[ig-obsidian\] hello$",
        )

    def test_main_error_output_is_timestamped(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["build", "--config", "missing-config.json"])
        self.assertEqual(exc.exception.code, 1)
        self.assertRegex(
            stderr.getvalue(),
            r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\] "
            r"\[ig-obsidian\] error: Config file not found: .+missing-config\.json.+\n$",
        )

    def test_main_cancel_output_is_timestamped(self) -> None:
        stderr = io.StringIO()
        with mock.patch("ig_obsidian.cli.load_config", side_effect=KeyboardInterrupt):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as exc:
                        main(["build", "--config", "config.json"])
        self.assertEqual(exc.exception.code, 130)
        self.assertRegex(
            stderr.getvalue(),
            r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\] "
            r"\[ig-obsidian\] Cancelled by user\.\n$",
        )


class CollectionsTest(unittest.TestCase):
    def test_normalize_shortcode_from_url(self) -> None:
        self.assertEqual(normalize_shortcode("https://www.instagram.com/reel/ABC123/"), "ABC123")

    def test_load_collection_map_supports_both_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "collections.json"
            path.write_text(
                json.dumps({"travels": ["ABC123", "https://www.instagram.com/p/XYZ789/"]}),
                encoding="utf-8",
            )
            self.assertEqual(
                load_collection_map(path),
                {"ABC123": ["travels"], "XYZ789": ["travels"]},
            )

            path.write_text(json.dumps({"ABC123": "food"}), encoding="utf-8")
            self.assertEqual(load_collection_map(path), {"ABC123": ["food"]})


class CategorizationTest(unittest.TestCase):
    def test_load_taxonomy_supports_object_and_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "taxonomy.json"
            path.write_text(json.dumps({"travel": "Trip ideas"}), encoding="utf-8")
            taxonomy = load_taxonomy(path, "uncategorized", ["travel", "food"])
            self.assertIn("travel", taxonomy.categories)
            self.assertIn("uncategorized", taxonomy.categories)
            self.assertEqual(taxonomy.dynamic_location_categories, ("travel",))

            path.write_text(json.dumps(["food", "shopping"]), encoding="utf-8")
            taxonomy = load_taxonomy(path, "uncategorized", ["travel", "food"])
            self.assertEqual(taxonomy.names, ["food", "shopping", "uncategorized"])
            self.assertEqual(taxonomy.dynamic_location_categories, ("food",))

    def test_discover_posts_loads_existing_category_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "saved"
            author_dir = root / "someuser"
            author_dir.mkdir(parents=True)

            (author_dir / "2026-04-08_12-01-02_ABC123.mp4").write_bytes(b"video")
            (author_dir / "2026-04-08_12-01-02_ABC123.ai_categories.json").write_text(
                json.dumps(
                    {
                        "primary_category": "travel",
                        "secondary_categories": ["food"],
                        "confidence": 0.9,
                        "reasoning": "Travel reel with food recommendations.",
                        "model": "gemma4:e4b",
                        "provider": "ollama",
                        "taxonomy_signature": "abc",
                        "categorized_at": "2026-04-08T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            posts = discover_posts(root, {})
            self.assertEqual(posts[0].category_result.primary_category, "travel")
            self.assertEqual(posts[0].category_result.secondary_categories, ["food"])

    def test_categorize_posts_writes_sidecars_with_fake_client(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.requests: list[dict[str, object]] = []

            def chat(self, payload: dict[str, object]) -> dict[str, object]:
                self.requests.append(payload)
                return {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "categorize_saved_post",
                                    "arguments": {
                                        "primary_category": "food/Lisbon, Portugal",
                                        "secondary_categories": ["travel/Lisbon, Portugal"],
                                        "confidence": 0.88,
                                        "reasoning": "Mostly restaurant-focused content.",
                                    },
                                }
                            }
                        ]
                    }
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_dir = base / "archive" / "saved"
            user_dir = archive_dir / "someuser"
            user_dir.mkdir(parents=True)
            video = user_dir / "2026-04-08_12-01-02_ABC123.mp4"
            caption = user_dir / "2026-04-08_12-01-02_ABC123.txt"
            video.write_bytes(b"video")
            caption.write_text("Great cafe in Lisbon", encoding="utf-8")

            taxonomy_path = base / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps({"food": "Restaurants and cafes", "travel": "Destinations"}),
                encoding="utf-8",
            )
            dynamic_roots = ["travel", "food"]
            taxonomy = load_taxonomy(taxonomy_path, "uncategorized", dynamic_roots)
            posts = discover_posts(archive_dir, {})
            client = FakeClient()
            with mock.patch.dict("os.environ", {"OLLAMA_MODEL": "gemma4:e4b"}, clear=False):
                written = categorize_posts(
                    posts,
                    CategorizationConfig(
                        enabled=True,
                        taxonomy_file=taxonomy_path,
                        fallback_category="uncategorized",
                        dynamic_location_categories=dynamic_roots,
                    ),
                    taxonomy,
                    client=client,
                )

            self.assertEqual(written, 1)
            self.assertEqual(posts[0].category_result.primary_category, "food/Lisbon, Portugal")
            self.assertEqual(posts[0].category_result.secondary_categories, ["travel/Lisbon, Portugal"])
            self.assertEqual(posts[0].category_result.model, "gemma4:e4b")
            self.assertEqual(posts[0].category_result.provider, "ollama")
            self.assertEqual(client.requests[0]["model"], "gemma4:e4b")
            self.assertFalse(client.requests[0]["think"])
            self.assertIn("tools", client.requests[0])
            self.assertTrue((user_dir / "2026-04-08_12-01-02_ABC123.ai_categories.json").exists())

    def test_categorize_posts_accepts_raw_json_fallback(self) -> None:
        class FakeClient:
            def chat(self, _: dict[str, object]) -> dict[str, object]:
                return {
                    "message": {
                        "content": json.dumps(
                            {
                                "primary_category": "travel",
                                "secondary_categories": ["food"],
                                "confidence": 0.72,
                                "reasoning": "Trip-focused reel with food suggestions.",
                            }
                        )
                    }
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_dir = base / "archive" / "saved"
            user_dir = archive_dir / "someuser"
            user_dir.mkdir(parents=True)
            video = user_dir / "2026-04-08_12-01-02_ABC123.mp4"
            caption = user_dir / "2026-04-08_12-01-02_ABC123.txt"
            video.write_bytes(b"video")
            caption.write_text("Weekend trip with great pastries", encoding="utf-8")

            taxonomy_path = base / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps({"food": "Restaurants and cafes", "travel": "Destinations"}),
                encoding="utf-8",
            )
            dynamic_roots = ["travel", "food"]
            taxonomy = load_taxonomy(taxonomy_path, "uncategorized", dynamic_roots)
            posts = discover_posts(archive_dir, {})
            written = categorize_posts(
                posts,
                CategorizationConfig(
                    enabled=True,
                    taxonomy_file=taxonomy_path,
                    fallback_category="uncategorized",
                    dynamic_location_categories=dynamic_roots,
                ),
                taxonomy,
                client=FakeClient(),
            )

            self.assertEqual(written, 1)
            self.assertEqual(posts[0].category_result.primary_category, "travel")
            self.assertEqual(posts[0].category_result.secondary_categories, ["food"])

    def test_categorize_posts_writes_failure_marker_and_continues(self) -> None:
        class FakeClient:
            def chat(self, payload: dict[str, object]) -> dict[str, object]:
                body = payload["messages"][1]["content"]
                if "Shortcode: BAD123" in body:
                    return {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "categorize_saved_post",
                                        "arguments": "",
                                    }
                                }
                            ]
                        }
                    }
                return {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "categorize_saved_post",
                                    "arguments": {
                                        "primary_category": "travel",
                                        "secondary_categories": ["food"],
                                        "confidence": 0.63,
                                        "reasoning": "Travel-heavy post with food context.",
                                    },
                                }
                            }
                        ]
                    }
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_dir = base / "archive" / "saved"
            user_dir = archive_dir / "someuser"
            user_dir.mkdir(parents=True)
            bad_video = user_dir / "2026-04-08_12-01-02_BAD123.mp4"
            good_video = user_dir / "2026-04-08_12-01-03_GOOD456.mp4"
            bad_caption = user_dir / "2026-04-08_12-01-02_BAD123.txt"
            good_caption = user_dir / "2026-04-08_12-01-03_GOOD456.txt"
            bad_video.write_bytes(b"bad video")
            good_video.write_bytes(b"good video")
            bad_caption.write_text("Bad caption", encoding="utf-8")
            good_caption.write_text("Good caption", encoding="utf-8")

            taxonomy_path = base / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps({"food": "Restaurants and cafes", "travel": "Destinations"}),
                encoding="utf-8",
            )
            dynamic_roots = ["travel", "food"]
            taxonomy = load_taxonomy(taxonomy_path, "uncategorized", dynamic_roots)
            posts = discover_posts(archive_dir, {})
            progress: list[str] = []

            written = categorize_posts(
                posts,
                CategorizationConfig(
                    enabled=True,
                    taxonomy_file=taxonomy_path,
                    fallback_category="uncategorized",
                    dynamic_location_categories=dynamic_roots,
                ),
                taxonomy,
                progress=progress.append,
                client=FakeClient(),
            )

            bad_post = next(post for post in posts if post.shortcode == "BAD123")
            good_post = next(post for post in posts if post.shortcode == "GOOD456")
            self.assertEqual(written, 1)
            self.assertIsNone(bad_post.category_result)
            self.assertEqual(good_post.category_result.primary_category, "travel")
            self.assertEqual(
                category_error_path(bad_post).read_text(encoding="utf-8"),
                "CategorizationResponseError: Ollama returned empty function-call arguments for categorization.\n",
            )
            self.assertTrue(any("Failed BAD123" in message for message in progress))
            self.assertIn(
                "Categorization summary: 1 AI category sidecar file(s), 1 failure marker(s).",
                progress,
            )

    def test_categorize_posts_skips_existing_failure_marker_on_rerun(self) -> None:
        class FakeClient:
            def chat(self, _: dict[str, object]) -> dict[str, object]:
                raise AssertionError("categorization should be skipped when a failure marker exists")

        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_dir = base / "archive" / "saved"
            user_dir = archive_dir / "someuser"
            user_dir.mkdir(parents=True)
            video = user_dir / "2026-04-08_12-01-02_BAD123.mp4"
            caption = user_dir / "2026-04-08_12-01-02_BAD123.txt"
            video.write_bytes(b"video")
            caption.write_text("Caption", encoding="utf-8")

            taxonomy_path = base / "taxonomy.json"
            taxonomy_path.write_text(json.dumps({"travel": "Destinations"}), encoding="utf-8")
            taxonomy = load_taxonomy(taxonomy_path, "uncategorized", ["travel"])
            posts = discover_posts(archive_dir, {})
            error_path = category_error_path(posts[0])
            error_path.write_text("CategorizationResponseError: broken response\n", encoding="utf-8")
            progress: list[str] = []

            written = categorize_posts(
                posts,
                CategorizationConfig(
                    enabled=True,
                    taxonomy_file=taxonomy_path,
                    fallback_category="uncategorized",
                    dynamic_location_categories=["travel"],
                ),
                taxonomy,
                progress=progress.append,
                client=FakeClient(),
            )

            self.assertEqual(written, 0)
            self.assertIn(
                "No posts need AI categorization. Existing category sidecars skipped: 0. Previous failures skipped: 1.",
                progress,
            )


class InstagramDiscoveryTest(unittest.TestCase):
    def test_extract_shortcode_from_instaloader_filename(self) -> None:
        self.assertEqual(
            extract_shortcode_from_name("2026-04-08_12-01-02_ABC123.mp4"),
            "ABC123",
        )
        self.assertEqual(
            extract_shortcode_from_name("2026-04-08_12-01-02_ABC123_1.jpg"),
            "ABC123",
        )

    def test_discover_posts_reads_caption_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "saved"
            author_dir = root / "someuser"
            author_dir.mkdir(parents=True)

            video = author_dir / "2026-04-08_12-01-02_ABC123.mp4"
            caption = author_dir / "2026-04-08_12-01-02_ABC123.txt"
            metadata = author_dir / "2026-04-08_12-01-02_ABC123.json"
            transcript = author_dir / "2026-04-08_12-01-02_ABC123.transcript.txt"

            video.write_bytes(b"video")
            caption.write_text(
                "A reel caption.\n\nurl=https://www.instagram.com/reel/ABC123/\nowner=someuser\ndate=2026-04-08T12:01:02",
                encoding="utf-8",
            )
            metadata.write_text(
                json.dumps({"product_type": "clips", "owner_username": "someuser"}),
                encoding="utf-8",
            )
            transcript.write_text("Line one\nLine two\n", encoding="utf-8")

            posts = discover_posts(root, {"ABC123": ["travels"]})
            self.assertEqual(len(posts), 1)

            post = posts[0]
            self.assertEqual(post.shortcode, "ABC123")
            self.assertEqual(post.author, "someuser")
            self.assertEqual(post.kind, "reel")
            self.assertEqual(post.collections, ["travels"])
            self.assertEqual(post.caption, "A reel caption.")
            self.assertEqual(post.transcript, "Line one\nLine two")
            self.assertEqual(post.url, "https://www.instagram.com/reel/ABC123/")
            self.assertEqual(post.date, datetime(2026, 4, 8, 12, 1, 2))

    def test_discover_posts_dedupes_duplicate_shortcode_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "saved"
            first_dir = root / "someuser"
            second_dir = root / "retry"
            first_dir.mkdir(parents=True)
            second_dir.mkdir(parents=True)

            first_video = first_dir / "2026-04-08_12-01-02_ABC123.mp4"
            second_video = second_dir / "2026-04-09_12-01-02_ABC123.mp4"
            metadata = second_dir / "2026-04-09_12-01-02_ABC123.json"

            first_video.write_bytes(b"small")
            second_video.write_bytes(b"this is the bigger duplicate file")
            metadata.write_text(
                json.dumps({"product_type": "clips", "owner_username": "someuser"}),
                encoding="utf-8",
            )

            posts = discover_posts(root, {})
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0].author, "someuser")
            self.assertEqual(posts[0].media_files, [second_video])

    def test_discover_posts_ignores_hidden_mac_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "saved"
            author_dir = root / "someuser"
            author_dir.mkdir(parents=True)

            video = author_dir / "2026-04-08_12-01-02_ABC123.mp4"
            caption = author_dir / "2026-04-08_12-01-02_ABC123.txt"
            hidden_video = author_dir / "._2026-04-08_12-01-02_ABC123.mp4"
            hidden_caption = author_dir / "._2026-04-08_12-01-02_ABC123.txt"

            video.write_bytes(b"real video")
            caption.write_text("Real caption", encoding="utf-8")
            hidden_video.write_bytes(b"this sidecar should never win duplicate selection")
            hidden_caption.write_bytes(b"AppleDouble\x00\xb0")

            posts = discover_posts(root, {})

            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0].media_files, [video])
            self.assertEqual(posts[0].caption, "Real caption")

    def test_discover_posts_ignores_category_failure_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "saved"
            author_dir = root / "someuser"
            author_dir.mkdir(parents=True)

            video = author_dir / "2026-04-08_12-01-02_ABC123.mp4"
            error_marker = author_dir / "2026-04-08_12-01-02_ABC123.ai_categories.error.txt"
            video.write_bytes(b"video")
            error_marker.write_text("CategorizationResponseError: empty arguments\n", encoding="utf-8")

            posts = discover_posts(root, {})

            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0].caption, "")
            self.assertIsNone(posts[0].category_file)


class TranscriptionTest(unittest.TestCase):
    def test_transcribe_posts_writes_failure_marker_and_continues(self) -> None:
        class FakeSegment:
            def __init__(self, text: str) -> None:
                self.text = text

        class FakeWhisperModel:
            def __init__(self, *_: object, **__: object) -> None:
                pass

            def transcribe(self, path: str, **_: object) -> tuple[list[FakeSegment], None]:
                if path.endswith("BAD123.mp4"):
                    raise IndexError("tuple index out of range")
                return ([FakeSegment("hello from whisper")], None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bad_video = root / "2026-04-08_12-01-02_BAD123.mp4"
            good_video = root / "2026-04-08_12-01-02_GOOD456.mp4"
            bad_video.write_bytes(b"bad video")
            good_video.write_bytes(b"good video")

            posts = [
                InstagramPost(shortcode="BAD123", author="someuser", media_files=[bad_video], video_files=[bad_video]),
                InstagramPost(
                    shortcode="GOOD456",
                    author="someuser",
                    media_files=[good_video],
                    video_files=[good_video],
                ),
            ]
            progress: list[str] = []

            with mock.patch.dict(
                sys.modules,
                {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeWhisperModel)},
            ):
                created = transcribe_posts(
                    posts,
                    TranscriptionConfig(enabled=True),
                    progress=progress.append,
                )

            self.assertEqual(created, 1)
            self.assertEqual(
                transcript_path_for_video(good_video).read_text(encoding="utf-8"),
                "hello from whisper\n",
            )
            self.assertEqual(
                transcript_error_path_for_video(bad_video).read_text(encoding="utf-8"),
                "IndexError: tuple index out of range\n",
            )
            self.assertTrue(any("Failed 2026-04-08_12-01-02_BAD123.mp4" in message for message in progress))
            self.assertTrue(any("Transcription summary: 1 transcript file(s), 0 empty marker(s), 1 failure marker(s)." == message for message in progress))

    def test_transcribe_posts_skips_existing_failure_marker_on_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            video = root / "2026-04-08_12-01-02_BAD123.mp4"
            video.write_bytes(b"bad video")
            transcript_error_path_for_video(video).write_text(
                "IndexError: tuple index out of range\n",
                encoding="utf-8",
            )

            posts = [
                InstagramPost(shortcode="BAD123", author="someuser", media_files=[video], video_files=[video])
            ]
            progress: list[str] = []

            created = transcribe_posts(
                posts,
                TranscriptionConfig(enabled=True),
                progress=progress.append,
            )

            self.assertEqual(created, 0)
            self.assertIn(
                "No videos need transcription. Existing transcripts skipped: 0. Previous failures skipped: 1.",
                progress,
            )


class NoteWritingTest(unittest.TestCase):
    def test_write_notes_symlinks_media_and_renders_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_dir = base / "archive" / "saved"
            media_source_dir = archive_dir / "someuser"
            media_source_dir.mkdir(parents=True)

            video = media_source_dir / "2026-04-08_12-01-02_ABC123.mp4"
            caption = media_source_dir / "2026-04-08_12-01-02_ABC123.txt"
            metadata = media_source_dir / "2026-04-08_12-01-02_ABC123.json"
            transcript = media_source_dir / "2026-04-08_12-01-02_ABC123.transcript.txt"

            video.write_bytes(b"video")
            caption.write_text("Caption", encoding="utf-8")
            metadata.write_text(
                json.dumps(
                    {
                        "product_type": "clips",
                        "location": {
                            "name": "Cafe Example",
                            "address": "123 Example Street",
                            "lat": 38.7223,
                            "lng": -9.1393,
                        },
                    }
                ),
                encoding="utf-8",
            )
            transcript.write_text("Transcript body", encoding="utf-8")

            config = AppConfig(
                paths=PathsConfig(
                    archive_dir=archive_dir,
                    vault_dir=base / "vault",
                    notes_dir="notes",
                    media_dir="media",
                ),
                transcription=TranscriptionConfig(enabled=False),
                obsidian=ObsidianConfig(use_symlinks=True, embed_media=True, base_tags=["instagram"]),
            )

            posts = discover_posts(archive_dir, {"ABC123": ["travels"]})
            posts = attach_locations(posts, {})
            posts[0].category_result = CategoryResult(
                primary_category="food/Lisbon, Portugal",
                secondary_categories=["travel/Lisbon, Portugal"],
                confidence=0.84,
                reasoning="Travel reel with dining recommendations.",
                model="gemma4:e4b",
            )
            written = write_notes(posts, config)
            self.assertEqual(written, 1)

            note_path = config.paths.notes_path / "2026-04-08_ABC123.md"
            media_target = config.paths.media_path / "someuser" / video.name

            self.assertTrue(note_path.exists())
            self.assertTrue(media_target.is_symlink())

            note_text = note_path.read_text(encoding="utf-8")
            self.assertIn('shortcode: "ABC123"', note_text)
            self.assertIn('saved_collection: "travels"', note_text)
            self.assertIn("![[../media/someuser/2026-04-08_12-01-02_ABC123.mp4]]", note_text)
            self.assertIn("## Transcript", note_text)
            self.assertIn("## Locations", note_text)
            self.assertIn("Cafe Example", note_text)
            self.assertIn('ai_primary_category: "food/Lisbon, Portugal"', note_text)
            self.assertIn('ai_secondary_categories:', note_text)
            self.assertIn('"category/food/lisbon-portugal"', note_text)
            self.assertIn('"category/travel/lisbon-portugal"', note_text)
            self.assertIn("## AI Categories", note_text)


class MapsExportTest(unittest.TestCase):
    def test_export_google_maps_csv_aggregates_duplicate_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_dir = base / "archive" / "saved"
            user_dir = archive_dir / "someuser"
            user_dir.mkdir(parents=True)

            first_video = user_dir / "2026-04-08_12-01-02_ABC123.mp4"
            second_video = user_dir / "2026-04-09_12-01-02_XYZ789.mp4"
            first_video.write_bytes(b"video one")
            second_video.write_bytes(b"video two")

            (user_dir / "2026-04-08_12-01-02_ABC123.txt").write_text("First caption", encoding="utf-8")
            (user_dir / "2026-04-09_12-01-02_XYZ789.txt").write_text("Second caption", encoding="utf-8")
            location_payload = {
                "product_type": "clips",
                "location": {
                    "name": "Cafe Example",
                    "address": "123 Example Street",
                    "lat": 38.7223,
                    "lng": -9.1393,
                },
            }
            (user_dir / "2026-04-08_12-01-02_ABC123.json").write_text(
                json.dumps(location_payload),
                encoding="utf-8",
            )
            (user_dir / "2026-04-09_12-01-02_XYZ789.json").write_text(
                json.dumps(location_payload),
                encoding="utf-8",
            )

            overrides_path = base / "locations.json"
            overrides_path.write_text(
                json.dumps(
                    {
                        "ABC123": {
                            "name": "Cafe Example",
                            "description": "Great brunch stop.",
                            "latitude": 38.7223,
                            "longitude": -9.1393,
                        }
                    }
                ),
                encoding="utf-8",
            )

            posts = discover_posts(archive_dir, {"ABC123": ["travels"], "XYZ789": ["food"]})
            posts = attach_locations(posts, load_location_overrides(overrides_path))

            output_path = base / "vault" / "exports" / "google-my-maps.csv"
            written = export_google_maps_csv(posts, output_path)
            self.assertEqual(written, 1)

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Name"], "Cafe Example")
            self.assertIn("Great brunch stop.", rows[0]["Description"])
            self.assertIn("ABC123 by someuser", rows[0]["Description"])
            self.assertIn("XYZ789 by someuser", rows[0]["Description"])
            self.assertEqual(rows[0]["Latitude"], "38.722300")
            self.assertEqual(rows[0]["Longitude"], "-9.139300")


if __name__ == "__main__":
    unittest.main()
