from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .config import TranscriptionConfig
from .models import InstagramPost


def transcript_path_for_video(video_path: Path) -> Path:
    return video_path.with_suffix(".transcript.txt")


def _emit(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)


def transcribe_posts(
    posts: list[InstagramPost],
    config: TranscriptionConfig,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> int:
    if not config.enabled or config.backend == "none":
        _emit(progress, "Transcription disabled; skipping.")
        return 0
    if config.backend != "faster-whisper":
        raise RuntimeError(
            f"Unsupported transcription backend: {config.backend}. Only 'faster-whisper' is implemented."
        )

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Reinstall with `pip install -e .[transcribe]`."
        ) from exc

    pending_videos = []
    skipped_existing = 0
    for post in posts:
        for video_path in post.video_files:
            output_path = transcript_path_for_video(video_path)
            if output_path.exists() and not config.overwrite:
                skipped_existing += 1
                continue
            pending_videos.append((post, video_path, output_path))

    if not pending_videos:
        _emit(
            progress,
            f"No videos need transcription. Existing transcripts skipped: {skipped_existing}.",
        )
        return 0

    _emit(
        progress,
        (
            f"Initializing Faster-Whisper model '{config.model}' "
            f"on {config.device} ({config.compute_type}). "
            "First run may download model files."
        ),
    )
    model = WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
    )
    _emit(
        progress,
        (
            f"Transcription model ready. Pending videos: {len(pending_videos)}. "
            f"Existing transcripts skipped: {skipped_existing}."
        ),
    )
    created = 0

    for index, (_, video_path, output_path) in enumerate(pending_videos, start=1):
        _emit(
            progress,
            f"[{index}/{len(pending_videos)}] Transcribing {video_path.name}",
        )
        segments, _ = model.transcribe(
            str(video_path),
            beam_size=config.beam_size,
            language=config.language,
        )
        lines = [segment.text.strip() for segment in segments if segment.text.strip()]
        text = "\n".join(lines).strip()
        if text:
            output_path.write_text(text + "\n", encoding="utf-8")
            created += 1
            _emit(progress, f"[{index}/{len(pending_videos)}] Wrote {output_path.name}")
        else:
            _emit(progress, f"[{index}/{len(pending_videos)}] No transcript text for {video_path.name}")

    return created
