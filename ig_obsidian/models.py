from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LocationRecord:
    name: str = ""
    description: str = ""
    address: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    source: str = "metadata"

    @property
    def map_key(self) -> tuple[str, ...]:
        if self.latitude is not None and self.longitude is not None:
            return (
                "coords",
                f"{self.latitude:.6f}",
                f"{self.longitude:.6f}",
            )
        if self.address.strip():
            return ("address", self.address.strip().lower())
        return ("name", self.name.strip().lower())


@dataclass
class CategoryResult:
    primary_category: str
    secondary_categories: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    reasoning: str = ""
    model: str = ""
    provider: str = "ollama"
    taxonomy_signature: str = ""
    categorized_at: str = ""


@dataclass
class InstagramPost:
    shortcode: str
    author: str
    date: Optional[datetime] = None
    kind: str = "post"
    url: str = ""
    caption: str = ""
    transcript: str = ""
    collections: List[str] = field(default_factory=list)
    media_files: List[Path] = field(default_factory=list)
    video_files: List[Path] = field(default_factory=list)
    image_files: List[Path] = field(default_factory=list)
    transcript_files: List[Path] = field(default_factory=list)
    caption_file: Optional[Path] = None
    metadata_file: Optional[Path] = None
    category_file: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    locations: List[LocationRecord] = field(default_factory=list)
    category_result: Optional[CategoryResult] = None

    @property
    def note_slug(self) -> str:
        if self.date is None:
            return self.shortcode
        return f"{self.date.date().isoformat()}_{self.shortcode}"
