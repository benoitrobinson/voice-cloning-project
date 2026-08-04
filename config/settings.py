"""Configuration. Every field can be overridden by an env var of the same name (upper-cased)."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, cast

ROOT = Path(__file__).resolve().parent.parent

T = TypeVar("T")


def _env(name: str, default: T) -> T:
    raw = os.environ.get(name.upper())
    if raw is None:
        return default
    if isinstance(default, bool):
        return cast(T, raw.lower() in ("1", "true", "yes"))
    if isinstance(default, int):
        return cast(T, int(raw))
    if isinstance(default, Path):
        return cast(T, Path(raw))
    return cast(T, raw)


@dataclass
class Settings:
    # Reference-audio handling
    sample_rate: int = field(default_factory=lambda: _env("sample_rate", 24000))
    min_audio_length: int = field(default_factory=lambda: _env("min_audio_length", 5))
    max_audio_length: int = field(default_factory=lambda: _env("max_audio_length", 600))

    # Storage
    voice_storage_path: Path = field(default_factory=lambda: _env("voice_storage_path", ROOT / "data/voices"))
    recording_path: Path = field(default_factory=lambda: _env("recording_path", ROOT / "data/recordings"))
    cache_dir: Path = field(default_factory=lambda: _env("cache_dir", ROOT / "data/cache"))
    output_dir: Path = field(default_factory=lambda: _env("output_dir", ROOT / "data/tracks"))

    # Synthesis
    engine: str = field(default_factory=lambda: _env("engine", "chatterbox"))
    device: str = field(default_factory=lambda: _env("device", "auto"))
    max_chunk_chars: int = field(default_factory=lambda: _env("max_chunk_chars", 280))

    # Output loudness. -14 LUFS is the streaming standard and holds up over
    # road noise; drop to -12 if you still can't hear it at pace.
    target_lufs: float = field(default_factory=lambda: float(_env("target_lufs", "-14")))

    def __post_init__(self):
        for p in (self.voice_storage_path, self.recording_path, self.cache_dir, self.output_dir):
            Path(p).mkdir(parents=True, exist_ok=True)
