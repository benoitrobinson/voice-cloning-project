"""Voice cloning: turn a reference recording into a reusable voice, then synthesize with it.

Cloning here is zero-shot — there is no training step. A "voice profile" is a
cleaned reference wav plus metadata; the engine conditions on it at generation
time.
"""
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from src.audio_processor import AudioProcessor
from src.engines import TTSEngine, get_engine

logger = logging.getLogger(__name__)

# Chatterbox conditions on a short prompt; more than ~30s adds time without
# improving similarity. Raise via ref_seconds if your delivery varies a lot.
DEFAULT_REF_SECONDS = 30


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "voice"


class VoiceCloner:
    def __init__(self, settings, engine: Optional[TTSEngine] = None):
        self.settings = settings
        self.storage = Path(settings.voice_storage_path)
        self.cache_dir = Path(settings.cache_dir)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._engine = engine
        self.audio_processor = AudioProcessor(settings)

    @property
    def engine(self) -> TTSEngine:
        if self._engine is None:
            self._engine = get_engine(self.settings.engine, self.settings.device)
        return self._engine

    # ---------- voice profiles ----------

    def create_voice_clone(
        self,
        audio_path,
        speaker_name: str,
        clean: bool = True,
        ref_seconds: int = DEFAULT_REF_SECONDS,
    ) -> dict[str, Any]:
        """Build a voice profile from a reference recording."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        quality = self.audio_processor.validate_recording_quality(audio_path)
        if not quality["passed"]:
            for rec in quality["recommendations"]:
                logger.warning("Reference audio: %s", rec)

        if clean:
            audio = self.audio_processor.process_audio(audio_path)
            sr = self.settings.sample_rate
        else:
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

        if ref_seconds and len(audio) > ref_seconds * sr:
            logger.info("Trimming reference to first %ds", ref_seconds)
            audio = audio[: ref_seconds * sr]

        voice_id = slugify(speaker_name)
        ref_wav = self.storage / f"{voice_id}.wav"
        if ref_wav.exists():
            logger.warning("Overwriting existing voice %r", voice_id)
        sf.write(ref_wav, audio, sr, subtype="PCM_16")

        profile = {
            "voice_id": voice_id,
            "speaker_name": speaker_name,
            "reference_wav": ref_wav.name,
            "sample_rate": sr,
            "duration": round(len(audio) / sr, 2),
            "source_recording": str(audio_path),
            "cleaned": clean,
            "quality": quality["metrics"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.storage / f"{voice_id}.json").write_text(json.dumps(profile, indent=2))
        logger.info("Created voice %r (%.1fs reference)", voice_id, profile["duration"])
        return profile

    def load_voice_profile(self, voice_id: str) -> dict[str, Any]:
        path = self.storage / f"{voice_id}.json"
        if not path.exists():
            available = ", ".join(v["voice_id"] for v in self.list_voices()) or "none"
            raise ValueError(f"Voice {voice_id!r} not found. Available: {available}")
        return json.loads(path.read_text())

    def list_voices(self) -> list[dict[str, Any]]:
        out = []
        for f in sorted(self.storage.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable profile %s", f)
        return out

    def reference_path(self, profile: dict[str, Any]) -> Path:
        return self.storage / profile["reference_wav"]

    # ---------- synthesis ----------

    def synthesize(
        self, text: str, profile: dict[str, Any], use_cache: bool = True, **params
    ) -> np.ndarray:
        """Synthesize one chunk of text. Cached on disk by text + voice + params."""
        ref = self.reference_path(profile)
        if not ref.exists():
            raise FileNotFoundError(f"Reference wav missing for {profile['voice_id']}: {ref}")

        key = hashlib.sha256(
            "|".join(
                [
                    self.engine.name,
                    profile["voice_id"],
                    str(profile["created_at"]),
                    self.engine.param_fingerprint(**params),
                    text,
                ]
            ).encode()
        ).hexdigest()[:24]
        cached = self.cache_dir / f"{key}.wav"

        if use_cache and cached.exists():
            audio, _ = sf.read(cached, dtype="float32")
            return audio

        audio = self.engine.synthesize(text, ref, **params)
        sf.write(cached, audio, self.engine.sample_rate, subtype="FLOAT")
        return audio
