"""ElevenLabs API engine. Optional paid alternative to the local model.

Set ELEVENLABS_API_KEY to use. Remote voices are created on first use from the
reference wav and cached locally by file hash so repeat runs reuse the voice.
"""
import hashlib
import io
import json
import logging
import os
from pathlib import Path

import numpy as np

from .base import TTSEngine

logger = logging.getLogger(__name__)

API = "https://api.elevenlabs.io/v1"


class ElevenLabsEngine(TTSEngine):
    name = "elevenlabs"

    def __init__(self, model_id: str = "eleven_multilingual_v2"):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        self.model_id = model_id
        self._voice_map_path = Path(__file__).resolve().parents[2] / "data/cache/elevenlabs_voices.json"

    @property
    def sample_rate(self) -> int:
        return 44100  # what we request below

    def _voice_map(self) -> dict:
        if self._voice_map_path.exists():
            return json.loads(self._voice_map_path.read_text())
        return {}

    def _remote_voice_id(self, ref_wav: Path) -> str:
        digest = hashlib.sha256(ref_wav.read_bytes()).hexdigest()[:16]
        cache = self._voice_map()
        if digest in cache:
            return cache[digest]

        import requests

        logger.info("Creating ElevenLabs voice from %s", ref_wav.name)
        resp = requests.post(
            f"{API}/voices/add",
            headers={"xi-api-key": self.api_key},
            data={"name": f"clone-{digest}"},
            files={"files": (ref_wav.name, ref_wav.read_bytes(), "audio/wav")},
            timeout=120,
        )
        resp.raise_for_status()
        voice_id = resp.json()["voice_id"]

        cache[digest] = voice_id
        self._voice_map_path.parent.mkdir(parents=True, exist_ok=True)
        self._voice_map_path.write_text(json.dumps(cache, indent=2))
        return voice_id

    def synthesize(self, text: str, ref_wav: Path, **params) -> np.ndarray:
        import requests
        import soundfile as sf

        voice_id = self._remote_voice_id(Path(ref_wav))
        resp = requests.post(
            f"{API}/text-to-speech/{voice_id}",
            headers={"xi-api-key": self.api_key, "Accept": "audio/wav"},
            json={
                "text": text,
                "model_id": self.model_id,
                "output_format": "pcm_44100",
                "voice_settings": {
                    "stability": params.get("stability", 0.4),
                    "similarity_boost": params.get("similarity_boost", 0.8),
                    "style": params.get("exaggeration", 0.5),
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        audio, _ = sf.read(io.BytesIO(resp.content), dtype="float32")
        return audio if audio.ndim == 1 else audio.mean(axis=1)
