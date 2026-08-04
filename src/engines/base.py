"""TTS engine interface. Implementations wrap a specific model or API."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class TTSEngine(ABC):
    """A zero-shot voice-cloning TTS backend.

    Implementations synthesize `text` in the voice of the speaker heard in
    `ref_wav`, returning mono float32 audio at `self.sample_rate`.
    """

    name: str = "base"

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def synthesize(self, text: str, ref_wav: Path, **params) -> np.ndarray:
        ...

    def param_fingerprint(self, **params) -> str:
        """Stable string describing params that affect output, for cache keys."""
        return ",".join(f"{k}={params[k]}" for k in sorted(params))
