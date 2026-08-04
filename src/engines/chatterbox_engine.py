"""Chatterbox (Resemble AI) local zero-shot cloning engine."""
import logging
import os
from pathlib import Path

import numpy as np

from .base import TTSEngine

logger = logging.getLogger(__name__)

# Chatterbox hits a few ops MPS doesn't implement; without this the model
# raises instead of falling back to CPU for those ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(device: str = "auto") -> str:
    """Pick a device.

    MPS is only worth it with headroom. Measured on an 8GB M2, MPS ran 59x
    realtime against CPU's 13.4x — the GPU path thrashes unified memory and
    stalls for 20s+ per decode step. Machines with >=16GB don't show this, so
    auto only takes MPS there. Force either with --device.
    """
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and _total_ram_gb() >= 16:
        return "mps"
    return "cpu"


def _total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError):
        return 0.0


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"

    # Defaults tuned for motivational delivery: exaggeration above the 0.5
    # baseline pushes energy up, lower cfg_weight keeps the pacing from
    # dragging when exaggeration is high (per Resemble's own guidance).
    DEFAULTS = {"exaggeration": 0.65, "cfg_weight": 0.4, "temperature": 0.8}

    def __init__(self, device: str = "auto"):
        self.device = resolve_device(device)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from chatterbox.tts import ChatterboxTTS

            logger.info("Loading Chatterbox on %s (first run downloads ~1GB)...", self.device)
            self._model = ChatterboxTTS.from_pretrained(self.device)
        return self._model

    @property
    def sample_rate(self) -> int:
        return self.model.sr

    def synthesize(self, text: str, ref_wav: Path, **params) -> np.ndarray:
        opts = {**self.DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
        wav = self.model.generate(text, audio_prompt_path=str(ref_wav), **opts)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
