"""Engine registry."""
from .base import TTSEngine


def get_engine(name: str = "chatterbox", device: str = "auto") -> TTSEngine:
    if name == "chatterbox":
        from .chatterbox_engine import ChatterboxEngine

        return ChatterboxEngine(device=device)
    if name == "elevenlabs":
        from .elevenlabs_engine import ElevenLabsEngine

        return ElevenLabsEngine()
    raise ValueError(f"Unknown engine {name!r}. Available: chatterbox, elevenlabs")


__all__ = ["TTSEngine", "get_engine"]
