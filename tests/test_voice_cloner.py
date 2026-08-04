import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src.engines.base import TTSEngine
from src.run_script import parse
from src.track_builder import build_timeline, export
from src.voice_cloner import VoiceCloner, slugify

SR = 24000


class FakeEngine(TTSEngine):
    """Returns 0.1s of tone per 10 chars, so durations are predictable."""

    name = "fake"

    def __init__(self):
        self.calls = []

    @property
    def sample_rate(self) -> int:
        return SR

    def synthesize(self, text, ref_wav, **params):
        self.calls.append(text)
        n = int(SR * 0.1 * max(1, len(text) / 10))
        return (0.2 * np.sin(np.linspace(0, 220 * 2 * np.pi, n))).astype(np.float32)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.voice_storage_path = tmp_path / "voices"
    s.cache_dir = tmp_path / "cache"
    s.output_dir = tmp_path / "tracks"
    s.recording_path = tmp_path / "rec"
    s.__post_init__()
    return s


@pytest.fixture
def reference_wav(tmp_path):
    """20s of amplitude-varying tone — long enough to pass the length check."""
    t = np.linspace(0, 20, 20 * SR, False)
    audio = (0.3 * np.sin(2 * np.pi * 160 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    path = tmp_path / "ref.wav"
    sf.write(path, audio, SR)
    return path


@pytest.fixture
def cloner(settings):
    return VoiceCloner(settings, engine=FakeEngine())


def test_slugify():
    assert slugify("Benoit R.") == "benoit-r"
    assert slugify("!!!") == "voice"


def test_create_voice_clone_writes_profile_and_reference(cloner, reference_wav):
    profile = cloner.create_voice_clone(reference_wav, "Benoit", clean=False)

    assert profile["voice_id"] == "benoit"
    assert cloner.reference_path(profile).exists()
    assert profile["duration"] == pytest.approx(20, abs=0.5)
    assert cloner.load_voice_profile("benoit")["speaker_name"] == "Benoit"


def test_reference_is_trimmed_to_ref_seconds(cloner, reference_wav):
    profile = cloner.create_voice_clone(reference_wav, "Benoit", clean=False, ref_seconds=5)
    assert profile["duration"] == pytest.approx(5, abs=0.2)


def test_audio_shorter_than_minimum_is_rejected(cloner, tmp_path, settings):
    short = tmp_path / "short.wav"
    sf.write(short, np.zeros(SR, dtype=np.float32), SR)  # 1 second
    with pytest.raises(ValueError, match="too short"):
        cloner.create_voice_clone(short, "Benoit", clean=True)


def test_missing_file_raises(cloner, tmp_path):
    with pytest.raises(FileNotFoundError):
        cloner.create_voice_clone(tmp_path / "nope.wav", "Benoit")


def test_unknown_voice_lists_available(cloner, reference_wav):
    cloner.create_voice_clone(reference_wav, "Benoit", clean=False)
    with pytest.raises(ValueError, match="Available: benoit"):
        cloner.load_voice_profile("nobody")


def test_synthesis_is_cached(cloner, reference_wav):
    profile = cloner.create_voice_clone(reference_wav, "Benoit", clean=False)

    first = cloner.synthesize("Keep going.", profile)
    second = cloner.synthesize("Keep going.", profile)

    assert cloner.engine.calls == ["Keep going."]  # second call served from disk
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_cache_key_separates_params(cloner, reference_wav):
    profile = cloner.create_voice_clone(reference_wav, "Benoit", clean=False)
    cloner.synthesize("Go.", profile, exaggeration=0.5)
    cloner.synthesize("Go.", profile, exaggeration=0.9)
    assert len(cloner.engine.calls) == 2


class TestTimeline:
    SCRIPT = "# T\n\n@0:00\nFirst cue here.\n\n@0:30\nSecond cue here.\n"

    def test_cues_land_at_their_timestamps(self, cloner, reference_wav):
        profile = cloner.create_voice_clone(reference_wav, "Benoit", clean=False)
        audio, sr, warnings = build_timeline(parse(self.SCRIPT), cloner, profile, 280)

        assert not warnings
        assert sr == SR
        # Silence between the end of cue 1 and the start of cue 2.
        assert np.abs(audio[int(20 * sr) : int(29 * sr)]).max() == 0
        assert np.abs(audio[int(30.05 * sr) : int(30.2 * sr)]).max() > 0

    def test_overlapping_cue_is_delayed_with_warning(self, cloner, reference_wav):
        profile = cloner.create_voice_clone(reference_wav, "Benoit", clean=False)
        # Two cues 1 second apart; the first is far longer than that.
        script = parse("# T\n\n@0:00\n" + "word " * 200 + "\n\n@0:01\nSecond.\n")
        _, _, warnings = build_timeline(script, cloner, profile, 280)

        assert any("overlapped" in w for w in warnings)

    def test_min_duration_pads_the_track(self, cloner, reference_wav):
        profile = cloner.create_voice_clone(reference_wav, "Benoit", clean=False)
        audio, sr, _ = build_timeline(parse(self.SCRIPT), cloner, profile, 280, min_duration=120)
        assert len(audio) == 120 * sr

    def test_export_falls_back_to_wav_without_ffmpeg(self, cloner, reference_wav, settings, monkeypatch):
        monkeypatch.setattr("src.track_builder.shutil.which", lambda _: None)
        out = export(np.zeros(SR, dtype=np.float32), SR, settings.output_dir / "t.mp3")
        assert out.suffix == ".wav" and out.exists()
