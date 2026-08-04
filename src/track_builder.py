"""Assemble a run script into a single playable track.

Pipeline: chunk each cue's text -> synthesize -> join -> lay out on a silent
timeline at the cue timestamps -> loudness-normalize -> MP3.
"""
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf

from src.run_script import Cue, RunScript

logger = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_PAUSE = 0.45  # seconds of silence between paragraphs within one cue
CHUNK_PAUSE = 0.18  # between sentence chunks

# Synthesized speech has a ~19dB crest factor, which leaves loudnorm unable to
# reach the target without breaching true peak. Gentle compression first both
# fixes that and is what you want for listening over traffic noise.
SPEECH_COMPRESSOR = "acompressor=threshold=-20dB:ratio=3:attack=5:release=120:makeup=2"

# loudnorm resamples to 192kHz internally for true-peak detection; without an
# explicit rate the output inherits it (a 24kHz source was landing at 48kHz).
OUTPUT_SAMPLE_RATE = 44100


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split into synthesis-sized chunks on sentence boundaries."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(text):
        # A single sentence longer than the limit gets split on commas, then
        # hard-wrapped as a last resort.
        pieces = [sentence] if len(sentence) <= max_chars else _split_long(sentence, max_chars)
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current = f"{current} {piece}"
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(sentence: str, max_chars: int) -> list[str]:
    parts, current = [], ""
    for piece in re.split(r"(?<=,)\s+", sentence):
        while len(piece) > max_chars:
            parts.append(piece[:max_chars])
            piece = piece[max_chars:]
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            parts.append(current)
            current = piece
    if current:
        parts.append(current)
    return parts


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(max(0, int(seconds * sr)), dtype=np.float32)


def render_cue(
    cue: Cue,
    cloner,
    profile: dict[str, Any],
    max_chunk_chars: int,
    params: dict[str, Any],
    on_chunk: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Synthesize one cue's paragraphs into a single audio segment."""
    sr = cloner.engine.sample_rate
    segments: list[np.ndarray] = []

    for p_idx, paragraph in enumerate(cue.paragraphs):
        if p_idx:
            segments.append(silence(PARAGRAPH_PAUSE, sr))
        chunks = chunk_text(paragraph, max_chunk_chars)
        for c_idx, chunk in enumerate(chunks):
            if c_idx:
                segments.append(silence(CHUNK_PAUSE, sr))
            if on_chunk:
                on_chunk(chunk)
            segments.append(cloner.synthesize(chunk, profile, **params))

    return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)


def build_timeline(
    script: RunScript,
    cloner,
    profile: dict[str, Any],
    max_chunk_chars: int,
    extra_params: Optional[dict[str, Any]] = None,
    min_duration: float = 0.0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[np.ndarray, int, list[str]]:
    """Render all cues onto one timeline. Returns (audio, sample_rate, warnings)."""
    params = {**script.synth_params, **(extra_params or {})}
    sr = cloner.engine.sample_rate
    warnings: list[str] = []

    rendered: list[tuple[float, np.ndarray]] = []
    total = len(script.cues)
    for idx, cue in enumerate(script.cues, 1):
        if on_progress:
            on_progress(idx, total, cue.text[:60])
        rendered.append((cue.at, render_cue(cue, cloner, profile, max_chunk_chars, params)))

    # Push cues later if the previous one is still talking.
    placed: list[tuple[int, np.ndarray]] = []
    cursor = 0
    for at, audio in rendered:
        start = int(at * sr)
        if start < cursor:
            warnings.append(
                f"Cue at {at / 60:.1f}min overlapped the previous one; "
                f"delayed by {(cursor - start) / sr:.1f}s"
            )
            start = cursor
        placed.append((start, audio))
        cursor = start + len(audio)

    total_samples = max(cursor, int(min_duration * sr))
    timeline = np.zeros(total_samples, dtype=np.float32)
    for start, audio in placed:
        timeline[start : start + len(audio)] += audio

    peak = float(np.abs(timeline).max())
    if peak > 1.0:
        timeline /= peak

    return timeline, sr, warnings


def export(
    audio: np.ndarray,
    sr: int,
    out_path: Path,
    target_lufs: float = -14.0,
    title: str = "Run",
    artist: str = "",
) -> Path:
    """Loudness-normalize and encode. MP3 via ffmpeg when available, else WAV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() != ".mp3" or not shutil.which("ffmpeg"):
        if out_path.suffix.lower() == ".mp3":
            logger.warning("ffmpeg not found; writing WAV instead of MP3")
            out_path = out_path.with_suffix(".wav")
        sf.write(out_path, audio, sr, subtype="PCM_16")
        return out_path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        sf.write(tmp_wav, audio, sr, subtype="PCM_16")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_wav),
            "-af", _filter_chain(tmp_wav, target_lufs),
            "-ar", str(OUTPUT_SAMPLE_RATE),
            "-c:a", "libmp3lame", "-q:a", "2",
            "-metadata", f"title={title}",
            "-metadata", f"artist={artist or 'Voice Clone'}",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        tmp_wav.unlink(missing_ok=True)

    return out_path


def _filter_chain(wav: Path, target_lufs: float) -> str:
    """Compressor + two-pass loudnorm.

    Single-pass loudnorm is a live estimator and undershoots: measured -16.9 LUFS
    against a -14 target. Measuring the compressed signal first and feeding the
    numbers back gets to about -15.
    """
    base = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
    single_pass = f"{SPEECH_COMPRESSOR},{base}"

    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav),
         "-af", f"{single_pass}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    try:
        # The JSON block is the last thing loudnorm writes to stderr.
        blob = probe.stderr[probe.stderr.rindex("{") : probe.stderr.rindex("}") + 1]
        m = json.loads(blob)
        return (
            f"{SPEECH_COMPRESSOR},{base}"
            f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
            f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
            f":offset={m['target_offset']}:linear=true"
        )
    except (ValueError, KeyError, json.JSONDecodeError):
        logger.warning("Loudness measurement failed; falling back to single-pass")
        return single_pass
