#!/usr/bin/env python3
"""CLI for the voice cloning running-track generator.

    python scripts/make_track.py clone data/recordings/raw/me.wav --name Benoit
    python scripts/make_track.py voices
    python scripts/make_track.py say "Keep going" --voice benoit
    python scripts/make_track.py make data/scripts/examples/long_run.run.md --voice benoit
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src import run_script
from src.track_builder import build_timeline, export
from src.voice_cloner import VoiceCloner


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("numba").setLevel(logging.WARNING)


def cmd_clone(args, settings):
    cloner = VoiceCloner(settings)
    profile = cloner.create_voice_clone(
        args.audio,
        args.name,
        clean=not args.no_clean,
        ref_seconds=args.ref_seconds,
    )
    print(f"\nVoice '{profile['voice_id']}' created from {profile['duration']}s of audio.")
    print(f"  reference: {cloner.reference_path(profile)}")
    print(f"  SNR: {profile['quality'].get('snr', float('nan')):.1f} dB")
    print(f"\nTry it:  python scripts/make_track.py say \"Let's go\" --voice {profile['voice_id']}")
    return 0


def cmd_voices(args, settings):
    voices = VoiceCloner(settings).list_voices()
    if not voices:
        print("No voices yet. Create one with:  make_track.py clone <recording.wav> --name <You>")
        return 1
    for v in voices:
        print(f"{v['voice_id']:<20} {v['duration']:>6.1f}s  {v['speaker_name']}  ({v['created_at'][:10]})")
    return 0


def cmd_say(args, settings):
    cloner = VoiceCloner(settings)
    profile = cloner.load_voice_profile(args.voice)
    params = {k: v for k, v in vars(args).items()
              if k in ("exaggeration", "cfg_weight", "temperature") and v is not None}

    started = time.time()
    audio = cloner.synthesize(args.text, profile, use_cache=not args.no_cache, **params)
    out = Path(args.output or settings.output_dir / "say.mp3")
    export(audio, cloner.engine.sample_rate, out, settings.target_lufs, "Sample", profile["speaker_name"])
    print(f"Wrote {out}  ({len(audio) / cloner.engine.sample_rate:.1f}s audio in {time.time() - started:.1f}s)")
    return 0


def cmd_make(args, settings):
    variables = dict(pair.split("=", 1) for pair in args.var)
    pace = run_script.parse_pace(args.pace) if args.pace else None

    script = run_script.load(args.script, variables=variables, pace=pace)

    cloner = VoiceCloner(settings)
    voice_id = args.voice or script.meta.get("voice")
    if not voice_id:
        print("No voice given. Use --voice <id> or add `voice: <id>` to the script header.", file=sys.stderr)
        return 1
    profile = cloner.load_voice_profile(voice_id)

    total_chars = sum(len(c.text) for c in script.cues)
    print(f"'{script.title}' — {len(script.cues)} cues, {total_chars} chars, voice '{voice_id}'")
    print("First run loads the model and can take a few minutes; synthesis is cached after that.\n")

    started = time.time()

    def progress(idx, total, preview):
        elapsed = time.time() - started
        print(f"  [{idx}/{total}] {elapsed:6.1f}s  {preview}...")

    extra = {k: v for k, v in vars(args).items()
             if k in ("exaggeration", "cfg_weight", "temperature") and v is not None}

    audio, sr, warnings = build_timeline(
        script, cloner, profile,
        max_chunk_chars=settings.max_chunk_chars,
        extra_params=extra,
        min_duration=args.duration * 60 if args.duration else 0.0,
        on_progress=progress,
    )

    out = Path(args.output) if args.output else settings.output_dir / f"{Path(args.script).stem}.mp3"
    out = export(audio, sr, out, settings.target_lufs, script.title, profile["speaker_name"])

    for w in warnings:
        print(f"  warning: {w}")
    print(f"\nWrote {out}  ({len(audio) / sr / 60:.1f} min, built in {time.time() - started:.0f}s)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="make_track", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--engine", help="chatterbox (default) or elevenlabs")
    parser.add_argument("--device", help="auto (default), mps, cpu, cuda")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("clone", help="create a voice from a reference recording")
    p.add_argument("audio", help="path to your recording")
    p.add_argument("--name", required=True, help="speaker name; becomes the voice id")
    p.add_argument("--no-clean", action="store_true", help="skip denoise/trim/normalize")
    p.add_argument("--ref-seconds", type=int, default=30, help="seconds of reference to keep")
    p.set_defaults(func=cmd_clone)

    p = sub.add_parser("voices", help="list saved voices")
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("say", help="synthesize one line (quick quality check)")
    p.add_argument("text")
    p.add_argument("--voice", required=True)
    p.add_argument("-o", "--output")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("make", help="build a full track from a run script")
    p.add_argument("script", help="path to a .run.md script")
    p.add_argument("--voice", help="voice id (or set `voice:` in the script header)")
    p.add_argument("-o", "--output")
    p.add_argument("--pace", help="MM:SS per km, for @km cues; overrides the script header")
    p.add_argument("--duration", type=float, help="pad the track to at least N minutes")
    p.add_argument("--var", action="append", default=[], metavar="KEY=VALUE",
                   help="fill {placeholders} in the script; repeatable")
    p.set_defaults(func=cmd_make)

    for p in (sub.choices["say"], sub.choices["make"]):
        p.add_argument("--exaggeration", type=float, help="emotional intensity, 0-1 (default 0.65)")
        p.add_argument("--cfg-weight", type=float, dest="cfg_weight", help="pacing, lower = faster (default 0.4)")
        p.add_argument("--temperature", type=float, help="variability (default 0.8)")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    settings = Settings()
    if args.engine:
        settings.engine = args.engine
    if args.device:
        settings.device = args.device

    try:
        return args.func(args, settings)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
