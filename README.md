# Running Coach — your own voice, on a timeline

Clone your voice from a short recording, then generate MP3 tracks of yourself
talking you through a run. Cues land at fixed times or distances, so at kilometre
15 you hear yourself telling you to keep going.

Runs locally on Apple Silicon. No account, no API key, no per-minute cost.

## Install

```bash
brew install ffmpeg                       # required for MP3 + loudness
uv venv --python 3.10 venv                # or: python3.10 -m venv venv
VIRTUAL_ENV=venv uv pip install -r requirements.txt
```

## Use it

**1. Record a reference.** Read [`data/scripts/recording_script.md`](data/scripts/recording_script.md)
— it explains why Section B (energetic) is the one you want for run tracks.

```bash
python scripts/record_voice.py
```

Any recording app works too. Mono, 24 kHz or better, 20–40 seconds, no music.

**2. Make it a voice.**

```bash
python scripts/make_track.py clone data/recordings/raw/section_b.wav --name Benoit
python scripts/make_track.py say "Two kilometres left, spend everything" --voice benoit
```

Listen to the sample. If it doesn't sound like you, re-record — the reference
dominates everything else.

**3. Build a track.**

```bash
python scripts/make_track.py make data/scripts/examples/long_run.run.md \
    --voice benoit --var name=Benoit --pace 5:30
```

Output lands in `data/tracks/`. Copy it to your phone and press play when you start.

## Writing a run script

```markdown
# Long Run 18k
pace: 5:30
exaggeration: 0.65

@0:00
Alright {name}. Eighteen kilometres.

First one is a warm-up. Slow.

@km 5
Five down. Keep the effort where it is.

@1:02:30
Nearly home.
```

- `@MM:SS`, `@H:MM:SS` — absolute time from the start of the track
- `@km N` — converted using `pace:` (or `--pace`, which overrides it)
- Blank line inside a cue = a short pause
- `{name}` and any other placeholder filled by `--var name=Benoit`
- Header keys `voice`, `pace`, `exaggeration`, `cfg_weight`, `temperature`

Cue times must increase. If one cue runs long, the next is pushed later and you
get a warning rather than overlapping speech.

## Tuning the delivery

| Flag | Default | Effect |
|---|---|---|
| `--exaggeration` | 0.65 | Emotional intensity. 0.3 calm, 0.8+ shouty. |
| `--cfg-weight` | 0.4 | Pacing. Lower = faster delivery. Drop it when raising exaggeration. |
| `--temperature` | 0.8 | Variability between takes. |

Synthesis is cached per chunk in `data/cache/`, keyed by text + voice + params.
Re-running a script after editing one cue only re-synthesizes that cue.

Output is compressed and loudness-normalized toward `target_lufs` (default -14,
lands around -15 on speech) at 44.1 kHz, so it holds up against road noise.
Set `TARGET_LUFS=-12` if you still can't hear it at pace.

## Speed — read this before you plan a session

Generation is slow. Measured on this 8 GB M2:

| | |
|---|---|
| Model load (once per run) | ~26 s |
| Synthesis | **~13× realtime on CPU** |
| Synthesis on MPS | ~59× realtime — *worse* |

So ~13 seconds of compute per second of speech. The bundled `long_run.run.md`
has roughly 4 minutes of speech in it, so budget **~45–60 minutes** for the first
build. After that it's cached: editing one cue only re-synthesizes that cue.

MPS being slower is not a mistake. On 8 GB the GPU path thrashes unified memory
and stalls 20 s+ on individual decode steps. `--device auto` therefore picks CPU
on machines under 16 GB. Force it either way with `--device mps` / `--device cpu`.

Practical approach: write the script, build it once while you do something else,
then iterate on individual cues cheaply.

## ElevenLabs instead (optional)

Better quality, costs money, needs a key:

```bash
export ELEVENLABS_API_KEY=...
python scripts/make_track.py --engine elevenlabs make <script> --voice benoit
```

## Layout

```
config/settings.py         env-overridable settings
src/voice_cloner.py        voice profiles + cached synthesis
src/run_script.py          run-script parser
src/track_builder.py       chunk -> synth -> timeline -> loudnorm -> MP3
src/audio_processor.py     reference-audio cleanup and quality checks
src/engines/               chatterbox (local), elevenlabs (API)
scripts/make_track.py      CLI: clone, voices, say, make
scripts/record_voice.py    interactive recorder
data/scripts/examples/     example run scripts
```

## Notes

- Chatterbox watermarks its output (inaudible, by Resemble AI). That's upstream.
- Voice profiles and cache stay on your machine; nothing is uploaded unless you
  opt into the ElevenLabs engine.
- Chatterbox is English-only. Multilingual needs a different engine.
