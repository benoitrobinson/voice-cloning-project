# Reference Recording Script

The model copies **how you sound in the reference**, not just your timbre. Record
this flat and sleepy and your motivational track will sound flat and sleepy.

## Setup

- Quiet room, soft furnishings. No fan, no traffic, no laptop fan on a hard desk.
- Mic 15–20 cm away, slightly off-axis so plosives don't pop.
- One take per section. Restart the section if you fumble; don't splice.
- Aim for 20–40 seconds per section. More is not better.

## Section A — Neutral (for calm, steady cues)

Read at your normal speaking pace, level and relaxed.

1. The quick brown fox jumps over the lazy dog near the riverbank.
2. She sells seashells by the seashore on sunny summer days.
3. Peter Piper picked a peck of pickled peppers perfectly.
4. Technology advances rapidly in the modern digital age.
5. I usually head out just after six, before the roads get busy.
6. Thirty-eight, ninety-one, fifteen, seventy-two, four hundred and six.

## Section B — Motivational (use this one for run tracks)

Same voice, but the energy you'd actually want in your ear at kilometre 15.
Volume up, pace brisk, mean it.

1. Come on, that's it, keep the rhythm going, you've got plenty left.
2. This is the part that counts. Right here. Don't back off now.
3. Head up, shoulders down, drive the arms, breathe out hard.
4. You've done this before and you'll do it again. Keep moving.
5. Two kilometres left. That's nothing. Spend everything you've got.
6. Last one. Go. Finish this properly.

## Then

```bash
python scripts/record_voice.py     # or record in any app, 24kHz+ mono wav
python scripts/make_track.py clone data/recordings/raw/section_b.wav --name Benoit
python scripts/make_track.py say "Keep going, you've got this" --voice benoit
```

If the sample sounds wrong, re-record rather than tweaking parameters — reference
quality dominates everything else.
