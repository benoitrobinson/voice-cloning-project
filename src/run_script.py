"""Parser for run scripts — the text format describing what gets said, and when.

Example:

    # Long Run 18k
    pace: 5:30
    exaggeration: 0.7

    @0:00
    Alright {name}, we're moving. Easy first kilometre.

    @km 5
    Five down. This is where it starts to feel real.

Cue markers are `@MM:SS`, `@H:MM:SS`, or `@km N` (converted using `pace`,
in min:sec per km). Everything until the next marker is that cue's text.
Blank lines inside a cue become short pauses. `{placeholders}` are filled
from `variables`.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CUE_TIME = re.compile(r"^@\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s*$")
CUE_KM = re.compile(r"^@\s*km\s*([\d.]+)\s*$", re.IGNORECASE)
META = re.compile(r"^([a-z_]+)\s*:\s*(.+)$", re.IGNORECASE)
PACE = re.compile(r"^(\d+):(\d{2})$")


class ScriptError(ValueError):
    pass


@dataclass
class Cue:
    at: float  # seconds from track start
    paragraphs: list[str]
    line_no: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.paragraphs)


@dataclass
class RunScript:
    title: str = "Run"
    cues: list[Cue] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def synth_params(self) -> dict[str, float]:
        """Engine params declared in the script header."""
        return {
            k: float(v)
            for k, v in self.meta.items()
            if k in ("exaggeration", "cfg_weight", "temperature")
        }


def parse_pace(value: str) -> float:
    """'5:30' -> 330.0 seconds per km."""
    m = PACE.match(value.strip())
    if not m:
        raise ScriptError(f"Bad pace {value!r}, expected MM:SS (e.g. 5:30)")
    return int(m.group(1)) * 60 + int(m.group(2))


def _parse_marker(line: str, pace: Optional[float], line_no: int) -> Optional[float]:
    m = CUE_TIME.match(line)
    if m:
        hours = int(m.group(1) or 0)
        return hours * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    m = CUE_KM.match(line)
    if m:
        if pace is None:
            raise ScriptError(
                f"Line {line_no}: '{line}' uses distance but no `pace:` was set "
                "in the script header (or --pace on the command line)"
            )
        return float(m.group(1)) * pace
    return None


def parse(text: str, variables: Optional[dict[str, str]] = None, pace: Optional[float] = None) -> RunScript:
    variables = variables or {}
    script = RunScript()
    current: Optional[Cue] = None
    buffer: list[str] = []
    in_header = True

    def flush_paragraph():
        if buffer and current is not None:
            current.paragraphs.append(" ".join(buffer))
        buffer.clear()

    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()

        if line.startswith("#"):
            if in_header and script.title == "Run":
                script.title = line.lstrip("#").strip() or "Run"
            continue

        if not line:
            flush_paragraph()
            continue

        at = _parse_marker(line, pace if pace is not None else script.meta.get("_pace_s"), line_no)
        if at is not None:
            flush_paragraph()
            if current is not None:
                script.cues.append(current)
            current = Cue(at=at, paragraphs=[], line_no=line_no)
            in_header = False
            continue

        if in_header:
            m = META.match(line)
            if m:
                key, value = m.group(1).lower(), m.group(2).strip()
                script.meta[key] = value
                if key == "pace":
                    script.meta["_pace_s"] = parse_pace(value)
                continue

        if current is None:
            raise ScriptError(
                f"Line {line_no}: text appears before the first cue marker. "
                "Start the script with a marker like `@0:00`."
            )

        try:
            buffer.append(line.format(**variables))
        except KeyError as e:
            raise ScriptError(f"Line {line_no}: no value supplied for {{{e.args[0]}}}") from None

    flush_paragraph()
    if current is not None:
        script.cues.append(current)

    script.cues = [c for c in script.cues if c.paragraphs]
    if not script.cues:
        raise ScriptError("Script contains no spoken lines")

    out_of_order = [
        (a.at, b.at) for a, b in zip(script.cues, script.cues[1:]) if b.at < a.at
    ]
    if out_of_order:
        raise ScriptError(f"Cue timestamps must increase; got {out_of_order[0]}")

    script.meta.pop("_pace_s", None)
    return script


def load(path, variables: Optional[dict[str, str]] = None, pace: Optional[float] = None) -> RunScript:
    return parse(Path(path).read_text(), variables=variables, pace=pace)
