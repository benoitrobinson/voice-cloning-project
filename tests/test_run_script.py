import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_script import ScriptError, parse, parse_pace
from src.track_builder import chunk_text

BASIC = """# Tempo Run
pace: 5:00
voice: benoit
exaggeration: 0.7

@0:00
Alright {name}, we're moving.

Easy first kilometre.

@km 5
Five down.

@1:02:30
Nearly home.
"""


def test_parses_header_and_cues():
    s = parse(BASIC, variables={"name": "Benoit"})
    assert s.title == "Tempo Run"
    assert s.meta["voice"] == "benoit"
    assert [c.at for c in s.cues] == [0, 1500, 3750]


def test_km_cue_uses_pace():
    s = parse(BASIC, variables={"name": "Benoit"})
    assert s.cues[1].at == 5 * parse_pace("5:00")


def test_cli_pace_overrides_header():
    s = parse(BASIC, variables={"name": "Benoit"}, pace=parse_pace("6:00"))
    assert s.cues[1].at == 5 * 360


def test_blank_lines_become_separate_paragraphs():
    s = parse(BASIC, variables={"name": "Benoit"})
    assert s.cues[0].paragraphs == ["Alright Benoit, we're moving.", "Easy first kilometre."]


def test_variable_substitution():
    assert "Benoit" in parse(BASIC, variables={"name": "Benoit"}).cues[0].text


def test_missing_variable_is_reported_with_line():
    with pytest.raises(ScriptError, match="no value supplied for {name}"):
        parse(BASIC)


def test_km_cue_without_pace_is_rejected():
    with pytest.raises(ScriptError, match="no `pace:`"):
        parse("@km 5\nGo.\n")


def test_text_before_first_cue_is_rejected():
    with pytest.raises(ScriptError, match="before the first cue"):
        parse("@0:00 is not a marker because of this text\nGo.")


def test_out_of_order_cues_are_rejected():
    with pytest.raises(ScriptError, match="must increase"):
        parse("@10:00\nLater.\n\n@5:00\nEarlier.\n")


def test_empty_script_is_rejected():
    with pytest.raises(ScriptError, match="no spoken lines"):
        parse("# Title only\n")


def test_synth_params_are_floats():
    assert parse(BASIC, variables={"name": "B"}).synth_params == {"exaggeration": 0.7}


@pytest.mark.parametrize("value,expected", [("5:30", 330), ("4:00", 240), ("10:05", 605)])
def test_parse_pace(value, expected):
    assert parse_pace(value) == expected


def test_parse_pace_rejects_garbage():
    with pytest.raises(ScriptError):
        parse_pace("fast")


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("Keep going.", 280) == ["Keep going."]

    def test_splits_on_sentence_boundaries(self):
        text = " ".join(f"Sentence number {i} here." for i in range(20))
        chunks = chunk_text(text, 80)
        assert all(len(c) <= 80 for c in chunks)
        assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

    def test_sentence_longer_than_limit_is_split(self):
        text = "word, " * 100
        chunks = chunk_text(text, 50)
        assert all(len(c) <= 50 for c in chunks)

    def test_empty_text_yields_nothing(self):
        assert chunk_text("   ", 280) == []
