"""Tests for the capture-series spec parser and plan reporting.

Runs as `uv run python scripts/test_capture_series.py` (or via pytest).
No hardware: exercises spec parsing, validation, the motion-key ban, and
the dry-run plan render.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from flakefinder.commands.capture_series import (
    SpecError,
    format_plan,
    load_series,
    parse_series,
)

BASE = {
    "defaults": {"lamp": 100, "exposure_ms": 10, "gain": 2.0, "white_balance": {"r": 1, "g": 1, "b": 1}, "frames": 3},
    "conditions": [
        {"name": "driftA_exp10"},
        {"name": "exp0.5", "exposure_ms": 0.5},
        {"name": "dark_exp1", "lamp": 0, "exposure_ms": 1, "frames": 5},
    ],
}


def _spec(**overrides):
    """BASE spec with top-level overrides applied."""
    return {**json.loads(json.dumps(BASE)), **overrides}


# --- Defaults merge with per-condition overrides ---


def test_defaults_merge_with_overrides():
    series = parse_series(BASE)
    drift, exp05, dark = series.conditions

    # Inherited from defaults
    assert drift.lamp == 100
    assert drift.exposure_ms == 10
    assert drift.gain == 2.0
    assert drift.frames == 3
    assert drift.white_balance == (1.0, 1.0, 1.0)

    # Per-condition overrides win, siblings still inherit
    assert exp05.exposure_ms == 0.5
    assert exp05.lamp == 100
    assert exp05.gain == 2.0

    assert dark.lamp == 0
    assert dark.exposure_ms == 1
    assert dark.frames == 5


def test_builtin_defaults_fill_unspecified_fields():
    series = parse_series({"conditions": [{"name": "solo"}]})
    (cond,) = series.conditions
    assert cond.lamp == 100
    assert cond.frames == 1
    assert cond.warmup_frames == 1
    assert cond.settle_s == 0.2
    assert cond.gamma == 1.0
    assert cond.auto_brightness is False
    assert cond.image_format == "png"


def test_notes_are_preserved():
    series = parse_series(_spec(notes={"chart_id": "cc-passport-01", "scope_id": "dm6m"}))
    assert series.notes["chart_id"] == "cc-passport-01"


# --- Unknown keys rejected (typo protection) ---


@pytest.mark.parametrize(
    ("where", "spec"),
    [
        ("condition", _spec(conditions=[{"name": "a", "exposure": 10}])),
        ("defaults", _spec(defaults={"exposure": 10})),
        ("top level", _spec(condition=[])),
    ],
)
def test_unknown_key_rejected(where, spec):
    with pytest.raises(SpecError, match="unknown key"):
        parse_series(spec)


def test_unknown_key_message_names_the_key():
    with pytest.raises(SpecError, match="'exposure'"):
        parse_series(_spec(conditions=[{"name": "a", "exposure": 10}]))


def test_white_balance_unknown_channel_rejected():
    with pytest.raises(SpecError, match="unknown key"):
        parse_series(_spec(conditions=[{"name": "a", "white_balance": {"r": 1, "g": 1, "b": 1, "w": 1}}]))


# --- Range validation ---


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"lamp": 101}, "lamp must be 0-100"),
        ({"lamp": -1}, "lamp must be 0-100"),
        ({"exposure_ms": 0}, "exposure_ms must be > 0"),
        ({"exposure_ms": -5}, "exposure_ms must be > 0"),
        ({"gain": 0}, "gain must be > 0"),
        ({"binning": 4}, "binning must be one of"),
        ({"binning": 0}, "binning must be one of"),
        ({"frames": 0}, "frames must be an integer >= 1"),
        ({"frames": 2.5}, "frames must be an integer >= 1"),
        ({"warmup_frames": -1}, "warmup_frames must be an integer >= 0"),
        ({"settle_s": -0.1}, "settle_s must be >= 0"),
        ({"gamma": 0}, "gamma must be > 0"),
        ({"format": "bmp"}, "format must be one of"),
        ({"quality": 0}, "quality must be an integer 1-100"),
        ({"auto_brightness": "no"}, "auto_brightness must be true or false"),
        ({"white_balance": {"r": 1, "g": 1}}, "missing 'b'"),
        ({"white_balance": {"r": 0, "g": 1, "b": 1}}, "white_balance.r must be > 0"),
        ({"white_balance": [1, 1, 1]}, "must be an object"),
    ],
)
def test_out_of_range_rejected(override, match):
    with pytest.raises(SpecError, match=match):
        parse_series(_spec(conditions=[{"name": "a", **override}]))


def test_missing_name_rejected():
    with pytest.raises(SpecError, match="'name' is required"):
        parse_series(_spec(conditions=[{"exposure_ms": 10}]))


def test_duplicate_names_rejected():
    with pytest.raises(SpecError, match="duplicate condition name 'a'"):
        parse_series(_spec(conditions=[{"name": "a"}, {"name": "a"}]))


def test_empty_conditions_rejected():
    with pytest.raises(SpecError, match="non-empty list"):
        parse_series(_spec(conditions=[]))


# --- Safety: motion keys are banned outright ---


@pytest.mark.parametrize("key", ["x", "y", "z", "objective_mag", "objective", "focus", "autofocus"])
def test_motion_key_rejected_in_condition(key):
    with pytest.raises(SpecError, match="never moves the stage"):
        parse_series(_spec(conditions=[{"name": "a", key: 100}]))


@pytest.mark.parametrize("key", ["z", "objective_mag"])
def test_motion_key_rejected_in_defaults(key):
    with pytest.raises(SpecError, match="never moves the stage"):
        parse_series(_spec(defaults={key: 100}))


def test_motion_key_rejected_at_top_level():
    with pytest.raises(SpecError, match="never moves the stage"):
        parse_series(_spec(objective_mag="2.5x"))


def test_motion_key_message_beats_unknown_key_message():
    """A revisit-style spec should get the safety message, not 'unknown key'."""
    with pytest.raises(SpecError) as excinfo:
        parse_series(_spec(conditions=[{"name": "a", "x": 5000, "typo": 1}]))
    assert "never moves the stage" in str(excinfo.value)


# --- Filenames and frame counts ---


def test_filenames_and_counts():
    series = parse_series(BASE)
    drift, exp05, dark = series.conditions

    assert drift.filenames() == ["driftA_exp10_f1.png", "driftA_exp10_f2.png", "driftA_exp10_f3.png"]
    assert exp05.filename(1) == "exp0.5_f1.png"
    assert len(dark.filenames()) == 5

    assert series.frame_count == 3 + 3 + 5
    assert series.warmup_count == 3  # builtin warmup_frames=1 per condition


def test_format_changes_extension():
    series = parse_series(_spec(defaults={"format": "tif"}, conditions=[{"name": "a"}]))
    assert series.conditions[0].filename(1) == "a_f1.tif"


def test_exposure_and_settle_totals():
    series = parse_series(
        {
            "defaults": {"frames": 2, "warmup_frames": 1, "settle_s": 0.2, "exposure_ms": 10},
            "conditions": [{"name": "a"}, {"name": "b", "exposure_ms": 20}],
        }
    )
    # a: 3 frames x 10ms, b: 3 frames x 20ms
    assert series.exposure_total_s == pytest.approx(0.03 + 0.06)
    assert series.settle_total_s == pytest.approx(0.4)


# --- Plan render / dry run ---


def test_format_plan_lists_every_condition():
    plan = format_plan(parse_series(BASE))
    for name in ("driftA_exp10", "exp0.5", "dark_exp1"):
        assert name in plan
    assert "3 conditions, 11 frames" in plan


def test_load_series_from_file_and_bad_json():
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.json"
        good.write_text(json.dumps(BASE))
        assert len(load_series(good).conditions) == 3

        bad = Path(tmp) / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(SpecError, match="invalid JSON"):
            load_series(bad)


def test_dry_run_end_to_end():
    """--dry-run renders the plan and exits without touching hardware."""
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "spec.json"
        spec.write_text(json.dumps(BASE))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flakefinder.commands.capture_series",
                "-o",
                str(Path(tmp) / "out"),
                "--spec",
                str(spec),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "no stage / Z / objective motion" in result.stdout
        assert "driftA_exp10" in result.stdout
        assert "(dry run -- exiting)" in result.stdout
        # Nothing was created
        assert not (Path(tmp) / "out").exists()


def test_dry_run_rejects_motion_spec():
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "spec.json"
        spec.write_text(json.dumps(_spec(conditions=[{"name": "a", "z": 19248.4}])))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flakefinder.commands.capture_series",
                "-o",
                str(Path(tmp) / "out"),
                "--spec",
                str(spec),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "never moves the stage" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
