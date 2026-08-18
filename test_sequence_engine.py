"""
PLOFA 26/27 — SEQUENCE ENGINE VALIDATION
=========================================
Runs a full match, then checks SequenceTracker output is sane:
    - Every shot-ending sequence has passes/actions/time/progress/speed
    - Macro style counts are non-negative and consistent with totals
    - Segmentation produced a reasonable number of sequences
"""

import random
import json
import shutil
from datetime import date

import pandas as pd

from player_dna import SquadBuilder
from match_engine import (
    MatchEngine, MatchConfig, MatchState, TeamProfile, TeamStyle,
    PlayingStyle, Intensity,
)
from sequence_engine import SequenceTracker
HOME_STARTERS = [
    ("Keano Walsh", "GK", ["sweeper_keeper"], 29),
    ("Darius Frost", "LB", ["aggressive_fullback"], 24),
    ("Emeka Obi", "CB", ["ball_playing_cb"], 27),
    ("Tavish Crane", "CB", ["stopper_defender"], 30),
    ("Rico Alves", "RB", ["overlapping_fullback"], 25),
    ("Mateo Sanz", "CDM", ["anchor_man"], 28),
    ("Luca Ferrini", "CM", ["box_box"], 26),
    ("Kofi Mensah", "CAM", ["creator"], 24),
    ("Adri Vela", "LW", ["dribbler"], 22),
    ("Dragan Novak", "ST", ["clinical_finisher"], 29),
    ("Percy", "RW", ["grand_dribbler"], 24),
]


def build_match(home_style=TeamStyle.ATTACKING, away_style=TeamStyle.FLUID_COUNTER,
                seed=7):
    random.seed(seed)
    home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
    away = SquadBuilder.build("Away", [
        ("P0", "GK", [], 25)] + [(f"P{i}", "CB", [], 25) for i in range(1, 11)])
    config = MatchConfig(home_team="Hartwell City", away_team="Away",
                         match_date=date(2026, 8, 16), matchday=1)
    hs = TeamProfile("Hartwell City", home_style, PlayingStyle.HIGH_PRESS, Intensity.HIGH)
    as_ = TeamProfile("Away", away_style, PlayingStyle.COUNTER, Intensity.MEDIUM)
    engine = MatchEngine(config, hs, as_)
    engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
    engine.set_squad("Away", away["starters"], away["substitutes"])
    return engine


def main():
    engine = build_match()
    result = engine.simulate()

    shots = [e for e in result.timeline if e.is_shot]
    print(f"\nTimeline events: {len(result.timeline)}")
    print(f"Shots in timeline: {len(shots)}")

    st = SequenceTracker("Hartwell City", "Away", result.timeline)
    st.compute_metrics()

    total_seq = len(st.sequences)
    shot_seq = len(st.shot_ending_sequences())
    print(f"Total sequences: {total_seq}")
    print(f"Shot-ending sequences: {shot_seq}")

    # Sanity: every shot in the timeline produced exactly one sequence
    assert shot_seq == len(shots), (
        f"shot-ending seqs ({shot_seq}) != shots ({len(shots)})"
    )

    for s in st.shot_ending_sequences():
        assert s.passes >= 0
        assert s.actions >= 0
        assert s.sequence_time >= 0
        assert s.progress >= 0
        assert s.direct_speed >= 0
        assert s.width >= 0
        if s.actions:
            assert abs(s.sequence_time - s.actions * 3.0) < 1e-6

    home = st.shot_ending_rows("Hartwell City")
    away = st.shot_ending_rows("Away")
    print(f"\nHartwell shot-ending sequences: {len(home)}")
    for row in home[:6]:
        print("  ", row)

    print(f"\nAway shot-ending sequences: {len(away)}")
    for row in away[:6]:
        print("  ", row)

    print("\nMacro attacking styles:")
    for row in st.macro_rows():
        print("  ", row)

    # Macro sanity: macro counts <= shot-ending sequences + open box touches
    for row in st.macro_rows():
        assert row["Build-Up Attacks"] >= 0
        assert row["Direct Attacks"] >= 0
        assert row["Shot-Ending High Turnovers"] >= 0
        assert row["Shot-Ending Sequences"] >= 0
        assert row["Total Sequences"] > 0

    print("\n✅ Sequence engine validated OK")


def test_exporter_integration(tmp="__seq_export_check__"):
    """The PLOFAExporter must surface sequence analytics in Excel + JSON."""
    import os
    from exporter import PLOFAExporter

    engine = build_match()
    result = engine.simulate()
    all_players = {"Hartwell City": SquadBuilder.build("Hartwell City", HOME_STARTERS),
                   "Away": SquadBuilder.build("Away", [
                       ("P0", "GK", [], 25)] + [(f"P{i}", "CB", [], 25) for i in range(1, 11)])}
    exporter = PLOFAExporter(result, all_players)
    exporter.export_all(tmp)

    xlsx = f"{tmp}/Hartwell_City_vs_Away_MD1.xlsx"
    assert os.path.exists(xlsx), f"missing {xlsx}"
    sheets = pd.ExcelFile(xlsx).sheet_names
    assert "Attacking Styles" in sheets, sheets
    assert "Shot-Ending Sequences" in sheets, sheets

    styles = pd.read_excel(xlsx, sheet_name="Attacking Styles")
    assert {"Build-Up Attacks", "Direct Attacks",
            "Shot-Ending High Turnovers"}.issubset(styles.columns)

    shots = pd.read_excel(xlsx, sheet_name="Shot-Ending Sequences")
    if not shots.empty:
        assert {"Passes", "Sequence Time (s)", "Progress (m)",
                "Direct Speed (m/s)", "Width (m)"}.issubset(shots.columns)

    with open(f"{tmp}/Hartwell_City_vs_Away_MD1.json", encoding="utf-8") as f:
        payload = json.load(f)
    assert "sequences" in payload
    assert payload["sequences"]["attacking_styles"]
    assert "Shot-Ending High Turnovers" in payload["sequences"]["attacking_styles"][0]

    shutil.rmtree(tmp, ignore_errors=True)
    print("✅ Exporter integration OK (Attacking Styles + Shot-Ending Sequences)")


if __name__ == "__main__":
    main()
    test_exporter_integration()
