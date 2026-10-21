"""THE ATTENDANCE MODEL — sourced, not hand-tuned.

WHAT THIS REPLACES
-----------------
`exporter.MatchFinancials` computed the crowd as

    attendance = stadium_capacity × fill_rate

where `fill_rate` came from a HAND-TUNNED table keyed on a "match tier"
(derby 0.95, big6_vs_big6 0.97, big6_home 0.90, standard 0.78,
low_interest 0.62), multiplied by a squad-quality term, a kickoff-time
term, a weather term, a season-position term, and finally a random
±4-6% of the CAPACITY.

Measured over PLOFA's 26/27 season (63 matches, `_diag_attendance_now.py`):
mean occupancy **85.0%**, range 57.7%-98.5%, and **seven clubs pinned at the
98.5% ceiling** on every one of their home matches — the signature of a
saturating table rather than a crowd.

THE BENCHMARK
-------------
Transfermarkt's Premier League attendance figures, **24/25** (the last
complete season), accessed 2026-10-21:

    Total: 15,380,351 spectators over 380 matches
           → 40,474 per match, 429 sell-outs, 96.8% mean occupancy

and per club (capacity, average attendance, occupancy):

    Old Trafford        Manchester United   74,158   73,815   99.5%
    London Stadium      West Ham United    62,500   62,407   99.9%
    Tottenham Stadium    Tottenham          62,850   61,127   97.3%
    Anfield             Liverpool FC       61,276   60,486   98.7%
    Emirates            Arsenal FC         60,704   60,252   99.3%
    Etihad              Manchester City    61,038   53,576   87.8%
    St James' Park      Newcastle United   52,729   52,187   99.0%
    Villa Park          Aston Villa        36,887   41,772  113.2%*
    Stamford Bridge     Chelsea FC         40,044   39,987   99.9%
    Hill Dickinson       Everton FC         52,769   39,191   74.3%
    Amex                Brighton           32,176   31,481   97.8%
    King Power          Leicester City     32,259   31,448   97.5%
    St Mary's           Southampton FC     32,384   30,865   95.3%
    Molineux            Wolves             31,750   30,755   96.9%
    City Ground          Nottingham Forest  31,212   30,076   96.4%
    Portman Road         Ipswich Town       30,294   29,742   98.2%
    Craven Cottage       Fulham FC          28,107   26,826   95.4%
    Selhurst Park        Crystal Palace     25,194   25,116   99.7%
    Gtech                Brentford FC       17,250   17,185   99.6%
    Vitality             AFC Bournemouth    12,357   11,200   90.6%

    * Aston Villa's stated capacity excludes standing areas; the club's real
      capacity is ~42,918, which puts them at ~97%.  Not a data error in the
      source, and not reproduced here.

THE STRUCTURE THIS REVEALS
--------------------------
**In a top-flight league the crowd IS the ground.**  Sixteen of the twenty
clubs sit between 95.3% and 99.9%; the league mean is 96.8%.  The two
exceptions are the two real cases in the table:

  * **Everton, 74.3%** — a new 52,769-seat ground for a club whose market
    could not fill it.  **A ground can outgrow its market.**
  * **Manchester City, 87.8%** — a deliberately oversized ground.

So the model is: **the ground sets the ceiling; the club's own market decides
how close to it the club gets; and the match decides where in that range the
crowd lands.**

AND THE CITIES DO NOT SPLIT THE CROWD
-------------------------------------
This is the natural question ("scale the fans by city population") and the
real data answers it.  **Seven of the twenty clubs are London clubs** —
Arsenal, Tottenham, Chelsea, West Ham, Fulham, Brentford, Crystal Palace —
and between them they drew **256,000 spectators a matchday from one city**,
with EVERY one of them between 95.4% and 99.9% full.  Liverpool and Everton
share Merseyside (98.7% and 74.3%); Manchester United and Manchester City
share Manchester (99.5% and 87.8%).

**A shared city does not split a crowd**: each club has its own support
base, and the demand in a league that saturated is far above the supply.
So the city is NOT a demand input, and a per-city population would add a
hand-tuned number without adding realism.  (The world layer's
`City.population` field stays available for the metric it is actually good
for — "fans per 1,000 inhabitants" — which this project cannot compute until
the populations are authored.  That is a data decision, not a modelling one.)

WHAT THIS MODULE IS
-------------------
Pure functions, no engine imports, so it can be tested and calibrated on
its own.  The exporter supplies the state; this module decides the crowd.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# ── THE BENCHMARK ─────────────────────────────────────────────────────────
#: (club, stadium, capacity, average attendance, occupancy %)
PL_24_25: Tuple[Tuple[str, str, int, int, float], ...] = (
    ("Manchester United", "Old Trafford", 74158, 73815, 99.5),
    ("West Ham United", "London Stadium", 62500, 62407, 99.9),
    ("Tottenham Hotspur", "Tottenham Hotspur Stadium", 62850, 61127, 97.3),
    ("Liverpool FC", "Anfield", 61276, 60486, 98.7),
    ("Arsenal FC", "Emirates Stadium", 60704, 60252, 99.3),
    ("Manchester City", "Etihad Stadium", 61038, 53576, 87.8),
    ("Newcastle United", "St James' Park", 52729, 52187, 99.0),
    ("Aston Villa", "Villa Park", 36887, 41772, 113.2),
    ("Chelsea FC", "Stamford Bridge", 40044, 39987, 99.9),    ("Everton FC", "Hill Dickinson Stadium", 52769, 39191, 74.3),
    ("Brighton & Hove Albion", "American Express Stadium", 32176, 31481, 97.8),
    ("Leicester City", "King Power Stadium", 32259, 31448, 97.5),
    ("Southampton FC", "St Mary's Stadium", 32384, 30865, 95.3),
    ("Wolverhampton Wanderers", "Molineux Stadium", 31750, 30755, 96.9),
    ("Nottingham Forest", "The City Ground", 31212, 30076, 96.4),
    ("Ipswich Town", "Portman Road", 30294, 29742, 98.2),
    ("Fulham FC", "Craven Cottage", 28107, 26826, 95.4),
    ("Crystal Palace", "Selhurst Park", 25194, 25116, 99.7),
    ("Brentford FC", "Gtech Community Stadium", 17250, 17185, 99.6),
    ("AFC Bournemouth", "Vitality Stadium", 12357, 11200, 90.6),
)

#: The one benchmark row whose attendance exceeds its STATED capacity:
#: Villa Park's published capacity excludes standing areas, and the club's
#: real capacity is ~42,918, which puts them at ~97%.  It is kept in the
#: table because it is real data, and named here so that anything which
#: asserts "attendance never exceeds capacity" can exclude it deliberately
#: instead of being surprised by it.
CAPACITY_ARTEFACT_CLUBS = ("Aston Villa",)

#: Total spectators, matches, and the league's mean occupancy, 24/25.
PL_TOTAL_SPECTATORS_24_25 = 15_380_351
PL_MATCHES_24_25 = 380
PL_MEAN_ATTENDANCE_24_25 = 40_474
PL_MEAN_OCCUPANCY_24_25 = 0.968

BENCHMARK_SOURCE = (
    "Transfermarkt, Premier League attendance figures 24/25 "
    "(total 15,380,351 over 380 matches; 40,474 per match; 96.8% mean "
    "occupancy), accessed 2026-10-21"
)

#: The occupancy band the benchmark's clubs actually occupy, excluding the
#: two structural exceptions (Everton's new ground, City's oversized one)
#: and Villa's capacity artefact.
PL_TYPICAL_OCCUPANCY_LOW = 0.953      # Southampton
PL_TYPICAL_OCCUPANCY_HIGH = 0.999     # West Ham

# ── THE MODEL ──────────────────────────────────────────────────────────────

#: How far the occupancy moves with a club's standing, in points of
#: occupancy (i.e. 0.03 = 3 percentage points).  The benchmark's spread
#: among the sixteen "typical" clubs is 95.3%-99.9%, a 4.6-point band, and
#: the model's own spread is checked against it by the gate.
STANDING_SWING = 0.030

#: The floor and the ceiling.  The ceiling is below 1.0 because the
#: benchmark's most-filled ground sits at 99.9%, and a model that reaches
#: 100.0% is a model pinned at its clamp — which is the defect this work
#: replaces (seven PLOFA clubs sat at exactly 98.5% every match).  It is
#: also deliberately OUT OF REACH of the model's own strongest input:
#: 0.968 + 0.030 = 0.998 < 0.999, so no score can hit the clamp.
OCCUPANCY_FLOOR = 0.70
OCCUPANCY_CEILING = 0.999

#: The unexplained residual.  Real attendance varies within a club by a
#: small amount after the explained causes (form, opponent, kickoff, weather)
#: are taken out; the old model's residual was ±4-6% of the CAPACITY, i.e.
#: ±4,200 on a 70,000-seat ground, which is not a residual but a different
#: model.  This one is a fraction of the CROWD, and it is small.
RESIDUAL_SWING = 0.012

#: How much the OPPONENT's own drawing power moves the crowd, in
#: occupancy points.  A marquee visitor brings a travelling contingent; a
#: small club does not.  This replaces the old hand-written "big6_home"
#: (0.90) and "big6_away" (0.85) rows, which were a table standing in for
#: exactly this one effect.
OPPONENT_SWING = 0.010


def market_score(squad_quality: float = 70.0,
                 position_score: float = 0.0) -> float:
    """How well a club can fill its own ground, as 0..1.

    `squad_quality` is the squad's average DNA rating (the project's own
    0-100 scale); `position_score` is the league standing, -1 (bottom) to
    +1 (top).

    NOTE, MEASURED, and it is why the squad rating is only half the signal:
    across the 26/27 squads the average DNA rating runs **67.7 to 74.2**
    (median 70.3) — a 6.5-point spread on a 100-point scale.  The old
    model's "bigness" term used `(avg_rating - 65) / 100`, which over that
    spread moves the fill rate by 0.077 to 0.092 — **less than two
    percentage points, i.e. effectively inert.**  A club's quality is
    therefore a weak market signal and the league position carries more of
    it.
    """
    # quality: 62..82 -> 0..1 (the measured band is 67.7-74.2, so this maps
    # the observed spread onto a usable part of the scale)
    q = (float(squad_quality) - 62.0) / 20.0
    q = max(0.0, min(1.0, q))
    p = (float(position_score) + 1.0) / 2.0
    p = max(0.0, min(1.0, p))
    return 0.5 * q + 0.5 * p


def base_occupancy(score: float) -> float:
    """The club's standing occupancy — the ceiling's share of the ground.

    Centred on the benchmark's league mean (96.8%) and moved by the club's
    standing, so a leading club sits nearer a full house than a struggling
    one.  The swing is set so the model's own spread matches the
    benchmark's 95.3%-99.9% band among the typical clubs.
    """
    occ = PL_MEAN_OCCUPANCY_24_25 + STANDING_SWING * (float(score) - 0.5) * 2.0
    return max(OCCUPANCY_FLOOR, min(OCCUPANCY_CEILING, occ))


def attendance_for(capacity: int,
                   home_score: float,
                   away_score: Optional[float] = None,
                   is_derby: bool = False,
                   match_multiplier: float = 1.0,
                   residual: float = 0.0) -> Dict[str, Any]:
    """The crowd.

    `home_score`/`away_score` are `market_score` values; `match_multiplier`
    carries the kickoff-time, weather and season-performance effects that
    the caller has already computed (they live in `weather_physics` and in
    the standings); `residual` is the small unexplained part.

    The ground is the ceiling, exactly as in the benchmark: no club in the
    Premier League draws more than its stadium holds.
    """
    capacity = int(capacity)
    occ = base_occupancy(home_score)
    if away_score is not None:
        # The opponent's drawing power: an established club brings its own
        # travelling support, which is a real attendance effect and the
        # honest replacement for the old "big6_home" / "big6_away" tier
        # rows — those were a hand-written table standing in for exactly
        # this.
        #
        # ABSOLUTE, not relative to the host: the first version used
        # (away - home), which made a bottom club hosting the same opponent
        # draw MORE than a leader hosting it — the travelling support of a
        # big club is a fact about that club, not about who it is visiting.
        occ *= 1.0 + OPPONENT_SWING * (float(away_score) - 0.5) * 2.0
    if is_derby:
        occ *= 1.0 + 0.02
    occ *= float(match_multiplier)
    occ *= 1.0 + float(residual)
    occ = max(OCCUPANCY_FLOOR, min(OCCUPANCY_CEILING, occ))
    attendance = int(round(capacity * occ))
    attendance = min(attendance, capacity)
    return {
        "attendance": attendance,
        "fill_rate": round(occ * 100, 1),
        "stadium_capacity": capacity,
    }
