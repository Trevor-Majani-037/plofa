"""calibrate_gps.py

Compare the virtual GPS physical-output per player against real-world
benchmarks drawn from published football GPS literature (Catapult /
STATSports / academic studies) so the simulation's movement model can be
tuned to realistic values.

Benchmarks (per-90 reference values, outfielders; pro level):

  Total distance       10.5 - 13.0 km  (avg ~11.5 km; CMs and wide players
                                        highest, CBs lowest)
  High-speed running   0.90 - 1.80 km  (> 19.8 km/h ≈ 5.5 m/s)
  Sprinting            0.30 - 0.80 km  (> 25.2 km/h ≈ 7.0 m/s)
  Sprint count         25 - 45
  Top speed            30.0 - 34.0 km/h (8.3 - 9.4 m/s), occasionally higher
  Distance per min     115 - 145 m/min

These are Euro top-5-league ranges (Di Salvo, Bradley, Carling, etc.).
The exact "right" number depends on position, playing style, and level —
the report flags OUTLIERS rather than declaring pass/fail.

Usage:
    python calibrate_gps.py <summary_csv> [--min-mins=10]
"""
import csv
import sys


# ── Real-world reference bands (per-90) ──────────────────────────────
REF = {
    "distance_m":        (10000.0, 13500.0, "total distance (m/90)"),
    "hsr_m":             (850.0, 1800.0, "high-speed running (m/90, >5.5 m/s)"),
    "sprint_m":          (300.0, 900.0, "sprint distance (m/90, >7.0 m/s)"),
    "sprint_count":      (25.0, 48.0, "sprint count (/90)"),
    "top_speed_mps":     (8.0, 9.6, "top speed (m/s)"),
}

# Reference per-minute for distance (used when a match has sub durations).
REF_DIST_PER_MIN = (115.0, 150.0)


def load_summary(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def est_high_speed_dist(row) -> float:
    """High-speed running distance directly from the GPS HSR accumulator."""
    return float(row.get("hsr_distance_m", 0) or 0)


def per90_dist(row) -> float:
    mins = float(row.get("duration_min", 0) or 0)
    if mins <= 0:
        return 0.0
    dist = float(row.get("distance_m", 0) or 0)
    return dist * (90.0 / mins)


def per90_metric(val, mins):
    if mins <= 0:
        return 0.0
    return val * (90.0 / mins)


def flag(value, band):
    lo, hi = band[0], band[1]
    if value < lo:
        return "LOW"
    if value > hi:
        return "HIGH"
    return "ok  "


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "output/gps_verify/gps_summary_m4242.csv"
    min_mins = 10.0
    for a in sys.argv[1:]:
        if a.startswith("--min-mins="):
            min_mins = float(a.split("=")[1])

    rows = load_summary(path)

    hdr = (f"{'Player':<20}{'Pos':<4}{'min':>6}{'dist/90 (km)':>13}"
           f"{'flag':>6}{'HSR m/90':>10}{'spr m/90':>10}"
           f"{'spr#/90':>9}{'tspd km/h':>10}{'flag':>6}")
    print(hdr)
    print("-" * len(hdr))

    problems = 0
    flagged = []
    for r in rows:
        if r.get("player") == "__ball__":
            continue
        mins = float(r.get("duration_min", 0) or 0)
        if mins < min_mins:
            continue
        dist90 = per90_dist(r)
        hsr90 = per90_metric(float(r.get("hsr_distance_m", 0) or 0), mins)
        spr_m90 = per90_metric(float(r.get("sprint_distance_m", 0) or 0), mins)
        sprints90 = per90_metric(float(r.get("sprint_count", 0) or 0), mins)
        tspd = float(r.get("top_speed_mps", 0) or 0)
        tspd_kmh = tspd * 3.6

        d90_km = dist90 / 1000.0
        f1 = flag(dist90, REF["distance_m"])
        f2 = flag(hsr90, REF["hsr_m"])
        f3 = flag(spr_m90, REF["sprint_m"])
        f4 = flag(sprints90, REF["sprint_count"])
        f5 = flag(tspd, REF["top_speed_mps"])
        flags = [f1, f2, f3, f4, f5]
        off = any(x in ("LOW", "HIGH") for x in flags)
        if off:
            flagged.append(r["player"])
            problems += 1

        print(f"{r['player']:<20}{r.get('position',''):<4}{mins:>6.0f}"
              f"{d90_km:>11.2f}{f1:>7}{hsr90:>8.0f}{spr_m90:>10.0f}"
              f"{sprints90:>9.1f}{tspd_kmh:>10.1f}{f5:>7}")

    if flagged:
        print(f"\n{problems} player(s) outside benchmark bands: "
              f"{', '.join(flagged)}")
    else:
        print("\nAll players within real-world benchmark bands.")


if __name__ == "__main__":
    main()