import glob, json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "plofa_output", "_roster_gate")

v7 = []
t1 = []

print("ROOT:", ROOT)
print("exists:", os.path.exists(ROOT))

for folder in sorted(glob.glob(os.path.join(ROOT, "seed*_brains_v7"))):
    print("v7 folder:", folder)
    for j in sorted(glob.glob(os.path.join(folder, "*.json"))):
        print("  json:", j)
        with open(j, encoding="utf-8") as f:
            d = json.load(f)
        if "match" in d:
            m = d["match"]
            parts = [p for p in str(m.get("score", "")).replace("-", "–").split("–") if p.strip().isdigit()]
            score = [int(parts[0]), int(parts[1])] if len(parts) == 2 else [0, 0]
            v7.append({
                "h": score[0], "a": score[1],
                "xg": m.get("home_xg", 0.0),
                "away_xg": m.get("away_xg", 0.0),
                "poss": m.get("home_possession_pct", 0.0)
            })
            print("  added:", score)
            break

for folder in sorted(glob.glob(os.path.join(ROOT, "seed*_brains"))):
    if folder.endswith("_brains_v7"):
        continue
    print("T1 folder:", folder)
    for j in sorted(glob.glob(os.path.join(folder, "*.json"))):
        print("  json:", j)
        with open(j, encoding="utf-8") as f:
            d = json.load(f)
        if "match" in d:
            m = d["match"]
            parts = [p for p in str(m.get("score", "")).replace("-", "–").split("–") if p.strip().isdigit()]
            score = [int(parts[0]), int(parts[1])] if len(parts) == 2 else [0, 0]
            t1.append({
                "h": score[0], "a": score[1],
                "xg": m.get("home_xg", 0.0),
                "away_xg": m.get("away_xg", 0.0),
                "poss": m.get("home_possession_pct", 0.0)
            })
            print("  added:", score)
            break

print("v7:")
for r in v7:
    print(f"  {r['h']}-{r['a']}  xG {r['xg']:.2f}-{r['away_xg']:.2f}  poss {r['poss']:.1f}%")

print("\nT1:")
for r in t1:
    print(f"  {r['h']}-{r['a']}  xG {r['xg']:.2f}-{r['away_xg']:.2f}  poss {r['poss']:.1f}%")

print()
v7_w = sum(1 for r in v7 if r["h"] > r["a"])
v7_d = sum(1 for r in v7 if r["h"] == r["a"])
v7_l = sum(1 for r in v7 if r["h"] < r["a"])
t1_w = sum(1 for r in t1 if r["h"] > r["a"])
t1_d = sum(1 for r in t1 if r["h"] == r["a"])
t1_l = sum(1 for r in t1 if r["h"] < r["a"])

print(f"v7 record: {v7_w}W {v7_d}D {v7_l}L  GF {sum(r['h'] for r in v7)} GA {sum(r['a'] for r in v7)}")
print(f"T1 record: {t1_w}W {t1_d}D {t1_l}L  GF {sum(r['h'] for r in t1)} GA {sum(r['a'] for r in t1)}")
print(f"v7 mean xG {sum(r['xg'] for r in v7)/len(v7):.2f}  T1 mean xG {sum(r['xg'] for r in t1)/len(t1):.2f}")
print(f"v7 mean poss {sum(r['poss'] for r in v7)/len(v7):.1f}%  T1 mean poss {sum(r['poss'] for r in t1)/len(t1):.1f}%")

sys.exit(0)