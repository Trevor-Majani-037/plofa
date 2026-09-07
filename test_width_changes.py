"""Quick test script to run 3 matches and check width distribution."""
import subprocess
import sys
from pathlib import Path
import json
import statistics

# Matches to test (edit teams in auto_run_match.py between runs)
test_matches = [
    ("Avada Zenith", "Pearls", 3),
    ("Claw", "Play City", 3),
    ("Ganester", "Uditon", 2),
]

print("This script needs you to manually edit auto_run_match.py for each test match")
print("Press Enter after updating the teams for each match, or Ctrl+C to skip to analysis")
input()

# Just analyze existing matches
print("\nAnalyzing ALL matches in plofa_output...")
results_old = []
results_new = []

for json_file in sorted(Path('plofa_output').glob('*/*.json')):
    try:
        with open(json_file, encoding='utf-8-sig') as f:
            data = json.load(f)
        
        timeline = data.get('timeline', [])
        y_coords = [evt['y'] for evt in timeline if evt.get('y') is not None]
        
        if not y_coords:
            continue
            
        center_pct = sum(1 for y in y_coords if 18 <= y <= 50) / len(y_coords) * 100
        touchlines_pct = sum(1 for y in y_coords if y < 10 or y > 58) / len(y_coords) * 100
        spread = statistics.stdev(y_coords)
        
        # Assume matches named with "MD01" or "MD1" in the LATEST folder are new tests
        # This is a hack - in practice you'd timestamp or flag them
        is_new = "Hartwell_City_vs_Thornfield" in str(json_file)
        
        result = {
            'file': json_file.stem,
            'center': center_pct,
            'touchlines': touchlines_pct,
            'spread': spread
        }
        
        if is_new:
            results_new.append(result)
        else:
            results_old.append(result)
            
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

print(f"\n{'='*90}")
print("BASELINE (matches before code changes):")
print(f"{'='*90}")
if results_old:
    print(f"Matches analyzed: {len(results_old)}")
    print(f"Average center concentration: {statistics.mean(r['center'] for r in results_old):.1f}%")
    print(f"Average touchline usage:      {statistics.mean(r['touchlines'] for r in results_old):.1f}%")
    print(f"Average width spread:         {statistics.mean(r['spread'] for r in results_old):.1f}m")

print(f"\n{'='*90}")
print("AFTER CHANGES (new test matches):")
print(f"{'='*90}")
if results_new:
    print(f"Matches analyzed: {len(results_new)}")
    print(f"Average center concentration: {statistics.mean(r['center'] for r in results_new):.1f}%")
    print(f"Average touchline usage:      {statistics.mean(r['touchlines'] for r in results_new):.1f}%")
    print(f"Average width spread:         {statistics.mean(r['spread'] for r in results_new):.1f}m")
    
    if results_old:
        old_center = statistics.mean(r['center'] for r in results_old)
        old_touch = statistics.mean(r['touchlines'] for r in results_old)
        old_spread = statistics.mean(r['spread'] for r in results_old)
        
        new_center = statistics.mean(r['center'] for r in results_new)
        new_touch = statistics.mean(r['touchlines'] for r in results_new)
        new_spread = statistics.mean(r['spread'] for r in results_new)
        
        print(f"\n{'='*90}")
        print("CHANGE:")
        print(f"{'='*90}")
        print(f"Center concentration: {new_center - old_center:+.1f}% (target: -5 to -10%)")
        print(f"Touchline usage:      {new_touch - old_touch:+.1f}% (target: +2 to +5%)")
        print(f"Width spread:         {new_spread - old_spread:+.1f}m (target: +1 to +3m)")
else:
    print("No new test matches found")

print(f"\n{'='*90}\n")
