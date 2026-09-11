"""
Diagnostic script to analyze y-coordinate (width) distribution in PLOFA matches.
Identifies if play is overly concentrated in the center vs wings/touchlines.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_match_width(json_path: str):
    """Analyze y-coordinate distribution from match JSON."""
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    events = data.get('timeline', [])  # Changed from 'events' to 'timeline'
    
    # Collect y-coordinates from events
    y_coords = []
    event_types = defaultdict(list)
    
    for evt in events:
        y = evt.get('y')
        if y is not None:
            y_coords.append(y)
            evt_type = evt.get('type', 'unknown')
            event_types[evt_type].append(y)
    
    if not y_coords:
        print("No y-coordinate data found in match JSON")
        return
    
    # Pitch dimensions: width = 68m, center at y=34.0
    # Left wing: 0-18m, Center: 18-50m, Right wing: 50-68m
    # Touchlines: 0-10m and 58-68m
    
    left_wing = [y for y in y_coords if y < 18]
    center = [y for y in y_coords if 18 <= y <= 50]
    right_wing = [y for y in y_coords if y > 50]
    
    left_touchline = [y for y in y_coords if y < 10]
    right_touchline = [y for y in y_coords if y > 58]
    
    total = len(y_coords)
    
    print(f"\n{'='*70}")
    print(f"PITCH WIDTH DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Match: {Path(json_path).stem}")
    print(f"Total events with coordinates: {total}")
    print(f"\nPitch zones (68m wide, center at 34m):")
    print(f"  Left Wing (0-18m):    {len(left_wing):4d} events ({len(left_wing)/total*100:5.1f}%)")
    print(f"  Center (18-50m):      {len(center):4d} events ({len(center)/total*100:5.1f}%)")
    print(f"  Right Wing (50-68m):  {len(right_wing):4d} events ({len(right_wing)/total*100:5.1f}%)")
    print(f"\nTouchline zones (extreme width):")
    print(f"  Left touchline (0-10m):   {len(left_touchline):4d} events ({len(left_touchline)/total*100:5.1f}%)")
    print(f"  Right touchline (58-68m): {len(right_touchline):4d} events ({len(right_touchline)/total*100:5.1f}%)")
    
    # Statistics
    mean_y = statistics.mean(y_coords)
    median_y = statistics.median(y_coords)
    stdev_y = statistics.stdev(y_coords) if len(y_coords) > 1 else 0
    
    print(f"\nStatistical summary:")
    print(f"  Mean y-coordinate:   {mean_y:.1f}m (center = 34.0m)")
    print(f"  Median y-coordinate: {median_y:.1f}m")
    print(f"  Std deviation:       {stdev_y:.1f}m")
    print(f"  Min/Max:             {min(y_coords):.1f}m / {max(y_coords):.1f}m")
    
    # Breakdown by event type
    print(f"\n{'─'*70}")
    print(f"Distribution by event type:")
    print(f"{'─'*70}")
    
    for evt_type in ['pass', 'carry', 'shot', 'cross', 'tackle', 'interception']:
        if evt_type in event_types:
            coords = event_types[evt_type]
            left = sum(1 for y in coords if y < 18)
            center_zone = sum(1 for y in coords if 18 <= y <= 50)
            right = sum(1 for y in coords if y > 50)
            total_type = len(coords)
            
            print(f"\n{evt_type.upper()} ({total_type} events):")
            print(f"  Left: {left:3d} ({left/total_type*100:5.1f}%)  "
                  f"Center: {center_zone:3d} ({center_zone/total_type*100:5.1f}%)  "
                  f"Right: {right:3d} ({right/total_type*100:5.1f}%)")
            
            if coords:
                print(f"  Mean: {statistics.mean(coords):.1f}m  "
                      f"Std: {statistics.stdev(coords) if len(coords) > 1 else 0:.1f}m")
    
    # Diagnosis
    print(f"\n{'='*70}")
    print(f"DIAGNOSIS:")
    print(f"{'='*70}")
    
    center_pct = len(center) / total * 100
    wings_pct = (len(left_wing) + len(right_wing)) / total * 100
    touchlines_pct = (len(left_touchline) + len(right_touchline)) / total * 100
    
    if center_pct > 70:
        print(f"⚠️  SEVERE CENTER BIAS: {center_pct:.1f}% of play in center (18-50m)")
        print(f"    Expected: ~50-60% center in balanced play")
    elif center_pct > 65:
        print(f"⚠️  MODERATE CENTER BIAS: {center_pct:.1f}% of play in center")
        print(f"    Expected: ~50-60% center in balanced play")
    else:
        print(f"✓  Center usage: {center_pct:.1f}% (reasonable)")
    
    if touchlines_pct < 5:
        print(f"⚠️  POOR TOUCHLINE USAGE: Only {touchlines_pct:.1f}% near touchlines")
        print(f"    Expected: ~8-12% for realistic wing play")
    elif touchlines_pct < 8:
        print(f"⚠️  LOW TOUCHLINE USAGE: {touchlines_pct:.1f}% near touchlines")
    else:
        print(f"✓  Touchline usage: {touchlines_pct:.1f}% (good)")
    
    if stdev_y < 12:
        print(f"⚠️  NARROW SPREAD: Std deviation only {stdev_y:.1f}m")
        print(f"    Expected: ~14-18m for good pitch width usage")
    elif stdev_y < 14:
        print(f"⚠️  MODERATE SPREAD: Std deviation {stdev_y:.1f}m")
    else:
        print(f"✓  Width spread: {stdev_y:.1f}m std deviation (good)")
    
    print(f"{'='*70}\n")
    
    return {
        'center_pct': center_pct,
        'wings_pct': wings_pct,
        'touchlines_pct': touchlines_pct,
        'mean_y': mean_y,
        'stdev_y': stdev_y,
    }


def analyze_multiple_matches(output_dir: str = "plofa_output"):
    """Analyze all matches in output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Output directory not found: {output_dir}")
        return
    
    json_files = list(output_path.glob("*/*.json"))
    
    if not json_files:
        print(f"No match JSON files found in {output_dir}")
        return
    
    print(f"Found {len(json_files)} match files to analyze\n")
    
    results = []
    for json_file in json_files[:5]:  # Analyze first 5 matches
        result = analyze_match_width(str(json_file))
        if result:
            results.append(result)
    
    if results:
        print(f"\n{'='*70}")
        print(f"AGGREGATE ANALYSIS ({len(results)} matches)")
        print(f"{'='*70}")
        
        avg_center = statistics.mean(r['center_pct'] for r in results)
        avg_touchlines = statistics.mean(r['touchlines_pct'] for r in results)
        avg_stdev = statistics.mean(r['stdev_y'] for r in results)
        
        print(f"Average center concentration: {avg_center:.1f}%")
        print(f"Average touchline usage:      {avg_touchlines:.1f}%")
        print(f"Average width spread (std):   {avg_stdev:.1f}m")
        
        if avg_center > 65:
            print(f"\n⚠️  CONSISTENT CENTER BIAS across matches")
        if avg_touchlines < 8:
            print(f"⚠️  CONSISTENTLY LOW touchline usage")
        if avg_stdev < 14:
            print(f"⚠️  CONSISTENTLY NARROW play distribution")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze specific match
        analyze_match_width(sys.argv[1])
    else:
        # Analyze multiple matches
        analyze_multiple_matches()
