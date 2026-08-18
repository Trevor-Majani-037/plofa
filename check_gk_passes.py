"""Check actual GK pass events in Tryox vs Oxton match."""
import json
with open('plofa_output/Tryox_City_vs_Oxton_MD34/Tryox_City_vs_Oxton_MD34.json') as f:
    data = json.load(f)

# Look for timeline or events list
for key in data:
    val = data[key]
    if isinstance(val, list) and len(val) > 100:
        print(f"Found large list: {key}, len={len(val)}")
        # Check if it contains events
        for item in val[:5]:
            if isinstance(item, dict):
                print(f"  Sample keys: {list(item.keys())[:10]}")
                break

# Check opta.activity for GK involvement
opta_activity = data.get('opta', {}).get('activity', [])
gk_events = [e for e in opta_activity if e.get('player') == 'Kal Grett' and e.get('involvements', 0) > 0]
print(f"\nKal Grett activity events with involvements > 0: {len(gk_events)}")
for e in gk_events[:10]:
    print(f"  {e.get('minute')}' involvements={e.get('involvements')}, ball_work={e.get('ball_work_m')}, distance={e.get('distance_m')}")

# Check opta.errors_to_shot_goal for GK passes
errors = data.get('opta', {}).get('errors_to_shot_goal', [])
gk_errors = [e for e in errors if e.get('player') == 'Kal Grett']
print(f"\nKal Grett error events: {len(gk_errors)}")
for e in gk_errors:
    print(f"  {e.get('minute')}' {e.get('error_type')} at x={e.get('x')}")
