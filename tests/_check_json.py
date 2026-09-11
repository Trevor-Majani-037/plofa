import json, os
f = 'plofa_output/Rodice_vs_Ganester_MD20/Rodice_vs_Ganester_MD20.json'
data = json.load(open(f, encoding='utf-8-sig'))
timeline = data.get('timeline', [])

# Check what's actually in the JSON
goals = [e for e in timeline if e.get('type') == 'GOAL']
print(f'Total timeline events: {len(timeline)}')
print(f'Goals in JSON: {len(goals)}')
for g in goals:
    print(f"  {g.get('minute')}' {g.get('player')} ({g.get('team')}) assist: {g.get('secondary_player')}")

# Check file modification time
mod_time = os.path.getmtime(f)
from datetime import datetime
print(f'\nJSON file modified: {datetime.fromtimestamp(mod_time)}')
