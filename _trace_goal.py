import json
f = 'plofa_output/Rodice_vs_Ganester_MD20/Rodice_vs_Ganester_MD20.json'
data = json.load(open(f, encoding='utf-8-sig'))
timeline = data.get('timeline', [])

# Find the 34' goal by Caelan Rustoh
goal_idx = None
for i, e in enumerate(timeline):
    if e.get('type') == 'GOAL' and e.get('minute') == 34:
        goal_idx = i
        break

if goal_idx is None:
    print('Goal not found')
else:
    print(f'Goal at index {goal_idx}: {timeline[goal_idx]}')
    print('\nEvents before goal (last 15):')
    for j in range(goal_idx - 1, max(-1, goal_idx - 15), -1):
        e = timeline[j]
        print(f"  {j}: {e.get('type'):25s} | {e.get('player', '?'):20s} | sec={e.get('secondary_player')} | team={e.get('team')}")
