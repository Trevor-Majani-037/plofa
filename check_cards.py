import json
import os

# Find latest output
dirs = [d for d in os.listdir('plofa_output') if os.path.isdir(f'plofa_output/{d}')]
dirs.sort()
latest = dirs[-1]
print(f'Checking: plofa_output/{latest}')

for fname in os.listdir(f'plofa_output/{latest}'):
    if fname.endswith('.json'):
        with open(f'plofa_output/{latest}/{fname}') as f:
            d = json.load(f)
        players = []
        for n, s in d.get('player_stats', {}).items():
            fc = s.get('fouls_committed', 0)
            fw = s.get('fouls_won', 0)
            y = s.get('yellow_cards', 0)
            r = s.get('red_cards', 0)
            if fc or fw or y or r:
                players.append((fc, fw, y, r, n))
        players.sort(reverse=True)
        for fc, fw, y, r, n in players:
            print(f'  {n:<25s} fouls={fc} won={fw} Y={y} R={r}')
        mr = d.get('match_report', {})
        print(f'  Match cards: {mr.get("Cards", "?")}')
        print()
