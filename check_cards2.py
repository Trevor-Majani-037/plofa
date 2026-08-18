import json
with open(r'plofa_output\Red_Wolves_vs_Pearls_MD11\Red_Wolves_vs_Pearls_MD11.json') as f:
    d = json.load(f)
print("Keys:", list(d.keys()))
ps = d.get('player_stats', {})
if ps:
    for n, s in list(ps.items())[:3]:
        fc = s.get('fouls_committed', '?')
        fw = s.get('fouls_won', '?')
        y = s.get('yellow_cards', '?')
        print(f'  {n}: fouls_committed={fc}, fouls_won={fw}, Y={y}')
