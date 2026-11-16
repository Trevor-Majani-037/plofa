import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('plofa_output/Pearls_vs_Justice_MD06/Pearls_vs_Justice_MD6.json', encoding='utf-8') as f:
    d = json.load(f)

# Look at events in minute 47, 48, 49
evs = [e for e in d['timeline'] if e.get('minute') in (47, 48, 49) and e.get('player') in ('Juan Massey', 'Francis Bonadi')]
for e in evs:
    print(f"m={e.get('minute')} | {e.get('player'):<15} | {e.get('type'):<15} | ({e.get('x'):.1f}, {e.get('y'):.1f}) -> end=({e.get('end_x')}, {e.get('end_y')})")
