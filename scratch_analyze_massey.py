import json

with open('plofa_output/Pearls_vs_Justice_MD06/Pearls_vs_Justice_MD6.json', encoding='utf-8') as f:
    d = json.load(f)

massey_events = [e for e in d['timeline'] if e.get('player') == 'Juan Massey']
print(f"Total events for Massey: {len(massey_events)}")
for e in massey_events[70:130]:
    t = e.get('type')
    m = e.get('minute')
    x = e.get('x')
    y = e.get('y')
    end_x = e.get('end_x')
    end_y = e.get('end_y')
    print(f"m={m:2d} | {t:<15} | ({x:5.1f}, {y:5.1f}) -> ({end_x}, {end_y})")


