import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('plofa_output/Pearls_vs_Justice_MD06/Pearls_vs_Justice_MD6.json', encoding='utf-8') as f:
    d = json.load(f)

events_m50 = [e for e in d['timeline'] if e.get('minute') == 50]
print(f"Events in minute 50 ({len(events_m50)}):")
for e in events_m50:
    t = e.get('type')
    p = e.get('player')
    x = e.get('x')
    y = e.get('y')
    rec = e.get('recipient')
    end_x = e.get('end_x')
    print(f"  {p:<18} | {t:<15} | ({x:5.1f}, {y:5.1f}) -> rec={rec} end_x={end_x}")
